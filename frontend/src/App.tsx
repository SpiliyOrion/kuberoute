import { useEffect, useRef, useState } from 'react'
import { discoverModels, GatewayError } from './api/gateway'
import { ChatMessage } from './components/ChatMessage'
import { MAX_MESSAGE_CHARS, useChat } from './hooks/useChat'
import { MAX_QUESTION_CHARS, useRag } from './hooks/useRag'
import type { RetrievalMode } from './types'
import './styles.css'

function modelLabel(id: string): string {
  if (id === 'auto') return 'Automatic routing'
  return id.startsWith('mock-') ? `${id} · simulated` : id
}

export default function App() {
  const [models, setModels] = useState<string[]>([])
  const [model, setModel] = useState('auto')
  const [modelError, setModelError] = useState<string | null>(null)
  const [loadingModels, setLoadingModels] = useState(true)
  const [reloadKey, setReloadKey] = useState(0)
  const [draft, setDraft] = useState('')
  const [mode, setMode] = useState<'chat' | 'rag'>('chat')
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>('hybrid')
  const [topK, setTopK] = useState(5)
  const [mobileNav, setMobileNav] = useState(false)
  const chat = useChat()
  const rag = useRag()
  const active = mode === 'chat' ? chat : rag
  const requestPending = chat.pending || rag.pending
  const inputLimit = mode === 'chat' ? MAX_MESSAGE_CHARS : MAX_QUESTION_CHARS
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const controller = new AbortController()
    setLoadingModels(true)
    discoverModels(controller.signal).then((ids) => {
      if (controller.signal.aborted) return
      setModels(ids)
      setModel((current) => current === 'auto' || ids.includes(current) ? current : 'auto')
      setModelError(null)
    }).catch((cause: unknown) => {
      if (controller.signal.aborted) return
      setModelError(cause instanceof GatewayError ? cause.message : 'Model discovery failed. Retry when the gateway is available.')
    }).finally(() => { if (!controller.signal.aborted) setLoadingModels(false) })
    return () => controller.abort()
  }, [reloadKey])

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [active.messages, active.pending])

  const submit = () => {
    if (mode === 'chat' ? chat.send(model, draft) : rag.send(model, draft, retrievalMode, topK)) setDraft('')
  }
  const reset = () => {
    active.newChat()
    setDraft('')
    setMobileNav(false)
  }

  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav ? 'sidebar-open' : ''}`} aria-label="Sidebar">
      <div className="brand"><div className="brand-mark">K<span>↗</span></div><div><strong>KubeRoute</strong><small>INFERENCE GATEWAY</small></div></div>
      <button className="new-chat" type="button" onClick={reset}><span aria-hidden="true">＋</span> New {mode === 'chat' ? 'Chat' : 'RAG session'}</button>
      <div className="sidebar-section-label">WORKSPACE</div>
      <div className="sidebar-item active"><span aria-hidden="true">◈</span> Current conversation</div>
      <div className="sidebar-bottom"><span className="status-dot" /> Local workspace <small>Questions and answers stay in this browser tab</small></div>
    </aside>

    <main className="main-panel">
      <header className="topbar">
        <button className="menu-button" type="button" aria-label="Toggle sidebar" onClick={() => setMobileNav(!mobileNav)}>☰</button>
        <div className="header-title"><span className="eyebrow">WORKSPACE / {mode.toUpperCase()}</span><strong>{mode === 'chat' ? 'Chat' : 'Document-grounded RAG'}</strong></div>
        <div className="toolbar-right">
          <div className="connection"><span className={`status-dot ${modelError ? 'offline' : ''}`} />{loadingModels ? 'Connecting' : modelError ? 'Offline' : 'Gateway connected'}</div>
          <label className="model-selector"><span>MODE</span><select aria-label="Mode" value={mode} disabled={requestPending} onChange={(event) => { setMode(event.target.value as 'chat' | 'rag'); setDraft('') }}>
            <option value="chat">Chat</option><option value="rag">RAG</option>
          </select></label>
          <label className="model-selector"><span>MODEL</span><select aria-label="Select model" value={model} disabled={requestPending} onChange={(event) => setModel(event.target.value)}>
            <option value="auto">Automatic routing</option>
            {models.map((id) => <option value={id} key={id}>{modelLabel(id)}</option>)}
          </select></label>
        </div>
      </header>

      {modelError && <div className="banner" role="alert"><span>{modelError}</span><button type="button" onClick={() => setReloadKey((key) => key + 1)}>Retry discovery</button></div>}
      {mode === 'rag' && <div className="rag-controls">
        <label>Retrieval mode <select aria-label="Retrieval mode" value={retrievalMode} disabled={requestPending} onChange={(event) => setRetrievalMode(event.target.value as RetrievalMode)}>
          <option value="hybrid">Hybrid search</option><option value="semantic">Semantic search</option><option value="lexical">Lexical / BM25 search</option>
        </select></label>
        <label>Top-K passages <select aria-label="Top-K passages" value={topK} disabled={requestPending} onChange={(event) => setTopK(Number(event.target.value))}>
          {Array.from({ length: 10 }, (_, index) => index + 1).map((value) => <option key={value} value={value}>{value}</option>)}
        </select></label>
        <p>Each RAG question is independent. Earlier questions stay visible in this tab but are not sent as conversation history.</p>
      </div>}
      <div className="conversation-scroll">
        <div className="conversation-content">
          {active.messages.length === 0 ? <div className="empty-state">
            <div className="empty-symbol">K<span>↗</span></div>
            <div className="eyebrow">WELCOME TO KUBEROUTE</div>
            <h1>{mode === 'chat' ? <>One gateway.<br /><em>Every model.</em></> : <>Ask your <em>local documents.</em></>}</h1>
            <p>{mode === 'chat' ? 'Choose a model, or let KubeRoute route your request. Ask anything to start a conversation.' : 'Ask one question at a time about explicitly configured local documents.'}</p>
            <div className="empty-note">{mode === 'chat' ? 'Automatic routing uses the gateway\'s configured policy. Simulated models are labeled in the selector.' : 'RAG must be enabled and indexed by the local gateway. References identify retrieved passages, not verified claims.'}</div>
          </div> : <div className="message-list">{active.messages.map((message) => <ChatMessage key={message.id} message={message} />)}
            {active.pending && <div className="generating" role="status"><span className="pulse" /> Generating response…</div>}
          </div>}
          <div ref={bottomRef} />
        </div>
      </div>
      <div className="composer-area">
        {active.error && <div className="request-error" role="alert"><span>{active.error}</span>{active.canRetry && <button type="button" onClick={active.retry}>Retry request</button>}</div>}
        <div className="composer"><textarea aria-label={mode === 'chat' ? 'Message' : 'RAG question'} placeholder={mode === 'chat' ? 'Ask KubeRoute anything…' : 'Ask about your local documents…'} value={draft} maxLength={inputLimit} disabled={requestPending} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); submit() }
        }} /><button type="button" aria-label="Send message" disabled={requestPending || !draft.trim()} onClick={submit}>↑</button></div>
        <div className="composer-footer"><span>Enter to send · Shift+Enter for a new line</span><span>{draft.length} / {inputLimit}</span></div>
      </div>
    </main>
  </div>
}
