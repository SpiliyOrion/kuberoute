import { useEffect, useRef, useState } from 'react'
import { discoverModels, GatewayError } from './api/gateway'
import { ChatMessage } from './components/ChatMessage'
import { MAX_MESSAGE_CHARS, useChat } from './hooks/useChat'
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
  const [mobileNav, setMobileNav] = useState(false)
  const chat = useChat()
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

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [chat.messages, chat.pending])

  const submit = () => {
    if (chat.send(model, draft)) setDraft('')
  }
  const reset = () => {
    chat.newChat()
    setDraft('')
    setMobileNav(false)
  }

  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav ? 'sidebar-open' : ''}`} aria-label="Sidebar">
      <div className="brand"><div className="brand-mark">K<span>↗</span></div><div><strong>KubeRoute</strong><small>INFERENCE GATEWAY</small></div></div>
      <button className="new-chat" type="button" onClick={reset}><span aria-hidden="true">＋</span> New Chat</button>
      <div className="sidebar-section-label">WORKSPACE</div>
      <div className="sidebar-item active"><span aria-hidden="true">◈</span> Current conversation</div>
      <div className="sidebar-bottom"><span className="status-dot" /> Local workspace <small>Chat stays in this browser tab</small></div>
    </aside>

    <main className="main-panel">
      <header className="topbar">
        <button className="menu-button" type="button" aria-label="Toggle sidebar" onClick={() => setMobileNav(!mobileNav)}>☰</button>
        <div className="header-title"><span className="eyebrow">WORKSPACE / CHAT</span><strong>Chat</strong></div>
        <div className="toolbar-right">
          <div className="connection"><span className={`status-dot ${modelError ? 'offline' : ''}`} />{loadingModels ? 'Connecting' : modelError ? 'Offline' : 'Gateway connected'}</div>
          <label className="model-selector"><span>MODEL</span><select aria-label="Select model" value={model} disabled={chat.pending} onChange={(event) => setModel(event.target.value)}>
            <option value="auto">Automatic routing</option>
            {models.map((id) => <option value={id} key={id}>{modelLabel(id)}</option>)}
          </select></label>
        </div>
      </header>

      {modelError && <div className="banner" role="alert"><span>{modelError}</span><button type="button" onClick={() => setReloadKey((key) => key + 1)}>Retry discovery</button></div>}
      <div className="conversation-scroll">
        <div className="conversation-content">
          {chat.messages.length === 0 ? <div className="empty-state">
            <div className="empty-symbol">K<span>↗</span></div>
            <div className="eyebrow">WELCOME TO KUBEROUTE</div>
            <h1>One gateway.<br /><em>Every model.</em></h1>
            <p>Choose a model, or let KubeRoute route your request. Ask anything to start a conversation.</p>
            <div className="empty-note">Automatic routing uses the gateway&apos;s configured policy. Simulated models are labeled in the selector.</div>
          </div> : <div className="message-list">{chat.messages.map((message) => <ChatMessage key={message.id} message={message} />)}
            {chat.pending && <div className="generating" role="status"><span className="pulse" /> Generating response…</div>}
          </div>}
          <div ref={bottomRef} />
        </div>
      </div>
      <div className="composer-area">
        {chat.error && <div className="request-error" role="alert"><span>{chat.error}</span>{chat.canRetry && <button type="button" onClick={chat.retry}>Retry request</button>}</div>}
        <div className="composer"><textarea aria-label="Message" placeholder="Ask KubeRoute anything…" value={draft} maxLength={MAX_MESSAGE_CHARS} disabled={chat.pending} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); submit() }
        }} /><button type="button" aria-label="Send message" disabled={chat.pending || !draft.trim()} onClick={submit}>↑</button></div>
        <div className="composer-footer"><span>Enter to send · Shift+Enter for a new line</span><span>{draft.length} / {MAX_MESSAGE_CHARS}</span></div>
      </div>
    </main>
  </div>
}
