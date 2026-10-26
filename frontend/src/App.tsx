import { useEffect, useRef, useState } from 'react'
import { discoverModels, GatewayError, listUploads, uploadDocument } from './api/gateway'
import { ChatMessage } from './components/ChatMessage'
import { SettingHelp } from './components/SettingHelp'
import { MAX_MESSAGE_CHARS, useChat } from './hooks/useChat'
import { MAX_QUESTION_CHARS, useRag } from './hooks/useRag'
import type { RagCollection, RetrievalMode, UploadedDocument } from './types'
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
  const [collection, setCollection] = useState<RagCollection>('curated')
  const [uploads, setUploads] = useState<UploadedDocument[]>([])
  const [uploadsError, setUploadsError] = useState<string | null>(null)
  const [uploadNotice, setUploadNotice] = useState<string | null>(null)
  const [uploadsPending, setUploadsPending] = useState(false)
  const [uploadsLoading, setUploadsLoading] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadsReloadKey, setUploadsReloadKey] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)
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

  useEffect(() => {
    if (mode !== 'rag' || collection !== 'uploads') return
    const controller = new AbortController()
    setUploadsLoading(true)
    listUploads(controller.signal).then((documents) => {
      if (!controller.signal.aborted) { setUploads(documents); setUploadsError(null) }
    }).catch((cause: unknown) => {
      if (!controller.signal.aborted) {
        setUploads([])
        setUploadsError(cause instanceof GatewayError ? cause.message : 'Could not list uploaded documents.')
      }
    }).finally(() => { if (!controller.signal.aborted) setUploadsLoading(false) })
    return () => controller.abort()
  }, [mode, collection, uploadsReloadKey])

  const submitUpload = async () => {
    if (!selectedFile || uploadsPending || requestPending) return
    setUploadsPending(true)
    setUploadsError(null)
    setUploadNotice(null)
    try {
      const indexed = await uploadDocument(selectedFile)
      setSelectedFile(null)
      if (fileInputRef.current) fileInputRef.current.value = ''
      setUploads((current) => current.some((document) => document.id === indexed.id) ? current : [...current, indexed])
      setUploadNotice(`Indexed ${indexed.name} (${indexed.chunkCount} ${indexed.chunkCount === 1 ? 'chunk' : 'chunks'}).`)
      try {
        setUploads(await listUploads())
      } catch {
        setUploadsError('Document indexed, but the list could not be refreshed. Use Refresh list.')
      }
    } catch (cause) {
      setUploadsError(cause instanceof GatewayError ? cause.message : 'The document could not be uploaded.')
    } finally {
      setUploadsPending(false)
    }
  }

  const submit = () => {
    if (mode === 'chat' ? chat.send(model, draft) : rag.send(model, draft, retrievalMode, topK, collection)) setDraft('')
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
          <div className="model-selector"><div className="setting-label"><label htmlFor="mode-select">MODE</label><SettingHelp label="Mode"><p><strong>Chat:</strong> Standard multi-turn conversations with a selected language model.</p><p><strong>RAG:</strong> Retrieves document passages to help answer research questions. Each question is independent; earlier RAG messages are not sent again.</p></SettingHelp></div><select id="mode-select" aria-label="Mode" value={mode} disabled={requestPending || uploadsPending} onChange={(event) => { setMode(event.target.value as 'chat' | 'rag'); setDraft(''); setSelectedFile(null) }}>
            <option value="chat">Chat</option><option value="rag">RAG</option>
          </select></div>
          <div className="model-selector"><div className="setting-label"><label htmlFor="model-select">MODEL</label><SettingHelp label="Model"><p><strong>Automatic:</strong> The gateway chooses an eligible real model using its configured policy. It is not guaranteed to pick the fastest or most accurate model.</p><p><strong>Specific model:</strong> Sends the request to that model.</p><p><strong>Mock models:</strong> Simulated responses for testing, not genuine AI inference.</p></SettingHelp></div><select id="model-select" aria-label="Select model" value={model} disabled={requestPending} onChange={(event) => setModel(event.target.value)}>
            <option value="auto">Automatic routing</option>
            {models.map((id) => <option value={id} key={id}>{modelLabel(id)}</option>)}
          </select></div>
        </div>
      </header>

      {modelError && <div className="banner" role="alert"><span>{modelError}</span><button type="button" onClick={() => setReloadKey((key) => key + 1)}>Retry discovery</button></div>}
      {mode === 'rag' && <div className="rag-controls">
        <div className="rag-setting"><label htmlFor="collection-select">Collection</label><select id="collection-select" aria-label="Collection" value={collection} disabled={requestPending || uploadsPending} onChange={(event) => {
          const next = event.target.value as RagCollection
          if (next !== collection) { rag.newChat(); setDraft(''); setSelectedFile(null); setUploadNotice(null); setUploadsError(null); setCollection(next) }
        }}><option value="curated">Research collection</option><option value="uploads">Uploaded documents</option></select></div>
        <div className="rag-setting"><div className="setting-label"><label htmlFor="retrieval-select">Retrieval mode</label><SettingHelp label="Retrieval mode"><p><strong>Hybrid:</strong> Combines semantic vector search and BM25 with reciprocal rank fusion. A balanced default, but not always best.</p><p><strong>Semantic:</strong> Embedding similarity finds related concepts even when wording differs; it can miss precise terms.</p><p><strong>Lexical / BM25:</strong> Favors matching important words and phrases, useful for technical or exact terminology; it can miss paraphrases.</p></SettingHelp></div><select id="retrieval-select" aria-label="Retrieval mode" value={retrievalMode} disabled={requestPending} onChange={(event) => setRetrievalMode(event.target.value as RetrievalMode)}>
          <option value="hybrid">Hybrid search</option><option value="semantic">Semantic search</option><option value="lexical">Lexical / BM25 search</option>
        </select></div>
        <div className="rag-setting"><div className="setting-label"><label htmlFor="top-k-select">Top-K passages</label><SettingHelp label="Top-K passages"><p>Top-K limits the number of retrieved passages considered for an answer. The default of 5 balances focus and coverage.</p><p>Lower values give more focused context but may miss useful evidence. Higher values broaden coverage but may add irrelevant context. Context limits or duplicate passages can reduce what reaches the model or appears in citations.</p></SettingHelp></div><select id="top-k-select" aria-label="Top-K passages" value={topK} disabled={requestPending} onChange={(event) => setTopK(Number(event.target.value))}>
          {Array.from({ length: 10 }, (_, index) => index + 1).map((value) => <option key={value} value={value}>{value}</option>)}
        </select></div>
        <p>Each RAG question is independent. Earlier questions stay visible in this tab but are not sent as conversation history.</p>
      </div>}
      {mode === 'rag' && collection === 'uploads' && <section className="upload-panel" aria-label="Uploaded documents">
        <div className="upload-heading"><div><strong>Your uploaded documents</strong><p>UTF-8 .txt or .md, up to 1 MiB each. Indexed in memory until this gateway restarts.</p></div><button type="button" className="upload-refresh" disabled={uploadsPending || uploadsLoading} onClick={() => setUploadsReloadKey((key) => key + 1)}>Refresh list</button></div>
        <div className="upload-actions"><input ref={fileInputRef} aria-label="Choose document" type="file" accept=".txt,.md,text/plain,text/markdown" disabled={uploadsPending || requestPending} onChange={(event) => { setSelectedFile(event.target.files?.[0] ?? null); setUploadNotice(null) }} /><button type="button" disabled={!selectedFile || uploadsPending || requestPending} onClick={() => void submitUpload()}>{uploadsPending ? 'Indexing…' : 'Upload and index'}</button></div>
        {uploadsError && <p className="upload-error" role="alert">{uploadsError}</p>}
        {uploadNotice && <p className="upload-notice" role="status">{uploadNotice}</p>}
        {uploadsLoading ? <p className="upload-empty">Loading uploaded documents…</p> : uploads.length ? <ul className="upload-list">{uploads.map((document) => <li key={document.id}>{document.name} <span>· {document.chunkCount} {document.chunkCount === 1 ? 'chunk' : 'chunks'}</span></li>)}</ul> : !uploadsError && <p className="upload-empty">No uploaded documents are indexed in this gateway.</p>}
      </section>}
      <div className="conversation-scroll">
        <div className="conversation-content">
          {active.messages.length === 0 ? <div className="empty-state">
            <div className="empty-symbol">K<span>↗</span></div>
            <div className="eyebrow">WELCOME TO KUBEROUTE</div>
            <h1>{mode === 'chat' ? <>One gateway.<br /><em>Every model.</em></> : <>Ask your <em>local documents.</em></>}</h1>
            <p>{mode === 'chat' ? 'Choose a model, or let KubeRoute route your request. Ask anything to start a conversation.' : collection === 'uploads' ? 'Upload a text document above, then ask one question at a time about your uploaded collection.' : 'Ask one question at a time about explicitly configured local documents.'}</p>
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
