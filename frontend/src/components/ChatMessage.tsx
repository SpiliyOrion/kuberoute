import { useState } from 'react'
import type { ConversationMessage } from '../types'

function MessageText({ content }: { content: string }) {
  const parts = content.split(/(```[\s\S]*?```)/g)
  return <div className="message-text">{parts.map((part, index) => part.startsWith('```') && part.endsWith('```')
    ? <pre key={index}><code>{part.slice(3, -3).replace(/^\w+\n/, '')}</code></pre>
    : <span key={index}>{part}</span>)}</div>
}

export function ChatMessage({ message }: { message: ConversationMessage }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(true)
    } catch {
      setCopied(false)
    }
  }
  return <article className={`message message-${message.role}`}>
    <div className="message-avatar" aria-hidden="true">{message.role === 'assistant' ? 'K' : 'Y'}</div>
    <div className="message-body">
      <div className="message-meta"><strong>{message.role === 'assistant' ? 'KubeRoute' : 'You'}</strong>
        {message.model && <span className="model-chip">Responded: {message.model}</span>}
        {message.rag && <span className="model-chip">Retrieval: {message.rag.retrievalMode}</span>}
        {message.finishReason === 'length' && <span className="length-chip">Output limit reached</span>}
      </div>
      <MessageText content={message.content} />
      {message.rag && <div className="rag-sources">
        <strong>Source references</strong>
        {message.rag.sources.length > 0 ? <ul>{message.rag.sources.map((source) => <li key={source.marker}>
          <strong>{source.marker}</strong>{' '}{source.reference ? <>
            {source.reference.authors}{source.reference.publicationYear !== null ? ` (${source.reference.publicationYear})` : ''} —{' '}
            <a href={source.reference.originalUrl} target="_blank" rel="noopener noreferrer" referrerPolicy="no-referrer">{source.reference.title}</a>
            {' '}({source.reference.sourceType}; {source.reference.catalogId}; chunk {source.chunkIndex})
          </> : <>Source <code>{source.sourceId}</code> · chunk <code>{source.chunkId}</code> · index {source.chunkIndex}</>}
        </li>)}</ul> : <p>No source references were returned. The answer has no cited support.</p>}
        <p>Citation markers identify retrieved passages; they do not prove that the answer&apos;s claims are grounded.</p>
      </div>}
      {message.role === 'assistant' && <button type="button" className="text-button copy-button" onClick={() => void copy()} aria-label="Copy response">{copied ? 'Copied' : 'Copy response'}</button>}
    </div>
  </article>
}
