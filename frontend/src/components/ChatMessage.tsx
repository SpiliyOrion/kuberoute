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
        {message.finishReason === 'length' && <span className="length-chip">Output limit reached</span>}
      </div>
      <MessageText content={message.content} />
      {message.role === 'assistant' && <button type="button" className="text-button copy-button" onClick={() => void copy()} aria-label="Copy response">{copied ? 'Copied' : 'Copy response'}</button>}
    </div>
  </article>
}
