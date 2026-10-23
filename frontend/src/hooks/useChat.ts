import { useCallback, useEffect, useRef, useState } from 'react'
import { completeChat, GatewayError } from '../api/gateway'
import type { ConversationMessage, GatewayMessage } from '../types'

export const MAX_MESSAGE_CHARS = 4000
export const MAX_HISTORY_MESSAGES = 16
export const MAX_REQUEST_CHARS = 16000
const MAX_VISIBLE_MESSAGES = 100
const REQUEST_TIMEOUT_MS = 180000

interface PendingRequest {
  model: string
  messages: GatewayMessage[]
}

function boundedHistory(messages: ConversationMessage[]): GatewayMessage[] {
  const selected: GatewayMessage[] = []
  let characters = 0
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (selected.length >= MAX_HISTORY_MESSAGES || characters + message.content.length > MAX_REQUEST_CHARS) break
    selected.unshift({ role: message.role, content: message.content })
    characters += message.content.length
  }
  return selected
}

export function useChat() {
  const [messages, setMessages] = useState<ConversationMessage[]>([])
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [retryRequest, setRetryRequest] = useState<PendingRequest | null>(null)
  const messagesRef = useRef<ConversationMessage[]>([])
  const controllerRef = useRef<AbortController | null>(null)
  const generationRef = useRef(0)
  const nextIdRef = useRef(0)

  const updateMessages = useCallback((next: ConversationMessage[]) => {
    const visible = next.slice(-MAX_VISIBLE_MESSAGES)
    messagesRef.current = visible
    setMessages(visible)
  }, [])

  const runRequest = useCallback(async (request: PendingRequest) => {
    if (controllerRef.current) return
    const controller = new AbortController()
    const generation = generationRef.current
    controllerRef.current = controller
    setPending(true)
    setError(null)
    setRetryRequest(null)
    const timer = window.setTimeout(() => controller.abort(new DOMException('Timed out', 'TimeoutError')), REQUEST_TIMEOUT_MS)
    try {
      const completion = await completeChat(request.model, request.messages, controller.signal)
      if (generation !== generationRef.current) return
      nextIdRef.current += 1
      updateMessages([...messagesRef.current, {
        id: String(nextIdRef.current),
        role: 'assistant',
        content: completion.content,
        model: completion.model,
        finishReason: completion.finishReason,
      }])
    } catch (cause) {
      if (generation !== generationRef.current) return
      if (controller.signal.aborted && controller.signal.reason instanceof DOMException && controller.signal.reason.name === 'TimeoutError') {
        setError('The request took too long. You can retry it.')
      } else if (!controller.signal.aborted) {
        setError(cause instanceof GatewayError ? cause.message : 'The request failed. You can retry it.')
      }
      setRetryRequest(request)
    } finally {
      window.clearTimeout(timer)
      if (generation === generationRef.current) {
        controllerRef.current = null
        setPending(false)
      }
    }
  }, [updateMessages])

  const send = useCallback((model: string, input: string): boolean => {
    const content = input.trim()
    if (controllerRef.current || !content || content.length > MAX_MESSAGE_CHARS) return false
    nextIdRef.current += 1
    const next = [...messagesRef.current, { id: String(nextIdRef.current), role: 'user' as const, content }]
    updateMessages(next)
    void runRequest({ model, messages: boundedHistory(next) })
    return true
  }, [runRequest, updateMessages])

  const retry = useCallback(() => {
    if (retryRequest && !controllerRef.current) void runRequest(retryRequest)
  }, [retryRequest, runRequest])

  const newChat = useCallback(() => {
    generationRef.current += 1
    controllerRef.current?.abort()
    controllerRef.current = null
    updateMessages([])
    setError(null)
    setRetryRequest(null)
    setPending(false)
  }, [updateMessages])

  useEffect(() => () => { controllerRef.current?.abort() }, [])

  return { messages, pending, error, canRetry: retryRequest !== null, send, retry, newChat }
}
