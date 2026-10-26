import { useCallback, useEffect, useRef, useState } from 'react'
import { completeRag, GatewayError } from '../api/gateway'
import type { ConversationMessage, RagCollection, RetrievalMode } from '../types'

export const MAX_QUESTION_CHARS = 2000
const MAX_VISIBLE_MESSAGES = 100
const REQUEST_TIMEOUT_MS = 180000

interface PendingQuestion {
  model: string
  question: string
  retrievalMode: RetrievalMode
  topK: number
  collection: RagCollection
}

export function useRag() {
  const [messages, setMessages] = useState<ConversationMessage[]>([])
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [retryQuestion, setRetryQuestion] = useState<PendingQuestion | null>(null)
  const messagesRef = useRef<ConversationMessage[]>([])
  const controllerRef = useRef<AbortController | null>(null)
  const generationRef = useRef(0)
  const nextIdRef = useRef(0)

  const updateMessages = useCallback((next: ConversationMessage[]) => {
    const visible = next.slice(-MAX_VISIBLE_MESSAGES)
    messagesRef.current = visible
    setMessages(visible)
  }, [])

  const runRequest = useCallback(async (request: PendingQuestion) => {
    if (controllerRef.current) return
    const controller = new AbortController()
    const generation = generationRef.current
    controllerRef.current = controller
    setPending(true)
    setError(null)
    setRetryQuestion(null)
    const timer = window.setTimeout(() => controller.abort(new DOMException('Timed out', 'TimeoutError')), REQUEST_TIMEOUT_MS)
    try {
      const result = await completeRag(request.model, request.question, request.retrievalMode, request.topK, controller.signal, request.collection)
      if (generation !== generationRef.current) return
      nextIdRef.current += 1
      updateMessages([...messagesRef.current, {
        id: String(nextIdRef.current), role: 'assistant', content: result.answer,
        model: result.model ?? undefined,
        finishReason: result.finishReason ?? undefined,
        rag: { retrievalMode: result.retrievalMode, sources: result.sources },
      }])
    } catch (cause) {
      if (generation !== generationRef.current) return
      if (controller.signal.aborted && controller.signal.reason instanceof DOMException && controller.signal.reason.name === 'TimeoutError') {
        setError('The RAG question took too long. You can retry it.')
      } else if (!controller.signal.aborted) {
        setError(cause instanceof GatewayError ? cause.message : 'The RAG question failed. You can retry it.')
      }
      setRetryQuestion(request)
    } finally {
      window.clearTimeout(timer)
      if (generation === generationRef.current) {
        controllerRef.current = null
        setPending(false)
      }
    }
  }, [updateMessages])

  const send = useCallback((model: string, input: string, retrievalMode: RetrievalMode, topK: number, collection: RagCollection): boolean => {
    const question = input.trim()
    if (controllerRef.current || !question || question.length > MAX_QUESTION_CHARS || !Number.isInteger(topK) || topK < 1 || topK > 10) return false
    nextIdRef.current += 1
    updateMessages([...messagesRef.current, { id: String(nextIdRef.current), role: 'user', content: question }])
    void runRequest({ model, question, retrievalMode, topK, collection })
    return true
  }, [runRequest, updateMessages])

  const retry = useCallback(() => {
    if (retryQuestion && !controllerRef.current) void runRequest(retryQuestion)
  }, [retryQuestion, runRequest])

  const newChat = useCallback(() => {
    generationRef.current += 1
    controllerRef.current?.abort()
    controllerRef.current = null
    updateMessages([])
    setError(null)
    setRetryQuestion(null)
    setPending(false)
  }, [updateMessages])

  useEffect(() => () => { controllerRef.current?.abort() }, [])

  return { messages, pending, error, canRetry: retryQuestion !== null, send, retry, newChat }
}
