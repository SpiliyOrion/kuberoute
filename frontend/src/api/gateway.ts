import type { Completion, GatewayMessage, RagCompletion, RagSource, RetrievalMode } from '../types'

const MAX_MODEL_ID_CHARS = 128

export class GatewayError extends Error {
  constructor(
    public readonly category: 'network' | 'model' | 'auto' | 'timeout' | 'upstream' | 'input' | 'response',
    message: string,
  ) {
    super(message)
    this.name = 'GatewayError'
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export async function discoverModels(signal?: AbortSignal): Promise<string[]> {
  let response: Response
  try {
    response = await fetch('/v1/models', { signal })
  } catch {
    throw new GatewayError('network', 'Cannot reach the gateway. Start the local gateway and try again.')
  }
  if (!response.ok) {
    throw new GatewayError('network', 'Model discovery failed. Check the local gateway and retry.')
  }
  let body: unknown
  try {
    body = await response.json()
  } catch {
    throw new GatewayError('response', 'The gateway returned an invalid model list.')
  }
  if (!isRecord(body) || body.object !== 'list' || !Array.isArray(body.data)) {
    throw new GatewayError('response', 'The gateway returned an invalid model list.')
  }
  const ids: string[] = []
  for (const item of body.data) {
    if (
      !isRecord(item) ||
      item.object !== 'model' ||
      typeof item.id !== 'string' ||
      !item.id ||
      item.id.length > MAX_MODEL_ID_CHARS ||
      item.id === 'auto' ||
      ids.includes(item.id)
    ) {
      throw new GatewayError('response', 'The gateway returned an invalid model list.')
    }
    ids.push(item.id)
  }
  return ids
}

export async function completeChat(
  model: string,
  messages: GatewayMessage[],
  signal: AbortSignal,
): Promise<Completion> {
  let response: Response
  try {
    response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, messages, max_tokens: 512, stream: false }),
      signal,
    })
  } catch {
    if (signal.aborted) throw signal.reason
    throw new GatewayError('network', 'Connection lost. Check the local gateway and retry.')
  }
  if (!response.ok) {
    switch (response.status) {
      case 404:
        throw new GatewayError('model', 'That model is unavailable. Refresh the model list or choose another.')
      case 503:
        throw new GatewayError('auto', 'Automatic routing has no eligible real model. Select an available model.')
      case 504:
        throw new GatewayError('timeout', 'Inference timed out. You can retry the request.')
      case 502:
        throw new GatewayError('upstream', 'The model server failed. You can retry or choose another model.')
      case 422:
        throw new GatewayError('input', 'The gateway rejected this request. Shorten the conversation and retry.')
      default:
        throw new GatewayError('upstream', 'The gateway could not complete this request.')
    }
  }
  let body: unknown
  try {
    body = await response.json()
  } catch {
    throw new GatewayError('response', 'The gateway returned an invalid completion.')
  }
  if (
    !isRecord(body) ||
    body.object !== 'chat.completion' ||
    typeof body.model !== 'string' ||
    !body.model ||
    !Array.isArray(body.choices) ||
    body.choices.length !== 1 ||
    !isRecord(body.choices[0]) ||
    body.choices[0].index !== 0 ||
    !isRecord(body.choices[0].message) ||
    body.choices[0].message.role !== 'assistant' ||
    typeof body.choices[0].message.content !== 'string' ||
    !body.choices[0].message.content.trim() ||
    (body.choices[0].finish_reason !== 'stop' && body.choices[0].finish_reason !== 'length')
  ) {
    throw new GatewayError('response', 'The gateway returned an invalid completion.')
  }
  return {
    content: body.choices[0].message.content,
    model: body.model,
    finishReason: body.choices[0].finish_reason,
  }
}

export async function completeRag(
  model: string,
  question: string,
  retrievalMode: RetrievalMode,
  topK: number,
  signal: AbortSignal,
): Promise<RagCompletion> {
  let response: Response
  try {
    response = await fetch('/v1/rag/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, question, retrieval_mode: retrievalMode, top_k: topK, stream: false }),
      signal,
    })
  } catch {
    if (signal.aborted) throw signal.reason
    throw new GatewayError('network', 'Connection lost. Check the local gateway and retry.')
  }
  if (!response.ok) {
    switch (response.status) {
      case 503:
        throw new GatewayError('upstream', 'RAG is disabled or its document index is unavailable. Check the local RAG setup.')
      case 413:
        throw new GatewayError('input', 'Retrieved context exceeds the gateway limit. Try fewer passages.')
      case 422:
        throw new GatewayError('input', 'The gateway rejected this RAG question or retrieval setting.')
      case 404:
        throw new GatewayError('model', 'That model is unavailable. Refresh the model list or choose another.')
      case 504:
        throw new GatewayError('timeout', 'Inference timed out. You can retry the question.')
      case 502:
        throw new GatewayError('upstream', 'Embedding or inference failed. You can retry the question.')
      default:
        throw new GatewayError('upstream', 'The gateway could not complete this RAG question.')
    }
  }
  let body: unknown
  try {
    body = await response.json()
  } catch {
    throw new GatewayError('response', 'The gateway returned an invalid RAG answer.')
  }
  if (
    !isRecord(body) ||
    typeof body.answer !== 'string' || !body.answer.trim() || body.answer.length > 32000 ||
    (body.model !== null && (typeof body.model !== 'string' || !body.model || body.model === 'auto' || body.model.length > MAX_MODEL_ID_CHARS)) ||
    (body.retrieval_mode !== 'hybrid' && body.retrieval_mode !== 'semantic' && body.retrieval_mode !== 'lexical') ||
    !Array.isArray(body.sources) || body.sources.length > 10 ||
    (body.finish_reason !== null && body.finish_reason !== 'stop' && body.finish_reason !== 'length') ||
    (body.model === null && (body.sources.length !== 0 || body.finish_reason !== null)) ||
    (body.model !== null && body.finish_reason === null)
  ) {
    throw new GatewayError('response', 'The gateway returned an invalid RAG answer.')
  }
  const sources: RagSource[] = []
  const seen = new Set<string>()
  for (const source of body.sources) {
    if (
      !isRecord(source) || typeof source.marker !== 'string' ||
      !/^\[(?:[1-9]|10)\]$/.test(source.marker) || seen.has(source.marker) ||
      typeof source.source_id !== 'string' || !/^[a-f0-9]{64}$/.test(source.source_id) ||
      typeof source.chunk_id !== 'string' || !/^[a-f0-9]{64}$/.test(source.chunk_id) ||
      typeof source.chunk_index !== 'number' || !Number.isSafeInteger(source.chunk_index) || source.chunk_index < 0
    ) {
      throw new GatewayError('response', 'The gateway returned an invalid RAG answer.')
    }
    seen.add(source.marker)
    sources.push({ marker: source.marker, sourceId: source.source_id, chunkId: source.chunk_id, chunkIndex: source.chunk_index })
  }
  return {
    answer: body.answer as string,
    model: body.model as string | null,
    retrievalMode: body.retrieval_mode as RetrievalMode,
    sources,
    finishReason: body.finish_reason as 'stop' | 'length' | null,
  }
}
