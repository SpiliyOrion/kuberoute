import type { Completion, GatewayMessage, RagCollection, RagCompletion, RagSource, RetrievalMode, UploadedDocument } from '../types'

const MAX_MODEL_ID_CHARS = 128
const MAX_UPLOAD_BYTES = 1_048_576
const SAFE_UPLOAD_NAME = /^[A-Za-z0-9][A-Za-z0-9._ -]{0,119}$/

function validUploadName(name: string): boolean {
  return SAFE_UPLOAD_NAME.test(name) && !name.includes('..') && /\.(?:txt|md)$/i.test(name)
}

function parseUploadedDocument(value: unknown): UploadedDocument {
  if (!isRecord(value) || typeof value.id !== 'string' || !/^[a-f0-9]{64}$/.test(value.id) ||
    typeof value.name !== 'string' || !validUploadName(value.name) ||
    typeof value.byte_size !== 'number' || !Number.isSafeInteger(value.byte_size) || value.byte_size < 1 || value.byte_size > MAX_UPLOAD_BYTES ||
    typeof value.chunk_count !== 'number' || !Number.isSafeInteger(value.chunk_count) || value.chunk_count < 1 || value.chunk_count > 2000) {
    throw new GatewayError('response', 'The gateway returned an invalid uploaded document.')
  }
  return { id: value.id, name: value.name, byteSize: value.byte_size, chunkCount: value.chunk_count }
}

export async function listUploads(signal?: AbortSignal): Promise<UploadedDocument[]> {
  let response: Response
  try {
    response = await fetch('/v1/rag/uploads', { signal, cache: 'no-store' })
  } catch {
    if (signal?.aborted) throw signal.reason
    throw new GatewayError('network', 'Cannot reach the gateway to list uploaded documents.')
  }
  if (response.status === 503) throw new GatewayError('upstream', 'Document uploads are disabled or unavailable. Enable uploads and restart the local gateway.')
  if (!response.ok) throw new GatewayError('upstream', 'Could not list uploaded documents.')
  let body: unknown
  try { body = await response.json() } catch { throw new GatewayError('response', 'The gateway returned an invalid upload list.') }
  if (!isRecord(body) || body.object !== 'list' || !Array.isArray(body.data) || body.data.length > 100) {
    throw new GatewayError('response', 'The gateway returned an invalid upload list.')
  }
  const documents = body.data.map(parseUploadedDocument)
  if (new Set(documents.map((document) => document.id)).size !== documents.length) {
    throw new GatewayError('response', 'The gateway returned an invalid upload list.')
  }
  return documents
}

export async function uploadDocument(file: File, signal?: AbortSignal): Promise<UploadedDocument> {
  if (!validUploadName(file.name)) throw new GatewayError('input', 'Choose a .txt or .md file with a short name using letters, numbers, spaces, dots, underscores, or hyphens.')
  if (file.size < 1 || file.size > MAX_UPLOAD_BYTES) throw new GatewayError('input', 'Choose a nonempty file no larger than 1 MiB.')
  let response: Response
  try {
    response = await fetch('/v1/rag/uploads', {
      method: 'POST',
      headers: { 'Content-Type': file.name.toLowerCase().endsWith('.md') ? 'text/markdown; charset=utf-8' : 'text/plain; charset=utf-8', 'X-Document-Name': file.name },
      body: file,
      signal,
      cache: 'no-store',
    })
  } catch {
    if (signal?.aborted) throw signal.reason
    throw new GatewayError('network', 'The upload could not reach the gateway. Try again.')
  }
  if (!response.ok) {
    switch (response.status) {
      case 503: throw new GatewayError('upstream', 'Document uploads are disabled or unavailable. Enable uploads and restart the local gateway.')
      case 413: throw new GatewayError('input', 'The file or upload collection exceeds the gateway limit.')
      case 415: throw new GatewayError('input', 'Only UTF-8 .txt and .md files are supported.')
      case 422: throw new GatewayError('input', 'The gateway rejected this file. Check its name and UTF-8 text.')
      case 502: throw new GatewayError('upstream', 'Indexing failed. Check that the local embedding model is running.')
      default: throw new GatewayError('upstream', 'The gateway could not index this document.')
    }
  }
  let body: unknown
  try { body = await response.json() } catch { throw new GatewayError('response', 'The gateway returned an invalid upload result.') }
  return parseUploadedDocument(body)
}

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

function safePublicationUrl(value: string): boolean {
  if (value.length > 500 || [...value].some((character) => character.charCodeAt(0) <= 32 || character.charCodeAt(0) === 127)) return false
  try {
    const url = new URL(value)
    return url.protocol === 'https:' && !url.username && !url.password && !url.port &&
      !url.search && !url.hash && url.href === value &&
      /^[a-z0-9-]+(?:\.[a-z0-9-]+)+$/i.test(url.hostname) &&
      !/\.(?:local|localhost|internal|test|invalid)$/i.test(url.hostname) &&
      !/^\d+(?:\.\d+){3}$/.test(url.hostname)
  } catch {
    return false
  }
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
  collection: RagCollection = 'curated',
): Promise<RagCompletion> {
  let response: Response
  try {
    response = await fetch('/v1/rag/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, question, retrieval_mode: retrievalMode, top_k: topK, stream: false, ...(collection === 'uploads' ? { collection } : {}) }),
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
    const metadataFields = ['catalog_id', 'title', 'authors', 'publication_year', 'source_type', 'original_url']
    const hasMetadata = metadataFields.some((field) => source[field] !== undefined)
    let uploadedName: string | undefined
    if (source.uploaded_name !== undefined) {
      if (hasMetadata || typeof source.uploaded_name !== 'string' ||
        !/^[A-Za-z0-9][A-Za-z0-9._ -]{0,119}$/.test(source.uploaded_name) ||
        source.uploaded_name.includes('..')) {
        throw new GatewayError('response', 'The gateway returned an invalid RAG answer.')
      }
      uploadedName = source.uploaded_name
    }
    let reference: RagSource['reference']
    if (hasMetadata) {
      if (
        typeof source.catalog_id !== 'string' || !/^[a-z][a-z0-9_]{2,63}$/.test(source.catalog_id) ||
        typeof source.title !== 'string' || !source.title.trim() || source.title.length > 300 ||
        typeof source.authors !== 'string' || !source.authors.trim() || source.authors.length > 300 ||
        typeof source.source_type !== 'string' || !source.source_type.trim() || source.source_type.length > 64 ||
        typeof source.original_url !== 'string' || !safePublicationUrl(source.original_url) ||
        (source.publication_year !== undefined && source.publication_year !== null &&
          (typeof source.publication_year !== 'number' || !Number.isInteger(source.publication_year) || source.publication_year < 1800 || source.publication_year > new Date().getFullYear()))
      ) throw new GatewayError('response', 'The gateway returned an invalid RAG answer.')
      reference = {
        catalogId: source.catalog_id, title: source.title, authors: source.authors,
        publicationYear: typeof source.publication_year === 'number' ? source.publication_year : null,
        sourceType: source.source_type,
        originalUrl: source.original_url,
      }
    }
    seen.add(source.marker)
    sources.push({ marker: source.marker, sourceId: source.source_id, chunkId: source.chunk_id, chunkIndex: source.chunk_index, ...(reference ? { reference } : {}), ...(uploadedName ? { uploadedName } : {}) })
  }
  return {
    answer: body.answer as string,
    model: body.model as string | null,
    retrievalMode: body.retrieval_mode as RetrievalMode,
    sources,
    finishReason: body.finish_reason as 'stop' | 'length' | null,
  }
}
