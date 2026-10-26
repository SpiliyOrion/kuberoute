export type Role = 'user' | 'assistant'

export interface ConversationMessage {
  id: string
  role: Role
  content: string
  model?: string
  finishReason?: 'stop' | 'length'
  rag?: { retrievalMode: RetrievalMode; sources: RagSource[] }
}

export type RetrievalMode = 'hybrid' | 'semantic' | 'lexical'
export type RagCollection = 'curated' | 'uploads'

export interface UploadedDocument {
  id: string
  name: string
  byteSize: number
  chunkCount: number
}

export interface RagSource {
  marker: string
  sourceId: string
  chunkId: string
  chunkIndex: number
  uploadedName?: string
  reference?: {
    catalogId: string
    title: string
    authors: string
    publicationYear: number | null
    sourceType: string
    originalUrl: string
  }
}

export interface RagCompletion {
  answer: string
  model: string | null
  retrievalMode: RetrievalMode
  sources: RagSource[]
  finishReason: 'stop' | 'length' | null
}

export interface GatewayMessage {
  role: Role
  content: string
}

export interface Completion {
  content: string
  model: string
  finishReason: 'stop' | 'length'
}
