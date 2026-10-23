export type Role = 'user' | 'assistant'

export interface ConversationMessage {
  id: string
  role: Role
  content: string
  model?: string
  finishReason?: 'stop' | 'length'
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
