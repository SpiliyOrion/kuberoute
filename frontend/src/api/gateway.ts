import type { Completion, GatewayMessage } from '../types'

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
