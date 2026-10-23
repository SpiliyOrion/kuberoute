import { afterEach, describe, expect, it, vi } from 'vitest'
import { completeChat, discoverModels, GatewayError } from './gateway'

const originalFetch = globalThis.fetch
afterEach(() => { globalThis.fetch = originalFetch })

function completion(model = 'mock-small', content = 'Hello', finishReason = 'stop') {
  return { object: 'chat.completion', model, choices: [{ index: 0, message: { role: 'assistant', content }, finish_reason: finishReason }] }
}

describe('gateway HTTP client', () => {
  it('discovers registered models, excluding no server-defined automatic selector', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ object: 'list', data: [{ id: 'mock-small', object: 'model' }, { id: 'real-1', object: 'model' }] }), { status: 200 }))
    expect(await discoverModels()).toEqual(['mock-small', 'real-1'])
  })

  it('rejects malformed model discovery without showing response details', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ object: 'list', data: [{ id: 'auto', object: 'model' }] }), { status: 200 }))
    await expect(discoverModels()).rejects.toThrow('invalid model list')
  })

  it('sends the selected model and existing non-streaming chat schema', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(completion('mock-large')), { status: 200 }))
    globalThis.fetch = fetchMock
    expect(await completeChat('mock-large', [{ role: 'user', content: 'Hi' }], new AbortController().signal)).toEqual({ content: 'Hello', model: 'mock-large', finishReason: 'stop' })
    const [url, options] = fetchMock.mock.calls[0]
    expect(url).toBe('/v1/chat/completions')
    expect(JSON.parse(options.body)).toEqual({ model: 'mock-large', messages: [{ role: 'user', content: 'Hi' }], max_tokens: 512, stream: false })
  })

  it.each([[404, 'model'], [503, 'auto'], [504, 'timeout'], [502, 'upstream'], [422, 'input']])('maps HTTP %i to a safe %s error', async (status, category) => {
    globalThis.fetch = vi.fn().mockResolvedValue(new Response('sensitive upstream details', { status }))
    try { await completeChat('auto', [], new AbortController().signal) } catch (error) {
      expect(error).toBeInstanceOf(GatewayError)
      expect((error as GatewayError).category).toBe(category)
      expect(String(error)).not.toContain('sensitive')
      return
    }
    throw new Error('Expected request failure')
  })

  it.each([completion('x', ''), completion('x', 'hi', 'unknown'), { ...completion(), choices: [] }, { ...completion(), choices: [completion().choices[0], completion().choices[0]] }])('rejects malformed completions', async (body) => {
    globalThis.fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status: 200 }))
    await expect(completeChat('auto', [], new AbortController().signal)).rejects.toThrow('invalid completion')
  })
})
