import { afterEach, describe, expect, it, vi } from 'vitest'
import { completeChat, completeRag, discoverModels, GatewayError, listUploads, uploadDocument } from './gateway'

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

describe('RAG HTTP client', () => {
  const result = {
    answer: 'A grounded answer [1].', model: 'qwen3:0.6b', retrieval_mode: 'hybrid',
    finish_reason: 'stop', sources: [{ marker: '[1]', source_id: 'a'.repeat(64), chunk_id: 'b'.repeat(64), chunk_index: 2 }],
  }

  it('accepts a safe uploaded name without inventing publication metadata', async () => {
    const source = { ...result.sources[0], uploaded_name: 'Synthetic Notes.md' }
    globalThis.fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ...result, sources: [source] }), { status: 200 }))
    const answer = await completeRag('auto', 'Question?', 'hybrid', 5, new AbortController().signal)
    expect(answer.sources[0].uploadedName).toBe('Synthetic Notes.md')
    expect(answer.sources[0].reference).toBeUndefined()
  })

  it.each(['../private.md', 'C:\\private.md', '<script>.md', 'name..md'])('rejects unsafe uploaded name %s', async (name) => {
    const source = { ...result.sources[0], uploaded_name: name }
    globalThis.fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ...result, sources: [source] }), { status: 200 }))
    await expect(completeRag('auto', 'Question?', 'hybrid', 5, new AbortController().signal)).rejects.toThrow('invalid RAG answer')
  })

  it('accepts verified metadata only with a safe HTTPS publication link', async () => {
    const source = { ...result.sources[0], catalog_id: 'verified_study', title: 'Synthetic study', authors: 'Research Group', publication_year: 2023, source_type: 'review', original_url: 'https://example.org/study' }
    globalThis.fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ...result, sources: [source] }), { status: 200 }))
    const answer = await completeRag('auto', 'Question?', 'hybrid', 5, new AbortController().signal)
    expect(answer.sources[0].reference).toEqual({ catalogId: 'verified_study', title: 'Synthetic study', authors: 'Research Group', publicationYear: 2023, sourceType: 'review', originalUrl: 'https://example.org/study' })
  })

  it.each(['javascript:alert(1)', 'file:///private', 'https://127.0.0.1/private', 'https://example.org@localhost/private', 'https://example.org/study#fragment'])('rejects unsafe publication URL %s', async (url) => {
    const source = { ...result.sources[0], catalog_id: 'verified_study', title: 'Synthetic study', authors: 'Research Group', source_type: 'review', original_url: url }
    globalThis.fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ...result, sources: [source] }), { status: 200 }))
    await expect(completeRag('auto', 'Question?', 'hybrid', 5, new AbortController().signal)).rejects.toThrow('invalid RAG answer')
  })

  it('rejects partial metadata without inventing a citation', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ...result, sources: [{ ...result.sources[0], title: 'Unverified study' }] }), { status: 200 }))
    await expect(completeRag('auto', 'Question?', 'hybrid', 5, new AbortController().signal)).rejects.toThrow('invalid RAG answer')
  })

  it('sends only the current question and bounded retrieval settings', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(result), { status: 200 }))
    globalThis.fetch = fetchMock
    expect(await completeRag('auto', 'Question?', 'hybrid', 5, new AbortController().signal)).toEqual({
      answer: 'A grounded answer [1].', model: 'qwen3:0.6b', retrievalMode: 'hybrid',
      finishReason: 'stop', sources: [{ marker: '[1]', sourceId: 'a'.repeat(64), chunkId: 'b'.repeat(64), chunkIndex: 2 }],
    })
    expect(fetchMock.mock.calls[0][0]).toBe('/v1/rag/chat/completions')
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      model: 'auto', question: 'Question?', retrieval_mode: 'hybrid', top_k: 5, stream: false,
    })
  })

  it('sends the uploaded collection only when selected', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(result), { status: 200 }))
    globalThis.fetch = fetchMock
    await completeRag('auto', 'Question?', 'hybrid', 5, new AbortController().signal, 'uploads')
    expect(JSON.parse(fetchMock.mock.calls[0][1].body).collection).toBe('uploads')
  })

  it('accepts a no-evidence abstention without inventing a model or source', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ answer: 'Insufficient information.', model: null, retrieval_mode: 'lexical', finish_reason: null, sources: [] }), { status: 200 }))
    const answer = await completeRag('auto', 'Question?', 'lexical', 1, new AbortController().signal)
    expect(answer.model).toBeNull()
    expect(answer.sources).toEqual([])
  })

  it.each([[503, 'disabled or its document index'], [413, 'context exceeds'], [422, 'rejected'], [502, 'Embedding or inference'], [504, 'timed out'], [500, 'could not complete']])('maps HTTP %i without exposing details', async (status, safeText) => {
    globalThis.fetch = vi.fn().mockResolvedValue(new Response('private path and upstream key', { status }))
    await expect(completeRag('auto', 'Question?', 'hybrid', 5, new AbortController().signal)).rejects.toThrow(safeText)
  })

  it.each([
    { ...result, answer: '' },
    { ...result, sources: [{ ...result.sources[0], marker: '[11]' }] },
    { ...result, sources: [result.sources[0], result.sources[0]] },
    { ...result, sources: [{ ...result.sources[0], chunk_index: -1 }] },
    { ...result, sources: [{ ...result.sources[0], source_id: 'C:\\private\\file.txt' }] },
    { ...result, model: null },
    { ...result, retrieval_mode: 'unknown' },
    { ...result, finish_reason: 'tool_calls' },
  ])('rejects malformed responses safely', async (body) => {
    globalThis.fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status: 200 }))
    await expect(completeRag('auto', 'Question?', 'hybrid', 5, new AbortController().signal)).rejects.toThrow('invalid RAG answer')
  })
})

describe('temporary document uploads', () => {
  const document = { id: 'a'.repeat(64), name: 'Safe Notes.md', byte_size: 18, chunk_count: 1 }

  it('sends a raw file with its safe name and validates the returned list', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify(document), { status: 201 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ object: 'list', data: [document] }), { status: 200 }))
    globalThis.fetch = fetchMock
    const file = new File(['Safe note content.'], document.name, { type: 'text/markdown' })
    expect(await uploadDocument(file)).toEqual({ id: document.id, name: document.name, byteSize: 18, chunkCount: 1 })
    expect(fetchMock.mock.calls[0][0]).toBe('/v1/rag/uploads')
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: 'POST', body: file, headers: { 'Content-Type': 'text/markdown; charset=utf-8', 'X-Document-Name': 'Safe Notes.md' } })
    expect(await listUploads()).toEqual([{ id: document.id, name: document.name, byteSize: 18, chunkCount: 1 }])
  })

  it('rejects unsafe or large files locally and sanitizes server errors', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('private path', { status: 502 }))
    globalThis.fetch = fetchMock
    await expect(uploadDocument(new File(['x'], '../private.md'))).rejects.toThrow('short name')
    await expect(uploadDocument(new File(['x'], 'report.pdf'))).rejects.toThrow('.txt or .md')
    await expect(uploadDocument(new File(['x'.repeat(1_048_577)], 'large.txt'))).rejects.toThrow('1 MiB')
    expect(fetchMock).not.toHaveBeenCalled()
    await expect(uploadDocument(new File(['valid text'], 'safe.txt'))).rejects.toThrow('Indexing failed')
    await expect(listUploads()).rejects.toThrow('Could not list uploaded documents')
  })

  it('rejects malformed upload metadata rather than displaying it', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ object: 'list', data: [{ ...document, name: '<script>.md' }] }), { status: 200 }))
    await expect(listUploads()).rejects.toThrow('invalid uploaded document')
  })
})
