import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'

const originalFetch = globalThis.fetch
beforeEach(() => { Element.prototype.scrollIntoView = vi.fn() })
afterEach(() => { cleanup(); globalThis.fetch = originalFetch })

function modelsResponse() {
  return new Response(JSON.stringify({ object: 'list', data: [{ object: 'model', id: 'mock-small' }, { object: 'model', id: 'mock-large' }, { object: 'model', id: 'qwen3:0.6b' }] }), { status: 200 })
}
function chatResponse(model = 'mock-small', content = 'Result') {
  return new Response(JSON.stringify({ object: 'chat.completion', model, choices: [{ index: 0, message: { role: 'assistant', content }, finish_reason: 'stop' }] }), { status: 200 })
}
function setupFetch(response = chatResponse()) {
  const mocked = vi.fn().mockImplementation((url: string) => Promise.resolve(url === '/v1/models' ? modelsResponse() : response))
  globalThis.fetch = mocked
  return mocked
}

it('explains Chat, RAG, routing, and mock models with accessible help controls', async () => {
  setupFetch()
  render(<App />)
  await screen.findByRole('option', { name: /mock-small/ })
  const modeHelp = screen.getByRole('button', { name: 'Mode help' })
  expect(modeHelp).toHaveAttribute('aria-expanded', 'false')
  await userEvent.click(modeHelp)
  expect(modeHelp).toHaveAttribute('aria-expanded', 'true')
  expect(screen.getByRole('note')).toHaveTextContent('Standard multi-turn conversations')
  expect(screen.getByRole('note')).toHaveTextContent('Each question is independent')
  await userEvent.keyboard('{Escape}')
  expect(modeHelp).toHaveAttribute('aria-expanded', 'false')

  await userEvent.click(screen.getByRole('button', { name: 'Model help' }))
  expect(screen.getByRole('note')).toHaveTextContent('not guaranteed to pick the fastest or most accurate')
  expect(screen.getByRole('note')).toHaveTextContent('Sends the request to that model')
  expect(screen.getByRole('note')).toHaveTextContent('Simulated responses for testing')
  await userEvent.click(screen.getByRole('combobox', { name: 'Select model' }))
  expect(screen.queryByRole('note')).not.toBeInTheDocument()
})

it('explains retrieval options and Top-K without changing the default selection', async () => {
  setupFetch()
  render(<App />)
  await screen.findByRole('option', { name: /mock-small/ })
  await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Mode' }), 'rag')
  const retrievalHelp = screen.getByRole('button', { name: 'Retrieval mode help' })
  fireEvent.pointerDown(retrievalHelp, { pointerType: 'touch' })
  await userEvent.click(retrievalHelp)
  expect(screen.getByRole('note')).toHaveTextContent('reciprocal rank fusion')
  expect(screen.getByRole('note')).toHaveTextContent('wording differs')
  expect(screen.getByRole('note')).toHaveTextContent('technical or exact terminology')
  await userEvent.click(retrievalHelp)
  await userEvent.click(screen.getByRole('button', { name: 'Top-K passages help' }))
  expect(screen.getByRole('note')).toHaveTextContent('default of 5')
  expect(screen.getByRole('note')).toHaveTextContent('may miss useful evidence')
  expect(screen.getByRole('note')).toHaveTextContent('irrelevant context')
  expect(screen.getByRole('combobox', { name: 'Retrieval mode' })).toHaveValue('hybrid')
  expect(screen.getByRole('combobox', { name: 'Top-K passages' })).toHaveValue('5')
})

it('loads real and simulated models and sends automatic routing by default', async () => {
  const mocked = setupFetch(chatResponse('qwen3:0.6b'))
  render(<App />)
  expect(await screen.findByRole('option', { name: /mock-small · simulated/ })).toBeInTheDocument()
  expect(screen.getByRole('option', { name: 'qwen3:0.6b' })).toBeInTheDocument()
  await userEvent.type(screen.getByRole('textbox', { name: 'Message' }), 'Hello{enter}')
  expect(await screen.findByText('Responded: qwen3:0.6b')).toBeInTheDocument()
  expect(JSON.parse(mocked.mock.calls.find((call) => call[0] === '/v1/chat/completions')![1].body).model).toBe('auto')
})

it('supports explicit selection, multi-turn context, and a fresh chat', async () => {
  const mocked = setupFetch()
  render(<App />)
  await screen.findByRole('option', { name: /mock-small/ })
  await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Select model' }), 'mock-small')
  const input = screen.getByRole('textbox', { name: 'Message' })
  await userEvent.type(input, 'First{enter}')
  await screen.findByText('Result')
  await userEvent.type(input, 'Second{enter}')
  await waitFor(() => expect(mocked.mock.calls.filter((call) => call[0] === '/v1/chat/completions')).toHaveLength(2))
  const requests = mocked.mock.calls.filter((call) => call[0] === '/v1/chat/completions')
  expect(JSON.parse(requests[1][1].body)).toMatchObject({ model: 'mock-small', messages: [{ role: 'user', content: 'First' }, { role: 'assistant', content: 'Result' }, { role: 'user', content: 'Second' }] })
  await userEvent.click(screen.getByRole('button', { name: /New Chat/ }))
  expect(screen.queryByText('First')).not.toBeInTheDocument()
  expect(screen.getByText(/One gateway/)).toBeInTheDocument()
})

it('keeps errors safe and allows retry without duplicating the user message', async () => {
  let calls = 0
  globalThis.fetch = vi.fn().mockImplementation((url: string) => {
    if (url === '/v1/models') return Promise.resolve(modelsResponse())
    calls += 1
    return Promise.resolve(calls === 1 ? new Response('private exception', { status: 503 }) : chatResponse())
  })
  render(<App />)
  await screen.findByRole('option', { name: /mock-small/ })
  await userEvent.type(screen.getByRole('textbox', { name: 'Message' }), 'Question{enter}')
  expect(await screen.findByRole('alert')).toHaveTextContent('no eligible real model')
  expect(screen.queryByText('private exception')).not.toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: 'Retry request' }))
  expect(await screen.findByText('Result')).toBeInTheDocument()
  expect(screen.getAllByText('Question')).toHaveLength(1)
})

it('reports discovery failure and offers retry', async () => {
  let attempts = 0
  globalThis.fetch = vi.fn().mockImplementation(() => {
    attempts += 1
    return attempts === 1 ? Promise.reject(new Error('secret URL')) : Promise.resolve(modelsResponse())
  })
  render(<App />)
  expect(await screen.findByRole('alert')).toHaveTextContent('Cannot reach the gateway')
  expect(screen.queryByText('secret URL')).not.toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: 'Retry discovery' }))
  expect(await screen.findByRole('option', { name: /mock-small/ })).toBeInTheDocument()
})

it('disables duplicate submission while pending and ignores a stale response after New Chat', async () => {
  let resolveChat!: (response: Response) => void
  globalThis.fetch = vi.fn().mockImplementation((url: string) => url === '/v1/models' ? Promise.resolve(modelsResponse()) : new Promise<Response>((resolve) => { resolveChat = resolve }))
  render(<App />)
  await screen.findByRole('option', { name: /mock-small/ })
  await userEvent.type(screen.getByRole('textbox', { name: 'Message' }), 'Old{enter}')
  expect(await screen.findByRole('status')).toHaveTextContent('Generating')
  expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled()
  await userEvent.click(screen.getByRole('button', { name: /New Chat/ }))
  await act(async () => { resolveChat(chatResponse()) })
  expect(screen.queryByText('Old')).not.toBeInTheDocument()
  expect(screen.queryByText('Result')).not.toBeInTheDocument()
})

it('renders hostile assistant text without interpreting HTML', async () => {
  setupFetch(chatResponse('mock-small', '<img src=x onerror=alert(1)>'))
  render(<App />)
  await screen.findByRole('option', { name: /mock-small/ })
  await userEvent.type(screen.getByRole('textbox', { name: 'Message' }), 'Hi{enter}')
  expect(await screen.findByText('<img src=x onerror=alert(1)>')).toBeInTheDocument()
  expect(document.querySelector('img')).toBeNull()
})

it('uses Shift+Enter for a newline without sending', async () => {
  const mocked = setupFetch()
  render(<App />)
  const input = screen.getByRole('textbox', { name: 'Message' })
  fireEvent.change(input, { target: { value: 'First' } })
  fireEvent.keyDown(input, { key: 'Enter', shiftKey: true })
  expect(mocked.mock.calls.filter((call) => call[0] === '/v1/chat/completions')).toHaveLength(0)
})

function ragResponse(answer = 'Synthetic answer [1]', sources: unknown[] = [{ marker: '[1]', source_id: 'a'.repeat(64), chunk_id: 'b'.repeat(64), chunk_index: 2 }]) {
  return new Response(JSON.stringify({ answer, model: 'qwen3:0.6b', retrieval_mode: 'hybrid', finish_reason: 'stop', sources }), { status: 200 })
}

it('shows uploaded citation names without inventing bibliography or links', async () => {
  const source = { marker: '[1]', source_id: 'a'.repeat(64), chunk_id: 'b'.repeat(64), chunk_index: 2,
    uploaded_name: 'Safe Notes.md' }
  globalThis.fetch = vi.fn().mockImplementation((url: string) => Promise.resolve(url === '/v1/models' ? modelsResponse() : ragResponse('Answer [1]', [source])))
  render(<App />)
  await screen.findByRole('option', { name: 'qwen3:0.6b' })
  await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Mode' }), 'rag')
  await userEvent.type(screen.getByRole('textbox', { name: 'RAG question' }), 'Question{enter}')
  const name = await screen.findByText(/Uploaded document: Safe Notes.md/)
  expect(name.closest('li')).toHaveTextContent('index 2')
  expect(name.closest('li')?.querySelector('a')).toBeNull()
  expect(screen.getByText(/Citation markers identify retrieved passages/)).toBeInTheDocument()
})

it('renders verified bibliographic citations as safe external links', async () => {
  const source = { marker: '[1]', source_id: 'a'.repeat(64), chunk_id: 'b'.repeat(64), chunk_index: 2,
    catalog_id: 'verified_study', title: '<Synthetic study>', authors: 'Research Group', publication_year: 2023,
    source_type: 'review', original_url: 'https://example.org/study' }
  globalThis.fetch = vi.fn().mockImplementation((url: string) => Promise.resolve(url === '/v1/models' ? modelsResponse() : ragResponse('Answer [1]', [source])))
  render(<App />)
  await screen.findByRole('option', { name: 'qwen3:0.6b' })
  await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Mode' }), 'rag')
  await userEvent.type(screen.getByRole('textbox', { name: 'RAG question' }), 'Question{enter}')
  const link = await screen.findByRole('link', { name: '<Synthetic study>' })
  expect(link).toHaveAttribute('href', 'https://example.org/study')
  expect(link).toHaveAttribute('target', '_blank')
  expect(link).toHaveAttribute('rel', 'noopener noreferrer')
  expect(link).toHaveAttribute('referrerpolicy', 'no-referrer')
  expect(link.closest('li')).toHaveTextContent('Research Group (2023)')
  expect(link.closest('li')).toHaveTextContent('verified_study')
  expect(document.querySelector('synthetic')).toBeNull()
  expect(screen.getByText(/Citation markers identify retrieved passages/)).toBeInTheDocument()
})

it('switches between chat and independent RAG questions without sending history', async () => {
  const mocked = vi.fn().mockImplementation((url: string) => Promise.resolve(url === '/v1/models' ? modelsResponse() : url === '/v1/rag/chat/completions' ? ragResponse() : chatResponse()))
  globalThis.fetch = mocked
  render(<App />)
  await screen.findByRole('option', { name: 'qwen3:0.6b' })
  await userEvent.type(screen.getByRole('textbox', { name: 'Message' }), 'Ordinary chat{enter}')
  await screen.findByText('Result')
  await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Mode' }), 'rag')
  expect(screen.queryByText('Ordinary chat')).not.toBeInTheDocument()
  expect(screen.getByText(/Each RAG question is independent/)).toBeInTheDocument()
  expect(screen.getByRole('combobox', { name: 'Retrieval mode' })).toHaveValue('hybrid')
  expect(screen.getByRole('combobox', { name: 'Top-K passages' })).toHaveValue('5')
  await userEvent.type(screen.getByRole('textbox', { name: 'RAG question' }), 'First question{enter}')
  await screen.findByText('Synthetic answer [1]')
  await userEvent.type(screen.getByRole('textbox', { name: 'RAG question' }), 'Second question{enter}')
  await waitFor(() => expect(mocked.mock.calls.filter((call) => call[0] === '/v1/rag/chat/completions')).toHaveLength(2))
  const ragCalls = mocked.mock.calls.filter((call) => call[0] === '/v1/rag/chat/completions')
  expect(JSON.parse(ragCalls[0][1].body)).toEqual({ model: 'auto', question: 'First question', retrieval_mode: 'hybrid', top_k: 5, stream: false })
  expect(JSON.parse(ragCalls[1][1].body)).toEqual({ model: 'auto', question: 'Second question', retrieval_mode: 'hybrid', top_k: 5, stream: false })
  expect(mocked.mock.calls.filter((call) => call[0] === '/v1/chat/completions')).toHaveLength(1)
  await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Mode' }), 'chat')
  expect(screen.getByText('Ordinary chat')).toBeInTheDocument()
})

it('sends selected retrieval controls and displays only provided source identifiers', async () => {
  const mocked = vi.fn().mockImplementation((url: string) => Promise.resolve(url === '/v1/models' ? modelsResponse() : ragResponse()))
  globalThis.fetch = mocked
  render(<App />)
  await screen.findByRole('option', { name: 'qwen3:0.6b' })
  await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Mode' }), 'rag')
  await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Select model' }), 'qwen3:0.6b')
  await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Retrieval mode' }), 'lexical')
  await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Top-K passages' }), '3')
  await userEvent.type(screen.getByRole('textbox', { name: 'RAG question' }), 'Question{enter}')
  expect(await screen.findByText('Synthetic answer [1]')).toBeInTheDocument()
  expect(screen.getByText('Responded: qwen3:0.6b')).toBeInTheDocument()
  expect(screen.getByText('Retrieval: hybrid')).toBeInTheDocument()
  expect(screen.getByText('a'.repeat(64)).closest('li')).toHaveTextContent('b'.repeat(64))
  expect(screen.getByText('a'.repeat(64)).closest('li')).toHaveTextContent('index 2')
  expect(screen.getByText(/Citation markers identify retrieved passages/)).toBeInTheDocument()
  const body = JSON.parse(mocked.mock.calls.find((call) => call[0] === '/v1/rag/chat/completions')![1].body)
  expect(body).toEqual({ model: 'qwen3:0.6b', question: 'Question', retrieval_mode: 'lexical', top_k: 3, stream: false })
})

it('shows no-citation and safe error states without rendering hostile HTML', async () => {
  let calls = 0
  globalThis.fetch = vi.fn().mockImplementation((url: string) => {
    if (url === '/v1/models') return Promise.resolve(modelsResponse())
    calls += 1
    return Promise.resolve(calls === 1 ? new Response('private path / key', { status: 503 }) : ragResponse('<img src=x onerror=alert(1)>', []))
  })
  render(<App />)
  await screen.findByRole('option', { name: 'qwen3:0.6b' })
  await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Mode' }), 'rag')
  await userEvent.type(screen.getByRole('textbox', { name: 'RAG question' }), 'Question{enter}')
  expect(await screen.findByRole('alert')).toHaveTextContent('RAG is disabled or its document index is unavailable')
  expect(screen.queryByText('private path / key')).not.toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: 'Retry request' }))
  expect(await screen.findByText('<img src=x onerror=alert(1)>')).toBeInTheDocument()
  expect(document.querySelector('img')).toBeNull()
  expect(screen.getByText('No source references were returned. The answer has no cited support.')).toBeInTheDocument()
  expect(screen.getAllByText('Question')).toHaveLength(1)
})
