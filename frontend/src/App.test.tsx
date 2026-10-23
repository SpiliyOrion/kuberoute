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
