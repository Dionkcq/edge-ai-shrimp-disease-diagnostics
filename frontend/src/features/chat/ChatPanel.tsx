import { useRef, useState } from 'react'
import { ApiError, sendChat } from '../../api/client'
import type { ChatTurn } from '../../api/types'

type Props = { available: boolean; file: File | null }

export function ChatPanel({ available, file }: Props) {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatTurn[]>([])
  const [draft, setDraft] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  async function submit() {
    const message = draft.trim()
    if (!message || busy || !available) return
    setDraft('')
    setError(null)
    setBusy(true)
    try {
      const response = await sendChat(message, sessionId, file)
      setSessionId(response.session_id)
      setMessages(response.messages.filter((item) => item.role !== 'tool'))
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'The chat request failed.')
      setDraft(message)
    } finally {
      setBusy(false)
      input.current?.focus()
    }
  }

  return (
    <section className="chat-panel" aria-labelledby="chat-title">
      <div className="section-kicker">03 · Screening assistant</div>
      <div className="chat-heading">
        <div>
          <h2 id="chat-title">Ask the pond-side agent</h2>
          <p>
            Upload an image above, then ask the local assistant to inspect it. The assistant can
            call the CNN screening tool and remembers this conversation briefly in the running
            service.
          </p>
        </div>
        <span className="chat-status">{available ? 'LOCAL AGENT' : 'OFF'}</span>
      </div>
      <div className="chat-transcript" aria-live="polite">
        {messages.length === 0 && (
          <p className="chat-empty">
            Try: “Please inspect this image and explain what I should do next.”
          </p>
        )}
        {messages.map((message, index) => (
          <div className={`chat-message chat-${message.role}`} key={`${message.role}-${index}`}>
            <span>{message.role === 'user' ? 'You' : 'Pondside agent'}</span>
            <p>{message.content}</p>
          </div>
        ))}
      </div>
      <div className="chat-compose">
        <input
          ref={input}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') void submit()
          }}
          placeholder={
            available ? 'Ask about the uploaded shrimp image…' : 'Enable the local agent to chat'
          }
          disabled={!available || busy}
          aria-label="Message the screening assistant"
        />
        <button
          className="button button-primary"
          type="button"
          onClick={() => void submit()}
          disabled={!available || busy || !draft.trim()}
        >
          {busy ? 'Thinking…' : 'Send'}
        </button>
      </div>
      {error && (
        <p className="inline-error" role="alert">
          {error}
        </p>
      )}
    </section>
  )
}
