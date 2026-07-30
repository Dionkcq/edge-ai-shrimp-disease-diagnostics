import { http, HttpResponse } from 'msw'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { App } from '../src/app/App'
import { server } from './support/server'
import { fixtureProblem } from './support/fixtures'

describe('API states', () => {
  it('shows fixture controls only when metadata marks fixture output', async () => {
    render(<App />)
    expect(await screen.findByText('DEMO FIXTURE — NOT MODEL OUTPUT')).toBeInTheDocument()
    expect(screen.getByLabelText(/demo scenario/i)).toBeInTheDocument()
  })

  it('humanizes problem+json and supports retry', async () => {
    server.use(
      http.post('/api/v1/screenings', () => HttpResponse.json(fixtureProblem(), { status: 503 })),
    )
    const user = userEvent.setup()
    render(<App />)
    await user.upload(
      screen.getByLabelText(/take or choose/i),
      new File(['x'], 'x.jpg', { type: 'image/jpeg' }),
    )
    await user.click(screen.getByRole('button', { name: /screen photograph/i }))
    expect(await screen.findByText(/already screening/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument()
  })

  it('reports malformed success bodies rather than guessing', async () => {
    server.use(http.post('/api/v1/screenings', () => HttpResponse.json({ decision: 'UNKNOWN' })))
    const user = userEvent.setup()
    render(<App />)
    await user.upload(
      screen.getByLabelText(/take or choose/i),
      new File(['x'], 'x.jpg', { type: 'image/jpeg' }),
    )
    await user.click(screen.getByRole('button', { name: /screen photograph/i }))
    await waitFor(() => expect(screen.getByText(/unexpected response/i)).toBeInTheDocument())
  })
})
