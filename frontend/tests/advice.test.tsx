import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { describe, expect, it } from 'vitest'
import { App } from '../src/app/App'
import { server } from './support/server'
import { fixtureAdvice, fixtureAdviceProblem, fixtureMeta } from './support/fixtures'

function image(): File {
  return new File(['image'], 'shrimp.jpg', { type: 'image/jpeg' })
}

async function screenPhotograph(user: ReturnType<typeof userEvent.setup>) {
  await user.upload(screen.getByLabelText(/take or choose/i), image())
  await user.click(screen.getByRole('button', { name: /screen photograph/i }))
  await screen.findByText('Screening result')
}

describe('locally generated advice', () => {
  it('is not offered at all when the service does not declare it', async () => {
    server.use(
      http.get('/api/v1/meta', () =>
        HttpResponse.json({ ...fixtureMeta(), advice_available: false }),
      ),
    )
    const user = userEvent.setup()
    render(<App />)
    await screenPhotograph(user)
    expect(screen.queryByRole('button', { name: /generate explanation/i })).not.toBeInTheDocument()
  })

  it('is generated only when asked, never as a side effect of screening', async () => {
    let calls = 0
    server.use(
      http.get('/api/v1/advice/:decision', () => {
        calls += 1
        return HttpResponse.json(fixtureAdvice())
      }),
    )
    const user = userEvent.setup()
    render(<App />)
    await screenPhotograph(user)
    expect(calls).toBe(0)

    await user.click(screen.getByRole('button', { name: /generate explanation/i }))
    expect(await screen.findByText(fixtureAdvice().summary)).toBeInTheDocument()
    expect(calls).toBe(1)
  })

  it('renders the unreviewed disclosure ahead of the generated text', async () => {
    const user = userEvent.setup()
    const { container } = render(<App />)
    await screenPhotograph(user)
    await user.click(screen.getByRole('button', { name: /generate explanation/i }))

    const disclosure = await screen.findByText(/ai generated · not reviewed/i)
    expect(screen.getByText(fixtureAdvice().review_note)).toBeInTheDocument()
    const summary = screen.getByText(fixtureAdvice().summary)
    expect(
      disclosure.compareDocumentPosition(summary) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()

    const [action] = fixtureAdvice().immediate_actions
    expect(screen.getByText(String(action))).toBeInTheDocument()
    expect(container.querySelector('.advice-provenance')?.textContent).toContain(
      fixtureAdvice().model_id,
    )
  })

  it('reports a failed generation without disturbing the reviewed guidance', async () => {
    server.use(
      http.get('/api/v1/advice/:decision', () =>
        HttpResponse.json(fixtureAdviceProblem(), { status: 503 }),
      ),
    )
    const user = userEvent.setup()
    render(<App />)
    await screenPhotograph(user)
    await user.click(screen.getByRole('button', { name: /generate explanation/i }))

    expect(
      await screen.findByText(/did not pass the safety check|safety check/i),
    ).toBeInTheDocument()
    expect(
      screen.getByRole('heading', { name: 'Both marker types were found' }),
    ).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /generate explanation/i })).toBeEnabled()
  })

  it('refuses advice that arrives without its disclosure rather than rendering it', async () => {
    const undisclosed: Record<string, unknown> = { ...fixtureAdvice() }
    delete undisclosed.review_status
    server.use(http.get('/api/v1/advice/:decision', () => HttpResponse.json(undisclosed)))
    const user = userEvent.setup()
    render(<App />)
    await screenPhotograph(user)
    await user.click(screen.getByRole('button', { name: /generate explanation/i }))

    await waitFor(() =>
      expect(screen.getByText(/did not match the expected contract/i)).toBeInTheDocument(),
    )
    expect(screen.queryByText(fixtureAdvice().summary)).not.toBeInTheDocument()
  })

  it('drops advice generated for a previous photograph', async () => {
    const user = userEvent.setup()
    render(<App />)
    await screenPhotograph(user)
    await user.click(screen.getByRole('button', { name: /generate explanation/i }))
    expect(await screen.findByText(fixtureAdvice().summary)).toBeInTheDocument()

    await user.upload(screen.getByLabelText(/replace image/i), image())
    expect(screen.queryByText(fixtureAdvice().summary)).not.toBeInTheDocument()
  })
})
