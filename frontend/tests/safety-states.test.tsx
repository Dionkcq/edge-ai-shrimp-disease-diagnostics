import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { delay, http, HttpResponse } from 'msw'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from '../src/app/App'
import { ErrorBoundary } from '../src/components/ErrorBoundary'
import { server } from './support/server'
import { fixtureGuidance, fixtureMeta, fixtureResult } from './support/fixtures'

function image(name = 'shrimp.jpg', bytes = 'image'): File {
  return new File([bytes], name, { type: 'image/jpeg' })
}

async function selectAndSubmit(user: ReturnType<typeof userEvent.setup>) {
  await user.upload(screen.getByLabelText(/take or choose/i), image())
  await user.click(screen.getByRole('button', { name: /screen photograph/i }))
}

describe('fail-safe interface states', () => {
  it('keeps selection available while clearly reporting unavailable metadata', async () => {
    server.use(http.get('/api/v1/meta', () => new HttpResponse(null, { status: 503 })))
    render(<App />)
    expect(await screen.findByText(/local screening service is unavailable/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/take or choose/i)).toBeEnabled()
  })

  it('rejects unsupported and oversized files before upload', async () => {
    const unrestricted = userEvent.setup({ applyAccept: false })
    const first = render(<App />)
    await unrestricted.upload(
      screen.getByLabelText(/take or choose/i),
      new File(['x'], 'notes.txt', { type: 'text/plain' }),
    )
    expect(screen.getByText(/choose a JPEG or PNG image/i)).toBeInTheDocument()
    first.unmount()

    server.use(
      http.get('/api/v1/meta', () => HttpResponse.json({ ...fixtureMeta(), max_upload_bytes: 1 })),
    )
    render(<App />)
    await screen.findByText('DEMO FIXTURE — NOT MODEL OUTPUT')
    await unrestricted.upload(screen.getByLabelText(/take or choose/i), image('large.jpg', 'xx'))
    expect(screen.getByText(/larger than the local limit/i)).toBeInTheDocument()
  })

  it('renders the successful local guidance and its reviewed source', async () => {
    const user = userEvent.setup()
    render(<App />)
    await selectAndSubmit(user)
    const [source] = fixtureGuidance().sources
    if (!source) throw new Error('guidance fixture must include a source')
    expect(
      await screen.findByRole('heading', { name: fixtureGuidance().headline }),
    ).toBeInTheDocument()
    expect(screen.getByRole('link', { name: source.title })).toHaveAttribute('rel', 'noreferrer')
  })

  it('retains the screening result if local guidance fails', async () => {
    server.use(
      http.get('/api/v1/guidance/:decision', () => new HttpResponse(null, { status: 500 })),
    )
    const user = userEvent.setup()
    render(<App />)
    await selectAndSubmit(user)
    expect(await screen.findByText(/guidance could not be loaded/i)).toBeInTheDocument()
    expect(screen.getByText('Both marker types seen')).toBeInTheDocument()
  })

  it('can cancel a request without inventing a result', async () => {
    server.use(
      http.post('/api/v1/screenings', async () => {
        await delay('infinite')
        return HttpResponse.json(fixtureResult())
      }),
    )
    const user = userEvent.setup()
    render(<App />)
    await user.upload(screen.getByLabelText(/take or choose/i), image())
    await user.click(screen.getByRole('button', { name: /screen photograph/i }))
    await user.click(await screen.findByRole('button', { name: /cancel screening/i }))
    expect(await screen.findByText(/screening cancelled/i)).toBeInTheDocument()
    expect(screen.queryByText('Screening result')).not.toBeInTheDocument()
  })

  it('accepts a dropped JPEG through the same local validation path', () => {
    const { container } = render(<App />)
    const dropZone = container.querySelector('.drop-zone')
    if (!dropZone) throw new Error('capture drop zone must render')
    fireEvent.dragEnter(dropZone)
    expect(dropZone).toHaveClass('is-dragging')
    fireEvent.drop(dropZone, { dataTransfer: { files: [image('drop.jpg')] } })
    expect(screen.getByText('drop.jpg')).toBeInTheDocument()
  })

  it('aborts the old screening before accepting a dropped replacement image', async () => {
    server.use(
      http.post('/api/v1/screenings', async () => {
        await delay('infinite')
        return HttpResponse.json(fixtureResult())
      }),
    )
    const user = userEvent.setup()
    const { container } = render(<App />)
    await user.upload(screen.getByLabelText(/take or choose/i), image('first.jpg'))
    await user.click(screen.getByRole('button', { name: /screen photograph/i }))
    const dropZone = container.querySelector('.drop-zone')
    if (!dropZone) throw new Error('capture drop zone must render')

    fireEvent.drop(dropZone, { dataTransfer: { files: [image('second.jpg')] } })

    expect(await screen.findByText('second.jpg')).toBeInTheDocument()
    await waitFor(() => expect(screen.queryByText('Screening result')).not.toBeInTheDocument())
    expect(screen.queryByText(/screening cancelled/i)).not.toBeInTheDocument()
  })
})

describe('interface crash boundary', () => {
  afterEach(() => vi.restoreAllMocks())

  it('fails visibly and does not preserve image history', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    function Broken(): never {
      throw new Error('test crash')
    }
    render(
      <ErrorBoundary>
        <Broken />
      </ErrorBoundary>,
    )
    expect(screen.getByRole('heading', { name: /could not continue safely/i })).toBeInTheDocument()
    expect(screen.getByText(/no image or screening history is preserved/i)).toBeInTheDocument()
  })
})
