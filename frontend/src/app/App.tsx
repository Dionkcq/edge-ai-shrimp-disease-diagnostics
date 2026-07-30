import { useEffect, useRef, useState } from 'react'
import { ApiError, getGuidance, getMeta, screenImage } from '../api/client'
import type { Guidance, Meta, ProblemCode, ScreeningResult } from '../api/types'
import { PrivacyPanel } from '../components/PrivacyPanel'
import { CapturePanel } from '../features/capture/CapturePanel'
import { ResultPanel } from '../features/result/ResultPanel'
import { StatusStrip } from '../features/status/StatusStrip'

const DEFAULT_MAX = 12 * 1024 * 1024
const scenarios = [
  ['multiple', 'Both marker types'],
  ['white_spot', 'White-spot-like marker'],
  ['gill', 'Dark-gill-like marker'],
  ['none', 'No target marker'],
  ['low_confidence', 'Low confidence'],
  ['overlapping', 'Overlapping boxes / NMS'],
] as const
const problemCopy: Record<ProblemCode, string> = {
  MALFORMED_REQUEST: 'The request was not understood. Choose the image again and retry.',
  PAYLOAD_TOO_LARGE: 'The file is larger than the server allows. Choose a smaller JPEG or PNG.',
  UNSUPPORTED_MEDIA_TYPE: 'The file content is not a supported JPEG or PNG.',
  UNDECODABLE_IMAGE:
    'The image is damaged or uses an unsupported format. Export it as a new JPEG or PNG.',
  SERVICE_BUSY: 'The service is already screening another photograph. Try again shortly.',
  NOT_FOUND: 'The requested local resource was not found.',
  INTERNAL_ERROR: 'The service could not complete the request. Try once more or check the server.',
}

export function App() {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [metaError, setMetaError] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<ScreeningResult | null>(null)
  const [guidance, setGuidance] = useState<Guidance | null>(null)
  const [guidanceState, setGuidanceState] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [scenario, setScenario] = useState('multiple')
  const request = useRef<AbortController | null>(null)
  const preview = useRef<string | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    void getMeta(controller.signal)
      .then(setMeta)
      .catch((caught: unknown) => {
        if (controller.signal.aborted || (caught instanceof Error && caught.name === 'AbortError'))
          return
        setMetaError(true)
      })
    return () => controller.abort()
  }, [])
  useEffect(
    () => () => {
      if (preview.current) URL.revokeObjectURL(preview.current)
      request.current?.abort()
    },
    [],
  )

  function choose(next: File | null) {
    const activeRequest = request.current
    request.current = null
    activeRequest?.abort()
    if (preview.current) URL.revokeObjectURL(preview.current)
    preview.current = null
    setPreviewUrl(null)
    setFile(null)
    setResult(null)
    setGuidance(null)
    setGuidanceState('idle')
    setError(null)
    if (!next) return
    if (!['image/jpeg', 'image/png'].includes(next.type)) {
      setError('Choose a JPEG or PNG image. The server will check the file content again.')
      return
    }
    if (next.size > (meta?.max_upload_bytes ?? DEFAULT_MAX)) {
      setError(
        'This file is larger than the local limit. Choose a JPEG or PNG under the stated size.',
      )
      return
    }
    const url = URL.createObjectURL(next)
    preview.current = url
    setPreviewUrl(url)
    setFile(next)
  }

  async function submit() {
    if (!file || busy) return
    const controller = new AbortController()
    request.current = controller
    setBusy(true)
    setError(null)
    setResult(null)
    setGuidance(null)
    setGuidanceState('idle')
    try {
      const next = await screenImage(
        file,
        meta?.is_demonstration_data ? scenario : null,
        controller.signal,
      )
      setResult(next)
      setGuidanceState('loading')
      try {
        const localGuidance = await getGuidance(next.decision, controller.signal)
        setGuidance(localGuidance)
        setGuidanceState('ready')
      } catch (guidanceError) {
        if (!(guidanceError instanceof DOMException && guidanceError.name === 'AbortError'))
          setGuidanceState('error')
      }
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') {
        if (request.current === controller)
          setError('Screening cancelled. The selected image remains ready to retry.')
      } else if (caught instanceof ApiError)
        setError(caught.problem ? problemCopy[caught.problem.code] : caught.message)
      else setError('An unexpected interface error stopped the request. No result was displayed.')
    } finally {
      if (request.current === controller) request.current = null
      setBusy(false)
    }
  }
  function cancel() {
    request.current?.abort()
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#operate">
        Skip to image capture
      </a>
      <header className="masthead">
        <div className="wordmark">
          <span className="wordmark-mark" aria-hidden="true">
            PS
          </span>
          <span>Pondside screen</span>
        </div>
        <p>Offline educational tool · visible markers only</p>
      </header>
      <StatusStrip meta={meta} result={result} metaError={metaError} />
      <main id="operate">
        <section className="intro" aria-labelledby="page-title">
          <div>
            <p className="eyebrow">Operate · photograph review</p>
            <h1 id="page-title">
              Check visible markers.
              <br />
              <em>Know when to stop.</em>
            </h1>
          </div>
          <p className="intro-copy">
            Photograph one shrimp. The local service checks whether the image can be assessed, marks
            visible evidence, and gives a cautious next action.
          </p>
        </section>
        {metaError && (
          <div className="service-message" role="alert">
            <strong>The local screening service is unavailable.</strong>
            <p>
              Image selection still works, but screening cannot start. Check that the FastAPI
              service is running on this device, then reload.
            </p>
          </div>
        )}
        {meta?.is_demonstration_data && (
          <div className="scenario-control">
            <label htmlFor="scenario">Demo scenario</label>
            <select id="scenario" value={scenario} onChange={(e) => setScenario(e.target.value)}>
              {scenarios.map(([value, name]) => (
                <option value={value} key={value}>
                  {name}
                </option>
              ))}
            </select>
            <p>
              This developer control selects canned contract data. It is shown only when the server
              declares fixture mode.
            </p>
          </div>
        )}
        <CapturePanel
          file={file}
          previewUrl={previewUrl}
          maxBytes={meta?.max_upload_bytes ?? DEFAULT_MAX}
          busy={busy}
          error={error}
          onFile={choose}
          onSubmit={() => void submit()}
          onCancel={cancel}
        />
        {error && file && !busy && (
          <div className="retry-row">
            <button className="button button-secondary" type="button" onClick={() => void submit()}>
              Try again
            </button>
          </div>
        )}
        {result && previewUrl && (
          <ResultPanel
            result={result}
            previewUrl={previewUrl}
            guidance={guidance}
            guidanceState={guidanceState}
          />
        )}
        <PrivacyPanel />
      </main>
      <footer>
        <p>
          For education and pond-side observation only. Not pathogen confirmation. Not
          lab-equivalent. No EMS/AHPND support.
        </p>
        <p>Images and results are not kept by this interface across reloads.</p>
      </footer>
    </div>
  )
}
