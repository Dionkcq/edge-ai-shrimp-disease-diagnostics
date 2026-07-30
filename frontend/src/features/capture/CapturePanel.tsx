import { useRef, useState, type DragEvent } from 'react'

type Props = {
  file: File | null
  previewUrl: string | null
  maxBytes: number
  busy: boolean
  error: string | null
  onFile: (file: File | null) => void
  onSubmit: () => void
  onCancel: () => void
}
export function CapturePanel({
  file,
  previewUrl,
  maxBytes,
  busy,
  error,
  onFile,
  onSubmit,
  onCancel,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  function receive(candidate: File | undefined) {
    if (candidate) onFile(candidate)
  }
  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setDragging(false)
    receive(event.dataTransfer.files[0])
  }
  return (
    <section className="capture" aria-labelledby="capture-title">
      <div className="section-kicker">01 · Capture</div>
      <h2 id="capture-title">One shrimp. One clear photograph.</h2>
      <p className="capture-instruction">
        Fill the frame, keep the whole animal visible, and use even light.
      </p>
      <div
        className={`drop-zone${dragging ? ' is-dragging' : ''}`}
        onDragEnter={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={drop}
      >
        {previewUrl && file ? (
          <div className="capture-preview">
            <img src={previewUrl} alt="Selected shrimp photograph preview" />
            <div className="file-line">
              <span className="filename">{file.name}</span>
              <span>Selected locally — not uploaded yet</span>
            </div>
          </div>
        ) : (
          <div className="capture-empty">
            <span className="frame-corners" aria-hidden="true" />
            <strong>Place one shrimp inside the frame</strong>
            <span>JPEG or PNG · up to {Math.floor(maxBytes / 1024 / 1024)} MB</span>
            <span className="desktop-hint">or drop a file here</span>
          </div>
        )}
        <input
          ref={inputRef}
          id="image-input"
          className="sr-only"
          type="file"
          accept="image/jpeg,image/png"
          capture="environment"
          disabled={busy}
          aria-describedby="file-help file-error"
          onChange={(e) => receive(e.currentTarget.files?.[0])}
        />
      </div>
      <p id="file-help" className="microcopy">
        The server checks file type, size, image quality, and whether screening is possible.
      </p>
      {error ? (
        <div id="file-error" className="inline-error" role="alert">
          {error}
        </div>
      ) : (
        <span id="file-error" />
      )}
      <div className="capture-actions">
        <label htmlFor="image-input" className="button button-secondary">
          {file ? 'Replace image' : 'Take or choose a photograph'}
        </label>
        {file && (
          <button
            className="button button-quiet"
            type="button"
            disabled={busy}
            onClick={() => {
              if (inputRef.current) inputRef.current.value = ''
              onFile(null)
            }}
          >
            Remove image
          </button>
        )}
        {file && !busy && (
          <button className="button button-primary" type="button" onClick={onSubmit}>
            Screen photograph
          </button>
        )}
        {busy && (
          <button className="button button-danger" type="button" onClick={onCancel}>
            Cancel screening
          </button>
        )}
      </div>
      {busy && (
        <div className="processing" role="status" aria-live="polite">
          <span className="processing-line" aria-hidden="true" />
          Checking image quality and visible markers…
        </div>
      )}
    </section>
  )
}
