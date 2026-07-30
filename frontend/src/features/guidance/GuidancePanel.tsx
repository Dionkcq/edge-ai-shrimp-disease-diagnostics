import type { Guidance } from '../../api/types'

export function GuidancePanel({
  guidance,
  state,
}: {
  guidance: Guidance | null
  state: 'idle' | 'loading' | 'ready' | 'error'
}) {
  if (state === 'loading')
    return (
      <section className="guidance">
        <h3>Reviewed local guidance</h3>
        <p role="status">Loading guidance stored on this device…</p>
      </section>
    )
  if (state === 'error')
    return (
      <section className="guidance">
        <h3>Reviewed local guidance</h3>
        <p>Guidance could not be loaded. The screening result remains non-diagnostic.</p>
      </section>
    )
  if (!guidance) return null
  return (
    <section className="guidance" aria-labelledby="guidance-title">
      <div className="section-kicker">03 · Learn</div>
      <h3 id="guidance-title">{guidance.headline}</h3>
      <p className="guidance-body">{guidance.body}</p>
      <div className="review-note">
        <strong>Review status</strong>
        <span>{guidance.review_status.replaceAll('_', ' ').toLowerCase()}</span>
        <p>{guidance.review_note}</p>
      </div>
      <h4>Sources stored with this guidance</h4>
      <ul className="source-list">
        {guidance.sources.map((source) => (
          <li key={source.id}>
            <a href={source.url} target="_blank" rel="noreferrer">
              {source.title}
            </a>
            <span>
              {source.publisher} · accessed {source.accessed_on}
            </span>
          </li>
        ))}
      </ul>
    </section>
  )
}
