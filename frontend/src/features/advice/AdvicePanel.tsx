import type { Advice } from '../../api/types'

export type AdviceState = 'idle' | 'loading' | 'ready' | 'error'

type Props = {
  /** `Meta.advice_available`. When false the endpoint answers 404 on this build and
   * the feature is not offered at all, rather than offered and then failed. */
  available: boolean
  advice: Advice | null
  state: AdviceState
  error: string | null
  onRequest: () => void
}

/** Optional, opt-in, unreviewed model output rendered strictly below the reviewed guidance.
 *
 * Two rules hold in every branch of this component:
 *
 * 1. Nothing is requested until a person asks for it. Generation runs on a local model and
 *    takes seconds to tens of seconds, so it is a deliberate action with a visible wait,
 *    never a side effect of screening a photograph.
 * 2. The generated text is never rendered without its disclosure. The banner precedes the
 *    summary in document order, so it cannot be scrolled past or clipped away. */
export function AdvicePanel({ available, advice, state, error, onRequest }: Props) {
  if (!available) return null
  return (
    <section className="advice" aria-labelledby="advice-title">
      <div className="section-kicker">04 · Optional · generated on this device</div>
      <h3 id="advice-title">A longer explanation, written by a local model</h3>
      <p className="advice-lede">
        This expands the reviewed guidance above into plain-language steps. It is generated on this
        device by a local language model, it is not reviewed by anyone, and it never replaces the
        cited guidance.
      </p>

      {state !== 'ready' && (
        <div className="advice-actions">
          <button
            className="button button-secondary"
            type="button"
            onClick={onRequest}
            disabled={state === 'loading'}
          >
            {state === 'loading' ? 'Generating…' : 'Generate explanation'}
          </button>
          {state === 'loading' && (
            <p role="status">
              Asking the local model. This can take up to a minute and nothing leaves this device.
            </p>
          )}
          {state === 'error' && error && (
            <p className="inline-error" role="alert">
              {error}
            </p>
          )}
        </div>
      )}

      {state === 'ready' && advice && (
        <>
          <div className="advice-disclosure" role="note">
            <strong>AI generated · not reviewed</strong>
            <p>{advice.review_note}</p>
          </div>
          <p className="advice-summary">{advice.summary}</p>
          <h4>Do now</h4>
          <ul className="advice-list">
            {advice.immediate_actions.map((action) => (
              <li key={action}>{action}</li>
            ))}
          </ul>
          <h4>Reduce the chance of it recurring</h4>
          <ul className="advice-list">
            {advice.prevention_actions.map((action) => (
              <li key={action}>{action}</li>
            ))}
          </ul>
          {advice.additional_considerations.length > 0 && (
            <>
              <h4>Also consider</h4>
              <ul className="advice-list">
                {advice.additional_considerations.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </>
          )}
          <h4>Sources this was grounded in</h4>
          <ul className="source-list">
            {advice.sources.map((source) => (
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
          <div className="advice-provenance">
            Expanded from guidance {advice.based_on_guidance_id} · generated locally by{' '}
            {advice.model_id} via {advice.provider} ·{' '}
            {advice.review_status.replaceAll('_', ' ').toLowerCase()}
          </div>
          <div className="advice-actions">
            <button className="button button-quiet" type="button" onClick={onRequest}>
              Generate again
            </button>
          </div>
        </>
      )}
    </section>
  )
}
