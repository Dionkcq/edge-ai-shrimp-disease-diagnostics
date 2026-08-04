import type { Advice, Guidance, ScreeningResult } from '../../api/types'
import { abstentionCopy, decisionMeta, qualityCopy } from '../../domain/decisions'
import { EvidenceOverlay } from '../inspect/EvidenceOverlay'
import { GuidancePanel } from '../guidance/GuidancePanel'
import { AdvicePanel, type AdviceState } from '../advice/AdvicePanel'

type Props = {
  result: ScreeningResult
  previewUrl: string
  guidance: Guidance | null
  guidanceState: 'idle' | 'loading' | 'ready' | 'error'
  adviceAvailable: boolean
  advice: Advice | null
  adviceState: AdviceState
  adviceError: string | null
  onRequestAdvice: () => void
}
export function ResultPanel({
  result,
  previewUrl,
  guidance,
  guidanceState,
  adviceAvailable,
  advice,
  adviceState,
  adviceError,
  onRequestAdvice,
}: Props) {
  const meta = decisionMeta[result.decision]
  return (
    <section className="result" aria-labelledby="result-title" aria-live="polite">
      {result.model.is_demonstration_data && (
        <div className="demo-stamp">DEMO FIXTURE — NOT MODEL OUTPUT</div>
      )}
      <div className={`decision-band decision-${meta.tone}`}>
        <div className="section-kicker">Screening result</div>
        <h2 id="result-title">{meta.title}</h2>
        <p>{meta.summary}</p>
      </div>
      <div className="result-next">
        <strong>Next safe action</strong>
        <p>{meta.next}</p>
        {result.abstention_reason && (
          <p className="specific-action">{abstentionCopy[result.abstention_reason]}</p>
        )}
        {result.quality.reasons.map((reason) => (
          <p className="specific-action" key={reason}>
            {qualityCopy[reason]}
          </p>
        ))}
      </div>
      <div
        className="confidence"
        aria-label={`Evidence confidence band: ${result.confidence_band.toLowerCase()}`}
      >
        <div>
          <strong>Evidence band</strong>
          <span>
            {result.confidence_band === 'NONE'
              ? 'Not available'
              : result.confidence_band.toLowerCase()}
          </span>
        </div>
        <div className="segments" aria-hidden="true">
          {(['LOW', 'MODERATE', 'HIGH'] as const).map((band) => (
            <span
              key={band}
              className={
                result.confidence_band === band ||
                result.confidence_band === 'HIGH' ||
                (result.confidence_band === 'MODERATE' && band === 'LOW')
                  ? 'active'
                  : ''
              }
            />
          ))}
        </div>
        <p>
          This band describes retained visual evidence. It is not a calibrated disease probability.
        </p>
      </div>
      <EvidenceOverlay previewUrl={previewUrl} result={result} />
      <GuidancePanel guidance={guidance} state={guidanceState} />
      <AdvicePanel
        available={adviceAvailable}
        advice={advice}
        state={adviceState}
        error={adviceError}
        onRequest={onRequestAdvice}
      />
      <div className="request-note">
        Request {result.request_id} · processed locally in {Math.round(result.timings_ms.total_ms)}{' '}
        ms · image not retained by this interface
      </div>
    </section>
  )
}
