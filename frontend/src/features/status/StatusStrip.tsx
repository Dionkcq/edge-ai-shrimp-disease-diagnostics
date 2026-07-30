import type { Meta, ScreeningResult } from '../../api/types'

type Props = { meta: Meta | null; result: ScreeningResult | null; metaError: boolean }
export function StatusStrip({ meta, result, metaError }: Props) {
  const provider = result?.model.provider ?? meta?.provider
  const demo = result?.model.is_demonstration_data ?? meta?.is_demonstration_data
  const unavailable = metaError || provider === 'unavailable' || meta?.model_available === false
  return (
    <div
      className={`status-strip${demo ? ' status-demo' : ''}${unavailable ? ' status-unavailable' : ''}`}
      role="status"
    >
      <div className="status-main">
        <span className="status-dot" aria-hidden="true" />
        {demo ? (
          <strong>DEMO FIXTURE — NOT MODEL OUTPUT</strong>
        ) : unavailable ? (
          <strong>MODEL UNAVAILABLE — SCREENING DISABLED</strong>
        ) : (
          <strong>LOCAL SCREENING READY</strong>
        )}
      </div>
      <div className="status-facts">
        <span>{provider ? `Provider: ${provider}` : 'Connecting locally…'}</span>
        <span>Offline-capable · same-origin only</span>
        {result?.model.version && <span>Version: {result.model.version}</span>}
        {result?.model.dataset_mapping_status === 'PROVISIONAL_UNCONFIRMED' && (
          <span>Dataset mapping: unconfirmed</span>
        )}
      </div>
    </div>
  )
}
