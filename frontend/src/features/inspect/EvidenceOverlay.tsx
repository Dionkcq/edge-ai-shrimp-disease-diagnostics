import { useState } from 'react'
import type { MarkerObservation, ScreeningResult } from '../../api/types'

const label = (marker: MarkerObservation) =>
  marker.role === 'WHITE_SPOT'
    ? 'White spot'
    : marker.role === 'GILL_DARKENING'
      ? 'Gill darkening'
      : marker.class_name.replaceAll('_', ' ')
const percent = (value: number) => `${Math.round(value * 100)}%`
export function EvidenceOverlay({
  previewUrl,
  result,
}: {
  previewUrl: string
  result: ScreeningResult
}) {
  const [selected, setSelected] = useState<number | null>(result.markers.length ? 0 : null)
  const active = selected === null ? null : (result.markers[selected] ?? null)
  return (
    <section className="evidence" aria-labelledby="evidence-title">
      <div className="section-kicker">02 · Inspect</div>
      <h3 id="evidence-title">Visible-marker evidence</h3>
      <div className="evidence-layout">
        <div
          className="image-stage"
          style={{ aspectRatio: `${result.image.width} / ${result.image.height}` }}
        >
          <img src={previewUrl} alt="Uploaded shrimp photograph with marker overlay" />
          {result.markers.map((marker, index) => (
            <button
              key={`${marker.class_index}-${index}`}
              type="button"
              className={`marker-box ${marker.role === 'GILL_DARKENING' ? 'marker-gill' : 'marker-spot'}`}
              style={{
                left: percent(marker.box.x1),
                top: percent(marker.box.y1),
                width: percent(marker.box.x2 - marker.box.x1),
                height: percent(marker.box.y2 - marker.box.y1),
              }}
              aria-label={`${label(marker)} marker ${index + 1}`}
              aria-pressed={selected === index}
              onClick={() => setSelected(index)}
            >
              <span>{index + 1}</span>
            </button>
          ))}
        </div>
        <div className="evidence-detail">
          {active ? (
            <>
              <div
                className="magnifier"
                role="img"
                aria-label={`Magnified marker ${Number(selected) + 1}`}
                style={{
                  backgroundImage: `url(${previewUrl})`,
                  backgroundSize: '400%',
                  backgroundPosition: `${(active.box.x1 + active.box.x2) * 50}% ${(active.box.y1 + active.box.y2) * 50}%`,
                }}
              >
                <span>4×</span>
              </div>
              <h4>Magnified marker {Number(selected) + 1}</h4>
              <p>
                {label(active)} · evidence band {result.confidence_band.toLowerCase()}
              </p>
            </>
          ) : (
            <p className="empty-evidence">No retained marker boxes to inspect.</p>
          )}
        </div>
      </div>
      {result.markers.length > 0 && (
        <ol className="marker-list" aria-label="Marker coordinates">
          {result.markers.map((marker, index) => (
            <li key={`text-${marker.class_index}-${index}`}>
              <button type="button" onClick={() => setSelected(index)}>
                <strong>
                  {index + 1}. {label(marker)}
                </strong>
                <span>
                  x {Math.round(marker.box.x1 * 100)}–{Math.round(marker.box.x2 * 100)}%, y{' '}
                  {Math.round(marker.box.y1 * 100)}–{Math.round(marker.box.y2 * 100)}% ·{' '}
                  {Math.round(marker.score * 100)} score units
                </span>
              </button>
            </li>
          ))}
        </ol>
      )}
      <p className="legend">
        <span className="line-solid" aria-hidden="true" /> solid: white-spot-like{' '}
        <span className="line-dashed" aria-hidden="true" /> dashed: dark-gill-like. Scores rank
        evidence; they are not disease probabilities.
      </p>
    </section>
  )
}
