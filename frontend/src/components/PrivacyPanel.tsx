export function PrivacyPanel() {
  return (
    <aside className="privacy-panel" aria-labelledby="privacy-title">
      <div>
        <span className="eyebrow">Safety boundary</span>
        <h2 id="privacy-title">A visible-marker screen, not a diagnosis.</h2>
      </div>
      <div className="privacy-grid">
        <p>
          <strong>What it does</strong>Checks one photograph for two visible appearances and can
          refuse uncertain images.
        </p>
        <p>
          <strong>What it cannot do</strong>It cannot identify a pathogen, replace laboratory
          testing, or support EMS/AHPND screening.
        </p>
        <p>
          <strong>Image handling</strong>Your selection stays in browser memory for this session.
          This interface stores no image or result history across reloads.
        </p>
      </div>
    </aside>
  )
}
