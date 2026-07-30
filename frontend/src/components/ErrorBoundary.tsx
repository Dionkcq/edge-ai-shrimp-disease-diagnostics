import { Component, type ErrorInfo, type ReactNode } from 'react'
export class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('UI boundary caught an error', error, info.componentStack)
  }
  render() {
    return this.state.failed ? (
      <main className="fatal">
        <p className="section-kicker">Interface error</p>
        <h1>This screen could not continue safely.</h1>
        <p>Reload the page. No image or screening history is preserved.</p>
        <button className="button button-primary" type="button" onClick={() => location.reload()}>
          Reload interface
        </button>
      </main>
    ) : (
      this.props.children
    )
  }
}
