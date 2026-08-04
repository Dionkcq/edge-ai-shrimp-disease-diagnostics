import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { run as runAxe } from 'axe-core'
import { describe, expect, it, vi } from 'vitest'
import { App } from '../src/app/App'
import { DECISIONS, decisionMeta } from '../src/domain/decisions'
import { ResultPanel } from '../src/features/result/ResultPanel'
import { fixtureResult } from './support/fixtures'

describe('operate surface', () => {
  for (const decision of DECISIONS) {
    it(`renders safe copy for ${decision}`, () => {
      const result = fixtureResult(decision)
      render(
        <ResultPanel
          result={result}
          previewUrl="blob:image"
          guidance={null}
          guidanceState="idle"
          adviceAvailable={false}
          advice={null}
          adviceState="idle"
          adviceError={null}
          onRequestAdvice={() => undefined}
        />,
      )
      expect(screen.getByText('Screening result')).toBeInTheDocument()
      expect(screen.getByRole('heading', { level: 2 })).toBeInTheDocument()
      expect(screen.getByText(decisionMeta[decision].summary)).toBeInTheDocument()
    })
  }

  it('starts accessible, with camera attributes and persistent safety language', async () => {
    const { container } = render(<App />)
    const input = screen.getByLabelText(/take or choose/i)
    expect(input).toHaveAttribute('accept', 'image/jpeg,image/png')
    expect(input).toHaveAttribute('capture', 'environment')
    expect(screen.getByText(/not pathogen confirmation/i)).toBeInTheDocument()
    const accessibility = await runAxe(container, {
      rules: { 'color-contrast': { enabled: false } },
    })
    expect(accessibility.violations).toHaveLength(0)
  })

  it('selects, removes and revokes a local preview without uploading', async () => {
    const user = userEvent.setup()
    render(<App />)
    const file = new File(['image'], 'pond-shrimp.jpg', { type: 'image/jpeg' })
    await user.upload(screen.getByLabelText(/take or choose/i), file)
    expect(screen.getByText('pond-shrimp.jpg')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /remove image/i }))
    const revokeObjectUrl = vi.mocked(URL.revokeObjectURL)
    expect(revokeObjectUrl).toHaveBeenCalledWith('blob:test-image')
  })
})
