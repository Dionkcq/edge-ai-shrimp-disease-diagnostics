import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { EvidenceOverlay } from '../src/features/inspect/EvidenceOverlay'
import { fixtureResult } from './support/fixtures'

describe('evidence overlay', () => {
  it('maps normalized original-frame xyxy boxes to percentages and textual evidence', () => {
    const result = fixtureResult('WHITE_SPOT_MARKER_DETECTED')
    render(<EvidenceOverlay previewUrl="blob:image" result={result} />)
    const marker = screen.getByRole('button', { name: /white spot marker 1/i })
    expect(marker).toHaveStyle({ left: '54%', top: '38%', width: '7%', height: '8%' })
    fireEvent.click(marker)
    expect(screen.getByText(/magnified marker/i)).toBeInTheDocument()
    expect(screen.getByText(/x 54–61%/i)).toBeInTheDocument()
  })
})
