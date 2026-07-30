import { describe, expect, it } from 'vitest'
import { DECISIONS, decisionMeta } from '../src/domain/decisions'
import { parseGuidance, parseMeta, parseProblem, parseScreeningResult } from '../src/api/validate'
import { fixtureGuidance, fixtureMeta, fixtureProblem, fixtureResult } from './support/fixtures'

describe('runtime contract validation', () => {
  it('accepts complete screening, guidance, metadata and problem responses', () => {
    expect(parseScreeningResult(fixtureResult()).decision).toBe('MULTIPLE_TARGET_MARKERS_DETECTED')
    expect(parseGuidance(fixtureGuidance()).sources).toHaveLength(1)
    expect(parseMeta(fixtureMeta()).decisions).toEqual(DECISIONS)
    expect(parseProblem(fixtureProblem()).code).toBe('SERVICE_BUSY')
  })

  it('rejects malformed and contradictory screening responses', () => {
    expect(() => parseScreeningResult({ decision: 'WSSV_CONFIRMED' })).toThrow('screening response')
    const invalid = fixtureResult()
    invalid.model.provider = 'unavailable'
    invalid.model.available = true
    expect(() => parseScreeningResult(invalid)).toThrow('screening response')
  })

  it('keeps metadata exhaustive for all five backend decisions', () => {
    expect(Object.keys(decisionMeta).sort()).toEqual([...DECISIONS].sort())
  })
})
