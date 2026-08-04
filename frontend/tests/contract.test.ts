import { describe, expect, it } from 'vitest'
import { DECISIONS, decisionMeta } from '../src/domain/decisions'
import {
  parseAdvice,
  parseGuidance,
  parseMeta,
  parseProblem,
  parseScreeningResult,
} from '../src/api/validate'
import {
  fixtureAdvice,
  fixtureGuidance,
  fixtureMeta,
  fixtureProblem,
  fixtureResult,
} from './support/fixtures'

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

  it('accepts a complete advice document', () => {
    expect(parseAdvice(fixtureAdvice()).review_status).toBe('AI_GENERATED_NOT_REVIEWED')
  })

  it('rejects advice that cannot be rendered with its full disclosure', () => {
    for (const mutate of [
      (a: Record<string, unknown>) => delete a.review_status,
      (a: Record<string, unknown>) => (a.review_status = 'EXPERT_REVIEWED'),
      (a: Record<string, unknown>) => (a.review_note = '   '),
      (a: Record<string, unknown>) => (a.provider = 'hosted-api'),
      (a: Record<string, unknown>) => (a.sources = []),
      (a: Record<string, unknown>) => (a.immediate_actions = []),
    ]) {
      const candidate = fixtureAdvice() as unknown as Record<string, unknown>
      mutate(candidate)
      expect(() => parseAdvice(candidate)).toThrow('advice response')
    }
  })

  it('keeps metadata exhaustive for all five backend decisions', () => {
    expect(Object.keys(decisionMeta).sort()).toEqual([...DECISIONS].sort())
  })
})
