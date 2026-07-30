import type { Decision, Guidance, Meta, Problem, ScreeningResult } from './types'
import { parseGuidance, parseMeta, parseProblem, parseScreeningResult } from './validate'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly problem: Problem | null = null,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}
async function json(response: Response): Promise<unknown> {
  try {
    return (await response.json()) as unknown
  } catch {
    throw new ApiError('The server returned unreadable data. No result was displayed.')
  }
}
export async function getMeta(signal?: AbortSignal): Promise<Meta> {
  const response = await fetch('/api/v1/meta', signal ? { signal } : {})
  if (!response.ok) throw new ApiError('The screening service is unavailable.')
  try {
    return parseMeta(await json(response))
  } catch {
    throw new ApiError('The server returned unexpected metadata. Reload before screening.')
  }
}
export async function screenImage(
  file: File,
  scenario: string | null,
  signal: AbortSignal,
): Promise<ScreeningResult> {
  const body = new FormData()
  body.append('image', file)
  const query = scenario ? `?scenario=${encodeURIComponent(scenario)}` : ''
  let response: Response
  try {
    response = await fetch(`/api/v1/screenings${query}`, { method: 'POST', body, signal })
  } catch (error) {
    if (
      signal.aborted ||
      (error instanceof DOMException && error.name === 'AbortError') ||
      (error instanceof Error && error.name === 'AbortError')
    ) {
      throw new DOMException('The screening request was cancelled.', 'AbortError')
    }
    throw new ApiError(
      'The service could not be reached. Check the local connection and try again.',
    )
  }
  const payload = await json(response)
  if (!response.ok) {
    let problem: Problem | null = null
    try {
      problem = parseProblem(payload)
    } catch {
      throw new ApiError('The service returned an unexpected error. No result was displayed.')
    }
    throw new ApiError(problem.detail, problem)
  }
  try {
    return parseScreeningResult(payload)
  } catch {
    throw new ApiError('The server returned an unexpected response. No result was displayed.')
  }
}
export async function getGuidance(decision: Decision, signal?: AbortSignal): Promise<Guidance> {
  const response = await fetch(`/api/v1/guidance/${decision}`, signal ? { signal } : {})
  if (!response.ok) throw new ApiError('Local guidance is unavailable for this result.')
  try {
    return parseGuidance(await json(response))
  } catch {
    throw new ApiError('Local guidance did not match the expected contract.')
  }
}
