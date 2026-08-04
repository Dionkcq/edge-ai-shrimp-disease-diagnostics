import { DECISIONS } from '../domain/decisions'
import type { Advice, ChatResponse, Guidance, Meta, Problem, ScreeningResult } from './types'

const decisionSet = new Set<string>([...DECISIONS])
const providerSet = new Set(['onnx', 'fixture', 'unavailable'])
const confidenceSet = new Set(['HIGH', 'MODERATE', 'LOW', 'NONE'])
const abstentionSet = new Set([
  'MODEL_UNAVAILABLE',
  'IMAGE_QUALITY_REJECTED',
  'LOW_CONFIDENCE',
  'INFERENCE_FAILED',
])
const qualityReasonSet = new Set([
  'IMAGE_TOO_SMALL',
  'IMAGE_TOO_BLURRY',
  'IMAGE_TOO_DARK',
  'IMAGE_TOO_BRIGHT',
  'IMAGE_LOW_CONTRAST',
])
const problemSet = new Set([
  'MALFORMED_REQUEST',
  'PAYLOAD_TOO_LARGE',
  'UNSUPPORTED_MEDIA_TYPE',
  'UNDECODABLE_IMAGE',
  'SERVICE_BUSY',
  'NOT_FOUND',
  'INTERNAL_ERROR',
  'ADVICE_UNAVAILABLE',
])

function object(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}
function string(value: unknown): value is string {
  return typeof value === 'string'
}
function number(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}
function strings(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(string)
}
function fail(name: string): never {
  throw new Error(`The server returned an unexpected ${name}. No result was displayed.`)
}
function citedSources(value: unknown): boolean {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every((item) => {
      const source = object(item)
      return (
        source !== null &&
        [source.id, source.title, source.publisher, source.url, source.accessed_on].every(string)
      )
    })
  )
}
function nonEmptyStrings(value: unknown): value is string[] {
  return strings(value) && value.length > 0 && value.every((item) => item.trim().length > 0)
}

export function parseScreeningResult(value: unknown): ScreeningResult {
  const root = object(value)
  if (!root) fail('screening response')
  const quality = object(root.quality)
  const metrics = object(quality?.metrics)
  const image = object(root.image)
  const model = object(root.model)
  const timings = object(root.timings_ms)
  if (
    root.schema_version !== '1.0.0' ||
    !string(root.request_id) ||
    !decisionSet.has(String(root.decision)) ||
    !confidenceSet.has(String(root.confidence_band)) ||
    !quality ||
    !metrics ||
    !image ||
    !model ||
    !timings ||
    !Array.isArray(root.markers) ||
    !strings(root.notices) ||
    !strings(root.guidance_refs) ||
    !strings(root.limitations)
  )
    fail('screening response')
  const abstention = root.abstention_reason
  if (
    (root.decision === 'UNABLE_TO_ASSESS') !==
    (string(abstention) && abstentionSet.has(abstention))
  )
    fail('screening response')
  if (
    !['PASS', 'FAIL'].includes(String(quality.status)) ||
    !Array.isArray(quality.reasons) ||
    !quality.reasons.every((r) => string(r) && qualityReasonSet.has(r)) ||
    !object(quality.metrics) ||
    !string(quality.policy_id) ||
    !string(quality.policy_hash)
  )
    fail('screening response')
  if (
    ![metrics.blur_score, metrics.mean_luminance, metrics.rms_contrast, metrics.min_side_px].every(
      number,
    ) ||
    !number(image.width) ||
    !number(image.height) ||
    !string(image.source_mode) ||
    typeof image.exif_transposed !== 'boolean'
  )
    fail('screening response')
  if (
    typeof model.available !== 'boolean' ||
    !providerSet.has(String(model.provider)) ||
    typeof model.is_demonstration_data !== 'boolean' ||
    (model.provider === 'unavailable' && model.available) ||
    (model.provider === 'fixture' && !model.is_demonstration_data) ||
    !object(model.class_names) ||
    !string(model.dataset_mapping_status)
  )
    fail('screening response')
  for (const item of root.markers) {
    const marker = object(item)
    const box = object(marker?.box)
    if (
      !marker ||
      !box ||
      !number(marker.class_index) ||
      !string(marker.class_name) ||
      !number(marker.score) ||
      ![box.x1, box.y1, box.x2, box.y2].every(number) ||
      Number(box.x2) < Number(box.x1) ||
      Number(box.y2) < Number(box.y1)
    )
      fail('screening response')
  }
  if (
    ![timings.intake_ms, timings.quality_ms, timings.inference_ms, timings.total_ms].every(number)
  )
    fail('screening response')
  return value as ScreeningResult
}

export function parseGuidance(value: unknown): Guidance {
  const root = object(value)
  if (
    !root ||
    !decisionSet.has(String(root.decision)) ||
    !string(root.id) ||
    !string(root.headline) ||
    !string(root.body) ||
    !citedSources(root.sources) ||
    !string(root.review_status) ||
    !string(root.review_note) ||
    !strings(root.limitations)
  )
    fail('guidance response')
  return value as Guidance
}
/** Reject any advice body that cannot be rendered with its full disclosure.
 *
 * `review_status`, `review_note` and `provider` are checked as strictly as the content
 * itself. An advice body that arrives without them is not "advice missing a label" this
 * interface could render anyway -- it is a response this interface refuses. */
export function parseAdvice(value: unknown): Advice {
  const root = object(value)
  if (
    !root ||
    !decisionSet.has(String(root.decision)) ||
    !string(root.summary) ||
    root.summary.trim().length === 0 ||
    !nonEmptyStrings(root.immediate_actions) ||
    !nonEmptyStrings(root.prevention_actions) ||
    !strings(root.additional_considerations) ||
    !string(root.based_on_guidance_id) ||
    !citedSources(root.sources) ||
    root.provider !== 'ollama' ||
    !string(root.model_id) ||
    root.review_status !== 'AI_GENERATED_NOT_REVIEWED' ||
    !string(root.review_note) ||
    root.review_note.trim().length === 0 ||
    !strings(root.limitations)
  )
    fail('advice response')
  return value as Advice
}
export function parseMeta(value: unknown): Meta {
  const root = object(value)
  if (!root) fail('service metadata')
  const decisions = root.decisions
  if (
    root.service !== 'shrimp-marker-screening' ||
    !string(root.schema_version) ||
    !string(root.environment) ||
    !providerSet.has(String(root.provider)) ||
    typeof root.model_available !== 'boolean' ||
    typeof root.is_demonstration_data !== 'boolean' ||
    !Array.isArray(decisions) ||
    decisions.length !== DECISIONS.length ||
    !decisions.every((d: unknown) => string(d) && decisionSet.has(d)) ||
    !string(root.guidance_review_status) ||
    !number(root.max_upload_bytes) ||
    typeof root.advice_available !== 'boolean' ||
    typeof root.chat_available !== 'boolean' ||
    root.offline !== true
  )
    fail('service metadata')
  return value as Meta
}
export function parseProblem(value: unknown): Problem {
  const root = object(value)
  if (
    !root ||
    !string(root.type) ||
    !string(root.title) ||
    !number(root.status) ||
    !string(root.detail) ||
    !string(root.code) ||
    !problemSet.has(root.code)
  )
    fail('error response')
  return {
    ...root,
    request_id: string(root.request_id) ? root.request_id : null,
    retry_after_seconds: number(root.retry_after_seconds) ? root.retry_after_seconds : null,
  } as Problem
}
export function parseChat(value: unknown): ChatResponse {
  const root = object(value)
  if (
    !root ||
    !string(root.session_id) ||
    !string(root.reply) ||
    !Array.isArray(root.messages) ||
    !root.messages.every((item) => {
      const message = object(item)
      return (
        message !== null &&
        ['user', 'assistant', 'tool'].includes(String(message.role)) &&
        string(message.content)
      )
    }) ||
    !Array.isArray(root.tool_calls) ||
    !root.tool_calls.every((item) => {
      const call = object(item)
      return (
        call !== null && string(call.name) && ['completed', 'failed'].includes(String(call.status))
      )
    }) ||
    (root.tool_result !== null &&
      root.tool_result !== undefined &&
      object(root.tool_result) === null)
  )
    fail('chat response')
  return root as ChatResponse
}
