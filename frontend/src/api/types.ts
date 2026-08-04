export const DECISIONS = [
  'GILL_DARKENING_MARKER_DETECTED',
  'WHITE_SPOT_MARKER_DETECTED',
  'MULTIPLE_TARGET_MARKERS_DETECTED',
  'NO_TARGET_MARKER_DETECTED',
  'UNABLE_TO_ASSESS',
] as const
export type Decision = (typeof DECISIONS)[number]
export type ConfidenceBand = 'HIGH' | 'MODERATE' | 'LOW' | 'NONE'
export type QualityReason =
  | 'IMAGE_TOO_SMALL'
  | 'IMAGE_TOO_BLURRY'
  | 'IMAGE_TOO_DARK'
  | 'IMAGE_TOO_BRIGHT'
  | 'IMAGE_LOW_CONTRAST'
export type AbstentionReason =
  'MODEL_UNAVAILABLE' | 'IMAGE_QUALITY_REJECTED' | 'LOW_CONFIDENCE' | 'INFERENCE_FAILED'
export type Provider = 'onnx' | 'fixture' | 'unavailable'
export type MarkerRole = 'WHITE_SPOT' | 'GILL_DARKENING'
export type Notice =
  | 'MODEL_NOT_INSTALLED'
  | 'DEMONSTRATION_DATA_NOT_A_REAL_RESULT'
  | 'DATASET_CLASS_MAPPING_UNCONFIRMED'
  | 'THRESHOLDS_UNCALIBRATED'

export type BoundingBox = { x1: number; y1: number; x2: number; y2: number }
export type MarkerObservation = {
  class_index: number
  class_name: string
  role: MarkerRole | null
  score: number
  box: BoundingBox
}
export type ScreeningResult = {
  schema_version: '1.0.0'
  request_id: string
  decision: Decision
  abstention_reason: AbstentionReason | null
  quality: {
    status: 'PASS' | 'FAIL'
    reasons: QualityReason[]
    metrics: {
      blur_score: number
      mean_luminance: number
      rms_contrast: number
      min_side_px: number
    }
    policy_id: string
    policy_hash: string
  }
  markers: MarkerObservation[]
  confidence_band: ConfidenceBand
  image: { width: number; height: number; source_mode: string; exif_transposed: boolean }
  model: {
    available: boolean
    provider: Provider
    model_id: string | null
    version: string | null
    output_layout: 'ultralytics_v8_detect_v1' | 'custom_yolo_anchor_v1' | null
    class_names: Record<string, string>
    dataset_mapping_status: 'PROVISIONAL_UNCONFIRMED' | 'AUTHOR_CONFIRMED' | 'NOT_APPLICABLE'
    is_demonstration_data: boolean
  }
  notices: Notice[]
  guidance_refs: string[]
  limitations: string[]
  timings_ms: { intake_ms: number; quality_ms: number; inference_ms: number; total_ms: number }
}
export type CitedSource = {
  id: string
  title: string
  publisher: string
  url: string
  accessed_on: string
}
export type Guidance = {
  decision: Decision
  id: string
  headline: string
  body: string
  sources: CitedSource[]
  review_status: string
  review_note: string
  limitations: string[]
}
/** Body of `GET /api/v1/advice/{decision}`. Optional, off by default; see `Meta.advice_available`.
 *
 * `review_status` and `provider` are fixed literals in the contract and are validated as such:
 * there is no arrangement in which this interface renders the text without also rendering the
 * fact that a language model wrote it and nobody reviewed it. */
export type Advice = {
  decision: Decision
  summary: string
  immediate_actions: string[]
  prevention_actions: string[]
  additional_considerations: string[]
  based_on_guidance_id: string
  sources: CitedSource[]
  provider: 'ollama'
  model_id: string
  review_status: 'AI_GENERATED_NOT_REVIEWED'
  review_note: string
  limitations: string[]
}
export type Meta = {
  service: 'shrimp-marker-screening'
  schema_version: string
  environment: string
  provider: Provider
  model_available: boolean
  is_demonstration_data: boolean
  decisions: Decision[]
  quality_policy_id: string
  quality_policy_hash: string
  decision_policy_id: string
  decision_policy_hash: string
  guidance_review_status: string
  max_upload_bytes: number
  advice_available: boolean
  offline: true
}
export type ProblemCode =
  | 'MALFORMED_REQUEST'
  | 'PAYLOAD_TOO_LARGE'
  | 'UNSUPPORTED_MEDIA_TYPE'
  | 'UNDECODABLE_IMAGE'
  | 'SERVICE_BUSY'
  | 'NOT_FOUND'
  | 'INTERNAL_ERROR'
  | 'ADVICE_UNAVAILABLE'
export type Problem = {
  type: string
  title: string
  status: number
  detail: string
  code: ProblemCode
  request_id: string | null
  retry_after_seconds: number | null
}
