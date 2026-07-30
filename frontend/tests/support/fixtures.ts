import type { Decision, Guidance, Meta, Problem, ScreeningResult } from '../../src/api/types'

export const decisions = [
  'GILL_DARKENING_MARKER_DETECTED',
  'WHITE_SPOT_MARKER_DETECTED',
  'MULTIPLE_TARGET_MARKERS_DETECTED',
  'NO_TARGET_MARKER_DETECTED',
  'UNABLE_TO_ASSESS',
] as const

export function fixtureMeta(): Meta {
  return {
    service: 'shrimp-marker-screening',
    schema_version: '1.0.0',
    environment: 'test',
    provider: 'fixture',
    model_available: true,
    is_demonstration_data: true,
    decisions: [...decisions],
    quality_policy_id: 'quality_policy_v1',
    quality_policy_hash: `sha256:${'a'.repeat(64)}`,
    decision_policy_id: 'decision_policy_v1',
    decision_policy_hash: `sha256:${'b'.repeat(64)}`,
    guidance_review_status: 'LITERATURE_REVIEWED_NOT_EXPERT_REVIEWED',
    max_upload_bytes: 12582912,
    offline: true,
  }
}

export function fixtureResult(
  decision: Decision = 'MULTIPLE_TARGET_MARKERS_DETECTED',
): ScreeningResult {
  const unable = decision === 'UNABLE_TO_ASSESS'
  const noMarker = decision === 'NO_TARGET_MARKER_DETECTED'
  const markers: ScreeningResult['markers'] =
    unable || noMarker
      ? []
      : [
          {
            class_index: 1,
            class_name: 'white_spot',
            role: 'WHITE_SPOT' as const,
            score: 0.79,
            box: { x1: 0.54, y1: 0.38, x2: 0.61, y2: 0.46 },
          },
          ...(decision === 'MULTIPLE_TARGET_MARKERS_DETECTED' ||
          decision === 'GILL_DARKENING_MARKER_DETECTED'
            ? [
                {
                  class_index: 0,
                  class_name: 'dark_gill',
                  role: 'GILL_DARKENING' as const,
                  score: 0.84,
                  box: { x1: 0.18, y1: 0.24, x2: 0.38, y2: 0.46 },
                },
              ]
            : []),
        ]
  return {
    schema_version: '1.0.0',
    request_id: '01JABCDEFGHJKMNPQRSTVWXYZ0',
    decision,
    abstention_reason: unable ? 'LOW_CONFIDENCE' : null,
    quality: {
      status: 'PASS',
      reasons: [],
      metrics: { blur_score: 200, mean_luminance: 120, rms_contrast: 40, min_side_px: 768 },
      policy_id: 'quality_policy_v1',
      policy_hash: `sha256:${'a'.repeat(64)}`,
    },
    markers,
    confidence_band: unable || noMarker ? 'NONE' : 'MODERATE',
    image: { width: 1024, height: 768, source_mode: 'RGB', exif_transposed: false },
    model: {
      available: true,
      provider: 'fixture',
      model_id: 'fixture-contract-v1',
      version: '0.0.0-synthetic',
      output_layout: 'ultralytics_v8_detect_v1',
      class_names: { '0': 'dark_gill', '1': 'white_spot' },
      dataset_mapping_status: 'PROVISIONAL_UNCONFIRMED',
      is_demonstration_data: true,
    },
    notices: [
      'DEMONSTRATION_DATA_NOT_A_REAL_RESULT',
      'DATASET_CLASS_MAPPING_UNCONFIRMED',
      'THRESHOLDS_UNCALIBRATED',
    ],
    guidance_refs: ['guide-v1'],
    limitations: ['lim-not-diagnostic'],
    timings_ms: { intake_ms: 10, quality_ms: 5, inference_ms: 25, total_ms: 40 },
  }
}

export function fixtureGuidance(): Guidance {
  return {
    decision: 'MULTIPLE_TARGET_MARKERS_DETECTED',
    id: 'guide-v1',
    headline: 'Both marker types were found',
    body: 'Repeat the capture under even light and inspect several animals.',
    sources: [
      {
        id: 'woah',
        title: 'Aquatic Manual',
        publisher: 'WOAH',
        url: 'https://www.woah.org/aquatic',
        accessed_on: '2026-07-30',
      },
    ],
    review_status: 'LITERATURE_REVIEWED_NOT_EXPERT_REVIEWED',
    review_note: 'Not reviewed by a qualified aquatic-animal health professional.',
    limitations: ['lim-not-diagnostic'],
  }
}

export function fixtureProblem(): Problem {
  return {
    type: '/problems/service-busy',
    title: 'Service busy',
    status: 503,
    detail: 'The service is already screening another photograph. Try again shortly.',
    code: 'SERVICE_BUSY',
    request_id: '01JABCDEFGHJKMNPQRSTVWXYZ0',
    retry_after_seconds: 2,
  }
}
