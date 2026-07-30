import type { AbstentionReason, Decision, QualityReason } from '../api/types'
export { DECISIONS } from '../api/types'

export type DecisionMeta = {
  title: string
  summary: string
  tone: 'marker' | 'clear' | 'indeterminate'
  next: string
}
export const decisionMeta: Record<Decision, DecisionMeta> = {
  GILL_DARKENING_MARKER_DETECTED: {
    title: 'Dark-gill-like marker seen',
    summary: 'The screen marked a dark region around the gills in this photograph.',
    tone: 'marker',
    next: 'Retake in even light, compare several shrimp, and document what you see.',
  },
  WHITE_SPOT_MARKER_DETECTED: {
    title: 'White-spot-like marker seen',
    summary: 'The screen marked a white-spot-like region in this photograph.',
    tone: 'marker',
    next: 'Check for reflections or debris, then photograph more than one shrimp.',
  },
  MULTIPLE_TARGET_MARKERS_DETECTED: {
    title: 'Both marker types seen',
    summary: 'The screen marked white-spot-like and dark-gill-like regions.',
    tone: 'marker',
    next: 'Repeat the capture and preserve clear records for a qualified professional.',
  },
  NO_TARGET_MARKER_DETECTED: {
    title: 'No target marker seen',
    summary:
      'Neither of the two visible markers was retained in this photograph. This does not mean the shrimp is healthy or disease-free.',
    tone: 'clear',
    next: 'Keep observing the pond. Escalate unusual behaviour or mortality.',
  },
  UNABLE_TO_ASSESS: {
    title: 'Unable to assess this image',
    summary: 'No marker result was produced for this photograph.',
    tone: 'indeterminate',
    next: 'Follow the specific instruction below, then try again.',
  },
}

export const qualityCopy: Record<QualityReason, string> = {
  IMAGE_TOO_SMALL: 'The image is too small — move closer and keep the whole shrimp in frame.',
  IMAGE_TOO_BLURRY: 'The image is blurred — steady the phone and tap to focus before retaking.',
  IMAGE_TOO_DARK: 'The image is too dark — move into brighter, even light or use flash.',
  IMAGE_TOO_BRIGHT: 'The image is overexposed — move out of direct glare and retake.',
  IMAGE_LOW_CONTRAST: 'The shrimp does not stand out — use an even, plain background and retake.',
}
export const abstentionCopy: Record<AbstentionReason, string> = {
  MODEL_UNAVAILABLE:
    'No screening model is installed. Image handling can be checked, but the shrimp cannot be assessed.',
  IMAGE_QUALITY_REJECTED:
    'The server rejected the image quality. Retake it using the instruction below.',
  LOW_CONFIDENCE:
    'The visible evidence was too weak or uncertain. Retake from another angle under even light.',
  INFERENCE_FAILED:
    'The screening step did not complete. Try once more; if it repeats, the service needs attention.',
}
