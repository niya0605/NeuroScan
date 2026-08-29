export const CLASS_NAMES = [
  "glioma",
  "meningioma",
  "notumor",
  "pituitary",
] as const;
export const LOW_CONFIDENCE_THRESHOLD = 85;

export function isNoTumor(value?: string) {
  return value?.toLowerCase().replace(/[\s_-]/g, "") === "notumor";
}

export function displayLabel(value: string) {
  return isNoTumor(value)
    ? "No tumor"
    : value.replace(/\b\w/g, letter => letter.toUpperCase());
}

export function resultLabel(prediction: string, confidence: number) {
  if (confidence < LOW_CONFIDENCE_THRESHOLD)
    return `Inconclusive · most likely ${displayLabel(prediction)}`;
  return isNoTumor(prediction)
    ? "No tumor detected"
    : `${displayLabel(prediction)} detected`;
}
