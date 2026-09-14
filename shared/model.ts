export const CLASS_NAMES = [
  "glioma",
  "meningioma",
  "notumor",
  "pituitary",
] as const;
export const LOW_CONFIDENCE_THRESHOLD = 85;

// glioma and meningioma are the pair this model confuses most often -- per
// model/training_report.json, glioma recall is ~90% and nearly all of the
// remaining misses land as meningioma. Unlike a low top-1 confidence score,
// the model can be highly (and wrongly) confident on these specific
// mix-ups, so this flag isn't gated on confidence -- it always shows for
// either class, independent of how sure the model claims to be.
const CONFUSABLE_CLASSES = new Set(["glioma", "meningioma"]);

export function isNoTumor(value?: string) {
  return value?.toLowerCase().replace(/[\s_-]/g, "") === "notumor";
}

export function isConfusablePrediction(value?: string) {
  return value ? CONFUSABLE_CLASSES.has(value.toLowerCase()) : false;
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
