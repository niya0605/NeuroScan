import { describe, expect, it } from "vitest";
import {
  CLASS_NAMES,
  LOW_CONFIDENCE_THRESHOLD,
  resultLabel,
} from "@shared/model";

describe("model contract", () => {
  it("keeps the trained four-class order stable", () => {
    expect(CLASS_NAMES).toEqual([
      "glioma",
      "meningioma",
      "notumor",
      "pituitary",
    ]);
  });

  it("does not present weak scores as definitive diagnoses", () => {
    expect(
      resultLabel("meningioma", LOW_CONFIDENCE_THRESHOLD - 0.01)
    ).toContain("Inconclusive");
    expect(resultLabel("notumor", 90)).toBe("No tumor detected");
  });
});
