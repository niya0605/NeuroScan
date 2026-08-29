import { randomUUID } from "node:crypto";
import { mkdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import type { Express } from "express";

const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/png", "image/jpeg"]);
const WORKER_TIMEOUT_MS = 45_000;

function hasSupportedImageSignature(buffer: Buffer) {
  const jpeg =
    buffer.length >= 3 &&
    buffer[0] === 0xff &&
    buffer[1] === 0xd8 &&
    buffer[2] === 0xff;
  const png =
    buffer.length >= 8 &&
    buffer
      .subarray(0, 8)
      .equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
  return jpeg || png;
}

export type InferenceResult = {
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  originalImage: string;
  gradcam: string;
  imageWidth: number;
  imageHeight: number;
};

function runWorker(imagePath: string): Promise<InferenceResult> {
  return new Promise((resolve, reject) => {
    const worker = spawn(
      "python3",
      [path.resolve(process.cwd(), "scripts/infer.py")],
      { stdio: ["pipe", "pipe", "pipe"] }
    );

    let stdout = "";
    let stderr = "";
    let settled = false;

    const finish = (callback: () => void) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      callback();
    };

    const timeout = setTimeout(() => {
      worker.kill("SIGTERM");
      finish(() =>
        reject(
          new Error(
            "Analysis timed out. Please upload a valid PNG or JPEG MRI image."
          )
        )
      );
    }, WORKER_TIMEOUT_MS);

    worker.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });

    worker.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });

    worker.once("error", error =>
      finish(() => reject(error))
    );

    worker.once("close", code => {
      finish(() => {
        if (code !== 0) {
          console.error("[Inference] Python worker error:", stderr);
          reject(
            new Error(
              "The model could not analyze this image. Please upload a valid PNG or JPEG MRI image."
            )
          );
          return;
        }

        try {
          const response = JSON.parse(stdout.trim()) as {
            ok: boolean;
            result?: InferenceResult;
            error?: string;
          };

          if (!response.ok || !response.result) {
            reject(
              new Error(
                response.error ||
                  "The model could not analyze this image."
              )
            );
            return;
          }

          resolve(response.result);
        } catch {
          console.error("[Inference] Invalid worker response:", stdout);
          reject(new Error("Inference worker returned an invalid response"));
        }
      });
    });

    worker.stdin.write(
      JSON.stringify({ image_path: imagePath }) + "\n"
    );

    worker.stdin.end();
  });
}

export async function analyzeImageDataUrl(
  imageDataUrl: string,
  mimeType?: string
): Promise<InferenceResult> {
  if (!imageDataUrl.includes(","))
    throw new Error("Upload a PNG or JPEG MRI image to analyze.");
  if (mimeType && !ALLOWED_TYPES.has(mimeType))
    throw new Error("Only PNG and JPEG MRI images are supported.");
  const encoded = imageDataUrl.split(",", 2)[1] ?? "";
  const imageBuffer = Buffer.from(encoded, "base64");
  if (imageBuffer.byteLength === 0 || imageBuffer.byteLength > MAX_IMAGE_BYTES)
    throw new Error("Please upload an image smaller than 12 MB.");
  if (!hasSupportedImageSignature(imageBuffer))
    throw new Error("The uploaded file is not a readable PNG or JPEG image.");
  const tempDir = path.join(os.tmpdir(), "neurolens");
  await mkdir(tempDir, { recursive: true });
  const tempPath = path.join(tempDir, `${randomUUID()}.upload`);
  try {
    await writeFile(tempPath, imageBuffer);
    return await runWorker(tempPath);
  } finally {
    await rm(tempPath, { force: true }).catch(() => undefined);
  }
}

export function registerInferenceRoute(app: Express) {
  app.post("/api/analyze", async (req, res) => {
    try {
      const result = await analyzeImageDataUrl(
        req.body.imageDataUrl,
        req.body.mimeType
      );
      res.json(result);
    } catch (error) {
      console.error(
        "[Inference] request failed",
        error instanceof Error ? error.message : error
      );
      res
        .status(400)
        .json({
          error:
            error instanceof Error && error.message.startsWith("Upload")
              ? error.message
              : "The model could not analyze this image. Please upload a valid PNG or JPEG MRI image.",
        });
    }
  });
}
