import { randomUUID } from "node:crypto";
import { mkdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import type { Express } from "express";

const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/png", "image/jpeg"]);
const WORKER_TIMEOUT_MS = 120_000;

export type InferenceResult = {
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  originalImage: string;
  gradcam: string;
  imageWidth: number;
  imageHeight: number;
};

type WorkerResponse = {
  ok: boolean;
  result?: InferenceResult;
  error?: string;
};

let workerProcess: ChildProcessWithoutNullStreams | null = null;
let workerBuffer = "";
let workerQueue: Promise<unknown> = Promise.resolve();

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

function startWorker() {
  if (workerProcess && !workerProcess.killed) {
    return workerProcess;
  }

  workerBuffer = "";

  const worker = spawn(
    "python3",
    [path.resolve(process.cwd(), "scripts/infer.py")],
    {
      stdio: ["pipe", "pipe", "pipe"],
    }
  );

  workerProcess = worker;

  worker.stderr.on("data", (chunk: Buffer) => {
    console.error("[Inference]", chunk.toString().trim());
  });

  worker.stdout.on("data", (chunk: Buffer) => {
    workerBuffer += chunk.toString();
  });

  worker.once("exit", (code, signal) => {
    console.error(
      `[Inference] Python worker exited. code=${code} signal=${signal}`
    );

    workerProcess = null;
    workerBuffer = "";
  });

  worker.once("error", error => {
    console.error("[Inference] Python worker error:", error);
    workerProcess = null;
    workerBuffer = "";
  });

  return worker;
}

function runWorker(imagePath: string): Promise<InferenceResult> {
  const execute = () =>
    new Promise<InferenceResult>((resolve, reject) => {
      const worker = startWorker();

      const timeout = setTimeout(() => {
        reject(
          new Error(
            "Analysis timed out. Please try again with a valid PNG or JPEG MRI image."
          )
        );
      }, WORKER_TIMEOUT_MS);

      const checkResponse = () => {
        const newlineIndex = workerBuffer.indexOf("\n");

        if (newlineIndex === -1) {
          return;
        }

        const line = workerBuffer.slice(0, newlineIndex).trim();
        workerBuffer = workerBuffer.slice(newlineIndex + 1);

        if (!line) {
          checkResponse();
          return;
        }

        clearTimeout(timeout);

        let response: WorkerResponse;

        try {
          response = JSON.parse(line) as WorkerResponse;
        } catch {
          reject(new Error("Inference worker returned an invalid response."));
          return;
        }

        if (!response.ok || !response.result) {
          reject(
            new Error(
              response.error || "The model could not analyze this image."
            )
          );
          return;
        }

        resolve(response.result);
      };

      const onData = () => checkResponse();

      worker.stdout.on("data", onData);

      worker.stdin.write(
        JSON.stringify({
          image_path: imagePath,
        }) + "\n"
      );

      const cleanup = () => {
        worker.stdout.off("data", onData);
      };

      void Promise.resolve().then(() => {
        if (!workerProcess) {
          clearTimeout(timeout);
          cleanup();
          reject(new Error("Inference worker is unavailable."));
        }
      });

      const originalResolve = resolve;
      const originalReject = reject;

      resolve = value => {
        cleanup();
        originalResolve(value);
      };

      reject = error => {
        cleanup();
        originalReject(error);
      };
    });

  const queued = workerQueue.then(execute, execute);

  workerQueue = queued.catch(() => undefined);

  return queued as Promise<InferenceResult>;
}

export async function analyzeImageDataUrl(
  imageDataUrl: string,
  mimeType?: string
): Promise<InferenceResult> {
  if (!imageDataUrl.includes(",")) {
    throw new Error("Upload a PNG or JPEG MRI image to analyze.");
  }

  if (mimeType && !ALLOWED_TYPES.has(mimeType)) {
    throw new Error("Only PNG and JPEG MRI images are supported.");
  }

  const encoded = imageDataUrl.split(",", 2)[1] ?? "";
  const imageBuffer = Buffer.from(encoded, "base64");

  if (
    imageBuffer.byteLength === 0 ||
    imageBuffer.byteLength > MAX_IMAGE_BYTES
  ) {
    throw new Error("Please upload an image smaller than 12 MB.");
  }

  if (!hasSupportedImageSignature(imageBuffer)) {
    throw new Error("The uploaded file is not a readable PNG or JPEG image.");
  }

  const tempDir = path.join(os.tmpdir(), "neurolens");
  await mkdir(tempDir, { recursive: true });

  const tempPath = path.join(
    tempDir,
    `${randomUUID()}.upload`
  );

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

      res.status(400).json({
        error:
          error instanceof Error &&
          error.message.startsWith("Upload")
            ? error.message
            : error instanceof Error
              ? error.message
              : "The model could not analyze this image.",
      });
    }
  });
}