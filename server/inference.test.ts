import { describe, expect, it } from "vitest";
import { appRouter } from "./routers";
import type { TrpcContext } from "./_core/context";

const ctx = {
  user: undefined,
  req: {} as TrpcContext["req"],
  res: {} as TrpcContext["res"],
} satisfies TrpcContext;

describe("analyze", () => {
  it("rejects an upload without a data URL payload", async () => {
    const caller = appRouter.createCaller(ctx);
    await expect(
      caller.analyze({ imageDataUrl: "not-an-image" })
    ).rejects.toThrow("Upload a PNG or JPEG MRI image to analyze.");
  });

  it("rejects malformed image bytes before starting the model worker", async () => {
    const caller = appRouter.createCaller(ctx);
    await expect(
      caller.analyze({
        imageDataUrl:
          "data:image/jpeg;base64,bm90LWEt dmFsaWQtanBlZw==".replace(" ", ""),
      })
    ).rejects.toThrow("not a readable PNG or JPEG");
  });
});
