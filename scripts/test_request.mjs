import { readFile } from "node:fs/promises";
const image = await readFile(
  "/home/ubuntu/webdev-static-assets/mri-editorial-texture.png"
);
const body = JSON.stringify({
  imageDataUrl: `data:image/png;base64,${image.toString("base64")}`,
  mimeType: "image/png",
});
const response = await fetch("http://localhost:3000/api/trpc/analyze?batch=1", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({
    0: {
      json: {
        imageDataUrl: `data:image/png;base64,${image.toString("base64")}`,
        mimeType: "image/png",
      },
    },
  }),
});
const text = await response.text();
console.log({
  status: response.status,
  responseBytes: text.length,
  preview: text.slice(0, 220),
});
