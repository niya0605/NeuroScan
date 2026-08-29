const response = await fetch("http://localhost:3000/api/trpc/analyze?batch=1", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({
    0: {
      json: {
        imageDataUrl: "data:image/png;base64,invalid",
        mimeType: "image/png",
      },
    },
  }),
});
const text = await response.text();
console.log({
  status: response.status,
  responseBytes: text.length,
  preview: text.slice(0, 260),
});
if (
  !text.includes("model could not analyze") &&
  !text.includes("cannot identify image file") &&
  !text.includes("Inference")
)
  process.exit(1);
