// Runs Basic Pitch (Spotify, Apache-2.0) off the page's main thread, with its own ONNX Runtime,
// so the live "sketch" never waits behind Transkun's GPU work (2026-11-13).
// Messages in:  { type: "init", model: Uint8Array } | { type: "window", id, samples: Float32Array(43844) }
// Messages out: { type: "ready" } | { type: "frames", id, note: Float32Array(172*88), onset: Float32Array(172*88) } | { type: "error", message }

importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js");
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = 1;

let session = null;
let chain = Promise.resolve();

self.onmessage = (e) => {
  const m = e.data;
  chain = chain
    .then(async () => {
      if (m.type === "init") {
        session = await ort.InferenceSession.create(m.model, { executionProviders: ["wasm"] });
        self.postMessage({ type: "ready" });
      } else if (m.type === "window" && session) {
        const out = await session.run({ "serving_default_input_2:0": new ort.Tensor("float32", m.samples, [1, 43844, 1]) });
        // ONNX output names → Basic Pitch's outputs (basic_pitch/inference.py): :1 note, :2 onset, :0 contour.
        const note = out["StatefulPartitionedCall:1"].data.slice(), onset = out["StatefulPartitionedCall:2"].data.slice(); // own buffers, to transfer
        self.postMessage({ type: "frames", id: m.id, note, onset }, [note.buffer, onset.buffer]);
      }
    })
    .catch((err) => self.postMessage({ type: "error", message: String(err) }));
};
