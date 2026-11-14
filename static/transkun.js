// In-browser piano transcription with Transkun V2 (Yujia Yan, MIT), via ONNX Runtime Web.
//
// The two networks run as ONNX graphs (MIT export: huggingface.co/ravetools/piano-visualizer):
//   transkun_score.onnx: 16s of 44.1kHz stereo audio → interval scores [90, T, T] + features [90, T, 256]
//   transkun_attr.onnx:  features of each chosen interval [n, 768] → velocity, onset/offset refinement
// Everything between them is ported from Transkun's Python (ModelTransformer.transcribe,
// CRF.viterbiBackward, Data.resolveOverlapping) and checked note-for-note against it.
//
// Needs the global `ort` (onnxruntime-web). Usage:
//   const tk = await Transkun.load({ base: "static/models/", ep: "webgpu" });
//   const audio = await Transkun.decodeAudio(arrayBuffer);    // → [Float32Array, Float32Array] at 44.1kHz
//   const notes = await tk.transcribe(audio, (done, total) => …);  // → [{start, end, pitch, velocity}]

(function () {
  const FS = 44100, HOP = 1024;
  const SEG_S = 16, STEP_S = 8;
  const PITCHES = [-64, -67].concat(Array.from({ length: 88 }, (_, i) => 21 + i)); // two pedals + 88 keys
  const NSYM = PITCHES.length;
  const D = 256;

  async function decodeAudio(arrayBuffer) {
    // Decoding through an OfflineAudioContext resamples to its rate (44.1kHz, what Transkun expects).
    const ctx = new OfflineAudioContext(2, FS, FS);
    const buf = await ctx.decodeAudioData(arrayBuffer);
    const a = buf.getChannelData(0);
    const b = buf.numberOfChannels > 1 ? buf.getChannelData(1) : a;
    return [Float32Array.from(a), Float32Array.from(b)];
  }

  // Semi-CRF Viterbi over intervals for one pitch (CRF/NeuralSemiCRFInterval.viterbiBackward).
  // score is the flat [NSYM, T, T] output; score[p][e][b] is the score of a note on pitch p
  // beginning at frame b and ending at frame e. The noise ("skip") score is zero in this model.
  function viterbiPitch(score, p, T, forcedStart) {
    const base = p * T * T;
    const at = (e, b) => score[base + e * T + b];
    const q = new Float64Array(T);
    const ptr = new Int32Array(T); // ptr[s]: -1 = skip frame s, k ≥ 0 = interval (s, s+1+k)
    const last = at(T - 1, T - 1);
    q[T - 1] = last > 0 ? last : 0;
    for (let s = T - 2; s >= 0; s--) {
      let best = q[s + 1], sel = -1;       // skip (noise score 0); ties keep the first option, like torch.max
      for (let x = s + 1; x < T; x++) {
        const v = q[x] + at(x, s);
        if (v > best) { best = v; sel = x - s - 1; }
      }
      ptr[s] = sel;
      const d = at(s, s);
      q[s] = best + (d > 0 ? d : 0);       // single-frame notes ride on the diagonal
    }
    const out = [];
    let j = forcedStart;
    while (j < T - 1) {
      if (at(j, j) > 0) out.push([j, j]);
      const sel = ptr[j];
      if (sel < 0) j += 1;
      else { const i = sel + j + 1; out.push([j, i]); j = i; }
    }
    if (last > 0) out.push([T - 1, T - 1]);
    return out;
  }

  // Data.resolveOverlapping: a note ends where the next note of the same pitch begins.
  function resolveOverlapping(notes) {
    notes.sort((a, b) => a.start - b.start || a.end - b.end || a.pitch - b.pitch);
    const lastIdx = new Map();
    notes.forEach((n, i) => {
      if (lastIdx.has(n.pitch)) {
        const prev = notes[lastIdx.get(n.pitch)];
        if (prev.end > n.start) prev.end = n.start;
      }
      lastIdx.set(n.pitch, i);
    });
    return notes.sort((a, b) => a.start - b.start || a.end - b.end || a.pitch - b.pitch);
  }

  // `bytes` (optional): { "transkun_score.onnx": Uint8Array, … } already downloaded, e.g. with progress.
  async function load({ base = "static/models/", ep = "webgpu", bytes = {} } = {}) {
    const opts = { executionProviders: [ep], graphOptimizationLevel: "all" };
    const src = (name) => bytes[name] || base + name;
    // One at a time: ONNX Runtime Web refuses to initialise its WASM backend concurrently.
    const scoreSess = await ort.InferenceSession.create(src("transkun_score.onnx"), opts);
    // The attribute network is tiny and its batch size varies; WASM is the safe home for it.
    const attrSess = await ort.InferenceSession.create(src("transkun_attr.onnx"), { executionProviders: ["wasm"] });

    // Geometry of the sliding window (independent of the recording's length).
    const pad = SEG_S - STEP_S;
    const padSamples = Math.ceil(pad * FS);
    const step = Math.ceil((STEP_S * FS) / HOP) * HOP;
    const seg = Math.ceil(SEG_S * FS);
    const lastFrame = Math.round(seg / HOP);
    const frameDur = HOP / FS;
    const stepFrames = Math.floor(step / HOP);
    const newState = () => ({ startPos: new Array(NSYM).fill(Math.floor((pad * FS) / HOP)), eventsByPitch: new Map() });

    // ONNX Runtime Web can't run sessions concurrently; every run goes through this queue
    // (also exported, so other models on the page can share it).
    let gate = Promise.resolve();
    const lock = (fn) => {
      const r = gate.then(fn);
      gate = r.catch(() => {});
      return r;
    };

    // One window starting at sample i (the body of ModelTransformer.transcribe's loop).
    // `audio(from, to)` gives samples [from, to) with zeros where there's no audio.
    async function runChunk(st, i, audio) {
      const chunk = new Float32Array(2 * seg);
      for (let c = 0; c < 2; c++) chunk.set(audio(c, i - padSamples, i - padSamples + seg), c * seg);
      const res = await scoreSess.run({ audio: new ort.Tensor("float32", chunk, [2, seg]) });
      const score = res.score.data, ctx = res.ctx.data;
      const T = res.score.dims[2];
      const begin = i / FS - pad;

      // Decode every pitch, then collect interval features for the attribute network.
      const intervals = [];
      for (let p = 0; p < NSYM; p++) intervals.push(viterbiPitch(score, p, T, st.startPos[p]));
      const nInt = intervals.reduce((s, iv) => s + iv.length, 0);
      let vel = [], of = [], pres = [];
      if (nInt > 0) {
        const feats = new Float32Array(nInt * 3 * D);
        let k = 0;
        intervals.forEach((iv, p) => iv.forEach(([a, b]) => {
          const ca = (p * T + a) * D, cb = (p * T + b) * D, o = k * 3 * D;
          for (let d = 0; d < D; d++) {
            const x = ctx[ca + d], y = ctx[cb + d];
            feats[o + d] = x; feats[o + D + d] = y; feats[o + 2 * D + d] = x * y;
          }
          k++;
        }));
        const r = await attrSess.run({ attr: new ort.Tensor("float32", feats, [nInt, 3 * D]) });
        vel = Array.from(r.velocity.data, Number); of = r.of.data; pres = r.presence.data;
      }

      // Notes for this chunk (ModelTransformer.transcribeFrames), then merge across chunks.
      const lastP = [], events = [];
      let k = 0;
      intervals.forEach((iv, p) => {
        let lastEnd = 0, curLast = 0;
        for (const [a, b] of iv) {
          let start = Math.max((a + of[2 * k]) * frameDur, lastEnd);
          let end = Math.max((b + of[2 * k + 1]) * frameDur, start + 1e-8);
          lastEnd = end;
          const hasOnset = a > 0 || !!pres[2 * k];
          const hasOffset = b < lastFrame || !!pres[2 * k + 1];
          if (hasOffset) curLast = b;
          start = Math.max(start + begin, 0);
          end = Math.max(end + begin, start);
          events.push({ start, end, pitch: PITCHES[p], velocity: vel[k], hasOnset, hasOffset });
          k++;
        }
        lastP.push(curLast);
      });
      events.sort((x, y) => x.start - y.start || x.end - y.end || x.pitch - y.pitch);
      st.startPos = lastP.map((v) => Math.max(v - stepFrames, 0));

      for (const e of events) {
        const list = st.eventsByPitch.get(e.pitch) || [];
        if (list.length) {
          const prev = list[list.length - 1];
          if (e.start < prev.end) {
            if (e.hasOnset) list[list.length - 1] = e;
            else { prev.hasOffset = e.hasOffset; prev.end = Math.max(e.end, prev.end); }
            continue;
          }
        }
        if (e.hasOnset) { list.push(e); st.eventsByPitch.set(e.pitch, list); }
      }
    }

    function finishNotes(st) {
      for (const list of st.eventsByPitch.values()) if (list.length) list[list.length - 1].hasOffset = true;
      const all = [];
      for (const list of st.eventsByPitch.values()) for (const e of list) if (e.hasOffset) all.push(e);
      return resolveOverlapping(all)
        .filter((e) => e.pitch >= 21)  // -64 / -67 are the sustain and soft pedals
        .map(({ start, end, pitch, velocity }) => ({ start, end, pitch, velocity }));
    }

    // A whole recording at once.
    async function transcribe(channels, onProgress) {
      const n0 = channels[0].length, n = n0 + 2 * padSamples, total = Math.ceil(n / step);
      const st = newState();
      const audio = (c, from, to) => {
        const out = new Float32Array(to - from);
        const a = Math.max(0, from), b = Math.min(n0, to);
        if (b > a) out.set(channels[c].subarray(a, b), a - from);
        return out;
      };
      let done = 0;
      for (let i = 0; i < n; i += step) {
        await lock(() => runChunk(st, i, audio));
        onProgress?.(++done, total);
      }
      return finishNotes(st);
    }

    // While it's being played: push 44.1kHz mono as it arrives; each window is decoded as soon as
    // its audio is complete (about 8s behind), so finishing only has the last window or two left.
    function stream() {
      const st = newState();
      let buf = new Float32Array(FS * 30), len = 0, next = 0, pumping = null;
      const audio = (_c, from, to) => {
        const out = new Float32Array(to - from);
        const a = Math.max(0, from), b = Math.min(len, to);
        if (b > a) out.set(buf.subarray(a, b), a - from);
        return out;
      };
      async function pump() {
        while (next - padSamples + seg <= len) {
          const i = next;
          next += step;
          await lock(() => runChunk(st, i, audio));
        }
      }
      return {
        push(x) {
          if (len + x.length > buf.length) {
            const bigger = new Float32Array(Math.max(buf.length * 2, len + x.length));
            bigger.set(buf.subarray(0, len));
            buf = bigger;
          }
          buf.set(x, len);
          len += x.length;
        },
        pump() {
          pumping = (pumping || Promise.resolve()).then(pump);
          return pumping;
        },
        // Notes found so far, including ones still sounding (for a live display).
        notes() {
          const all = [];
          for (const list of st.eventsByPitch.values()) for (const e of list) if (e.pitch >= 21) all.push(e);
          return all.map(({ start, end, pitch, velocity }) => ({ start, end, pitch, velocity }));
        },
        async finish() {
          await (pumping || Promise.resolve());
          const n = len + 2 * padSamples;
          while (next < n) {
            const i = next;
            next += step;
            await lock(() => runChunk(st, i, audio));
          }
          return finishNotes(st);
        },
        get seconds() { return len / FS; },
        get audio() { return buf.subarray(0, len); },
      };
    }

    return { transcribe, stream, lock };
  }

  window.Transkun = { load, decodeAudio };
})();
