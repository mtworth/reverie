// The live "sketch" (2026-11-13): Basic Pitch (Spotify, Apache-2.0, 230KB) transcribes the last
// couple of seconds as you play, ~30x faster than real time on the CPU, in a worker. Its notes are
// drawn as faint pencil strokes a second or two behind you; Transkun's notes ink over them as each
// of its 8s windows lands (~10s behind). The saved sketch is always Transkun's.
//
// Windowing, unwrapping and note creation are ports of basic_pitch/inference.py and
// basic_pitch/note_creation.py (output_to_notes_polyphonic, with inferred onsets and the melodia
// trick), at Basic Pitch's default thresholds.
//
//   await Sketch.load(modelBytes);
//   const s = Sketch.stream(); s.push(audio44k); await s.pump(); s.notes(fromSeconds) → [[pitch, onset_s, dur_s, velocity]]

window.Sketch = (() => {
  const SR = 22050, FFT_HOP = 256, N = SR * 2 - FFT_HOP; // AUDIO_N_SAMPLES = 43844
  const FPS = Math.floor(SR / FFT_HOP); // ANNOTATIONS_FPS = 86
  const FRAMES = FPS * 2; // ANNOT_N_FRAMES = 172 per window
  const N_OLAP = 30, OLAP_LEN = N_OLAP * FFT_HOP, STEP = N - OLAP_LEN, TRIM = N_OLAP / 2, KEEP = FRAMES - 2 * TRIM;
  const WINDOW_OFFSET = (FFT_HOP / SR) * (FRAMES - N / FFT_HOP) + 0.0018; // their "magic number" alignment
  const NF = 88, MIDI_OFFSET = 21, ONSET_T = 0.5, FRAME_T = 0.3, ENERGY_TOL = 11;
  const MIN_LEN = Math.round((127.7 / 1000) * (SR / FFT_HOP)); // 11 frames

  // ---------- the worker ----------
  let worker = null, ready = null, seq = 0;
  const waiting = new Map();
  function load(model) {
    if (ready) return ready;
    worker = new Worker("static/sketch_worker.js");
    ready = new Promise((resolve, reject) => {
      worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === "ready") resolve();
        else if (m.type === "frames") {
          waiting.get(m.id)?.(m);
          waiting.delete(m.id);
        } else if (m.type === "error") {
          console.error("sketch worker", m.message);
          reject(new Error(m.message));
        }
      };
      worker.postMessage({ type: "init", model });
    });
    return ready;
  }
  const runWindow = (samples) =>
    new Promise((resolve) => {
      const id = ++seq;
      waiting.set(id, resolve);
      worker.postMessage({ type: "window", id, samples }, [samples.buffer]);
    });

  // Frame index → seconds (note_creation.model_frames_to_time).
  const frameTime = (f) => (f * FFT_HOP) / SR - WINDOW_OFFSET * Math.floor(f / FRAMES);

  // ---------- note creation (output_to_notes_polyphonic) on frames [f0, f1) ----------
  function notesFrom(note, onset, f0, f1) {
    const n = f1 - f0;
    if (n < 3) return [];
    const at = (t, f) => (t + f0) * NF + f;
    const frames = new Float32Array(n * NF), ons = new Float32Array(n * NF);
    for (let t = 0; t < n; t++) for (let f = 0; f < NF; f++) { frames[t * NF + f] = note[at(t, f)]; ons[t * NF + f] = onset[at(t, f)]; }
    // Inferred onsets: big rises in frame energy (min of 1- and 2-frame differences), rescaled.
    const diff = new Float32Array(n * NF);
    let maxOn = 0, maxDiff = 0;
    for (let t = 0; t < n; t++)
      for (let f = 0; f < NF; f++) {
        const x = frames[t * NF + f];
        const d1 = x - (t >= 1 ? frames[(t - 1) * NF + f] : 0), d2 = x - (t >= 2 ? frames[(t - 2) * NF + f] : 0);
        const d = t < 2 ? 0 : Math.max(0, Math.min(d1, d2));
        diff[t * NF + f] = d;
        maxDiff = Math.max(maxDiff, d);
        maxOn = Math.max(maxOn, ons[t * NF + f]);
      }
    if (maxDiff > 0) for (let i = 0; i < diff.length; i++) ons[i] = Math.max(ons[i], (maxOn * diff[i]) / maxDiff);
    // Onset peaks (strict local maxima in time) above the threshold, latest first.
    const peaks = [];
    for (let t = 1; t < n - 1; t++)
      for (let f = 0; f < NF; f++) {
        const x = ons[t * NF + f];
        if (x >= ONSET_T && x > ons[(t - 1) * NF + f] && x > ons[(t + 1) * NF + f]) peaks.push([t, f]);
      }
    peaks.reverse();
    const energy = Float32Array.from(frames);
    const out = [];
    const clearAround = (t, f) => {
      energy[t * NF + f] = 0;
      if (f < NF - 1) energy[t * NF + f + 1] = 0;
      if (f > 0) energy[t * NF + f - 1] = 0;
    };
    const add = (a, b, f) => {
      let s = 0;
      for (let t = a; t < b; t++) s += frames[t * NF + f];
      const amp = s / Math.max(1, b - a);
      const start = frameTime(a + f0), end = frameTime(b + f0);
      out.push([f + MIDI_OFFSET, start, Math.max(0.05, end - start), Math.max(1, Math.min(127, Math.round(amp * 127)))]);
    };
    for (const [t0, f] of peaks) {
      if (t0 >= n - 1) continue;
      let i = t0 + 1, k = 0;
      while (i < n - 1 && k < ENERGY_TOL) {
        k = energy[i * NF + f] < FRAME_T ? k + 1 : 0;
        i++;
      }
      i -= k;
      if (i - t0 <= MIN_LEN) continue;
      for (let t = t0; t < i; t++) clearAround(t, f);
      add(t0, i, f);
    }
    // The melodia trick: notes with no clear onset, grown out from the strongest remaining energy.
    for (;;) {
      let best = -1, bi = -1;
      for (let j = 0; j < energy.length; j++) if (energy[j] > best) { best = energy[j]; bi = j; }
      if (best <= FRAME_T) break;
      const iMid = Math.floor(bi / NF), f = bi % NF;
      energy[bi] = 0;
      let i = iMid + 1, k = 0;
      while (i < n - 1 && k < ENERGY_TOL) {
        k = energy[i * NF + f] < FRAME_T ? k + 1 : 0;
        clearAround(i, f);
        i++;
      }
      const iEnd = i - 1 - k;
      i = iMid - 1;
      k = 0;
      while (i > 0 && k < ENERGY_TOL) {
        k = energy[i * NF + f] < FRAME_T ? k + 1 : 0;
        clearAround(i, f);
        i--;
      }
      const iStart = i + 1 + k;
      if (iEnd - iStart <= MIN_LEN) continue;
      add(iStart, iEnd, f);
    }
    return out.sort((a, b) => a[1] - b[1] || a[0] - b[0]);
  }

  // ---------- a stream ----------
  function stream() {
    let buf = new Float32Array(SR * 30), len = OLAP_LEN / 2; // Basic Pitch pads the start with half an overlap
    let carry = null, k = 0, nFrames = 0, chain = Promise.resolve();
    let note = new Float32Array(FPS * 30 * NF), onset = new Float32Array(FPS * 30 * NF);
    const grow = (a, need) => {
      if (need <= a.length) return a;
      const b = new Float32Array(Math.max(a.length * 2, need));
      b.set(a);
      return b;
    };
    return {
      // 44.1kHz in, halved to 22.05kHz (pairs averaged: plenty for a preview).
      push(x44) {
        let x = x44;
        if (carry !== null) {
          const y = new Float32Array(x.length + 1);
          y[0] = carry;
          y.set(x, 1);
          x = y;
          carry = null;
        }
        const half = Math.floor(x.length / 2);
        if (x.length % 2) carry = x[x.length - 1];
        buf = grow(buf, len + half);
        for (let i = 0; i < half; i++) buf[len + i] = 0.5 * (x[2 * i] + x[2 * i + 1]);
        len += half;
      },
      // Run every window whose audio is complete (a new one every ~1.6s of playing).
      pump() {
        chain = chain.then(async () => {
          while (k * STEP + N <= len) {
            const w = buf.slice(k * STEP, k * STEP + N);
            k++;
            const r = await runWindow(w);
            note = grow(note, (nFrames + KEEP) * NF);
            onset = grow(onset, (nFrames + KEEP) * NF);
            note.set(r.note.subarray(TRIM * NF, (TRIM + KEEP) * NF), nFrames * NF);
            onset.set(r.onset.subarray(TRIM * NF, (TRIM + KEEP) * NF), nFrames * NF);
            nFrames += KEEP;
          }
        });
        return chain;
      },
      // Notes starting after `from` seconds (a little earlier is read, so notes crossing it survive).
      notes(from = 0) {
        const f0 = Math.max(0, Math.floor((from - 0.5) * FPS));
        return notesFrom(note, onset, f0, nFrames).filter((x) => x[1] >= from);
      },
      get seconds() {
        return nFrames / FPS;
      },
    };
  }

  return { load, stream };
})();
