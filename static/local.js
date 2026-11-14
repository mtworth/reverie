// Reverie entirely in the browser (2026-11-12). Loaded before reverie.js on play.html. It
// answers the app's /api/... requests from IndexedDB, runs the pipeline for each burst
// (Transkun transcription → facts and a one-line description → fingerprints), and plays sketches
// back from stored audio. Nothing leaves the browser.
//
// Recognition is a suggestion, not a claim: on the user's own library no similarity cut-off
// separated repeats from different ideas (ideas.md), so each sketch is its own idea, and each
// idea's page lists what it "sounds close to" with a one-tap merge. Merges and splits are
// logged as labels — ground truth for "the same idea".

(() => {
  const MODELS = "static/models/";

  // ---------- IndexedDB ----------

  const dbp = new Promise((resolve, reject) => {
    const req = indexedDB.open("reverie", 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      db.createObjectStore("entries", { keyPath: "id" });
      db.createObjectStore("ideas", { keyPath: "id" });
      db.createObjectStore("audio");
      db.createObjectStore("labels", { autoIncrement: true });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  async function tx(store, mode, fn) {
    const db = await dbp;
    return new Promise((resolve, reject) => {
      const t = db.transaction(store, mode);
      const r = fn(t.objectStore(store));
      t.oncomplete = () => resolve(r?.result);
      t.onerror = () => reject(t.error);
    });
  }
  const getAll = (s) => tx(s, "readonly", (o) => o.getAll());
  const put = (s, v, k) => tx(s, "readwrite", (o) => (k === undefined ? o.put(v) : o.put(v, k)));
  const del = (s, k) => tx(s, "readwrite", (o) => o.delete(k));
  const getOne = (s, k) => tx(s, "readonly", (o) => o.get(k));

  // In-memory copy of the small tables (entries without audio, ideas).
  const state = { entries: new Map(), ideas: new Map(), ready: null };
  state.ready = (async () => {
    for (const e of await getAll("entries")) state.entries.set(e.id, e);
    for (const i of await getAll("ideas")) state.ideas.set(i.id, i);
    // A tab closed mid-transcription: try those again.
    for (const e of state.entries.values()) if (e.status === "processing") queue(e.id);
  })();
  const uid = (p) => `${p}_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
  const changed = () => window.dispatchEvent(new Event("reverie:changed"));
  const status = { models: "not loaded", step: "idle" }; // for ?debug

  // ---------- models (loaded on first use) ----------

  // Models: downloaded once into Cache Storage (so later visits start in a second or two),
  // with progress for the loading screen, then loaded and run once on silence so the first real
  // sketch doesn't pay for GPU shader compilation.
  // Half-precision Transkun: same notes (checked on six sketches), half the download, ~1.3x faster on a
  // GPU. Full precision is fetched only if the GPU can't run half precision (and for the CPU path).
  const MODEL_FILES = ["transkun_score_fp16.onnx", "transkun_attr.onnx", "student.onnx", "basic_pitch.onnx"];
  async function fetchModels(onProgress, files = MODEL_FILES) {
    const cache = "caches" in window ? await caches.open("reverie-models-v1") : null;
    const sizes = {}, got = {}, bytes = {};
    // (Reported once every file's size is known, so the total doesn't grow as it goes.)
    const report = () => Object.keys(sizes).length === files.length &&
      onProgress?.(Object.values(got).reduce((a, b) => a + b, 0), Object.values(sizes).reduce((a, b) => a + b, 0));
    await Promise.all(files.map(async (name) => {
      const url = MODELS + name;
      let res = cache && (await cache.match(url));
      const cached = Boolean(res);
      if (!res) res = await fetch(url);
      if (!res.ok) throw new Error(`${name}: ${res.status}`);
      sizes[name] = Number(res.headers.get("content-length")) || 0;
      got[name] = 0;
      const reader = res.clone().body.getReader();
      const parts = [];
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        parts.push(value);
        got[name] += value.length;
        if (!sizes[name] || got[name] > sizes[name]) sizes[name] = got[name];
        report();
      }
      const buf = new Uint8Array(got[name]);
      let o = 0;
      for (const p of parts) {
        buf.set(p, o);
        o += p.length;
      }
      bytes[name] = buf;
      if (cache && !cached) await cache.put(url, new Response(buf, { headers: { "content-type": "application/octet-stream" } }));
    }));
    return bytes;
  }

  // How far along the models are, for the page's quiet cues: downloading is most of the wait the
  // first time; warming up (compiling for the GPU, one practice run) is the rest.
  const modelState = { phase: "idle", progress: 0 };
  const reportModels = (phase, progress = modelState.progress) => {
    Object.assign(modelState, { phase, progress });
    window.dispatchEvent(new CustomEvent("reverie:models", { detail: { ...modelState } }));
  };

  let models = null;
  function loadModels(onProgress = () => {}) {
    if (!models) {
      status.models = "loading";
      reportModels("download", 0);
      const downloaded = (done, total) => {
        onProgress("download", done, total);
        if (total) reportModels("download", Math.max(modelState.progress, 0.85 * Math.min(1, done / total)));
      };
      models = (async () => {
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
        const bytes = await fetchModels(downloaded);
        onProgress("warm");
        reportModels("warm", 0.88);
        let tk, ep = "webgpu", precision = "half";
        try {
          if (!navigator.gpu) throw new Error("no WebGPU");
          tk = await Transkun.load({ bytes: { ...bytes, "transkun_score.onnx": bytes["transkun_score_fp16.onnx"] }, ep });
        } catch (err) {
          console.warn("half-precision Transkun unavailable; using full precision", err);
          precision = "full";
          Object.assign(bytes, await fetchModels((done, total) => onProgress("download", done, total), ["transkun_score.onnx"]));
          reportModels("warm", 0.9);
          try {
            if (!navigator.gpu) throw new Error("no WebGPU");
            tk = await Transkun.load({ bytes, ep });
          } catch (err2) {
            console.warn("Transkun on WebGPU unavailable, using the CPU", err2);
            ep = "wasm";
            tk = await Transkun.load({ bytes, ep });
          }
        }
        await Sketch.load(bytes["basic_pitch.onnx"]).catch((err) => console.warn("no live sketch", err));
        const student = await Student.load({ url: bytes["student.onnx"], ep: "wasm" });
        reportModels("warm", 0.94);
        await tk.transcribe([new Float32Array(44100), new Float32Array(44100)]); // warm up
        await student.embed([[60, 0, 0.5, 64], [64, 0.5, 0.5, 64], [67, 1, 0.5, 64]]);
        status.models = `ready (${ep === "webgpu" ? "GPU" : "CPU"}, ${precision} precision)`;
        reportModels("ready", 1);
        return { tk, student };
      })();
      models.catch(() => reportModels("failed"));
    }
    return models;
  }
  window.ReverieModels = { load: loadModels, state: modelState };

  // ---------- audio helpers ----------

  async function resample(pcm, from, to) {
    const len = Math.max(1, Math.round((pcm.length * to) / from));
    const ctx = new OfflineAudioContext(1, len, to);
    const buf = ctx.createBuffer(1, pcm.length, from);
    buf.copyToChannel(pcm, 0);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.start();
    return (await ctx.startRendering()).getChannelData(0);
  }

  function wav(pcm, rate) {
    const out = new DataView(new ArrayBuffer(44 + pcm.length * 2));
    const str = (o, s) => [...s].forEach((c, i) => out.setUint8(o + i, c.charCodeAt(0)));
    str(0, "RIFF");
    out.setUint32(4, 36 + pcm.length * 2, true);
    str(8, "WAVEfmt ");
    out.setUint32(16, 16, true);
    out.setUint16(20, 1, true);
    out.setUint16(22, 1, true);
    out.setUint32(24, rate, true);
    out.setUint32(28, rate * 2, true);
    out.setUint16(32, 2, true);
    out.setUint16(34, 16, true);
    str(36, "data");
    out.setUint32(40, pcm.length * 2, true);
    for (let i = 0; i < pcm.length; i++) out.setInt16(44 + i * 2, Math.max(-1, Math.min(1, pcm[i])) * 0x7fff, true);
    return new Blob([out.buffer], { type: "audio/wav" });
  }

  function peaks(pcm, n = 400) {
    const out = new Array(n).fill(0), step = pcm.length / n;
    for (let i = 0; i < n; i++) {
      let m = 0;
      for (let j = Math.floor(i * step); j < Math.floor((i + 1) * step); j++) m = Math.max(m, Math.abs(pcm[j]));
      out[i] = m;
    }
    const top = Math.max(...out) || 1;
    return out.map((x) => +(x / top).toFixed(3));
  }

  // 12x12 pitch-class transition counts (compared under all 12 rotations: "same notes, any key").
  function transitions(notes) {
    const pcs = [...notes].sort((a, b) => a[1] - b[1]).map((n) => n[0] % 12);
    const m = new Array(144).fill(0);
    for (let i = 1; i < pcs.length; i++) m[pcs[i - 1] * 12 + pcs[i]]++;
    return m;
  }
  function anyKey(a, b) {
    let best = 0;
    const na = Math.hypot(...a) || 1, nb = Math.hypot(...b) || 1;
    for (let k = 0; k < 12; k++) {
      let s = 0;
      for (let i = 0; i < 12; i++) for (let j = 0; j < 12; j++) s += a[i * 12 + j] * b[((i + k) % 12) * 12 + ((j + k) % 12)];
      best = Math.max(best, s / (na * nb));
    }
    return best;
  }

  // ---------- the pipeline: one burst at a time ----------

  let chain = Promise.resolve();
  const pending = new Map(); // entry id -> { pcm, rate } until it's processed
  function queue(id) {
    const once = () =>
      navigator.locks
        ? navigator.locks.request(`reverie-sketch-${id}`, { ifAvailable: true }, (lock) => (lock ? process(id) : null))
        : process(id);
    chain = chain.then(once).catch((err) => console.error("pipeline", err));
  }
  const plog = []; // pipeline timings, for watching a session
  const pnote = (event, extra = {}) => {
    plog.push({ at: new Date().toLocaleTimeString(), event, ...extra });
    if (plog.length > 200) plog.shift();
  };

  async function process(id) {
    const e = state.entries.get(id);
    if (!e) return;
    try {
      let mono44;
      const p = pending.get(id);
      if (p?.live) mono44 = null; // transcribed while it was played
      else if (p) mono44 = await resample(p.pcm, p.rate, 44100);
      else {
        // After a reload: decode the stored recording (decoding resamples to the context's 44.1kHz).
        const blob = await getOne("audio", id);
        if (!blob) {
          // The tab closed while this sketch was still being played: nothing to keep.
          Object.assign(e, { status: "discarded" });
          await put("entries", e);
          return;
        }
        mono44 = (await new OfflineAudioContext(1, 1, 44100).decodeAudioData(await blob.arrayBuffer())).getChannelData(0);
      }
      const t0 = performance.now();
      const { tk, student } = await loadModels();
      const tModels = performance.now();
      let found;
      if (p?.live) {
        // Only the last window or two are left to do.
        status.step = "finishing a sketch";
        const L = p.live;
        L.finishing = true;
        liveChanged(L);
        await ensureStream(L);
        found = await L.stream.finish(p.pcm.length / p.rate + 0.5); // the sketch ends a moment after its last sound
        // The whole sketch is inked now; the line keeps showing it until it turns into a finished line.
        L.notes = found.map((n) => [n.pitch, n.start, Math.max(0.05, n.end - n.start), Math.round(n.velocity)]);
        L.covered = Infinity;
        liveChanged(L);
        mono44 = to44(p.pcm, p.rate);
      } else {
        status.step = `transcribing ${(mono44.length / 44100).toFixed(0)}s`;
        found = await tk.transcribe([mono44, mono44], (done, total) => (status.step = `transcribing ${done}/${total}`));
      }
      const notes = found.map((n) => [n.pitch, +n.start.toFixed(3), +(n.end - n.start).toFixed(3), Math.round(n.velocity)]);
      pnote(p?.live ? "finished live sketch" : "transcribed", { id, seconds: +(mono44.length / 44100).toFixed(1), notes: notes.length,
                             waitedForModels: +((tModels - t0) / 1000).toFixed(1), took: +((performance.now() - tModels) / 1000).toFixed(1) });
      if (notes.length < 3) {
        Object.assign(e, { status: "discarded" });
        await put("entries", e);
        await del("audio", id);
        return;
      }
      const facts = Facts.of(notes);
      const vec = Array.from(await student.embed(notes.slice(0, 256)));
      // Every sketch starts as its own idea (merging is the user's call).
      const idea = { id: uid("idea"), name: uniqueName(Facts.defaultName(facts)), name_source: "auto", created_at: e.created_at };
      state.ideas.set(idea.id, idea);
      await put("ideas", idea);
      Object.assign(e, { status: "ready", notes, line: Facts.describe(facts), vec, trans: transitions(notes), idea: idea.id,
                         duration: +(mono44.length / 44100).toFixed(2), peaks: peaks(mono44) });
      await put("entries", e);
    } catch (err) {
      console.error(err);
      Object.assign(e, { status: "failed" });
      await put("entries", e);
    } finally {
      if (p?.live) live.delete(p.live.liveId);
      pending.delete(id);
      status.step = "idle";
      changed();
    }
  }

  function uniqueName(base) {
    const taken = new Set([...state.ideas.values()].map((i) => i.name.toLowerCase()));
    const roman = ["", " II", " III", " IV", " V", " VI", " VII", " VIII", " IX", " X"];
    for (let k = 0; k < 200; k++) {
      const name = k < roman.length ? base + roman[k] : `${base} ${k + 1}`;
      if (!taken.has(name.toLowerCase())) return name;
    }
    return base;
  }

  // ---------- live: transcribe a sketch while it's being played ----------
  // Each open sketch gets a Transkun stream; audio is pushed as it arrives and every 8s window is
  // decoded as soon as it's complete (about 8s behind, ~4.5s of work on a GPU), so its notes can
  // be drawn as a live piano roll and, when it closes, only the last window or two remain.

  const live = new Map(); // live id -> { liveId, entryId, rate, stream, waiting, notes, covered, pumping, finishing }
  const liveChanged = (L) => window.dispatchEvent(new CustomEvent("reverie:live", { detail: { id: L.entryId } }));

  // Linear resampling for browsers that won't run the microphone at 44.1kHz.
  const to44 = (pcm, rate) => {
    if (rate === 44100) return pcm;
    const out = new Float32Array(Math.round((pcm.length * 44100) / rate));
    for (let i = 0; i < out.length; i++) {
      const x = (i * rate) / 44100, j = Math.floor(x), f = x - j;
      out[i] = (pcm[j] || 0) * (1 - f) + (pcm[j + 1] || pcm[j] || 0) * f;
    }
    return out;
  };

  function ensureStream(L) {
    if (!L.ready) {
      L.ready = loadModels().then(({ tk }) => {
        L.stream = tk.stream();
        L.sketch = Sketch.stream();
        for (const x of L.waiting) {
          L.stream.push(x);
          L.sketch.push(x);
        }
        L.waiting = [];
        pump(L);
      });
    }
    return L.ready;
  }

  // The sketch: Basic Pitch, a second or two behind; only ahead of where Transkun has settled.
  function sketchPump(L) {
    if (!L.sketch) return;
    L.sketching = (L.sketching || Promise.resolve())
      .then(() => L.sketch.pump())
      .then(() => {
        L.sketchNotes = L.sketch.notes(Math.min(L.covered || 0, L.sketch.seconds));
        liveChanged(L);
      })
      .catch((err) => console.warn("sketch", err));
  }

  function pump(L) {
    if (!L.stream) return;
    sketchPump(L);
    L.pumping = (L.pumping || Promise.resolve())
      .then(() => {
        status.step = `listening along (${L.stream.seconds.toFixed(0)}s)`;
        return L.stream.pump();
      })
      .then(() => {
        if (L.covered === Infinity) return; // finished meanwhile
        // The ink and how far it reaches are taken together, so the sketch gives way to the ink
        // only where the ink has actually arrived.
        L.notes = L.stream.notes().map((n) => [n.pitch, n.start, Math.max(0.05, n.end - n.start), Math.round(n.velocity)]);
        L.covered = L.stream.covered;
        liveChanged(L);
        if (status.step.startsWith("listening along")) status.step = "idle";
      })
      .catch((err) => console.error("live", err));
    return L.pumping;
  }

  Capture.on("open", async ({ id, at, pcm, rate }) => {
    const L = { liveId: id, entryId: uid("sketch"), rate, stream: null, waiting: [to44(pcm, rate)], notes: [] };
    live.set(id, L);
    ensureStream(L);
    await state.ready;
    const e = { id: L.entryId, created_at: at.toISOString(), status: "processing", live: true };
    state.entries.set(e.id, e);
    await put("entries", e);
    changed();
  });

  Capture.on("frames", ({ id, pcm }) => {
    const L = live.get(id);
    if (!L) return;
    const x = to44(pcm, L.rate);
    if (L.stream) {
      L.stream.push(x);
      L.sketch?.push(x);
      pump(L);
    } else L.waiting.push(x);
  });

  Capture.on("burst", async ({ id, pcm, rate, at }) => {
    await state.ready;
    const L = id && live.get(id);
    if (L) {
      L.closed = true;
      liveChanged(L);
    }
    let e = L && state.entries.get(L.entryId);
    if (!e) {
      // Injected for testing (or the live entry never got created): the whole-recording path.
      e = { id: uid("sketch"), created_at: at.toISOString(), status: "processing" };
      state.entries.set(e.id, e);
    }
    pending.set(e.id, L ? { pcm, rate, live: L } : { pcm, rate });
    // Keep the recording, for playback and so a reload mid-processing can recover it.
    const half = await resample(pcm, rate, 22050);
    await put("audio", wav(half, 22050), e.id);
    await put("entries", e);
    changed();
    queue(e.id);
  });

  // Notes found so far for a sketch still being played (for the live piano roll), and where it's at.
  // { ink: Transkun's settled notes, sketch: Basic Pitch's notes beyond them, covered: seconds settled,
  //   seconds: heard so far, phase: "tuning" (models still loading) | "live" | "finishing" | "inked" }
  window.LocalLive = (entryId) => {
    for (const L of live.values()) {
      if (L.entryId !== entryId) continue;
      const covered = L.covered || 0;
      const seconds = L.stream ? L.stream.seconds : L.waiting.reduce((a, x) => a + x.length, 0) / 44100;
      const phase = L.covered === Infinity ? "inked" : !L.stream ? "tuning" : L.finishing ? "finishing" : L.closed ? "queued" : "live";
      return { ink: L.notes || [], sketch: (L.sketchNotes || []).filter((n) => n[1] >= covered), covered, seconds, phase };
    }
    return null;
  };

  // ---------- the API the app expects ----------

  const ready = () => [...state.entries.values()].filter((e) => e.status === "ready");
  const occurrencesOf = (ideaId) => ready().filter((e) => e.idea === ideaId).sort((a, b) => a.created_at.localeCompare(b.created_at));
  const titleOf = (e) => state.ideas.get(e.idea)?.name || "a sketch";

  function publicEntry(e) {
    const idea = state.ideas.get(e.idea);
    const occ = idea ? occurrencesOf(idea.id) : [];
    return {
      id: e.id, created_at: e.created_at, status: e.status, stage: null, error: null, title: e.status === "ready" ? titleOf(e) : "",
      title_source: "auto", summary: e.line || "", duration: e.duration || 0, sitting_offset: 0, transcript: [], attached_notes: [],
      session_id: "local", has_audio: e.status === "ready",
      motifs: idea ? [{ id: idea.id, name: idea.name, count: occ.length, highlighted: false, intended_instrument: null,
                        elsewhere: occ.filter((x) => x.id !== e.id).map((x) => ({ entry_id: x.id })) }] : [],
    };
  }

  // Which ideas a given idea sounds close to: fused ranking (small model + any-key counts),
  // the same fusion that scored best on the identity tests.
  function closest(ideaId, k = 3) {
    const mine = occurrencesOf(ideaId)[0];
    if (!mine) return [];
    const others = [...state.ideas.values()].filter((i) => i.id !== ideaId).map((i) => ({ i, e: occurrencesOf(i.id)[0] })).filter((x) => x.e);
    if (others.length < 2) return others.map((x) => ({ motif_id: x.i.id, name: x.i.name }));
    const s = others.map((x) => x.e.vec.reduce((acc, v, j) => acc + v * mine.vec[j], 0));
    const a = others.map((x) => anyKey(mine.trans, x.e.trans));
    const z = (xs) => {
      const m = xs.reduce((p, q) => p + q, 0) / xs.length;
      const sd = Math.sqrt(xs.reduce((p, q) => p + (q - m) ** 2, 0) / xs.length) || 1;
      return xs.map((x) => (x - m) / sd);
    };
    const zs = z(s), za = z(a);
    return others
      .map((x, n) => ({ motif_id: x.i.id, name: x.i.name, score: zs[n] + za[n] }))
      .sort((p, q) => q.score - p.score)
      .slice(0, k)
      .map(({ motif_id, name }) => ({ motif_id, name }));
  }

  function motifsPayload() {
    const out = [];
    for (const idea of state.ideas.values()) {
      const occ = occurrencesOf(idea.id);
      if (!occ.length) continue;
      out.push({
        id: idea.id, name: idea.name, name_source: idea.name_source, description: occ[0].line || "", harmony: "",
        created_at: occ[0].created_at, first_heard: { entry_id: occ[0].id, phrase_id: "p1" }, highlighted: false,
        intended_instrument: null, last_heard: occ[occ.length - 1].created_at, entry_count: occ.length, related: [],
        closest: closest(idea.id),
        occurrences: occ.map((e) => ({ entry_id: e.id, session_id: "local", entry_title: titleOf(e), entry_created_at: e.created_at,
                                       phrase_id: "p1", start: 0, end: e.duration, variation_note: null, shared: [],
                                       highlighted: false, intended_instrument: null })),
      });
    }
    return { motifs: out.sort((a, b) => b.last_heard.localeCompare(a.last_heard)) };
  }

  function snippetsPayload() {
    return {
      snippets: ready().map((e) => {
        const idea = state.ideas.get(e.idea);
        const occ = occurrencesOf(e.idea);
        return { id: "p1", start: 0, end: e.duration, kind: "motif", name: idea?.name, description: "", harmony: "",
                 variation_note: null, shared: null, motif_relation: occ[0]?.id === e.id ? "new" : "same", highlighted: false,
                 intended_instrument: null, user_annotations: [], entry_id: e.id, entry_title: titleOf(e), created_at: e.created_at,
                 theme: idea ? { id: idea.id, name: idea.name, count: occ.length } : null, related: [], stanza: null };
      }),
    };
  }

  const label = (x) => put("labels", { at: new Date().toISOString(), ...x });
  const json = (x, status = 200) => new Response(JSON.stringify(x), { status, headers: { "Content-Type": "application/json" } });
  const realFetch = window.fetch.bind(window);

  window.fetch = async (input, opts = {}) => {
    const url = typeof input === "string" ? input : input.url;
    if (!url.startsWith("/api/")) return realFetch(input, opts);
    await state.ready;
    const method = (opts.method || "GET").toUpperCase();
    const body = opts.body ? JSON.parse(opts.body) : {};
    let m;
    if (url === "/api/journal") {
      const entries = [...state.entries.values()].sort((a, b) => a.created_at.localeCompare(b.created_at)).map(publicEntry);
      return json({ sittings: entries.length ? [{ id: "local", started_at: entries[0].created_at, entries }] : [] });
    }
    if (url === "/api/motifs") return json(motifsPayload());
    if (url === "/api/snippets") return json(snippetsPayload());
    if ((m = url.match(/^\/api\/entries\/([\w-]+)\/peaks\?n=(\d+)$/))) {
      const p = state.entries.get(m[1])?.peaks;
      if (!p) return json(null, 404);
      const n = Number(m[2]);
      return json(Array.from({ length: n }, (_, i) => Math.max(...p.slice(Math.floor((i * p.length) / n), Math.max(Math.floor(((i + 1) * p.length) / n), Math.floor((i * p.length) / n) + 1)))));
    }
    if ((m = url.match(/^\/api\/motifs\/([\w-]+)$/)) && method === "PATCH") {
      const idea = state.ideas.get(m[1]);
      if (!idea || !body.name?.trim()) return json({ detail: "not found" }, 404);
      Object.assign(idea, { name: body.name.trim(), name_source: "user" });
      await put("ideas", idea);
      return json({ ok: true });
    }
    if ((m = url.match(/^\/api\/motifs\/([\w-]+)\/merge$/)) && method === "POST") {
      // This idea is the same as `into`: its sketches join that idea.
      const from = state.ideas.get(m[1]), into = state.ideas.get(body.into);
      if (!from || !into || from.id === into.id) return json({ detail: "not found" }, 404);
      for (const e of occurrencesOf(from.id)) {
        e.idea = into.id;
        await put("entries", e);
      }
      if (from.name_source === "user" && into.name_source !== "user") Object.assign(into, { name: from.name, name_source: "user" });
      await put("ideas", into);
      state.ideas.delete(from.id);
      await del("ideas", from.id);
      await label({ kind: "same", idea: into.id, merged: from.id });
      return json({ ok: true, motif_id: into.id });
    }
    if ((m = url.match(/^\/api\/entries\/([\w-]+)\/phrases\/[\w-]+\/not-motif$/)) && method === "POST") {
      const e = state.entries.get(m[1]);
      if (!e) return json({ detail: "not found" }, 404);
      const old = e.idea;
      const idea = { id: uid("idea"), name: uniqueName(Facts.defaultName(Facts.of(e.notes))), name_source: "auto", created_at: e.created_at };
      state.ideas.set(idea.id, idea);
      await put("ideas", idea);
      e.idea = idea.id;
      await put("entries", e);
      await label({ kind: "not_same", entry: e.id, idea: old });
      return json({ ok: true, motif_id: idea.id });
    }
    if ((m = url.match(/^\/api\/entries\/([\w-]+)\/retry$/)) && method === "POST") {
      const e = state.entries.get(m[1]);
      if (e) {
        e.status = "processing";
        queue(e.id);
      }
      return json({ ok: true });
    }
    return json({ detail: "not available in the browser version" }, 404);
  };

  // Notes straight from the store (views.js asks for them by entry id).
  window.LocalNotes = async (id) => {
    await state.ready;
    return state.entries.get(id)?.notes || null;
  };

  // ---------- playback from stored audio, behind an <audio>-like interface ----------

  class LocalPlayer extends EventTarget {
    constructor() {
      super();
      this.a = new Audio();
      this.a.preload = "metadata";
      for (const ev of ["loadedmetadata", "play", "pause", "ended", "error"]) this.a.addEventListener(ev, () => this.dispatchEvent(new Event(ev)));
      this._src = "";
      this._url = null;
    }
    get src() {
      return this._src;
    }
    set src(v) {
      this._src = v;
    }
    async load() {
      const m = this._src.match(/entries\/([\w-]+)\/audio/);
      const blob = m && (await getOne("audio", m[1]));
      if (!blob) return this.dispatchEvent(new Event("error"));
      if (this._url) URL.revokeObjectURL(this._url);
      this._url = URL.createObjectURL(blob);
      this.a.src = this._url;
      this.a.load();
    }
    play() {
      return this.a.play();
    }
    pause() {
      this.a.pause();
    }
    get paused() {
      return this.a.paused;
    }
    get currentTime() {
      return this.a.currentTime;
    }
    set currentTime(v) {
      this.a.currentTime = v;
    }
    get duration() {
      return this.a.duration;
    }
    get readyState() {
      return this.a.readyState;
    }
  }
  window.LocalPlayer = new LocalPlayer();
  window.LocalStatus = status;
  window.ReverieLog = () => ({ listener: Capture.log, pipeline: plog, status: { ...status }, queue: [...state.entries.values()].filter((e) => e.status === "processing").length });


})();
