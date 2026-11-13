// Listening for the in-browser Reverie (2026-11-12). Same silence detector as capture.js, but
// nothing is uploaded: the microphone feeds a two-minute ring buffer in memory, and each burst
// is handed over as plain samples.
//
//   Capture.on("state", (state) => …)          // off | starting | listening | paused | blocked | needs-gesture
//   Capture.on("burst", ({ pcm, rate, at }) …)  // Float32Array mono, its sample rate, Date it began
//   Capture.ignorePlayback(audioLike); Capture.start(); Capture.pause();

window.Capture = (() => {
  const TICK_MS = 100;
  const SILENCE_MS = 12000;
  const PREROLL_MS = 1000;
  const TAIL_MS = 1500;
  const MIN_ACTIVE_MS = 600;
  const MAX_BURST_MS = 2 * 60 * 1000; // longer playing is split into two-minute sketches
  const RING_S = 130;
  const THRESHOLD_ABOVE_FLOOR_DB = 12;
  const MIN_THRESHOLD_DB = -65;
  const PLAYBACK_GUARD_MS = 1500;

  const cap = { state: "off", stream: null, ctx: null, analyser: null, buf: null, node: null, ring: null, written: 0,
                t0: 0, burst: null, floorDb: null, timer: null, playbackUntil: 0, players: [] };
  const handlers = { state: [], burst: [] };
  const emit = (type, x) => handlers[type].forEach((fn) => fn(x));
  const now = () => performance.now() - cap.t0;
  const setState = (s) => {
    if (cap.state !== s) {
      cap.state = s;
      emit("state", s);
    }
  };

  // Every audio frame goes into the ring; `written` counts samples since listening began.
  const TAP = `class Tap extends AudioWorkletProcessor {
    process(inputs) { const ch = inputs[0] && inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true; }
  }
  registerProcessor("reverie-tap", Tap);`;

  async function start() {
    if (cap.state === "starting" || cap.state === "listening") return;
    setState("starting");
    try {
      if (!cap.stream) {
        cap.stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
        });
      }
    } catch (err) {
      console.error(err);
      return setState("blocked");
    }
    if (!cap.ctx) {
      cap.ctx = new AudioContext();
      const source = cap.ctx.createMediaStreamSource(cap.stream);
      cap.analyser = cap.ctx.createAnalyser();
      cap.analyser.fftSize = 2048;
      cap.buf = new Float32Array(cap.analyser.fftSize);
      source.connect(cap.analyser);
      await cap.ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([TAP], { type: "application/javascript" })));
      cap.node = new AudioWorkletNode(cap.ctx, "reverie-tap");
      cap.ring = new Float32Array(Math.round(cap.ctx.sampleRate * RING_S));
      cap.node.port.onmessage = (e) => {
        if (cap.state !== "listening") return;
        const x = e.data, n = cap.ring.length;
        for (let i = 0; i < x.length; i++) cap.ring[(cap.written + i) % n] = x[i];
        cap.written += x.length;
      };
      source.connect(cap.node);
      const sink = cap.ctx.createGain(); // keep the worklet pulled without making sound
      sink.gain.value = 0;
      cap.node.connect(sink).connect(cap.ctx.destination);
    }
    if (cap.ctx.state === "suspended") {
      try {
        await cap.ctx.resume();
      } catch (_) {}
    }
    if (cap.ctx.state === "suspended") return setState("needs-gesture");
    cap.t0 = performance.now();
    cap.written = 0;
    cap.timer = setInterval(tick, TICK_MS);
    setState("listening");
  }

  function levelDb() {
    cap.analyser.getFloatTimeDomainData(cap.buf);
    let s = 0;
    for (let i = 0; i < cap.buf.length; i++) s += cap.buf[i] * cap.buf[i];
    return 20 * Math.log10(Math.sqrt(s / cap.buf.length) + 1e-9);
  }

  function tick() {
    const t = now();
    const db = levelDb();
    cap.lastDb = db;
    if (cap.floorDb === null) cap.floorDb = db;
    const threshold = Math.max(cap.floorDb + THRESHOLD_ABOVE_FLOOR_DB, MIN_THRESHOLD_DB);
    const heard = db > threshold;
    if (db < cap.floorDb) cap.floorDb = cap.floorDb * 0.7 + db * 0.3;
    else if (!heard) cap.floorDb += 0.03;
    if (cap.players.some((p) => !p.paused) || t < cap.playbackUntil) {
      if (cap.burst) closeBurst(t);
      return;
    }
    if (heard) {
      if (!cap.burst) cap.burst = { startMs: Math.max(0, t - PREROLL_MS), lastActiveMs: t, activeMs: 0, at: new Date() };
      cap.burst.lastActiveMs = t;
      cap.burst.activeMs += TICK_MS;
    }
    if (cap.burst) {
      if (t - cap.burst.lastActiveMs > SILENCE_MS) closeBurst();
      else if (t - cap.burst.startMs > MAX_BURST_MS) closeBurst(t);
    }
  }

  function closeBurst(forcedEndMs) {
    const b = cap.burst;
    cap.burst = null;
    if (!b || b.activeMs < MIN_ACTIVE_MS) return;
    const rate = cap.ctx.sampleRate, n = cap.ring.length;
    const end = Math.min(cap.written, Math.round(((forcedEndMs ?? Math.min(now(), b.lastActiveMs + TAIL_MS)) / 1000) * rate));
    const start = Math.max(0, cap.written - n + rate, Math.round((b.startMs / 1000) * rate));
    if (end - start < rate) return;
    const pcm = new Float32Array(end - start);
    for (let i = 0; i < pcm.length; i++) pcm[i] = cap.ring[(start + i) % n];
    emit("burst", { pcm, rate, at: b.at });
  }

  async function pause() {
    if (cap.state !== "listening") return;
    clearInterval(cap.timer);
    if (cap.burst) closeBurst(now());
    setState("paused");
  }

  function ignorePlayback(p) {
    cap.players.push(p);
    p.addEventListener("pause", () => (cap.playbackUntil = now() + PLAYBACK_GUARD_MS));
  }

  const debug = () => ({ state: cap.state, written: cap.written, db: cap.lastDb, floor: cap.floorDb, burst: cap.burst && { ...cap.burst }, rate: cap.ctx?.sampleRate, ctx: cap.ctx?.state });
  // For testing without a microphone: hand the pipeline a recording as if it had just been heard.
  const inject = (pcm, rate) => emit("burst", { pcm, rate, at: new Date() });
  return { start, pause, ignorePlayback, debug, inject, on: (type, fn) => handlers[type].push(fn), get state() { return cap.state; } };
})();
