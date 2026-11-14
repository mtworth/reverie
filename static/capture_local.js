// Listening for the in-browser Reverie (2026-11-12; streaming 2026-11-13). The microphone feeds
// a ring buffer in memory (nothing is uploaded). A level detector decides when there's sound:
// while a sketch is open its audio is handed over as it arrives, so it can be transcribed live;
// silence is never transcribed (that saves the GPU, the battery and the heat).
//
//   Capture.on("state", (state) => …)              // off | starting | listening | hearing | paused | blocked | elsewhere | needs-gesture
//   Capture.on("open",  ({ id, at, pcm, rate }) …)  // a sketch began; pcm = its first audio (incl. a second before)
//   Capture.on("frames", ({ id, pcm }) …)           // more audio for the open sketch
//   Capture.on("burst", ({ id, pcm, rate, at }) …)  // the sketch closed; pcm = all of it (id: null when injected)
//   Capture.ignorePlayback(audioLike); Capture.start(); Capture.pause();

window.Capture = (() => {
  const TICK_MS = 100;
  const SILENCE_MS = 12000; // this long without sound closes a sketch
  const PREROLL_S = 1;
  const TAIL_S = 1.5;
  const MIN_ACTIVE_MS = 600; // less sound than this is a cough, not a sketch
  const MAX_SKETCH_S = 60; // continuous playing is cut into one-minute sketches
  const RING_S = 90;
  const THRESHOLD_ABOVE_FLOOR_DB = 12;
  const KEEP_GOING_DB = 6;
  const MIN_THRESHOLD_DB = -65;
  const PLAYBACK_GUARD_MS = 1500;
  const RATE = 44100; // what the transcriber expects

  const cap = { state: "off", on: false, stream: null, ctx: null, analyser: null, buf: null, node: null, ring: null, written: 0,
                t0: 0, burst: null, floorDb: null, timer: null, playbackUntil: 0, players: [], lastDb: null, threshold: null };
  const handlers = { state: [], open: [], frames: [], burst: [] };
  const log = []; // what the listener did, for ?debug and for watching a session
  const note = (event, extra = {}) => {
    log.push({ at: new Date().toLocaleTimeString(), event, db: cap.lastDb && +cap.lastDb.toFixed(1), threshold: cap.threshold && +cap.threshold.toFixed(1), ...extra });
    if (log.length > 400) log.shift();
  };
  const emit = (type, x) => handlers[type].forEach((fn) => fn(x));
  const now = () => performance.now() - cap.t0;
  const setState = (s) => {
    if (cap.state !== s) {
      cap.state = s;
      emit("state", s);
    }
  };
  let nextId = 0;

  // Every audio frame goes into the ring; `written` counts samples since listening began.
  const TAP = `class Tap extends AudioWorkletProcessor {
    process(inputs) { const ch = inputs[0] && inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true; }
  }
  registerProcessor("reverie-tap", Tap);`;

  // Only one tab listens at a time; the lock is held until this tab pauses or closes.
  let releaseLock = null;
  async function takeLock() {
    if (releaseLock || !navigator.locks) return true;
    return new Promise((resolve) => {
      navigator.locks.request("reverie-listening", { ifAvailable: true }, (lock) => {
        if (!lock) return resolve(false);
        resolve(true);
        return new Promise((r) => (releaseLock = r));
      });
    });
  }

  async function start() {
    if (cap.state === "starting" || cap.on) return;
    if (!(await takeLock())) return setState("elsewhere");
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
      try {
        cap.ctx = new AudioContext({ sampleRate: RATE });
      } catch (_) {
        cap.ctx = new AudioContext(); // the pipeline resamples if the browser won't run at 44.1kHz
      }
      const source = cap.ctx.createMediaStreamSource(cap.stream);
      cap.analyser = cap.ctx.createAnalyser();
      cap.analyser.fftSize = 2048;
      cap.buf = new Float32Array(cap.analyser.fftSize);
      source.connect(cap.analyser);
      await cap.ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([TAP], { type: "application/javascript" })));
      cap.node = new AudioWorkletNode(cap.ctx, "reverie-tap");
      cap.ring = new Float32Array(Math.round(cap.ctx.sampleRate * RING_S));
      cap.node.port.onmessage = (e) => {
        if (!cap.on) return;
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
    cap.on = true;
    cap.timer = setInterval(tick, TICK_MS);
    setState("listening");
  }

  function levelDb() {
    cap.analyser.getFloatTimeDomainData(cap.buf);
    let s = 0;
    for (let i = 0; i < cap.buf.length; i++) s += cap.buf[i] * cap.buf[i];
    return 20 * Math.log10(Math.sqrt(s / cap.buf.length) + 1e-9);
  }

  // Samples [from, to) out of the ring.
  function take(from, to) {
    const n = cap.ring.length, out = new Float32Array(Math.max(0, to - from));
    for (let i = 0; i < out.length; i++) out[i] = cap.ring[(from + i) % n];
    return out;
  }

  function tick() {
    const t = now();
    const db = levelDb();
    cap.lastDb = db;
    if (cap.floorDb === null) cap.floorDb = Math.min(db, -55); // a room, not whatever was already playing
    cap.threshold = Math.max(cap.floorDb + THRESHOLD_ABOVE_FLOOR_DB, MIN_THRESHOLD_DB);
    // Hysteresis: it takes a clear sound to start a sketch, but a softer one keeps it going.
    const heard = db > (cap.burst ? cap.threshold - KEEP_GOING_DB : cap.threshold);
    if (db < cap.floorDb) cap.floorDb = cap.floorDb * 0.7 + db * 0.3;
    else if (!heard) cap.floorDb += 0.03;
    if (cap.players.some((p) => !p.paused) || t < cap.playbackUntil) {
      if (cap.burst) closeBurst(true, "playback");
      return;
    }
    const rate = cap.ctx.sampleRate;
    if (heard) {
      if (!cap.burst) {
        const oldest = Math.max(0, cap.written - cap.ring.length + rate);
        cap.burst = { id: null, startSample: Math.max(oldest, cap.written - Math.round(PREROLL_S * rate)), sent: 0,
                      lastActiveMs: t, lastActiveSample: cap.written, activeMs: 0, at: new Date() };
        note("sound began");
      }
      cap.burst.lastActiveMs = t;
      cap.burst.lastActiveSample = cap.written;
      cap.burst.activeMs += TICK_MS;
    }
    const b = cap.burst;
    if (b) {
      // Once it's clearly a sketch, hand its audio over as it arrives.
      if (!b.id && b.activeMs >= MIN_ACTIVE_MS) {
        b.id = `live${++nextId}`;
        b.sent = cap.written;
        emit("open", { id: b.id, at: b.at, pcm: take(b.startSample, cap.written), rate });
      } else if (b.id && cap.written > b.sent) {
        emit("frames", { id: b.id, pcm: take(b.sent, cap.written) });
        b.sent = cap.written;
      }
      if (t - b.lastActiveMs > SILENCE_MS) closeBurst(false, "quiet");
      else if (cap.written - b.startSample > MAX_SKETCH_S * rate) closeBurst(true, "one-minute cut");
    }
    setState(b && b.activeMs >= MIN_ACTIVE_MS && t - b.lastActiveMs < 1500 ? "hearing" : "listening");
  }

  // forced: cut right now (limit, playback, pause); otherwise end a moment after the last sound.
  function closeBurst(forced, why) {
    const b = cap.burst;
    cap.burst = null;
    if (!b) return;
    if (!b.id) return note("too brief, ignored", { heardMs: b.activeMs });
    const rate = cap.ctx.sampleRate;
    const end = forced ? cap.written : Math.min(cap.written, b.lastActiveSample + Math.round(TAIL_S * rate));
    if (end > b.sent) emit("frames", { id: b.id, pcm: take(b.sent, end) });
    const pcm = take(b.startSample, end);
    note(`sketch closed (${why})`, { seconds: +(pcm.length / rate).toFixed(1) });
    emit("burst", { id: b.id, pcm, rate, at: b.at });
    // Continuous playing: the next sketch starts at once, where this one was cut.
    if (forced && why === "one-minute cut") {
      cap.burst = { id: null, startSample: end, sent: 0, lastActiveMs: b.lastActiveMs, lastActiveSample: b.lastActiveSample,
                    activeMs: 0, at: new Date() };
    }
  }

  async function pause() {
    if (!cap.on) return;
    clearInterval(cap.timer);
    if (cap.burst) closeBurst(true, "paused");
    cap.on = false;
    releaseLock?.();
    releaseLock = null;
    setState("paused");
  }

  function ignorePlayback(p) {
    cap.players.push(p);
    p.addEventListener("pause", () => (cap.playbackUntil = now() + PLAYBACK_GUARD_MS));
  }

  const debug = () => ({ state: cap.state, written: cap.written, db: cap.lastDb, floor: cap.floorDb, threshold: cap.threshold,
                         burst: cap.burst && { ...cap.burst, sinceMs: ((cap.written - cap.burst.startSample) / (cap.ctx?.sampleRate || RATE)) * 1000,
                                               quietMs: now() - cap.burst.lastActiveMs },
                         rate: cap.ctx?.sampleRate, ctx: cap.ctx?.state });
  // For testing without a microphone: hand the pipeline a recording as if it had just been heard.
  const inject = (pcm, rate) => emit("burst", { id: null, pcm, rate, at: new Date() });
  // For testing the live path without a microphone: play a recording in as if it were arriving.
  // `soundS`: where the sound ends (the closed sketch is trimmed there, as the listener does).
  async function injectLive(pcm, rate, chunkS = 1, speed = 1, soundS = null) {
    const id = `live${++nextId}`, step = Math.round(chunkS * rate), at = new Date();
    emit("open", { id, at, pcm: pcm.slice(0, step), rate });
    for (let i = step; i < pcm.length; i += step) {
      await new Promise((r) => setTimeout(r, (chunkS * 1000) / speed));
      emit("frames", { id, pcm: pcm.slice(i, i + step) });
    }
    emit("burst", { id, pcm: soundS ? pcm.slice(0, Math.round((soundS + 1.5) * rate)) : pcm, rate, at });
  }
  return { start, pause, ignorePlayback, debug, inject, injectLive, log,
           on: (type, fn) => handlers[type].push(fn), get state() { return cap.on ? "listening" : cap.state; } };
})();
