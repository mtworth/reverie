// Where one idea gives way to another while the playing goes on (2026-11-14). Replaces cutting
// continuous playing into one-minute sketches by the clock.
//
// A cut needs two things at once:
//   - a pause: at least MIN_GAP_S from one note to the next, and
//   - a change in the music across it: the small model's fingerprints of 4 s windows, a second
//     apart, compared for the ~8 s before against the ~8 s after (Foote's checkerboard novelty).
// Either alone over-cuts. In the user's "C minor line" (14 Nov, 6:35 pm), which turns to jazz at
// 17.7 s, there were pauses at 9.3, 17.7, 33.7 and 55.4 s, and the change curve's next-highest peaks
// (5 s, 37 s) fell inside one idea (the user: 37 s is "same"). Only 17.7 s had both: change 0.066
// against 0.018–0.027 at the other pauses. One example, so these numbers are a starting point for
// the user's corrections to tune.
// A change can only be judged once ~13 s after it has been transcribed, so a cut is made that long
// after the switch. A return that lasts less (that sketch's last 5 s, back to its opening line, the
// user confirmed) isn't cut: the fingerprints can't tell it from what it follows (0.903 vs 0.898).
//
//   const seg = Segment.create(embed);          // embed(notes) → Float32Array (Student.embed)
//   const cut = await seg.cutAt(notes, settled); // notes [[pitch, onset_s, dur_s, vel]], times from
//                                                // the part's start; settled: seconds transcribed
//   → null, or { at, gap, change, why: "switch" | "long" }  (at: the first note of the next part)

window.Segment = (() => {
  const WIN = 4; // seconds per fingerprinted window
  const HOP = 1;
  const HALF = 8; // windows on each side of a boundary
  const MIN_GAP_S = 1.5;
  const MIN_CHANGE = 0.04;
  const MIN_PART_S = 8;
  const HEARD_AFTER_S = HALF + WIN + 1; // transcribed past a pause before it's judged
  const MAX_PART_S = 180; // no switch in three minutes: cut at the longest recent pause anyway
  const LONG_LOOKBACK_S = 30;

  const dot = (a, b) => {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  };

  function create(embed) {
    const vecs = new Map(); // window start → unit vector, or null when it holds fewer than 3 notes
    let judgedTo = 0; // pauses up to here have been judged (and weren't a switch)

    async function windowVec(sorted, s) {
      if (vecs.has(s)) return vecs.get(s);
      const ns = sorted.filter((n) => n[1] >= s && n[1] < s + WIN).map((n) => [n[0], n[1] - s, n[2], n[3]]);
      let v = null;
      if (ns.length >= 3) {
        const e = await embed(ns);
        const norm = Math.sqrt(dot(e, e)) || 1;
        v = e.map((x) => x / norm);
      }
      vecs.set(s, v);
      return v;
    }

    // Checkerboard novelty at window i of `keep` (windows with notes): mean similarity within the
    // HALF windows before and within the HALF after, less the mean similarity across.
    function novelty(keep, i) {
      const a = Math.max(0, i - HALF), b = Math.min(keep.length, i + HALF);
      if (i - a < 3 || b - i < HALF) return NaN;
      const mean = (r0, r1, c0, c1) => {
        let s = 0;
        for (let p = r0; p < r1; p++) for (let q = c0; q < c1; q++) s += dot(keep[p].v, keep[q].v);
        return s / ((r1 - r0) * (c1 - c0));
      };
      return (mean(a, i, a, i) + mean(i, b, i, b)) / 2 - mean(a, i, i, b);
    }

    async function cutAt(notes, settled) {
      const sorted = [...notes].sort((x, y) => x[1] - y[1]);
      const keep = [];
      for (let s = 0; s + WIN <= settled; s += HOP) {
        const v = await windowVec(sorted, s);
        if (v) keep.push({ s, v });
      }
      for (let j = 1; j < sorted.length; j++) {
        const t = sorted[j][1], gap = t - sorted[j - 1][1];
        if (gap < MIN_GAP_S || t < MIN_PART_S || t <= judgedTo) continue;
        if (t + HEARD_AFTER_S > settled) break; // not heard enough after this pause yet
        judgedTo = t;
        let change = -Infinity;
        keep.forEach((w, i) => {
          if (w.s >= t - 2 && w.s <= t + 1) {
            const c = novelty(keep, i);
            if (c > change) change = c;
          }
        });
        if (change >= MIN_CHANGE) return { at: t, gap, change, why: "switch" };
      }
      if (settled >= MAX_PART_S) {
        let best = null;
        for (let j = 1; j < sorted.length; j++) {
          const t = sorted[j][1], gap = t - sorted[j - 1][1];
          if (t < Math.max(MIN_PART_S, settled - LONG_LOOKBACK_S) || t > settled - 2) continue;
          if (!best || gap > best.gap) best = { at: t, gap, change: null, why: "long" };
        }
        return best || { at: settled - 2, gap: 0, change: null, why: "long" };
      }
      return null;
    }

    return { cutAt };
  }

  return { create, MIN_GAP_S, MIN_CHANGE, MAX_PART_S };
})();
