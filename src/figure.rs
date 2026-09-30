//! Figure engine. STRICT band->effect mapping over the 24 audio bands:
//!   bands 0-3  (sub/bass)         -> ring DIAMETER (size + beat pump)
//!   bands 4-7  (low-mid body)     -> rim BUMPS / vibration at fixed slots
//!   bands 8-9  (snare body)       -> heavy slow EMBERS (red)
//!   bands 10-12 (voice/guitar)    -> rim TEARS (rupture, then heal)
//!   bands 13-15 (violin/flute lo) -> KNOTS: strands darting out of the rim
//!   bands 16-18 (violin harmonics)-> INNER rim-threader orbiters (cyan)
//!   bands 19-21 (brass/air)       -> OUTER moon orbiters (blue)
//!   bands 22-23 (cymbals/air)     -> fast thin STREAKS (cyan)
//! Every effect lands at its band's golden-angle slot, so each instrument
//! range owns a distinct sector AND a distinct anomaly type. The rim itself
//! stays a clean tilted circle apart from the capped bands 4-7 bumps.

use std::f32::consts::TAU;

/// ratio table indexed by argmax of the 5 column-group activities
pub const RATIOS: [(u32, u32); 5] = [(1, 1), (1, 2), (2, 3), (3, 4), (3, 5)];

const NH: usize = 4; // harmonics per axis
const NS: usize = crate::audio::NB; // one slot per audio band

/// slot angle on the disk (golden-angle spread => every instrument range
/// owns a distinct, non-overlapping sector)
fn slot_ang(b: usize) -> f32 {
    b as f32 * 2.399963 + 0.3
}

/// gaussian width of a bump / tear (radians)
const SLOT_W: f32 = 0.45;

/// first anomaly band: 0-3 = diameter, 4-7 = bumps, 8+ = anomalies
const ANOM_BAND0: usize = 8;

const MAX_PARTS: usize = 96;
const MAX_ORBS: usize = 64;

/// expanding shockwave ring (drums + bass body combo)
struct Pulse {
    r0: f32, // rim radius at spawn (base units)
    life: f32,
    max: f32,
}

struct Part {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    life: f32,
    max: f32,
    band: u8, // source audio band: sets color + spark style
}

/// tiny particle circling the disk
struct Orb {
    a: f32,   // angle (disk frame)
    r: f32,   // radius (outer) or rim-factor (inner)
    va: f32,  // angular velocity
    life: f32,
    max: f32,
    inner: bool, // hugs the rim while orbiting it (strings), vs flying outside
    glow: f32,   // spawn-time brightness
}

pub struct Figure {
    // ratio state machine (hysteresis + min dwell); displayed in the status
    // line; the shape itself stays circular regardless of ratio
    cur: usize,
    dwell: f32, // seconds since last accepted ratio change
    // slow per-group running means: dominance is judged RELATIVE to each
    // group's own floor, so drums poke through a constant bass line
    gavg: [f32; 5],
    // harmonic shape params (smoothed toward targets each frame)
    hx: [f32; NH],
    hy: [f32; NH],
    hx_t: [f32; NH],
    hy_t: [f32; NH],
    phx: [f32; NH],
    phy: [f32; NH],
    // per-band activity slots (kick, snare, hats, violins, voice...):
    // fast attack, slow release, one per audio band
    slot: [f32; NS],
    // slow per-slot mean: effects fire on the SURGE above it
    savg: [f32; NS],
    // per-band rim tears (anomaly bands only): rupture, then heal
    tear: [f32; NS],
    // bass body: slow size envelope + fast bass-following pump envelope
    prev_bass: f32, // last frame's bass level (flux for beat/tempo)
    size_sm: f32,
    beat: f32, // pump envelope of bass-band surges (attack fast, release slow)
    bavg: f32, // slow mean of bass-band activity (surge reference)
    // debris flung off active regions
    parts: Vec<Part>,
    // tiny particles circling the disk
    orbs: Vec<Orb>,
    // shockwave rings (combo effect: drums 8-9 + bass body 4-7)
    pulses: Vec<Pulse>,
    rot: f32,
    rot_goal: f32,  // stepped a quarter turn per beat; rot chases it (on-beat)
    rot_rate: f32,  // measured rad/s, couples orbiters to the disk
    tilt: f32,      // 3D spin phase; projected as cos(tilt) squash on one axis
    tilt_goal: f32, // stepped per beat like rot_goal
    tilt_axis: f32, // angle of the tilt axis in the disk plane (drifts -> 3D tumble)
    pub intensity: f32,
    /// 1.0 = alive; shrinks toward 0 fast when the field dies, so the disk
    /// visibly collapses into a point instead of popping out of existence
    collapse: f32,
    t: f32, // drawing clock (tracer position)
    // deterministic rotation: tempo estimate from bass-hit intervals
    tempo: f32,      // smoothed bass hits/sec
    last_onset: f32, // t of last accepted bass hit
    onset_armed: bool,
    bd_avg: f32, // running mean of positive bass deltas (adaptive threshold)
    dir: f32, // slow EMA of E-I balance -> rotation direction
    static_on: bool, // static mode: disk does not rotate
    spin_mul: f32,  // user rotation-speed multiplier (',' / '.')
    swirl_mul: f32, // user inner-swirl (phase drift) multiplier ('a' / 'd')
    size_mul: f32,  // user disk-size multiplier ('z' / 'x')
    mapping: u8,
}

const SPD_X: [f32; NH] = [0.31, -0.53, 0.83, -1.21];
const SPD_Y: [f32; NH] = [-0.41, 0.67, -0.97, 1.37];

/// shortest signed angular distance a-b
fn ang_diff(a: f32, b: f32) -> f32 {
    (a - b + TAU * 1.5) % TAU - TAU * 0.5
}

impl Figure {
    pub fn new() -> Self {
        Self {
            cur: 0,
            dwell: 0.0,
            gavg: [0.1; 5],
            hx: [0.0; NH],
            hy: [0.0; NH],
            hx_t: [0.0; NH],
            hy_t: [0.0; NH],
            phx: [0.0; NH],
            phy: [0.0; NH],
            slot: [0.0; NS],
            savg: [0.0; NS],
            tear: [0.0; NS],
            size_sm: 0.55,
            prev_bass: 0.0,
            beat: 0.0,
            bavg: 0.05,
            parts: Vec::with_capacity(MAX_PARTS),
            orbs: Vec::with_capacity(MAX_ORBS),
            pulses: Vec::with_capacity(8),
            rot: 0.0,
            rot_goal: 0.0,
            rot_rate: 0.0,
            tilt: 0.0,
            tilt_goal: 0.0,
            tilt_axis: 0.0,
            intensity: 0.0,
            collapse: 1.0,
            t: 0.0,
            tempo: 0.0,
            last_onset: -1.0,
            onset_armed: true,
            bd_avg: 0.005,
            dir: 1.0,
            static_on: false,
            spin_mul: 1.0,
            swirl_mul: 1.0,
            size_mul: 1.0,
            mapping: 0,
        }
    }

    pub fn set_mapping(&mut self, m: u8) {
        self.mapping = m;
    }

    /// Hard reset of every visible envelope: the end state of a collapse.
    pub fn snap_off(&mut self) {
        self.hx = [0.0; NH];
        self.hy = [0.0; NH];
        self.hx_t = [0.0; NH];
        self.hy_t = [0.0; NH];
        self.beat = 0.0;
        self.bavg = 0.05;
        self.prev_bass = 0.0;
        self.slot = [0.0; NS];
        self.savg = [0.0; NS];
        self.tear = [0.0; NS];
        self.intensity = 0.0;
        self.size_sm = 0.55;
        self.rot_rate = 0.0;
        self.rot_goal = self.rot;
        self.tilt_goal = self.tilt;
        self.tempo = 0.0;
        self.last_onset = -1.0;
        self.onset_armed = true;
        self.bd_avg = 0.005;
        self.parts.clear();
        self.orbs.clear();
        self.pulses.clear();
    }

    /// static mode: freeze disk rotation (tone-map regions stay put)
    pub fn set_static(&mut self, on: bool) {
        self.static_on = on;
    }

    /// user spin-speed multiplier (0 = frozen .. 3 = triple)
    pub fn set_spin_mul(&mut self, m: f32) {
        self.spin_mul = m.clamp(0.0, 3.0);
    }

    /// user inner-swirl multiplier: how fast the harmonic phases drift
    pub fn set_swirl_mul(&mut self, m: f32) {
        self.swirl_mul = m.clamp(0.0, 3.0);
    }

    /// user disk-size multiplier on top of the bass-driven size
    pub fn set_size_mul(&mut self, m: f32) {
        self.size_mul = m.clamp(0.35, 3.0);
    }

    pub fn ratio_str(&self) -> String {
        let (a, b) = RATIOS[self.cur];
        format!("{}:{}", a, b)
    }

    pub fn rot_snapshot(&self) -> f32 {
        self.rot
    }

    /// debris particles: (x, y, vx, vy, brightness 0..1, source band);
    /// velocity included so the renderer can draw fire-style motion trails;
    /// the band lets the renderer color each instrument range differently
    pub fn particles(&self) -> impl Iterator<Item = (f32, f32, f32, f32, f32, u8)> + '_ {
        self.parts
            .iter()
            .map(|p| (p.x, p.y, p.vx, p.vy, p.life / p.max, p.band))
    }

    /// knots: short radial strands that dart OUT of the rim, kink, and come
    /// back: bands 13-15 (violin/flute low). Yields (x1, y1, x2, y2, bright)
    /// as an out-and-back polyline drawn in the rim's own frame.
    pub fn knots(&self) -> impl Iterator<Item = (f32, f32, f32, f32, f32)> + '_ {
        (13..16.min(NS)).filter_map(move |b| {
            let a = self.slot[b];
            if a < 0.2 {
                return None;
            }
            let u = slot_ang(b) / TAU;
            let (x, y) = self.shape_point(u);
            // outward direction in screen space (from the disk center)
            let r = (x * x + y * y).sqrt().max(1e-4);
            let (ux, uy) = (x / r, y / r);
            // strand length breathes with the band; the kink wiggles fast
            let len = a * 0.10 * (0.6 + 0.4 * (self.t * 9.0 + b as f32).sin());
            let (x2, y2) = (x + ux * len, y + uy * len);
            Some((x, y, x2, y2, (a * 1.2).min(1.0)))
        })
    }

    /// orbiting particles: (x, y, trail_x, trail_y, brightness, inner).
    /// trail_* is the orbiter's position a moment ago; the renderer draws
    /// the fire-style trail from there to (x, y). String orbiters thread the
    /// donut in 3D (over/under the ring); outer ones fly further out.
    pub fn orbits(&self) -> impl Iterator<Item = (f32, f32, f32, f32, f32, bool)> + '_ {
        let t = self.t;
        let place = move |a: f32, r: f32, inner: bool| {
            let (bx, by) = self.base_point((a / TAU) % 1.0);
            let (s, c) = self.rot.sin_cos();
            let (x, y) = (bx * c - by * s, bx * s + by * c);
            let z = if inner {
                0.22 * (a * 2.0 + t * 2.5).sin() // over/under the ring
            } else {
                0.08 * (t * 3.0 + a * 3.0).sin()
            };
            self.project(x * r, y * r, z)
        };
        self.orbs.iter().map(move |o| {
            let (px, py) = place(o.a, o.r, o.inner);
            let (tx, ty) = place(o.a - o.va * 0.18, o.r, o.inner);
            (px, py, tx, ty, o.glow * (o.life / o.max).min(1.0), o.inner)
        })
    }

    /// shockwave rings: (rim_radius_base, scale, brightness). Renderer draws
    /// the rim polyline scaled by `scale` (a fading copy expanding outward).
    pub fn pulses(&self) -> impl Iterator<Item = (f32, f32, f32)> + '_ {
        self.pulses.iter().map(|p| {
            let k = 1.0 - p.life / p.max; // 0..1 as it expands
            (p.r0, 1.0 + k * 0.55, (1.0 - k) * 0.8)
        })
    }

    /// tether: a SINGLE chord strung across the disk between the hottest
    /// voice slot (10-12) and the hottest strings slot (13-15); one taut
    /// string, not a web of criss-crossing lines.
    /// Yields (x1, y1, x2, y2, bright).
    pub fn tethers(&self) -> impl Iterator<Item = (f32, f32, f32, f32, f32)> + '_ {
        let mut best: Option<(usize, usize, f32)> = None;
        for b1 in 10..13.min(NS) {
            for b2 in 13..16.min(NS) {
                let c = self.slot[b1].min(self.slot[b2]);
                if c > 0.40 && best.map_or(true, |(.., bc)| c > bc) {
                    best = Some((b1, b2, c));
                }
            }
        }
        best.into_iter().map(move |(b1, b2, c)| {
            let (x1, y1) = self.shape_point(slot_ang(b1) / TAU);
            let (x2, y2) = self.shape_point(slot_ang(b2) / TAU);
            (x1, y1, x2, y2, (c * 0.8).min(0.7))
        })
    }

    /// corona flares: tall straight radial spikes at the TWO hottest slots of
    /// bands 16-21, only while BOTH violin harmonics (16-18) and brass (19-21)
    /// are hot. Capped at two: more reads as a starburst, not a corona.
    /// Yields (x1, y1, x2, y2, bright).
    pub fn flares(&self) -> impl Iterator<Item = (f32, f32, f32, f32, f32)> + '_ {
        let c = self.ract(16, 19).min(self.ract(19, 22));
        let mut out = Vec::new();
        if c > 0.35 {
            // pick the two hottest slots in 16..22
            let mut ranked: Vec<usize> = (16..22.min(NS)).collect();
            ranked.sort_by(|&a, &b| self.slot[b].total_cmp(&self.slot[a]));
            for &b in ranked.iter().take(2) {
                if self.slot[b] < 0.25 {
                    continue;
                }
                let (x, y) = self.shape_point(slot_ang(b) / TAU);
                let r = (x * x + y * y).sqrt().max(1e-4);
                let len = c * 0.22 * (0.7 + 0.3 * (self.t * 5.0 + b as f32 * 2.1).sin());
                out.push((x, y, x + x / r * len, y + y / r * len, c));
            }
        }
        out.into_iter()
    }

    /// glitch dashes: short horizontal tears scattered inside the disk while
    /// cymbals (22-23) AND bass (0-3) are both hot. Positions are a
    /// deterministic hash of the frame clock, so they flicker, not crawl.
    /// Yields (x1, y1, x2, y2, bright).
    pub fn glitches(&self) -> impl Iterator<Item = (f32, f32, f32, f32, f32)> + '_ {
        let c = self.ract(22, 24).min(self.ract(0, 4));
        let mut out = Vec::new();
        if c > 0.35 {
            let (bx, by) = self.base_point(0.0);
            let rim = (bx * bx + by * by).sqrt();
            let frame = (self.t * 24.0) as u32; // new dash set every ~40 ms
            for i in 0..2u32 {
                let mut h = frame.wrapping_mul(2654435761) ^ i.wrapping_mul(40503);
                h ^= h >> 13;
                h = h.wrapping_mul(0x5bd1e995);
                h ^= h >> 15;
                let fy = (h % 1000) as f32 / 1000.0 * 2.0 - 1.0; // -1..1
                let fx = ((h >> 10) % 1000) as f32 / 1000.0;
                let y = fy * rim * 0.85;
                let half = (rim * rim - y * y).max(0.0).sqrt(); // disk chord
                let x0 = -half + fx * half * 1.6;
                let len = half * (0.15 + 0.25 * ((h >> 20) % 100) as f32 / 100.0);
                // rotate + tilt-project so dashes stay inside the drawn disk
                let (s, co) = self.rot.sin_cos();
                let (ax, ay) = self.project(x0 * co - y * s, x0 * s + y * co, 0.0);
                let (bx2, by2) =
                    self.project((x0 + len) * co - y * s, (x0 + len) * s + y * co, 0.0);
                out.push((ax, ay, bx2, by2, c));
            }
        }
        out.into_iter()
    }

    /// hottest slot in band range [a, b)
    fn ract(&self, a: usize, b: usize) -> f32 {
        (a..b.min(NS)).fold(0.0f32, |m, i| m.max(self.slot[i]))
    }

    /// is the rim torn open near loop parameter u? (renderer skips segments)
    pub fn is_torn(&self, u: f32) -> bool {
        let th = TAU * u + self.phx[0];
        (10..13.min(NS)).any(|b| {
            self.tear[b] > 0.5 && ang_diff(th, slot_ang(b)).abs() < SLOT_W * 0.5
        })
    }

    /// Update all params from pool signals (per frame). `onset` may trigger a
    /// morph. dt = frame seconds.
    pub fn update(
        &mut self,
        pools: &crate::sim::Sim,
        _onset: f32,
        _tonal: f32, // legacy param: orbiters are per-band now
        _bass_share: f32, // legacy param: size comes from bass band slots
        volume: f32, // raw pre-AGC loudness 0..1: scales the disk size
        dt: f32,
        rng: &mut impl FnMut() -> f32,
    ) {
        let (lo, hi, l, r, rate) = (pools.act_lo, pools.act_hi, pools.pool_l, pools.pool_r, pools.rate);
        let groups = &pools.col_groups;

        // -- rarity tracking: each bracket's recurrence floor is a ~30 s EMA
        // of its own activity. rel[g] = activity vs its own floor: common
        // sounds (steady bass) sit at ~1, rare entrances spike well above.
        for g in 0..5 {
            self.gavg[g] += (groups[g] - self.gavg[g]) * 0.0006; // ~30 s at 60 fps
        }
        let mut rel = [0.0f32; 5];
        for g in 0..5 {
            rel[g] = groups[g] / (self.gavg[g] + 0.05);
        }
        let target = if self.mapping == 0 {
            let mut best = 0;
            for g in 1..5 {
                if rel[g] > rel[best] {
                    best = g;
                }
            }
            best
        } else if lo > hi {
            1
        } else {
            3
        };
        self.dwell += dt;
        let dominated = if self.mapping == 0 {
            rel[target] > rel[self.cur] + 0.4
        } else {
            (lo - hi).abs() > 0.15 && target != self.cur
        };
        if target != self.cur && (self.dwell > 0.8 || dominated) {
            self.cur = target;
            self.dwell = 0.0;
        }

        // -- field liveness: dead field (silence / COUPLING=0) => collapse
        let live = (rate * 12.0).min(1.0);
        // collapse: when the field dies, shrink the whole disk into a point
        // over ~0.4 s (instead of popping out of existence); regrow from
        // that point when audio returns. Fully dead => snap everything off.
        if live < 0.02 {
            self.collapse *= (1.0 - 6.0 * dt).max(0.0);
            if self.collapse < 0.02 {
                self.snap_off();
                self.collapse = 0.0;
            }
        } else {
            self.collapse += (1.0 - self.collapse) * (8.0 * dt).min(1.0);
        }

        // -- harmonic weight targets. The rim is a PURE CIRCLE: only the
        // fundamental (k=0) carries amplitude; higher harmonics stay zero.
        // All mutation lives in overlays (strands, tears, debris, orbiters).
        let a = 0.62 * live;
        let (e, i_) = (pools.pool_e, pools.pool_i);
        if self.mapping == 0 {
            self.hx_t = [a * (0.80 + 0.20 * lo), 0.0, 0.0, 0.0];
            self.hy_t = [a * (0.80 + 0.20 * hi), 0.0, 0.0, 0.0];
        } else {
            self.hx_t = [a * (0.80 + 0.20 * l), 0.0, 0.0, 0.0];
            self.hy_t = [a * (0.80 + 0.20 * r), 0.0, 0.0, 0.0];
        }
        for k in 0..NH {
            self.hx[k] += (self.hx_t[k] - self.hx[k]) * 0.10;
            self.hy[k] += (self.hy_t[k] - self.hy[k]) * 0.10;
        }

        // -- phase drift (inner swirl): E-I balance sets the base rate,
        // user's swirl multiplier scales it on top
        let drift = if self.mapping == 0 { e - i_ } else { l - r };
        let spin = (0.35 + 1.6 * drift.abs().min(1.0)) * self.swirl_mul;

        // -- DIAMETER: bands 0-2 (sub/bass) set the ring size, scaled by the
        // RAW loudness (volume knob matters); user's size_mul stays on top.
        let bass = pools.band_act[0..4].iter().sum::<f32>() / 4.0;
        let vol = 0.30 + 0.70 * volume.clamp(0.0, 1.0);
        let size_t = (0.55 + 0.65 * (bass * 2.0).min(1.0)) * vol * live;
        self.size_sm += (size_t - self.size_sm) * 0.18;
        self.bavg += (bass - self.bavg) * 0.01;
        let dev0 = ((bass - self.bavg) / (self.bavg + 0.05)).clamp(0.0, 1.2);
        let batk = if dev0 > self.beat { 0.55 } else { 0.10 };
        self.beat += (dev0 - self.beat) * batk;

        // -- per-band slots: fast-attack slow-release activity per band.
        // Bands 4-7 become rim bumps; bands 8+ drive anomalies. Level-based:
        // the audio floor normalization already keeps levels meaningful, so
        // a playing instrument KEEPS producing its effect (no fade-out).
        for b in 0..NS {
            let tgt = (pools.band_act[b] * 3.0).min(1.3) * live;
            let atk = if tgt > self.slot[b] { 0.65 } else { 0.12 };
            self.slot[b] += (tgt - self.slot[b]) * atk;
            self.savg[b] += (self.slot[b] - self.savg[b]) * 0.008; // ~2 s
            // rim tears: ONLY bands 10-12 (voice/guitar range) rupture the
            // rim; every other anomaly range gets its own effect instead
            if (10..13).contains(&b) {
                if self.slot[b] > 0.55 && self.tear[b] < 0.2 {
                    self.tear[b] = 1.0;
                }
                self.tear[b] *= (1.0 - 2.2 * dt).max(0.0);
            }
        }

        if !self.static_on {
            for k in 0..NH {
                self.phx[k] += SPD_X[k] * spin * dt;
                self.phy[k] += SPD_Y[k] * spin * dt;
            }
        }

        // -- beat: BASS-HIT detection. A beat is a positive jump in the bass
        // bands (0-3), judged against the bass's own running delta-mean so
        // loud dense songs don't saturate a fixed threshold and skip hits.
        // Each accepted hit pumps the size envelope AND steps the rotation.
        self.dir += ((if drift >= 0.0 { 1.0 } else { -1.0 }) - self.dir) * 0.005;
        let dir = if self.dir >= 0.0 { 1.0 } else { -1.0 };
        let bass_d = bass - self.prev_bass; // per-frame bass delta
        self.prev_bass = bass;
        if bass_d > 0.0 {
            self.bd_avg += (bass_d - self.bd_avg) * 0.04; // ~0.4 s mean
        }
        let mut beat_fired = false;
        if bass_d > self.bd_avg * 1.8 + 0.008 && self.onset_armed {
            self.onset_armed = false;
            beat_fired = true;
            let d = self.t - self.last_onset;
            self.last_onset = self.t;
            if (0.15..2.0).contains(&d) {
                let inst = 1.0 / d; // hits/sec, 0.5..~6.7 (30..400 bpm 4/4)
                self.tempo += (inst - self.tempo) * 0.25;
            }
            // kick => sharp pump impulse (the bass envelope alone saturates
            // flat at high BPM; this articulates every hit)
            self.beat = self.beat.max(0.9);
        } else if bass_d < self.bd_avg * 0.5 {
            self.onset_armed = true;
        }
        // stale tempo decays when hits stop
        if self.t - self.last_onset > 2.0 {
            self.tempo *= 1.0 - 0.5 * dt;
        }

        // -- rotation & spin lock to the BEAT, and ONLY to the beat: every
        // detected onset steps the target angle by a quarter turn and the
        // disk chases it smoothly, landing ON the beat. No idle drift;
        // during monotone segments there are no onsets, so the disk holds
        // perfectly still until the beat comes back. Static mode freezes.
        if beat_fired && !self.static_on {
            self.rot_goal += dir * TAU * 0.25 * self.spin_mul;
            self.tilt_goal += dir * TAU * 0.15 * self.spin_mul;
        }
        let k = 1.0 - (-8.0 * dt).exp();
        let prev_rot = self.rot;
        self.rot += (self.rot_goal - self.rot) * k;
        self.tilt += (self.tilt_goal - self.tilt) * k;
        self.rot_rate = (self.rot - prev_rot) / dt.max(1e-4);
        // tilt axis precesses (tempo-scaled, direction from E-I balance) so
        // the 3D tumble is multidirectional
        if !self.static_on {
            self.tilt_axis += dir * (0.08 + 0.35 * (self.tempo / 3.0).min(1.0)) * live * dt;
        }

        // -- debris: anomaly bands (8+) shed sparks while they play; a hit
        // (surge above the slot's own mean) sheds a bigger burst. Spark STYLE
        // depends on the band range: low-mids (8-9) shed slow heavy embers,
        // air bands (22-23) shed fast thin streaks, mids stay in between.
        for b in ANOM_BAND0..NS {
            let surge = (self.slot[b] - self.savg[b]).max(0.0);
            let rate = self.slot[b] * 0.10 + surge * 0.8;
            if self.slot[b] > 0.15 && self.parts.len() < MAX_PARTS && rng() < rate {
                let u = slot_ang(b) / TAU;
                let (bx, by) = self.base_point(u);
                let r = (bx * bx + by * by).sqrt().max(1e-4);
                let (ux, uy) = (bx / r, by / r);
                // tangential jitter so debris shears off, not just radial
                let tj = (rng() - 0.5) * 0.6;
                let sp = 0.35 + 0.5 * rng();
                let (sp, life) = if b < 10 {
                    (sp * 0.55, 0.9 + 1.3 * rng()) // heavy embers
                } else if b >= 22 {
                    (sp * 1.7, 0.25 + 0.35 * rng()) // thin fast streaks
                } else {
                    (sp, 0.4 + 0.8 * rng())
                };
                let (s, c) = self.rot.sin_cos();
                let px = bx * 1.06 * c - by * 1.06 * s;
                let py = bx * 1.06 * s + by * 1.06 * c;
                let vx = (ux * sp - uy * tj) * c - (uy * sp + ux * tj) * s;
                let vy = (ux * sp - uy * tj) * s + (uy * sp + ux * tj) * c;
                self.parts.push(Part { x: px, y: py, vx, vy, life, max: life, band: b as u8 });
            }
        }
        let drag = (1.0 - 1.6 * dt).max(0.0);
        for p in self.parts.iter_mut() {
            p.life -= dt;
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            p.vx *= drag;
            p.vy *= drag;
        }
        self.parts.retain(|p| p.life > 0.0);

        // -- orbiters: purely per-band, two instrument ranges own them:
        //   bands 16-18 (violin harmonics) -> INNER rim-threaders that weave
        //   over/under the ring; bands 19-21 (brass/air) -> OUTER moons
        //   flying detached orbits. No tonal gate: the band's own activity
        //   slot decides, so an instrument entering lights its own orbiters.
        for b in 16..19.min(NS) {
            if self.slot[b] > 0.25 && self.orbs.len() < MAX_ORBS && rng() < self.slot[b] * 0.05 {
                let life = 2.5 + 3.5 * rng();
                self.orbs.push(Orb {
                    a: slot_ang(b) + (rng() - 0.5) * SLOT_W,
                    r: 1.0, // exactly rim radius: weaving is 3D height, not width
                    va: (0.4 + 0.8 * rng()) * if rng() < 0.7 { 1.0 } else { -1.0 },
                    life,
                    max: life,
                    inner: true,
                    glow: self.slot[b],
                });
            }
        }
        for b in 19..22.min(NS) {
            if self.slot[b] > 0.25 && self.orbs.len() < MAX_ORBS && rng() < self.slot[b] * 0.04 {
                let u = slot_ang(b) / TAU;
                let (bx, by) = self.base_point(u);
                let life = 2.0 + 3.0 * rng();
                self.orbs.push(Orb {
                    a: slot_ang(b) + (rng() - 0.5) * SLOT_W,
                    r: (bx * bx + by * by).sqrt() * (1.18 + 0.28 * rng()),
                    va: (0.5 + 1.0 * rng()) * if rng() < 0.85 { 1.0 } else { -1.0 },
                    life,
                    max: life,
                    inner: false,
                    glow: self.slot[b],
                });
            }
        }
        for o in self.orbs.iter_mut() {
            // orbiters fade with their own schedule; a burst dies when its
            // band goes quiet because new spawns simply stop arriving
            o.life -= dt;
            o.a += (o.va + self.rot_rate) * dt;
        }
        self.orbs.retain(|o| o.life > 0.0);

        // -- COMBO anomalies: two ranges hot at once => an effect neither
        // produces alone. Range activity = hottest slot in the range.
        //   drums 8-9  + bass body 4-7   -> shockwave ring pulses
        //   strings 13-15 + cymbals 22-23 -> white sparkles inside the disk
        // (tethers, corona flares and glitch dashes are stateless; computed
        // on the fly by the render iterators from the current slots)
        let c_shock = self.ract(8, 10).min(self.ract(4, 8));
        if c_shock > 0.45 && self.pulses.len() < 8 && rng() < c_shock * 0.10 {
            let (bx, by) = self.base_point(0.0);
            self.pulses.push(Pulse {
                r0: (bx * bx + by * by).sqrt(),
                life: 0.8,
                max: 0.8,
            });
        }
        let c_spark = self.ract(13, 16).min(self.ract(22, 24));
        if c_spark > 0.35 && self.parts.len() < MAX_PARTS {
            // pinpoint flashes scattered inside the disk, zero velocity
            for _ in 0..2 {
                if rng() > c_spark * 0.5 {
                    continue;
                }
                let a = rng() * TAU;
                let rr = rng().sqrt() * 0.8;
                let (bx, by) = self.base_point(0.0);
                let rim = (bx * bx + by * by).sqrt().max(1e-4);
                let (s, c) = self.rot.sin_cos();
                let (lx, ly) = (rim * rr * a.sin(), rim * rr * a.cos());
                self.parts.push(Part {
                    x: lx * c - ly * s,
                    y: lx * s + ly * c,
                    vx: 0.0,
                    vy: 0.0,
                    life: 0.18 + 0.15 * rng(),
                    max: 0.33,
                    band: 30, // sparkle marker => white
                });
            }
        }
        for p in self.pulses.iter_mut() {
            p.life -= dt;
        }
        self.pulses.retain(|p| p.life > 0.0);

        // -- intensity from global spike rate; while collapsing, keep the
        // beam lit enough to see the shrink all the way down to the point
        self.intensity += ((rate * 4.0).min(1.0) - self.intensity) * 0.2;
        if self.collapse > 0.0 && self.collapse < 1.0 {
            self.intensity = self.intensity.max(self.collapse * 0.4);
        }

        self.t += dt;
    }

    /// Base shape with bass size + beat, before region bumps and rotation.
    /// Fundamentally CIRCULAR: every harmonic is a small quadrature
    /// epicycle (its own little circle) stacked on the main circle, so the
    /// result is always a smooth rounded disk; never a starfish.
    fn base_point(&self, u: f32) -> (f32, f32) {
        let a0 = (self.hx[0] + self.hy[0]) * 0.5;
        let th0 = TAU * u + self.phx[0];
        let (mut x, mut y) = (a0 * th0.sin(), a0 * th0.cos());
        for k in 1..NH {
            let m = (k + 1) as f32;
            let w = (self.hx[k] + self.hy[k]) * 0.12; // small, capped upstream
            // alternating spin direction keeps the wobble smooth
            let dir = if k % 2 == 1 { -1.0 } else { 1.0 };
            x += w * (dir * m * TAU * u + self.phx[k]).sin();
            y += w * (dir * m * TAU * u + self.phy[k]).cos();
        }
        // bands 3-5: rim bumps at each band's golden-angle slot, with a fast
        // vibration term so the rim buzzes while those bands play. Amplitude
        // is capped: bumps deform the circle, they never break it.
        let mut bump = 0.0f32;
        for b in 4..8 {
            let d = ang_diff(th0, slot_ang(b));
            let vib = 1.0 + 0.15 * (self.t * 7.0 + b as f32 * 1.7).sin();
            bump += self.slot[b] * vib * (-d * d / (2.0 * SLOT_W * SLOT_W)).exp();
        }
        let bump = (bump * 0.09).min(0.12);
        let s = self.size_sm * self.size_mul * (1.0 + 0.20 * self.beat) * self.collapse
            * (1.0 + bump);
        (x * s, y * s)
    }

    /// 3D projection: tilt the disk by `tilt` about the axis lying in the
    /// disk plane at angle `tilt_axis` (which slowly precesses, so the
    /// tumble is multidirectional; the squash axis wanders around the disk).
    fn project(&self, x: f32, y: f32, z: f32) -> (f32, f32) {
        let (sa, ca) = self.tilt_axis.sin_cos();
        let (st, ct) = self.tilt.sin_cos();
        // rotate into the tilt-axis frame
        let x1 = x * ca + y * sa;
        let y1 = -x * sa + y * ca;
        // tilt about that axis
        let y2 = y1 * ct - z * st;
        // rotate back to screen frame
        (x1 * ca - y2 * sa, x1 * sa + y2 * ca)
    }

    /// Point on the closed curve; u in [0,1] around the loop. ALWAYS a pure
    /// tilted circle; rim integrity is never compromised by anomalies.
    pub fn shape_point(&self, u: f32) -> (f32, f32) {
        let (bx, by) = self.base_point(u);
        let (s, c) = self.rot.sin_cos();
        self.project(bx * c - by * s, bx * s + by * c, 0.0)
    }

    /// Tracer point travelling along the current shape (used by selftest and
    /// for the bright beam dot).
    pub fn beam_at(&self, toff: f32) -> (f32, f32) {
        let u = ((self.t + toff) * 0.35) % 1.0;
        self.shape_point(u)
    }
}
