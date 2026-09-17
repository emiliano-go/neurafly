//! Leaky integrate-and-fire spiking circuit on the REAL FlyWire v783
//! connectome (data/flywire_net.bin, produced by tools/preprocess.py).
//! Audio only ever injects current into neurons; nothing downstream reads
//! audio. Falls back to a small synthetic geometric network when the
//! connectome file is missing.

pub struct Sim {
    pub n: usize,
    pub grid_w: usize,
    pub grid_h: usize,
    pub v: Vec<f32>,
    pub act: Vec<f32>,
    pub i_buf: Vec<f32>,
    refr: Vec<i8>,
    inhib: Vec<bool>,
    px: Vec<u16>,
    py: Vec<u16>,
    band: Vec<u8>,     // tonotopic audio band 0..9 owning this neuron
    thr: Vec<f32>,     // per-neuron firing threshold
    jg: Vec<f32>,      // per-neuron injection gain
    indptr: Vec<u32>,  // CSR adjacency (real FlyWire synapses)
    tgt: Vec<u32>,
    wgt: Vec<f32>,     // signed weights (negative = inhibitory)
    band_members: [Vec<u32>; crate::audio::NB],
    cmd: Vec<u32>,     // random ~1% "command" neurons for global onset pulses
    rng_state: u64,
    pub spikes: usize,
    pub rate: f32, // smoothed spike fraction 0..1
    // pool activities, written every tick
    pub pool_l: f32,
    pub pool_r: f32,
    pub pool_e: f32,
    pub pool_i: f32,
    pub act_lo: f32,
    pub act_hi: f32,
    pub col_groups: [f32; 5], // band pairs (0-1, 2-3, ..., 8-9), mean act
    pub band_act: [f32; crate::audio::NB], // per-band mean act
}

/// connectome search order: cwd data dir (repo checkout), then the
/// user-wide data dir the installer populates, then the executable's dir
fn net_candidates() -> Vec<std::path::PathBuf> {
    let mut v = vec![std::path::PathBuf::from("data/flywire_net.bin")];
    if let Some(home) = std::env::var_os("HOME") {
        v.push(std::path::Path::new(&home).join(".local/share/neurafly/flywire_net.bin"));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push(dir.join("flywire_net.bin"));
        }
    }
    v
}

impl Sim {
    pub fn new(seed: u64) -> Self {
        match Self::load_connectome(seed) {
            Some(s) => {
                eprintln!("connectome: {} neurons, {} edges (FlyWire v783)", s.n, s.tgt.len());
                s
            }
            None => {
                eprintln!("connectome file not found, using synthetic fallback");
                Self::synthetic(seed)
            }
        }
    }

    fn finish(mut self) -> Self {
        for b in 0..crate::audio::NB {
            self.band_members[b] = (0..self.n as u32)
                .filter(|&i| self.band[i as usize] == b as u8)
                .collect();
        }
        self
    }

    fn load_connectome(seed: u64) -> Option<Self> {
        let raw = net_candidates()
            .into_iter()
            .find_map(|p| std::fs::read(p).ok())?;
        if raw.len() < 24 || &raw[0..8] != b"NBSNET01" {
            return None;
        }
        let u32at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
        let (n, gw, gh, e) = (
            u32at(8) as usize,
            u32at(12) as usize,
            u32at(16) as usize,
            u32at(20) as usize,
        );
        let rec = 10; // per-neuron record: u16 u16 u8 u8 f32
        let base = 24;
        if raw.len() < base + n * rec + (n + 1) * 4 + e * 8 {
            return None;
        }
        let mut px = vec![0u16; n];
        let mut py = vec![0u16; n];
        let mut band = vec![0u8; n];
        let mut inhib = vec![false; n];
        let mut thr = vec![1.0f32; n];
        for i in 0..n {
            let o = base + i * rec;
            px[i] = u16::from_le_bytes(raw[o..o + 2].try_into().unwrap());
            py[i] = u16::from_le_bytes(raw[o + 2..o + 4].try_into().unwrap());
            band[i] = raw[o + 4];
            inhib[i] = raw[o + 5] != 0;
            thr[i] = f32::from_le_bytes(raw[o + 6..o + 10].try_into().unwrap());
        }
        let off_indptr = base + n * rec;
        let off_tgt = off_indptr + (n + 1) * 4;
        let off_w = off_tgt + e * 4;
        let mut indptr = Vec::with_capacity(n + 1);
        for i in 0..=n {
            indptr.push(u32at(off_indptr + i * 4));
        }
        let mut tgt = Vec::with_capacity(e);
        let mut wgt = Vec::with_capacity(e);
        for i in 0..e {
            tgt.push(u32at(off_tgt + i * 4));
            wgt.push(f32::from_le_bytes(raw[off_w + i * 4..off_w + i * 4 + 4].try_into().unwrap()));
        }
        let mut s = Self {
            n,
            grid_w: gw,
            grid_h: gh,
            v: vec![0.0; n],
            act: vec![0.0; n],
            i_buf: vec![0.0; n],
            refr: vec![0; n],
            inhib,
            px,
            py,
            band,
            thr,
            jg: vec![1.0; n],
            indptr,
            tgt,
            wgt,
            band_members: Default::default(),
            cmd: Vec::new(),
            rng_state: seed | 1,
            spikes: 0,
            rate: 0.0,
            pool_l: 0.0,
            pool_r: 0.0,
            pool_e: 0.0,
            pool_i: 0.0,
            act_lo: 0.0,
            act_hi: 0.0,
            col_groups: [0.0; 5],
            band_act: [0.0; crate::audio::NB],
        };
        for i in 0..n {
            s.jg[i] = 0.45 + s.rand() * 1.10;
            if s.rand() < 0.01 {
                s.cmd.push(i as u32);
            }
        }
        Some(s.finish())
    }

    /// Small synthetic geometric fallback (same dynamics, no real data).
    fn synthetic(seed: u64) -> Self {
        let (gw, gh) = (40usize, 20usize);
        let n = gw * gh;
        let mut s = Self {
            n,
            grid_w: gw,
            grid_h: gh,
            v: vec![0.0; n],
            act: vec![0.0; n],
            i_buf: vec![0.0; n],
            refr: vec![0; n],
            inhib: vec![false; n],
            px: vec![0; n],
            py: vec![0; n],
            band: vec![0; n],
            thr: vec![1.0; n],
            jg: vec![1.0; n],
            indptr: vec![0; n + 1],
            tgt: Vec::new(),
            wgt: Vec::new(),
            band_members: Default::default(),
            cmd: Vec::new(),
            rng_state: seed | 1,
            spikes: 0,
            rate: 0.0,
            pool_l: 0.0,
            pool_r: 0.0,
            pool_e: 0.0,
            pool_i: 0.0,
            act_lo: 0.0,
            act_hi: 0.0,
            col_groups: [0.0; 5],
            band_act: [0.0; crate::audio::NB],
        };
        let mut edges: Vec<(u32, f32)> = Vec::new();
        for i in 0..n {
            s.px[i] = (i % gw) as u16;
            s.py[i] = (i / gw) as u16;
            s.band[i] = ((i % gw) * crate::audio::NB / gw) as u8;
            s.inhib[i] = s.rand() < 0.2;
            s.thr[i] = 0.70 + s.rand() * 0.65;
            s.jg[i] = 0.45 + s.rand() * 1.10;
            if s.rand() < 0.15 {
                s.cmd.push(i as u32);
            }
            let lam = if s.inhib[i] { 2.0 } else { 4.0 };
            let mut row: Vec<(u32, f32)> = Vec::new();
            for _ in 0..6 {
                let mut best = ((i + 1) % n) as u32;
                let mut best_score = -1.0f32;
                for _ in 0..8 {
                    let j = (s.rand() * n as f32) as usize % n;
                    if j == i {
                        continue;
                    }
                    let dx = s.px[j] as f32 - s.px[i] as f32;
                    let dy = s.py[j] as f32 - s.py[i] as f32;
                    let score = (-dx.hypot(dy) / lam).exp() * s.rand();
                    if score > best_score {
                        best_score = score;
                        best = j as u32;
                    }
                }
                let w = if s.inhib[i] { -1.6 * 0.28 } else { 0.28 };
                row.push((best, w));
            }
            edges.append(&mut row);
        }
        // build CSR
        let mut counts = vec![0u32; n];
        // edges were appended per-source in order already; just fill
        for (i, e) in edges.chunks(6).enumerate() {
            s.indptr[i + 1] = s.indptr[i] + e.len() as u32;
            for &(t, w) in e {
                s.tgt.push(t);
                s.wgt.push(w);
            }
            let _ = &mut counts;
        }
        s.finish()
    }

    fn rand(&mut self) -> f32 {
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f32 / (1u64 << 53) as f32
    }

    /// neuron grid position (for the map view)
    pub fn pos(&self, i: usize) -> (u16, u16) {
        (self.px[i], self.py[i])
    }

    /// 'r' key: the real connectome's synapses are fixed data, so this just
    /// re-jitters the per-neuron injection gains (the only randomness left).
    pub fn rewire(&mut self) {
        for i in 0..self.n {
            self.jg[i] = 0.45 + self.rand() * 1.10;
        }
    }

    /// Hard silence: zero all membrane/activity state so pools read dead
    /// immediately (used when the audio engine reports true silence).
    pub fn kill(&mut self) {
        self.v.fill(0.0);
        self.act.fill(0.0);
        self.i_buf.fill(0.0);
        self.spikes = 0;
        self.rate = 0.0;
        self.pool_l = 0.0;
        self.pool_r = 0.0;
        self.pool_e = 0.0;
        self.pool_i = 0.0;
        self.act_lo = 0.0;
        self.act_hi = 0.0;
        self.col_groups = [0.0; 5];
        self.band_act = [0.0; crate::audio::NB];
    }

    /// One 100 Hz tick. `bands` = 10 band energies 0..1, `bonsets` = per-band
    /// onset pulses, `onset` global pulse 0..1, `leak`/`noise`/`coupling` from
    /// knobs. coupling==0 kills the whole field.
    pub fn tick(&mut self, bands: &[f32; crate::audio::NB], bonsets: &[f32; crate::audio::NB], onset: f32, leak: f32, noise: f32, coupling: f32) {
        let leak_rate = 0.02 + (0.15 - 0.02) * leak;

        // integrate with current from previous tick's injection + spikes
        for i in 0..self.n {
            self.v[i] += -self.v[i] * leak_rate + self.i_buf[i];
            self.i_buf[i] = 0.0;
        }

        // spike detection + delivery along REAL synapses
        let mut spikes = 0usize;
        for i in 0..self.n {
            if self.refr[i] > 0 {
                self.refr[i] -= 1;
                self.v[i] = 0.0;
            } else if self.v[i] > self.thr[i] {
                self.v[i] = 0.0;
                self.refr[i] = 2;
                self.act[i] = 1.0;
                spikes += 1;
                for e in self.indptr[i]..self.indptr[i + 1] {
                    self.i_buf[self.tgt[e as usize] as usize] += self.wgt[e as usize] * coupling;
                }
            }
            self.act[i] *= 0.90;
        }

        // audio injection (tonotopic): band b -> neurons assigned to band b
        if coupling > 0.0 {
            let inj = 0.55 * coupling;
            for b in 0..crate::audio::NB {
                let e = bands[b];
                let hit = bonsets[b] * 0.6; // sharp extra kick on band onsets
                if e < 0.001 && hit < 0.01 {
                    continue;
                }
                let amp = e * inj + hit * coupling;
                for &i in &self.band_members[b] {
                    self.i_buf[i as usize] += amp * self.jg[i as usize];
                }
            }
            if onset > 0.02 {
                let amp = onset * 0.4 * coupling;
                for &i in &self.cmd {
                    self.i_buf[i as usize] += amp * self.jg[i as usize];
                }
            }
        }
        if noise > 0.0 {
            let amp = noise * 0.06;
            for i in 0..self.n {
                self.i_buf[i] += (self.rand() - 0.5) * amp;
            }
        }

        self.spikes = spikes;
        let r = spikes as f32 / self.n as f32;
        self.rate += (r - self.rate) * 0.1;

        // pools (mean act over masks)
        let (mut sl, mut sr, mut se, mut si) = (0.0f32, 0.0, 0.0, 0.0);
        let (mut nl, mut nr, mut ne, mut ni) = (0usize, 0usize, 0usize, 0usize);
        let half = (self.grid_w / 2) as u16;
        for i in 0..self.n {
            let a = self.act[i];
            if self.px[i] < half {
                sl += a;
                nl += 1;
            } else {
                sr += a;
                nr += 1;
            }
            if self.inhib[i] {
                si += a;
                ni += 1;
            } else {
                se += a;
                ne += 1;
            }
        }
        self.pool_l = sl / nl.max(1) as f32;
        self.pool_r = sr / nr.max(1) as f32;
        self.pool_e = se / ne.max(1) as f32;
        self.pool_i = si / ni.max(1) as f32;
        for b in 0..crate::audio::NB {
            let m = &self.band_members[b];
            if m.is_empty() {
                self.band_act[b] = 0.0;
                continue;
            }
            let mut s = 0.0f32;
            for &i in m {
                s += self.act[i as usize];
            }
            self.band_act[b] = s / m.len() as f32;
        }
        self.act_lo = self.band_act[0..4].iter().sum::<f32>() / 4.0;
        self.act_hi = self.band_act[20..24].iter().sum::<f32>() / 4.0;
        // 5 display brackets from 24 bands: [0-4] [5-9] [10-13] [14-18] [19-23]
        const GS: [(usize, usize); 5] = [(0, 5), (5, 10), (10, 14), (14, 19), (19, 24)];
        for g in 0..5 {
            let (a, b) = GS[g];
            self.col_groups[g] = self.band_act[a..b].iter().sum::<f32>() / (b - a) as f32;
        }
    }
}
