//! Audio: ring buffer of stereo samples -> analyzer -> bands/onset.
//! Modes: DEMO (synth loop), MIC (cpal input), FILE (symphonia decode),
//! SYS (per-app system audio via pw-record, ignore list).

use std::collections::{HashMap, VecDeque};
use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{num_complex::Complex32, FftPlanner};

pub const SR: u32 = 44100;
pub const FFT_N: usize = 2048;
/// 24 log-spaced bands 40 Hz .. 12 kHz; one per instrument family slot:
/// 0-3 sub/bass (diameter), 4-7 low-mid body (rim bumps), 8+ mids..air
/// (anomalies: each band = one golden-angle slot = one instrument range).
pub const NB: usize = 24;
pub const BAND_EDGES: [f32; 25] = [
    40.0, 50.7, 64.3, 81.6, 103.5, 131.3, 166.5, 211.1, 267.8, 339.6, 430.7,
    546.3, 692.8, 878.7, 1114.4, 1413.4, 1792.6, 2273.5, 2883.4, 3656.9,
    4638.0, 5882.2, 7460.3, 9461.7, 12000.0,
];
const RING_CAP: usize = (SR as usize) / 4;

#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Demo,
    Mic,
    File,
    Sys,
    Silent,
}

impl Mode {
    pub fn name(&self) -> &'static str {
        match self {
            Mode::Demo => "DEMO",
            Mode::Mic => "MIC",
            Mode::File => "FILE",
            Mode::Sys => "SYS",
            Mode::Silent => "SILENT",
        }
    }
}

enum SysMsg {
    Add(String, Arc<Mutex<VecDeque<(f32, f32)>>>),
    Remove(String),
}

struct SysNode {
    proc: Child,
    q: Arc<Mutex<VecDeque<(f32, f32)>>>,
}

pub struct AudioEngine {
    pub bands: [f32; NB],
    pub onset: f32,
    pub mode: Mode,
    pub sys_apps: Vec<String>, // apps currently captured (HUD/debug)
    ring: Arc<Mutex<VecDeque<(f32, f32)>>>,
    gonio: VecDeque<(f32, f32)>, // recent stereo samples for GONIO view
    win: Vec<f32>,
    hann: Vec<f32>,
    fft: Arc<dyn rustfft::Fft<f32>>,
    fft_buf: Vec<Complex32>,
    flux_avg: f32, // running mean spectral flux for adaptive onset threshold
    band_prev: [f32; NB], // per-band spectral shape, for band-wise flux
    band_favg: [f32; NB], // per-band running flux mean (adaptive thresholds)
    /// per-band onset pulses 0..1 (kick/snare/hat fire in their own bands)
    pub bonset: [f32; NB],
    // cava-style gravity state: instant rise, quadratic falloff from peak
    gpeak: [f32; NB],
    gfall: [f32; NB],
    /// sustained-tonal-instrument level 0..1 (strings/pads): high when the
    /// string bands are loud AND not jumping (percussion pumps, bows hold)
    pub tonal: f32,
    /// last frame's normalized per-band peakiness (diagnostics/tuning)
    pub last_pk: [f32; NB],
    /// share of total spectral energy in bands 0-3 (volume-independent,
    /// never saturates); drives the disk DIAMETER and the beat pump
    pub bass_share: f32,
    agc_rms: f32, // slow-decay running peak for auto-gain
    /// per-band rolling spectral floor (dB): bands are judged against their
    /// OWN history, so violins 30 dB under the bass still light their band
    band_floor: [f32; NB],
    /// 1 while real samples are arriving, decays to 0 in ~1 s of silence;
    /// used to gate the sim's noise floor so the field dies when audio stops
    pub act_env: f32,
    /// smoothed RAW (pre-AGC) loudness 0..1: follows the actual volume knob,
    /// unlike the normalized bands; drives disk size
    pub loud_raw: f32,
    // playback plumbing (demo/file)
    playbuf: Arc<Vec<(f32, f32)>>,
    playhead: Arc<AtomicUsize>,
    out_stream: Option<cpal::Stream>,
    in_stream: Option<cpal::Stream>,
    demo_loop: Option<Vec<(f32, f32)>>,
    // sys mode
    sys_rx: Option<mpsc::Receiver<SysMsg>>,
    sys_qs: HashMap<String, Arc<Mutex<VecDeque<(f32, f32)>>>>,
    sys_stop: Option<Arc<AtomicBool>>,
    ignore: Vec<String>,
}

// ---------------------------------------------------------------- demo synth
const BPM: f32 = 120.0;
const BASS_PAT: [f32; 8] = [55.0, 55.0, 65.41, 55.0, 49.0, 55.0, 65.41, 49.0];
const PAD_CHORDS: [[f32; 3]; 4] = [
    [220.0, 261.63, 329.63],
    [174.61, 220.0, 261.63],
    [196.0, 246.94, 293.66],
    [164.81, 196.0, 246.94],
];
const PI2: f32 = 2.0 * std::f32::consts::PI;

fn place(buf: &mut [(f32, f32)], t0: f32, l: &[f32], r: &[f32]) {
    let i0 = (t0 * SR as f32) as usize;
    for k in 0..l.len() {
        if i0 + k >= buf.len() {
            break;
        }
        buf[i0 + k].0 += l[k];
        buf[i0 + k].1 += r[k];
    }
}

fn kick() -> Vec<f32> {
    let n = (0.30 * SR as f32) as usize;
    let mut ph = 0.0f32;
    (0..n)
        .map(|i| {
            let t = i as f32 / SR as f32;
            ph += (45.0 + 105.0 * (-t / 0.04).exp()) / SR as f32;
            (PI2 * ph).sin() * (-t / 0.08).exp() * 0.9
        })
        .collect()
}

fn bass(f: f32, dur: f32, detune: f32, seed: u64) -> Vec<f32> {
    let n = (dur * SR as f32) as usize;
    let cutoff = 300.0 + (((seed ^ 0x9E37).wrapping_mul(2654435761) >> 40) as f32 / 16777216.0) * 600.0;
    let a = (-PI2 * cutoff / SR as f32).exp();
    let f = f * (1.0 + detune / 1200.0);
    let mut acc = 0.0f32;
    (0..n)
        .map(|i| {
            let t = i as f32 / SR as f32;
            acc = (1.0 - a) * (2.0 * ((t * f) % 1.0) - 1.0) + a * acc;
            acc * (-t / 0.18).exp() * 0.5
        })
        .collect()
}

fn hat() -> Vec<f32> {
    let n = (0.06 * SR as f32) as usize;
    let mut x = 0x12345u64;
    let mut prev = 0.0f32;
    (0..n)
        .map(|i| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            let v = ((x >> 40) as f32 / 16777216.0) * 2.0 - 1.0;
            let hp = v - prev;
            prev = v;
            hp * (-(i as f32) / SR as f32 / 0.012).exp() * 0.09
        })
        .collect()
}

fn pad(freqs: [f32; 3], dur: f32) -> Vec<f32> {
    let n = (dur * SR as f32) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / SR as f32;
            let env = (t / (0.4 * dur)).min(1.0) * (-t / (0.8 * dur)).exp();
            let mut s = 0.0f32;
            for f in freqs {
                s += (PI2 * f * 0.995 * t).sin() + (PI2 * f * 1.005 * t).sin();
            }
            s * env * 0.02
        })
        .collect()
}

/// 4-bar 120 BPM stereo loop; bass slightly detuned L/R for gonio content.
pub fn make_demo_loop() -> Vec<(f32, f32)> {
    let step = 60.0 / BPM / 4.0;
    let dur = 4.0 * 4.0 * 60.0 / BPM; // 8 s
    let mut buf = vec![(0.0f32, 0.0f32); (dur * SR as f32) as usize];
    let k = kick();
    let h = hat();
    let mut stp = 0usize;
    let mut t = 0.0f32;
    while t < dur - step {
        if stp % 4 == 0 {
            place(&mut buf, t, &k, &k);
        }
        if stp % 2 == 1 {
            place(&mut buf, t, &h, &h);
        }
        let f = BASS_PAT[(stp >> 1) % 8];
        place(
            &mut buf,
            t,
            &bass(f, step * 1.9, -6.0, stp as u64),
            &bass(f, step * 1.9, 6.0, (stp + 7) as u64),
        );
        if stp % 16 == 0 {
            let p = pad(PAD_CHORDS[(stp / 16) % 4], step * 16.0);
            place(&mut buf, t, &p, &p);
        }
        stp += 1;
        t += step;
    }
    let peak = buf.iter().fold(0.0f32, |m, s| m.max(s.0.abs()).max(s.1.abs()));
    if peak > 0.0 {
        for s in buf.iter_mut() {
            s.0 *= 0.85 / peak;
            s.1 *= 0.85 / peak;
        }
    }
    buf
}

// ---------------------------------------------------------------- file decode
fn decode_file(path: &str) -> Result<Vec<(f32, f32)>, String> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::default::{get_codecs, get_probe};

    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let probed = get_probe()
        .format(&Default::default(), mss, &Default::default(), &Default::default())
        .map_err(|e| e.to_string())?;
    let mut format = probed.format;
    let track = format.default_track().ok_or("no audio track")?;
    let src_rate = track.codec_params.sample_rate.unwrap_or(SR);
    let mut decoder = get_codecs()
        .make(&track.codec_params, &Default::default())
        .map_err(|e| e.to_string())?;

    let max_frames = (SR * 300) as usize; // cap at 5 minutes
    let mut raw: Vec<(f32, f32)> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let spec = *decoded.spec();
        let frames = decoded.frames() as u64;
        let mut sb = SampleBuffer::<f32>::new(frames, spec);
        sb.copy_interleaved_ref(decoded);
        let ch = spec.channels.count().max(1);
        for frame in sb.samples().chunks(ch) {
            let l = frame[0];
            let r = if ch > 1 { frame[1] } else { l };
            raw.push((l, r));
        }
        if raw.len() > max_frames * src_rate as usize / SR as usize {
            break;
        }
    }
    if raw.is_empty() {
        return Err("decoded nothing".into());
    }
    Ok(resample_stereo(&raw, src_rate, SR))
}

fn resample_stereo(src: &[(f32, f32)], from: u32, to: u32) -> Vec<(f32, f32)> {
    if from == to || src.is_empty() {
        return src.to_vec();
    }
    let n = src.len() * to as usize / from as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let p = i as f64 * from as f64 / to as f64;
        let i0 = p as usize;
        let f = (p - i0 as f64) as f32;
        let i1 = (i0 + 1).min(src.len() - 1);
        out.push((
            src[i0].0 + (src[i1].0 - src[i0].0) * f,
            src[i0].1 + (src[i1].1 - src[i0].1) * f,
        ));
    }
    out
}

// ---------------------------------------------------------------- engine
impl AudioEngine {
    pub fn new() -> Self {
        let mut planner = FftPlanner::new();
        let ignore = std::env::var("NBS_IGNORE")
            .unwrap_or_else(|_| "discord,vesktop".into())
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_lowercase())
            .collect();
        Self {
            bands: [0.0; NB],
            onset: 0.0,
            mode: Mode::Silent,
            sys_apps: Vec::new(),
            ring: Arc::new(Mutex::new(VecDeque::new())),
            gonio: VecDeque::new(),
            win: vec![0.0; FFT_N],
            hann: (0..FFT_N)
                .map(|i| 0.5 - 0.5 * (PI2 * i as f32 / FFT_N as f32).cos())
                .collect(),
            fft: planner.plan_fft_forward(FFT_N),
            fft_buf: vec![Complex32::ZERO; FFT_N],
            flux_avg: 0.0,
            band_prev: [0.0; NB],
            band_favg: [0.0; NB],
            bonset: [0.0; NB],
            gpeak: [0.0; NB],
            gfall: [0.0; NB],
            tonal: 0.0,
            last_pk: [1.0; NB],
            bass_share: 0.0,
            agc_rms: 0.0,
            band_floor: [-120.0; NB],
            act_env: 0.0,
            loud_raw: 0.0,
            playbuf: Arc::new(Vec::new()),
            playhead: Arc::new(AtomicUsize::new(0)),
            out_stream: None,
            in_stream: None,
            demo_loop: None,
            sys_rx: None,
            sys_qs: HashMap::new(),
            sys_stop: None,
            ignore,
        }
    }

    fn open_output(&mut self, buf: Vec<(f32, f32)>) -> Result<(), String> {
        let host = cpal::default_host();
        let dev = host.default_output_device().ok_or("no output device")?;
        let cfg = dev.default_output_config().map_err(|e| e.to_string())?;
        if cfg.sample_format() != cpal::SampleFormat::F32 {
            return Err("output device not f32".into());
        }
        let dev_sr = cfg.sample_rate().0;
        let ch = cfg.channels() as usize;
        let buf = if dev_sr != SR { resample_stereo(&buf, SR, dev_sr) } else { buf };
        self.playbuf = Arc::new(buf);
        self.playhead.store(0, Ordering::Relaxed);
        let ring = self.ring.clone();
        let playbuf = self.playbuf.clone();
        let playhead = self.playhead.clone();
        let stream = dev
            .build_output_stream(
                &cfg.config(),
                move |data: &mut [f32], _| {
                    let buf = &*playbuf;
                    if buf.is_empty() {
                        data.fill(0.0);
                        return;
                    }
                    let mut head = playhead.load(Ordering::Relaxed);
                    let nframes = data.len() / ch.max(1);
                    let mut pushed: Vec<(f32, f32)> = Vec::with_capacity(nframes);
                    for frame in data.chunks_mut(ch.max(1)) {
                        if head >= buf.len() {
                            head = 0; // loop
                        }
                        let (l, r) = buf[head];
                        head += 1;
                        for (ci, s) in frame.iter_mut().enumerate() {
                            *s = if ci % 2 == 0 { l } else { r };
                        }
                        pushed.push((l, r));
                    }
                    playhead.store(head, Ordering::Relaxed);
                    if let Ok(mut ring) = ring.try_lock() {
                        if ring.len() + pushed.len() < RING_CAP {
                            ring.extend(pushed);
                        }
                    }
                },
                |_| {},
                None,
            )
            .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;
        self.out_stream = Some(stream);
        Ok(())
    }

    fn open_mic(&mut self) -> Result<(), String> {
        let host = cpal::default_host();
        let dev = host.default_input_device().ok_or("no input device")?;
        let cfg = dev.default_input_config().map_err(|e| e.to_string())?;
        if cfg.sample_format() != cpal::SampleFormat::F32 {
            return Err("input device not f32".into());
        }
        let ch = cfg.channels() as usize;
        let ring = self.ring.clone();
        let stream = dev
            .build_input_stream(
                &cfg.config(),
                move |data: &[f32], _| {
                    if let Ok(mut ring) = ring.try_lock() {
                        if ring.len() + data.len() < RING_CAP {
                            for frame in data.chunks(ch.max(1)) {
                                let l = frame[0];
                                let r = if ch > 1 { frame[1] } else { l };
                                ring.push_back((l, r));
                            }
                        }
                    }
                },
                |_| {},
                None,
            )
            .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;
        self.in_stream = Some(stream);
        Ok(())
    }

    // ----- SYS: per-app capture via pw-record -----
    fn sys_list_apps() -> Option<Vec<(String, String)>> {
        // -> [(node_name, lowercased app+node+media names)]
        let out = Command::new("pactl").args(["list", "sink-inputs"]).output().ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        let mut apps = Vec::new();
        for block in text.split("Sink Input #").skip(1) {
            let get = |key: &str| {
                let pat = format!("{} = \"", key);
                block.split(&pat).nth(1).and_then(|r| r.split('"').next())
            };
            let node = get("node.name").unwrap_or_default().to_string();
            let names = format!(
                "{} {} {}",
                get("application.name").unwrap_or_default().to_lowercase(),
                get("node.name").unwrap_or_default().to_lowercase(),
                get("media.name").unwrap_or_default().to_lowercase()
            );
            if !node.is_empty() {
                apps.push((node, names));
            }
        }
        Some(apps)
    }

    fn sys_spawn(node: &str) -> Option<SysNode> {
        let mut proc = Command::new("pw-record")
            .args([
                &format!("--target={}", node),
                &format!("--rate={}", SR),
                "--channels=2",
                "--format=f32",
                "--container=raw",
                "--latency=25ms",
                "-",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let q = Arc::new(Mutex::new(VecDeque::new()));
        let q2 = q.clone();
        let mut stdout = proc.stdout.take()?;
        thread::spawn(move || {
            let mut buf = vec![0u8; 8192 * 8];
            loop {
                match stdout.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        if let Ok(mut q) = q2.lock() {
                            for fr in 0..(n / 8) {
                                let b = &buf[fr * 8..fr * 8 + 8];
                                let l = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                                let r = f32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                                q.push_back((l, r));
                            }
                            while q.len() > RING_CAP {
                                q.pop_front();
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        Some(SysNode { proc, q })
    }

    fn open_sys(&mut self) -> Result<(), String> {
        if Self::sys_list_apps().is_none() {
            return Err("no pactl".into());
        }
        let stop = Arc::new(AtomicBool::new(false));
        self.sys_stop = Some(stop.clone());
        let (tx, rx) = mpsc::channel::<SysMsg>();
        self.sys_rx = Some(rx);
        let ignore = self.ignore.clone();
        thread::spawn(move || {
            let mut nodes: HashMap<String, SysNode> = HashMap::new();
            while !stop.load(Ordering::Relaxed) {
                if let Some(apps) = Self::sys_list_apps() {
                    let live: std::collections::HashSet<&str> =
                        apps.iter().map(|(n, _)| n.as_str()).collect();
                    let dead: Vec<String> =
                        nodes.keys().filter(|n| !live.contains(n.as_str())).cloned().collect();
                    for n in dead {
                        if let Some(mut h) = nodes.remove(&n) {
                            let _ = h.proc.kill();
                            let _ = tx.send(SysMsg::Remove(n));
                        }
                    }
                    for (node, names) in apps {
                        if nodes.contains_key(&node) {
                            continue;
                        }
                        if ignore.iter().any(|bad| names.contains(bad)) {
                            continue;
                        }
                        if let Some(h) = Self::sys_spawn(&node) {
                            let _ = tx.send(SysMsg::Add(node.clone(), h.q.clone()));
                            nodes.insert(node, h);
                        }
                    }
                }
                thread::sleep(Duration::from_millis(1500));
            }
            for (_, mut h) in nodes {
                let _ = h.proc.kill();
            }
        });
        Ok(())
    }

    pub fn close(&mut self) {
        if let Some(stop) = self.sys_stop.take() {
            stop.store(true, Ordering::Relaxed);
        }
        self.sys_rx = None;
        self.sys_qs.clear();
        self.sys_apps.clear();
        self.out_stream = None;
        self.in_stream = None;
    }

    pub fn set_mode(&mut self, mode: Mode, filepath: Option<&str>) {
        self.close();
        self.mode = mode;
        let result = match mode {
            Mode::Demo => {
                let buf = self.demo_loop.get_or_insert_with(make_demo_loop).clone();
                self.open_output(buf)
            }
            Mode::Mic => self.open_mic(),
            Mode::File => match filepath {
                Some(p) => decode_file(p).and_then(|buf| self.open_output(buf)),
                None => {
                    self.mode = Mode::Demo;
                    let buf = self.demo_loop.get_or_insert_with(make_demo_loop).clone();
                    self.open_output(buf)
                }
            },
            Mode::Sys => self.open_sys(),
            Mode::Silent => Ok(()),
        };
        if result.is_err() {
            self.mode = Mode::Silent;
        }
    }

    // ----- analyzer (main loop, ~60 Hz) -----
    pub fn update(&mut self) {
        // ingest sys node add/remove messages
        if let Some(rx) = &self.sys_rx {
            let mut msgs = Vec::new();
            while let Ok(m) = rx.try_recv() {
                msgs.push(m);
            }
            for m in msgs {
                match m {
                    SysMsg::Add(n, q) => {
                        self.sys_qs.insert(n.clone(), q);
                        self.sys_apps.push(n);
                    }
                    SysMsg::Remove(n) => {
                        self.sys_qs.remove(&n);
                        self.sys_apps.retain(|a| a != &n);
                    }
                }
            }
        }

        // build this frame's sample batch
        let mut batch: Vec<(f32, f32)> = Vec::new();
        if self.mode == Mode::Sys {
            let mut chunks: Vec<Vec<(f32, f32)>> = Vec::new();
            for q in self.sys_qs.values() {
                if let Ok(mut q) = q.lock() {
                    // real-time bound: never process audio older than ~25 ms.
                    // Bursts (startup, scheduling hiccups) would otherwise
                    // build a backlog that makes every beat arrive late.
                    let keep = SR as usize / 40; // ~25 ms
                    let excess = q.len().saturating_sub(keep);
                    if excess > 0 {
                        q.drain(..excess);
                    }
                    let n = q.len().min(2048);
                    if n > 0 {
                        chunks.push(q.drain(..n).collect());
                    }
                }
            }
            if !chunks.is_empty() {
                let n = chunks.iter().map(|c| c.len()).min().unwrap_or(0);
                if n > 0 {
                    batch = vec![(0.0, 0.0); n];
                    for c in &chunks {
                        for (i, s) in c[..n].iter().enumerate() {
                            batch[i].0 = (batch[i].0 + s.0).clamp(-1.0, 1.0);
                            batch[i].1 = (batch[i].1 + s.1).clamp(-1.0, 1.0);
                        }
                    }
                }
            }
        } else if let Ok(mut ring) = self.ring.try_lock() {
            let n = ring.len().min(4096);
            batch.extend(ring.drain(..n));
        }

        let got = batch.len();
        // auto-gain: track a fast-adapting RMS envelope and normalize every
        // source toward a fixed RMS, so the visuals are independent of both
        // app volume and system volume (only the spectral shape drives them).
        if got > 0 {
            let rms = (batch.iter().map(|s| s.0 * s.0 + s.1 * s.1).sum::<f32>()
                / (2 * got) as f32)
                .sqrt();
            if rms > 1e-5 {
                self.act_env = 1.0;
                // raw loudness BEFORE the AGC: this is what makes the disk
                // size follow the actual volume (bands stay normalized)
                self.loud_raw += ((rms * 5.0).min(1.0) - self.loud_raw) * 0.20;
                // ~0.8 s time constant at 60 fps: slow enough to preserve
                // beat-to-beat dynamics (fast AGC pumps the gain and creates
                // phantom/skipped onsets), fast enough to track volume knobs.
                self.agc_rms += (rms - self.agc_rms) * 0.02;
            } else {
                self.act_env *= 0.80; // ~100 ms fade on digital silence
                self.loud_raw *= 0.90;
            }
            if self.agc_rms > 1e-4 {
                let gain = (0.12 / self.agc_rms).clamp(0.02, 64.0);
                for s in batch.iter_mut() {
                    s.0 *= gain;
                    s.1 *= gain;
                }
            }
        }
        for (l, r) in &batch {
            self.gonio.push_back((*l, *r));
        }
        while self.gonio.len() > 8192 {
            self.gonio.pop_front();
        }
        let mono: Vec<f32> = batch.iter().map(|(l, r)| (l + r) * 0.5).collect();
        self.feed_window(&mono);
        if got == 0 {
            self.act_env *= 0.80;
            self.loud_raw *= 0.90;
            self.onset *= 0.93;
            self.tonal *= 0.90;
            for b in self.bands.iter_mut() {
                *b *= 0.98;
            }
            return;
        }
        self.analyze();
    }

    fn feed_window(&mut self, samples: &[f32]) {
        let mut off = 0;
        while off < samples.len() {
            let n = (samples.len() - off).min(FFT_N);
            if n == FFT_N {
                self.win.copy_from_slice(&samples[off..off + FFT_N]);
            } else {
                self.win.copy_within(n.., 0);
                self.win[FFT_N - n..].copy_from_slice(&samples[off..]);
            }
            off += n;
        }
    }

    fn analyze(&mut self) {
        for i in 0..FFT_N {
            self.fft_buf[i] = Complex32::new(self.win[i] * self.hann[i], 0.0);
        }
        self.fft.process(&mut self.fft_buf);
        let freqs_step = SR as f32 / FFT_N as f32;
        // band mean log-energies -> per-frame min-max normalized spectral shape
        let mut d = [0.0f32; NB];
        let mut eraw = [0.0f32; NB]; // pre-EQ mean bin energy (for bass share)
        let mut pk = [15.0f32; NB]; // per-band peakiness (default: pass gate)
        for b in 0..NB {
            let i0 = (BAND_EDGES[b] / freqs_step).ceil() as usize;
            let i1 = ((BAND_EDGES[b + 1] / freqs_step) as usize).min(FFT_N / 2 - 1);
            let mut e = 0.0f32;
            let mut emax = 0.0f32;
            let mut n = 0usize;
            for i in i0..=i1.max(i0) {
                let m = self.fft_buf[i].norm_sqr();
                e += m;
                if m > emax {
                    emax = m;
                }
                n += 1;
            }
            eraw[b] = e / n.max(1) as f32;
            d[b] = 10.0 * ((e / n.max(1) as f32) + 1e-12).log10();
            // cava-style equal-loudness EQ: f^0.85 magnitude boost
            // (~+2.6 dB/octave) so treble isn't structurally starved
            let fc = (BAND_EDGES[b] * BAND_EDGES[b + 1]).sqrt();
            d[b] += 8.5 * (fc / 40.0).log10();
            // tonal sounds pile energy into few bins; noise spreads evenly
            // with expected peak/mean ≈ ln(n)+0.58. Gate each band against
            // ITS OWN bin-count expectation so the gate works the same in
            // narrow low bands and wide high bands.
            if n >= 4 && e > 0.0 {
                pk[b] = emax * n as f32 / e / ((n as f32).ln() + 0.577);
            }
        }
        let rms = (self.win.iter().map(|v| v * v).sum::<f32>() / FFT_N as f32).sqrt();
        let loud = (rms * 3.0).min(1.0);
        // bass share of total spectrum energy (volume-independent by
        // construction, and it cannot saturate like the floored bands do)
        let etot: f32 = eraw.iter().sum::<f32>().max(1e-12);
        let share: f32 = eraw[0..4].iter().sum::<f32>() / etot;
        self.bass_share += (share - self.bass_share) * 0.15;
        // Onset detection: Böck-style log-domain spectral flux. Positive
        // diffs are computed on the log-magnitude bands directly; linear or
        // normalized values squash exactly the transients we want, which is
        // what skipped beats in loud/dense passages.
        let mut lflux = 0.0f32;
        let mut str_act = 0.0f32;
        for b in 0..NB {
            // Per-band self-normalization: each band is judged against its
            // OWN rolling floor. The up-rate is very slow (~40 s, cava's
            // autosens is 20:1 asymmetric the same way) so a sustained
            // string section keeps playing instead of being absorbed after
            // ~10 s; the down rate (~2 s) resets quickly between songs.
            let up = 0.0004;
            let down = 0.008;
            let rate = if d[b] > self.band_floor[b] { up } else { down };
            self.band_floor[b] += (d[b] - self.band_floor[b]) * rate;
            // +4 dB over floor -> 0, +18 dB over floor -> 1
            let act = ((d[b] - self.band_floor[b] - 4.0) / 14.0).clamp(0.0, 1.0);
            // soft tonality tilt only: the per-band floor already absorbs
            // STATIONARY noise (it normalizes to it). pk just adds a mild
            // preference for tonal content (music sits at pk 1.2-2.1, pure
            // noise at ~1.0) without collapsing dense mixes.
            self.last_pk[b] = pk[b];
            let w = ((pk[b] - 1.1) / 1.6).clamp(0.0, 1.0);
            let raw = act * loud * (0.35 + 0.65 * w);
            let df = (d[b] - self.band_prev[b]).max(0.0); // d is 10*log10(E)
            self.band_prev[b] = d[b];
            lflux += df;
            if (11..18).contains(&b) {
                str_act += raw;
            }
            // per-band onset: fires when THIS band jumps vs its own running
            // flux. The floor (0.4 ≈ 9 dB) keeps broadband kicks from
            // firing every band at once; each drum owns its own bands.
            self.band_favg[b] += (df - self.band_favg[b]) * 0.05;
            let bthr = self.band_favg[b] * 1.8 + 0.4;
            if df > bthr && self.bonset[b] < 0.4 {
                self.bonset[b] = 1.0;
            }
            self.bonset[b] *= 0.88;
            // cava gravity: instant rise; on fall, quadratic decay from the
            // last peak (~0.4 s to zero) instead of a fixed exponential.
            if raw >= self.bands[b] {
                self.gpeak[b] = raw;
                self.gfall[b] = 0.0;
                self.bands[b] = raw;
            } else {
                self.gfall[b] += 0.028;
                self.bands[b] =
                    (self.gpeak[b] * (1.0 - self.gfall[b] * self.gfall[b] * 2.6)).max(0.0);
            }
        }
        // sustained-tonal detector: string bands (546 Hz - 2.9 kHz) holding
        // level. Slow EMA (~1.5 s): percussion spikes don't accumulate,
        // bows and pads do. (A flux penalty was tried and rejected; it
        // kills rhythmic string ostinatos, which ARE strings.)
        let str_tgt = (str_act / 7.0 * 2.2).min(1.0);
        self.tonal += (str_tgt - self.tonal) * 0.025;
        self.tonal *= 0.998_f32.max(1.0 - 0.4 * (1.0 - loud)); // die fast when quiet
        lflux /= NB as f32;
        // adaptive threshold over a short window (Böck): beats must stand out
        // from the recent flux average, whatever the song's density.
        self.flux_avg += (lflux - self.flux_avg) * 0.05;
        let thr = self.flux_avg * 1.3 + 0.12;
        if lflux > thr && self.onset < 0.5 {
            self.onset = 1.0;
        }
        self.onset *= 0.88; // ~90 ms refractory => catches kicks up to ~650 BPM
    }

    /// recent stereo samples for the GONIO view
    pub fn gonio_samples(&self) -> &VecDeque<(f32, f32)> {
        &self.gonio
    }

    /// Offline profiler: decode a file and run it through the exact same
    /// analyzer at the same ~60 Hz cadence, collecting per-band statistics.
    /// Prints a tuning table. Used by `--profile <file>`.
    pub fn profile(path: &str) -> Result<(), String> {
        let buf = decode_file(path)?;
        let mut eng = AudioEngine::new();
        let hop = SR as usize / 60; // one analyzer step per frame
        let mut mean = [0.0f64; NB];
        let mut p90: Vec<Vec<f32>> = vec![Vec::new(); NB];
        let mut pks: Vec<Vec<f32>> = vec![Vec::new(); NB];
        let mut bonsets = [0u32; NB];
        let mut frames = 0u64;
        let mut tonal_frames = 0u64;
        let mut pos = 0usize;
        while pos + hop <= buf.len() {
            let chunk: Vec<(f32, f32)> = buf[pos..pos + hop].to_vec();
            pos += hop;
            let rms = (chunk.iter().map(|s| s.0 * s.0 + s.1 * s.1).sum::<f32>()
                / (2 * hop) as f32)
                .sqrt();
            if rms > 1e-5 {
                eng.agc_rms += (rms - eng.agc_rms) * 0.02;
            }
            let mut scaled = chunk.clone();
            if eng.agc_rms > 1e-4 {
                let gain = (0.12 / eng.agc_rms).clamp(0.02, 64.0);
                for s in scaled.iter_mut() {
                    s.0 *= gain;
                    s.1 *= gain;
                }
            }
            let mono: Vec<f32> = scaled.iter().map(|(l, r)| (l + r) * 0.5).collect();
            eng.feed_window(&mono);
            eng.analyze();
            frames += 1;
            for b in 0..NB {
                mean[b] += eng.bands[b] as f64;
                p90[b].push(eng.bands[b]);
                pks[b].push(eng.last_pk[b]);
            }
            for b in 0..NB {
                if eng.bonset[b] > 0.7 {
                    bonsets[b] += 1;
                }
            }
            if eng.tonal > 0.3 {
                tonal_frames += 1;
            }
        }
        let secs = frames as f32 / 60.0;
        println!("file: {path}");
        println!("duration {:.0}s, {} frames", secs, frames);
        println!("band | f_range        | mean  p90   | onsets/s | pk p50");
        for b in 0..NB {
            let v = &mut p90[b];
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p = v[(v.len() * 9 / 10).min(v.len() - 1)];
            let v2 = &mut pks[b];
            v2.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pk50 = v2[v2.len() / 2];
            println!(
                "{:4} | {:7.0}-{:7.0} | {:.3} {:.3} | {:.2} | {:.2}",
                b,
                BAND_EDGES[b],
                BAND_EDGES[b + 1],
                mean[b] / frames as f64,
                p,
                bonsets[b] as f32 / secs,
                pk50
            );
        }
        println!("strings-detector active: {:.0}% of frames", 100.0 * tonal_frames as f32 / frames as f32);
        Ok(())
    }
}
