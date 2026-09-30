//! NEURAFLY v0.2 (was neural-beam-scope), Aloboi edition (Rust)
//! audio → bands/onsets → current injection → spiking network → pool activities
//!       → figure params → oscillator bank → beam → braille phosphor → terminal

mod audio;
mod canvas;
mod figure;
mod sim;

use std::io::{stdout, Write};
use std::time::{Duration, Instant};

use crossterm::{
    cursor, event::{self, Event, KeyCode, KeyEventKind}, execute, queue,
    style::Print, terminal,
};

use audio::{AudioEngine, Mode, NB};
use canvas::Canvas;
use figure::Figure;
use sim::Sim;

const SIM_DT: f32 = 0.01; // 100 Hz fixed sim tick
const BARS: &str = "▁▂▃▄▅▆▇█";

struct Knobs {
    coupling: f32,
    leak: f32,
    persist: f32,
    noise: f32,
    sel: usize,
}

#[derive(PartialEq, Clone, Copy)]
enum View {
    Scope,
    Gonio,
    Grid,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f32 / (1u64 << 53) as f32
    }
}

const KNOB_DEFS: [(&str, f32, f32, f32); 4] = [
    ("COUPLING", 0.2, 2.5, 0.05),
    ("LEAK", 0.0, 1.0, 0.02),
    ("PERSIST", 0.06, 0.25, 0.005),
    ("NOISE", 0.0, 1.0, 0.02),
];

struct App {
    sim: Sim,
    fig: Figure,
    audio: AudioEngine,
    canvas: Canvas,
    grid_buf: Vec<f32>,
    knobs: Knobs,
    view: View,
    mapping: u8,
    paused: bool,
    hud_on: bool,
    static_disk: bool,
    zoom: f32,
    spin: f32,
    swirl: f32,
    help_on: bool,
    last_rows: Vec<String>,
    sim_time: f32,
    rng: Rng,
    w: u16,
    h: u16,
    filepath: Option<String>,
}

impl App {
    fn new(filepath: Option<String>) -> Self {
        let sim = Sim::new(0xC0FFEE);
        let grid_buf = vec![0.0f32; sim.n];
        Self {
            sim,
            fig: Figure::new(),
            audio: AudioEngine::new(),
            canvas: Canvas::new(10, 10),
            grid_buf,
            knobs: Knobs { coupling: 1.0, leak: 0.5, persist: 0.16, noise: 0.15, sel: 0 },
            view: View::Scope,
            mapping: 0,
            paused: false,
            hud_on: true,
            static_disk: false,
            zoom: 1.0,
            spin: 1.0,
            swirl: 1.0,
            help_on: false,
            last_rows: Vec::new(),
            sim_time: 0.0,
            rng: Rng(0xBADC0DE),
            w: 0,
            h: 0,
            filepath,
        }
    }

    fn layout(&mut self, w: u16, h: u16) {
        self.w = w;
        self.h = h;
        self.canvas = Canvas::new(w as usize, (h as usize).saturating_sub(1));
        self.grid_buf.iter_mut().for_each(|v| *v = 0.0);
        self.last_rows.clear();
    }

    fn status_line(&self) -> String {
        let mut band = String::new();
        for b in 0..NB {
            let i = (self.audio.bands[b] * 8.0).min(7.0) as usize;
            band.push(BARS.chars().nth(i).unwrap());
        }
        let vals = [self.knobs.coupling, self.knobs.leak, self.knobs.persist, self.knobs.noise];
        let names = ["c", "l", "p", "n"];
        let mut knobs = String::new();
        for (i, v) in vals.iter().enumerate() {
            let seg = format!("{} {:.2}", names[i], v);
            if i == self.knobs.sel {
                knobs.push_str(&format!("\x1b[1m{}\x1b[0m", seg));
            } else {
                knobs.push_str(&seg);
            }
            if i < 3 {
                knobs.push_str("  ");
            }
        }
        format!(
            " {}  {}  {:<6} z{:.2} v{:.1} w{:.1}  {}",
            band,
            self.fig.ratio_str(),
            self.audio.mode.name().to_lowercase(),
            self.zoom,
            self.spin,
            self.swirl,
            knobs
        )
    }

    // ----- scope / gonio drawing
    fn draw_beam(&mut self, dt: f32) {
        self.canvas.decay(self.knobs.persist);
        self.fig.update(&self.sim, self.audio.onset, self.audio.tonal, self.audio.bass_share, self.audio.loud_raw, dt, &mut | | self.rng.next());

        let (pw, ph) = (self.canvas.px_w as f32, self.canvas.px_h as f32);
        let (cx, cy) = (pw / 2.0, ph / 2.0);
        let intensity = 0.3 + 0.9 * self.fig.intensity;
        // uniform scale on the SHORTER axis: braille dots are ~square, so a
        // unit radius is a true circle and the default disk fits with
        // headroom for rim anomalies and detached orbiters
        let sc = 0.36 * pw.min(ph);
        let to_px = |p: (f32, f32)| {
            ((cx + p.0 * sc).round() as i32, (cy - p.1 * sc).round() as i32)
        };
        // sample the current polyline
        let steps = 220usize;
        let poly: Vec<(f32, f32)> = (0..=steps)
            .map(|s| self.fig.shape_point(s as f32 / steps as f32))
            .collect();
        let torn: Vec<bool> = (0..=steps)
            .map(|s| self.fig.is_torn(s as f32 / steps as f32))
            .collect();
        // full closed circle; torn sectors are skipped (rim rupture glitch).
        // In near-silence nothing is drawn: even a faint constant line would
        // accumulate through phosphor persistence into a visible ghost.
        let quiet = self.fig.intensity < 0.05;
        let mut prev = to_px(poly[0]);
        for s in 1..=steps {
            let p = to_px(poly[s]);
            if !quiet && !torn[s] && !torn[s - 1] {
                self.canvas.line(prev.0, prev.1, p.0, p.1, intensity * 0.22, 0);
            }
            prev = p;
        }
        // shockwave rings: fading copies of the rim expanding outward
        // (combo: drums 8-9 + bass body 4-7)
        for (_r0, scale, bright) in self.fig.pulses() {
            let mut prev = to_px((poly[0].0 * scale, poly[0].1 * scale));
            for s in 1..=steps {
                let p = to_px((poly[s].0 * scale, poly[s].1 * scale));
                self.canvas.line(prev.0, prev.1, p.0, p.1, intensity * 0.30 * bright, 6);
                prev = p;
            }
        }
        // tethers: chords across the disk (combo: voice 10-12 + strings 13-15)
        for (x1, y1, x2, y2, bright) in self.fig.tethers() {
            let p = to_px((x1, y1));
            let q = to_px((x2, y2));
            self.canvas.line(p.0, p.1, q.0, q.1, intensity * 0.45 * bright, 4);
        }
        // corona flares: tall radial spikes (combo: violin 16-18 + brass 19-21)
        for (x1, y1, x2, y2, bright) in self.fig.flares() {
            let p = to_px((x1, y1));
            let q = to_px((x2, y2));
            self.canvas.line(p.0, p.1, q.0, q.1, intensity * 0.8 * bright, 7);
        }
        // glitch dashes: horizontal tears inside the disk (combo: cymbals 22-23
        // + bass 0-3), flickering every frame tick
        for (x1, y1, x2, y2, bright) in self.fig.glitches() {
            let p = to_px((x1, y1));
            let q = to_px((x2, y2));
            self.canvas.line(p.0, p.1, q.0, q.1, intensity * 0.7 * bright, 2);
        }
        // debris particles: fire-style motion trails, colored by source
        // band range: red embers (8-9), yellow mids (10-17), cyan air (18+),
        // white sparkles (30, combo strings+cymbals)
        for (px, py, vx, vy, bright, band) in self.fig.particles() {
            let p = to_px((px, py));
            let q = to_px((px - vx * 0.12, py - vy * 0.12));
            let col = if band >= 30 { 7 } else if band < 10 { 1 } else if band < 18 { 3 } else { 6 };
            self.canvas.line(q.0, q.1, p.0, p.1, intensity * 0.5 * bright, col);
            self.canvas
                .dot(p.0 as f32, p.1 as f32, intensity * 0.9 * bright, col);
        }
        // knots: strands darting out of the rim and back (bands 13-15,
        // violin/flute low), drawn in magenta so they read as filaments
        for (x1, y1, x2, y2, bright) in self.fig.knots() {
            let p = to_px((x1, y1));
            let q = to_px((x2, y2));
            self.canvas.line(p.0, p.1, q.0, q.1, intensity * 0.7 * bright, 5);
            self.canvas
                .dot(q.0 as f32, q.1 as f32, intensity * 0.9 * bright, 5);
        }
        // orbiters: cyan ring-threaders / blue outer moons, each with a
        // short trail along its orbit
        for (px, py, tx, ty, bright, inner) in self.fig.orbits() {
            let p = to_px((px, py));
            let q = to_px((tx, ty));
            let col = if inner { 6 } else { 4 }; // cyan / blue
            self.canvas.line(q.0, q.1, p.0, p.1, intensity * 0.35 * bright, col);
            self.canvas
                .dot(p.0 as f32, p.1 as f32, intensity * 0.6 * bright, col);
        }
        // bright tracer travelling along the shape (magenta head)
        if self.fig.intensity > 0.15 {
            let p = self.fig.beam_at(0.0);
            let q = self.fig.beam_at(-0.06);
            let hp = to_px(p);
            let tp = to_px(q);
            self.canvas
                .line(tp.0, tp.1, hp.0, hp.1, intensity * 0.6, 5);
            self.canvas
                .dot(hp.0 as f32, hp.1 as f32, intensity, 5);
        }
    }

    fn draw_gonio(&mut self, dt: f32) {
        // network controls gonio rotation, scale, intensity; freeze when silent
        self.fig.update(&self.sim, self.audio.onset, self.audio.tonal, self.audio.bass_share, self.audio.loud_raw, dt, &mut | | self.rng.next());
        self.canvas.decay(self.knobs.persist);
        let act_mean = (self.sim.act_lo + self.sim.act_hi) * 0.5;
        let scale = (0.2 + 0.8 * act_mean).min(1.0);
        let inten = 0.3 + 0.9 * self.fig.intensity;
        if self.fig.intensity < 0.02 {
            return; // frozen to a point
        }
        let (pw, ph) = (self.canvas.px_w as f32, self.canvas.px_h as f32);
        let (cx, cy) = (pw / 2.0, ph / 2.0);
        let samples = self.audio.gonio_samples();
        let n = samples.len().min(2048);
        if n < 8 {
            return;
        }
        let rot = self.fig_rot_snapshot();
        let (rs, rc) = rot.sin_cos();
        let step = 8usize;
        let mut prev: Option<(i32, i32)> = None;
        for k in 0..(n / step) {
            let (l, r) = samples[samples.len() - n + k * step];
            let x = l * scale;
            let y = r * scale;
            let xr = x * rc - y * rs;
            let yr = x * rs + y * rc;
            let (px, py) = ((cx + xr * pw * 0.42).round() as i32, (cy - yr * ph * 0.42).round() as i32);
            if let Some((qx, qy)) = prev {
                self.canvas.line(qx, qy, px, py, inten * 0.5, 0);
            }
            prev = Some((px, py));
        }
    }

    fn fig_rot_snapshot(&self) -> f32 {
        self.fig.rot_snapshot()
    }

    /// Neuron map: every neuron owns a fixed cell (its connectome layout
    /// position), mapped onto the WHOLE screen; many neurons may share a
    /// cell, which then lights with the max of them. The glyph never moves
    /// and never changes; always '·'; it only lights up (bold) while the
    /// neuron fires, then dims back.
    fn draw_grid_rows(&mut self) -> Vec<String> {
        for i in 0..self.sim.n {
            self.grid_buf[i] = (self.grid_buf[i] * 0.90).max(self.sim.act[i]);
        }
        let (aw, ah) = (self.canvas.cols.saturating_sub(2).max(1), self.canvas.rows.max(1));
        let (gw, gh) = (self.sim.grid_w.max(1), self.sim.grid_h.max(1));
        let mut lit = vec![0.0f32; aw * ah];
        let mut used = vec![false; aw * ah];
        for i in 0..self.sim.n {
            let (gx, gy) = self.sim.pos(i);
            let cx = (gx as usize).min(gw - 1) * (aw - 1) / (gw - 1).max(1);
            let cy = (gy as usize).min(gh - 1) * (ah - 1) / (gh - 1).max(1);
            let c = cy * aw + cx;
            used[c] = true;
            let a = self.grid_buf[i];
            if a > lit[c] {
                lit[c] = a;
            }
        }
        let mut rows = Vec::with_capacity(ah);
        for cy in 0..ah {
            let mut s = String::with_capacity(aw * 2);
            let mut bold = false;
            for cx in 0..aw {
                let c = cy * aw + cx;
                if !used[c] {
                    if bold {
                        s.push_str("\x1b[0m");
                        bold = false;
                    }
                    s.push(' ');
                    continue;
                }
                let on = lit[c] >= 0.45;
                if on != bold {
                    s.push_str(if on { "\x1b[1m" } else { "\x1b[0m" });
                    bold = on;
                }
                s.push('·');
            }
            if bold {
                s.push_str("\x1b[0m");
            }
            rows.push(s);
        }
        rows
    }

    fn write_frame(&mut self) -> std::io::Result<()> {
        let mut rows: Vec<String> = Vec::new();
        match self.view {
            View::Scope | View::Gonio => rows.extend(self.canvas.render_rows()),
            View::Grid => {
                let grew = self.draw_grid_rows();
                let canvas_rows = self.canvas.rows;
                let gh = grew.len();
                let offy = (canvas_rows.saturating_sub(gh)) / 2;
                for _ in 0..offy {
                    rows.push(String::new());
                }
                for r in grew {
                    rows.push(format!(" {}", r));
                }
                for _ in (offy + gh)..canvas_rows {
                    rows.push(String::new());
                }
            }
        }
        rows.push(if self.hud_on { self.status_line() } else { String::new() });

        // help overlay: keybinds on the lower half (toggle 'h')
        if self.help_on {
            const HELP: [&str; 12] = [
                "── keys ─────────────────────────",
                " q / esc   quit          space   pause",
                " tab       neuron map    g       gonio view",
                " 1-4       source demo/mic/file/sys",
                " s         static disk   m       alt mapping",
                " z / x     disk size     r       rewire net",
                " , / .     rotation -/+  a / d   swirl -/+",
                " [ ]       select knob   - =     adjust knob",
                " h         this help     H       status line",
                "──────────────────────────────────",
                " knobs: c coupling  l leak  p persist  n noise",
                " status: z size  v rotation  w swirl",
            ];
            let start = rows.len().saturating_sub(1) / 2;
            for (i, line) in HELP.iter().enumerate() {
                let r = start + i;
                if r + 1 < rows.len() {
                    rows[r] = format!(" {}", line);
                }
            }
        }

        let mut out = stdout();
        for (i, r) in rows.iter().enumerate() {
            if self.last_rows.get(i) != Some(r) {
                queue!(out, cursor::MoveTo(0, i as u16), Print(r), terminal::Clear(terminal::ClearType::UntilNewLine))?;
            }
        }
        self.last_rows = rows;
        out.flush()
    }

    fn handle_key(&mut self, c: char) -> bool {
        match c {
            'q' | 'Q' => return false,
            '\t' => {
                self.view = match self.view {
                    View::Scope => View::Grid,
                    View::Grid | View::Gonio => View::Scope,
                };
                self.canvas.clear();
                self.last_rows.clear();
            }
            'g' | 'G' => {
                self.view = if self.view == View::Gonio { View::Scope } else { View::Gonio };
                self.canvas.clear();
                self.last_rows.clear();
            }
            '1' => self.audio.set_mode(Mode::Demo, None),
            '2' => self.audio.set_mode(Mode::Mic, None),
            '3' => self.audio.set_mode(Mode::File, self.filepath.as_deref()),
            '4' => self.audio.set_mode(Mode::Sys, None),
            'm' | 'M' => {
                self.mapping = 1 - self.mapping;
                self.fig.set_mapping(self.mapping);
            }
            'r' | 'R' => self.sim.rewire(),
            's' | 'S' => {
                self.static_disk = !self.static_disk;
                self.fig.set_static(self.static_disk);
            }
            'z' | 'Z' => {
                self.zoom = (self.zoom * 1.2).min(3.0);
                self.fig.set_size_mul(self.zoom);
            }
            'x' | 'X' => {
                self.zoom = (self.zoom / 1.2).max(0.35);
                self.fig.set_size_mul(self.zoom);
            }
            ',' => {
                self.spin = (self.spin / 1.25).max(0.0);
                self.fig.set_spin_mul(self.spin);
            }
            '.' => {
                self.spin = (self.spin * 1.25).min(3.0);
                self.fig.set_spin_mul(self.spin);
            }
            'a' | 'A' => {
                self.swirl = (self.swirl / 1.25).max(0.0);
                self.fig.set_swirl_mul(self.swirl);
            }
            'd' | 'D' => {
                self.swirl = (self.swirl * 1.25).min(3.0);
                self.fig.set_swirl_mul(self.swirl);
            }
            ' ' => self.paused = !self.paused,
            'h' => {
                self.help_on = !self.help_on;
                self.last_rows.clear();
            }
            'H' => {
                self.hud_on = !self.hud_on;
                self.last_rows.clear();
            }
            '[' => self.knobs.sel = (self.knobs.sel + 3) % 4,
            ']' => self.knobs.sel = (self.knobs.sel + 1) % 4,
            '-' | '=' => {
                let (_, lo, hi, step) = KNOB_DEFS[self.knobs.sel];
                let d = step * if c == '=' { 1.0 } else { -1.0 };
                let v = match self.knobs.sel {
                    0 => &mut self.knobs.coupling,
                    1 => &mut self.knobs.leak,
                    2 => &mut self.knobs.persist,
                    _ => &mut self.knobs.noise,
                };
                *v = (*v + d).clamp(lo, hi);
            }
            _ => {}
        }
        true
    }

    fn run(&mut self) -> std::io::Result<()> {
        let mut out = stdout();
        // initial mode: SYS, fall back to DEMO
        self.audio.set_mode(Mode::Sys, None);
        if self.audio.mode == Mode::Silent {
            self.audio.set_mode(Mode::Demo, None);
        }
        if let Some(p) = &self.filepath.clone() {
            self.audio.set_mode(Mode::File, Some(p));
        }

        let (mut w, mut h) = terminal::size()?;
        self.layout(w, h);

        let frame = Duration::from_micros(1_000_000 / 60);
        let mut acc = 0.0f32;
        let mut prev = Instant::now();
        loop {
            let now = Instant::now();
            let dt = (now - prev).as_secs_f32().min(0.1);
            prev = now;

            // resize
            let (nw, nh) = terminal::size()?;
            if (nw, nh) != (w, h) {
                w = nw;
                h = nh;
                self.layout(w, h);
            }
            if w < 60 || h < 15 {
                queue!(out, cursor::MoveTo(0, 0), terminal::Clear(terminal::ClearType::All),
                    Print(format!(" enlarge your terminal ({}x{}, need 60x15)", w, h)))?;
                out.flush()?;
                std::thread::sleep(Duration::from_millis(200));
                continue;
            }

            // input (drain all pending keys). NB: poll(ZERO) never reads the
            // OS fd on crossterm 0.28, so keys would never arrive; use 1ms.
            while event::poll(Duration::from_millis(1))? {
                if let Event::Key(k) = event::read()? {
                    if k.kind == KeyEventKind::Press {
                        // Ctrl+C quits like 'q'
                        if k.code == KeyCode::Char('c')
                            && k.modifiers.contains(event::KeyModifiers::CONTROL)
                        {
                            return Ok(());
                        }
                        let c = match k.code {
                            KeyCode::Char(c) => c,
                            KeyCode::Tab => '\t',
                            KeyCode::Esc => 'q',
                            _ => continue,
                        };
                        if !self.handle_key(c) {
                            return Ok(());
                        }
                    }
                }
            }

            if !self.paused {
                self.sim_time += dt;
                self.audio.update();
                // true silence: kill the field immediately; the figure sees
                // live==0 and collapses the disk into a point on its own
                if self.audio.act_env < 0.1 {
                    self.sim.kill();
                }
                // NBS_DEBUG=1: dump band levels once a second for diagnosis
                if std::env::var_os("NBS_DEBUG").is_some() && (self.sim_time * 2.0) as u64 != ((self.sim_time - dt) * 2.0) as u64 {
                    use std::io::Write as _;
                    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/nbs_debug.log") {
                        let b: Vec<String> = self.audio.bands.iter().map(|v| format!("{:.2}", v)).collect();
                        let _ = writeln!(f, "t={:.1} bands [{}]", self.sim_time, b.join(" "));
                    }
                }
                acc += dt;
                while acc >= SIM_DT {
                    self.sim.tick(&self.audio.bands, &self.audio.bonset, self.audio.onset, self.knobs.leak, self.knobs.noise * self.audio.act_env, self.knobs.coupling);
                    acc -= SIM_DT;
                }
                match self.view {
                    View::Scope => self.draw_beam(dt),
                    View::Gonio => self.draw_gonio(dt),
                    View::Grid => {}
                }
            }
            self.write_frame()?;
            std::thread::sleep(frame.saturating_sub(now.elapsed()));
        }
    }
}

// ---------------------------------------------------------------- selftest
/// Mandatory-model invariant: COUPLING=0 for 2s => beam speed -> ~0.
/// And COUPLING=1 with synthetic bands => beam moves.
fn selftest() {
    let mut sim = Sim::new(42);
    let mut fig = Figure::new();
    let mut bands = [0.0f32; NB]; for (b, v) in bands.iter_mut().enumerate() { *v = 0.85 * (0.82f32).powi(b as i32); }
    let mut rng = Rng(7);

    // phase 1: coupling 1.0, expect motion
    for _ in 0..200 {
        sim.tick(&bands, &[0.0; NB], 0.0, 0.5, 0.0, 1.0);
    }
    let mut speed_hot = 0.0f32;
    let mut last = fig.beam_at(0.0);
    for i in 0..120 {
        fig.update(&sim, 0.0, 0.0, 0.5, 0.8, 1.0 / 60.0, &mut | | rng.next());
        let p = fig.beam_at(0.0);
        speed_hot += ((p.0 - last.0).powi(2) + (p.1 - last.1).powi(2)).sqrt();
        last = p;
        sim.tick(&bands, &[0.0; NB], 0.0, 0.5, 0.0, 1.0);
        let _ = i;
    }
    println!("hot beam speed sum:  {:.3} (want > 0.5)", speed_hot);

    // phase 2: coupling 0 for 4s, then measure the final window: must be frozen
    for _ in 0..400 {
        sim.tick(&bands, &[0.0; NB], 0.0, 0.5, 0.0, 0.0);
    }
    let mut speed_cold = 0.0f32;
    let mut last = fig.beam_at(0.0);
    for i in 0..120 {
        fig.update(&sim, 0.0, 0.0, 0.5, 0.8, 1.0 / 60.0, &mut | | rng.next());
        let p = fig.beam_at(0.0);
        if i >= 90 {
            speed_cold += ((p.0 - last.0).powi(2) + (p.1 - last.1).powi(2)).sqrt();
        }
        last = p;
        sim.tick(&bands, &[0.0; NB], 0.0, 0.5, 0.0, 0.0);
    }
    println!("cold beam speed sum: {:.3} (want < 0.01)", speed_cold);

    let ok = speed_hot > 0.5 && speed_cold < 0.01;
    println!("MANDATORY-MODEL TEST: {}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}

// ---------------------------------------------------------------- diag
/// Band selectivity probe: drive the sim with ONE band at a time, print the
/// resulting per-band activity. If off-diagonal bands light up nearly as much
/// as the driven one, recurrence is smearing activity network-wide (which
/// would explain "all effects present at all times").
fn diag() {
    for drive in 0..NB {
        let mut sim = Sim::new(42);
        let mut bands = [0.0f32; NB];
        bands[drive] = 0.7;
        for _ in 0..300 {
            sim.tick(&bands, &[0.0; NB], 0.0, 0.5, 0.0, 1.0);
        }
        let b: Vec<String> = sim.band_act.iter().map(|v| format!("{:.2}", v)).collect();
        let g: Vec<String> = sim.col_groups.iter().map(|v| format!("{:.2}", v)).collect();
        println!(
            "drive band {} -> rate {:.3} | bands [{}] | groups [{}]",
            drive,
            sim.rate,
            b.join(" "),
            g.join(" ")
        );
    }
    std::process::exit(0);
}

/// Decay probe: drive hard for 5 s, then cut ALL input (coupling stays 1)
/// and print how the network rate decays. If the field is self-sustaining,
/// the rate never falls, which would explain the disk lingering after the
/// music stops.
fn diag_decay() {
    let mut sim = Sim::new(42);
    let mut bands = [0.0f32; NB]; for (b, v) in bands.iter_mut().enumerate() { *v = 0.85 * (0.82f32).powi(b as i32); }
    for _ in 0..500 {
        sim.tick(&bands, &[0.0; NB], 0.0, 0.5, 0.0, 1.0);
    }
    println!("driven rate: {:.3}", sim.rate);
    let silent = [0.0f32; NB];
    for t in 0..3000 {
        sim.tick(&silent, &[0.0; NB], 0.0, 0.5, 0.0, 1.0);
        if t % 100 == 0 {
            println!("t={:>4.1}s rate {:.4}", t as f32 / 100.0, sim.rate);
        }
    }
    std::process::exit(0);
}

// ---------------------------------------------------------------- main
fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--selftest") {
        selftest();
        return Ok(());
    }
    if args.iter().any(|a| a == "--diag") {
        diag();
        return Ok(());
    }
    if args.iter().any(|a| a == "--diag-decay") {
        diag_decay();
        return Ok(());
    }
    if let Some(i) = args.iter().position(|a| a == "--profile") {
        let Some(p) = args.get(i + 1) else {
            eprintln!("usage: --profile <audio-file>");
            std::process::exit(2);
        };
        if let Err(e) = audio::AudioEngine::profile(p) {
            eprintln!("profile failed: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }
    let filepath = args.get(1).cloned();

    let mut out = stdout();
    execute!(out, terminal::EnterAlternateScreen, cursor::Hide)?;
    terminal::enable_raw_mode()?;
    let mut app = App::new(filepath);
    let res = app.run();
    app.audio.close();
    execute!(out, cursor::Show, terminal::LeaveAlternateScreen)?;
    terminal::disable_raw_mode()?;
    res
}
