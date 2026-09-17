//! Braille phosphor canvas: 2x4 dot blocks per cell, float buffer with
//! persistence decay. Terminal default colors only; bright blocks render BOLD.

/// [row][col] bit values for braille U+2800
const DOT: [[u8; 2]; 4] = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]];

pub struct Canvas {
    pub cols: usize,
    pub rows: usize,
    pub px_w: usize,
    pub px_h: usize,
    pub buf: Vec<f32>,
    colbuf: Vec<u8>,   // per-dot ANSI color idx (1..=7, 0 = default fg)
    code: Vec<u8>,     // per-cell braille bitmask (reused)
    bright: Vec<bool>, // per-cell bright flag (reused)
    ccol: Vec<u8>,     // per-cell color (reused)
}

impl Canvas {
    pub fn new(cols: usize, rows: usize) -> Self {
        Self {
            cols,
            rows,
            px_w: cols * 2,
            px_h: rows * 4,
            buf: vec![0.0; cols * rows * 8],
            colbuf: vec![0; cols * rows * 8],
            code: vec![0; cols * rows],
            bright: vec![false; cols * rows],
            ccol: vec![0; cols * rows],
        }
    }

    pub fn decay(&mut self, persistence: f32) {
        let keep = 1.0 - persistence;
        for v in self.buf.iter_mut() {
            *v *= keep;
        }
    }

    pub fn clear(&mut self) {
        self.buf.iter_mut().for_each(|v| *v = 0.0);
        self.colbuf.iter_mut().for_each(|v| *v = 0);
    }

    #[inline]
    pub fn dot(&mut self, x: f32, y: f32, intensity: f32, color: u8) {
        let (xi, yi) = (x.round() as i32, y.round() as i32);
        if xi >= 0 && xi < self.px_w as i32 && yi >= 0 && yi < self.px_h as i32 {
            let i = yi as usize * self.px_w + xi as usize;
            self.buf[i] += intensity;
            self.colbuf[i] = color;
        }
    }

    pub fn line(&mut self, mut x0: i32, mut y0: i32, x1: i32, y1: i32, intensity: f32, color: u8) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let (w, h) = (self.px_w as i32, self.px_h as i32);
        loop {
            if x0 >= 0 && x0 < w && y0 >= 0 && y0 < h {
                let i = y0 as usize * self.px_w + x0 as usize;
                self.buf[i] += intensity;
                self.colbuf[i] = color;
            }
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    /// Fold 2x4 blocks into braille row strings. Bright blocks (max >= 0.6)
    /// get BOLD; colored blocks use the terminal's own ANSI palette (30-37),
    /// so everything adapts to the user's theme. Default fg otherwise.
    pub fn render_rows(&mut self) -> Vec<String> {
        // compute codes + brightness + color per cell
        for cy in 0..self.rows {
            for cx in 0..self.cols {
                let cell = cy * self.cols + cx;
                let mut code = 0u8;
                let mut max = 0.0f32;
                let mut col = 0u8;
                let mut any = false;
                for r in 0..4 {
                    let py = cy * 4 + r;
                    let row = py * self.px_w;
                    for c in 0..2 {
                        let i = row + cx * 2 + c;
                        let v = self.buf[i];
                        if v > 0.18 {
                            code |= DOT[r][c];
                            any = true;
                            if self.colbuf[i] != 0 {
                                col = self.colbuf[i];
                            }
                        }
                        if v > max {
                            max = v;
                        }
                    }
                }
                self.code[cell] = code;
                self.bright[cell] = any && max >= 0.6;
                self.ccol[cell] = col;
            }
        }
        let mut rows = Vec::with_capacity(self.rows);
        for cy in 0..self.rows {
            let mut s = String::with_capacity(self.cols * 4);
            let mut bold = false;
            let mut col = 0u8;
            for cx in 0..self.cols {
                let cell = cy * self.cols + cx;
                let b = self.bright[cell];
                let cl = self.ccol[cell];
                if b != bold || cl != col {
                    let esc = match (b, cl) {
                        (true, c) if c > 0 => format!("\x1b[1;{}m", 30 + c),
                        (false, c) if c > 0 => format!("\x1b[{}m", 30 + c),
                        (true, _) => "\x1b[0;1m".to_string(),
                        (false, _) => "\x1b[0m".to_string(),
                    };
                    s.push_str(&esc);
                    bold = b;
                    col = cl;
                }
                s.push(char::from_u32(0x2800 + self.code[cell] as u32).unwrap());
            }
            if bold || col > 0 {
                s.push_str("\x1b[0m");
            }
            rows.push(s);
        }
        rows
    }
}
