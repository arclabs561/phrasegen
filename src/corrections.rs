//! Correction-aware utilities (backspace + total elapsed time).
//!
//! This module is intentionally conservative: it helps *analyze* correction behavior without
//! polluting the “clean” digraph timing model (which is fit from uninterrupted rows by default).

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearFit3 {
    /// Intercept term.
    pub a: f64,
    /// Coefficient for `predicted_ms` (from a timing model).
    pub b: f64,
    /// Coefficient for `backspaces` (count).
    pub c: f64,
    /// Number of samples used.
    pub n: usize,
    /// Root-mean-square error in ms.
    pub rmse_ms: f64,
}

fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    // Gaussian elimination with partial pivoting.
    for i in 0..3 {
        // Pivot.
        let mut p = i;
        let mut best = a[i][i].abs();
        for (r, row) in a.iter().enumerate().skip(i + 1) {
            let v = row[i].abs();
            if v > best {
                best = v;
                p = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if p != i {
            a.swap(i, p);
            b.swap(i, p);
        }
        // Eliminate below.
        for r in (i + 1)..3 {
            let f = a[r][i] / a[i][i];
            if f == 0.0 {
                continue;
            }
            let row_i = a[i];
            for c in i..3 {
                a[r][c] -= f * row_i[c];
            }
            b[r] -= f * b[i];
        }
    }
    // Back-substitute.
    let mut x = [0.0f64; 3];
    for i in (0..3).rev() {
        let mut s = b[i];
        for j in (i + 1)..3 {
            s -= a[i][j] * x[j];
        }
        let d = a[i][i];
        if d.abs() < 1e-12 {
            return None;
        }
        x[i] = s / d;
    }
    Some(x)
}

/// Fit a simple linear model:
///
/// `total_ms ≈ a + b * predicted_ms + c * backspaces`
///
/// Inputs are tuples `(predicted_ms, backspaces, total_ms)`.
pub fn fit_total_ms_linear(samples: &[(f64, f64, f64)]) -> Option<LinearFit3> {
    if samples.len() < 3 {
        return None;
    }

    let mut n = 0.0f64;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    let mut s11 = 0.0f64;
    let mut s22 = 0.0f64;
    let mut s12 = 0.0f64;
    let mut sy = 0.0f64;
    let mut s1y = 0.0f64;
    let mut s2y = 0.0f64;

    for &(x1, x2, y) in samples {
        if !(x1.is_finite() && x2.is_finite() && y.is_finite()) {
            continue;
        }
        n += 1.0;
        s1 += x1;
        s2 += x2;
        s11 += x1 * x1;
        s22 += x2 * x2;
        s12 += x1 * x2;
        sy += y;
        s1y += x1 * y;
        s2y += x2 * y;
    }

    if n < 3.0 {
        return None;
    }

    let a = [[n, s1, s2], [s1, s11, s12], [s2, s12, s22]];
    let b = [sy, s1y, s2y];
    let x = solve_3x3(a, b)?;
    let (aa, bb, cc) = (x[0], x[1], x[2]);

    // RMSE.
    let mut se = 0.0f64;
    let mut used = 0usize;
    for &(x1, x2, y) in samples {
        if !(x1.is_finite() && x2.is_finite() && y.is_finite()) {
            continue;
        }
        let yhat = aa + bb * x1 + cc * x2;
        let e = yhat - y;
        se += e * e;
        used += 1;
    }
    if used == 0 {
        return None;
    }

    Some(LinearFit3 {
        a: aa,
        b: bb,
        c: cc,
        n: used,
        rmse_ms: (se / (used as f64)).sqrt(),
    })
}
