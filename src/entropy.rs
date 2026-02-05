use std::collections::HashMap;

/// Compute \(\log_2(x)\) for `f64`.
#[inline]
pub fn log2_f64(x: f64) -> f64 {
    x.ln() / 2f64.ln()
}

/// Plugin estimates of output entropies from sampled counts.
///
/// Inputs:
/// - `counts`: map from outcome to observed count
/// - `n`: total samples (should equal sum(counts))
///
/// Returns `(H1, H2, Hinf, sum_p2)` in bits, where `sum_p2 = \sum_s \hat p(s)^2`.
pub fn entropy_from_counts(counts: &HashMap<String, u32>, n: u64) -> (f64, f64, f64, f64) {
    if n == 0 || counts.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n_f = n as f64;
    let mut h1 = 0.0f64;
    let mut s_p2 = 0.0f64;
    let mut p_max = 0.0f64;
    for &c in counts.values() {
        let p = (c as f64) / n_f;
        if p > 0.0 {
            h1 -= p * log2_f64(p);
            s_p2 += p * p;
            p_max = p_max.max(p);
        }
    }
    let h2 = if s_p2 > 0.0 { -log2_f64(s_p2) } else { 0.0 };
    let h_inf = if p_max > 0.0 { -log2_f64(p_max) } else { 0.0 };
    (h1, h2, h_inf, s_p2)
}

/// Number of equal pairs among `n` draws: \(\binom{n}{2}\).
#[inline]
pub fn n_pairs(n: u64) -> u64 {
    n.saturating_mul(n.saturating_sub(1)) / 2
}

/// Observed collision pairs from outcome counts: \(\sum_s \binom{c_s}{2}\).
pub fn collision_pairs(counts: &HashMap<String, u32>) -> u64 {
    counts
        .values()
        .map(|&c| {
            let c = c as u64;
            c.saturating_mul(c.saturating_sub(1)) / 2
        })
        .sum()
}

/// If we observe 0 collisions among \(\binom{n}{2}\) pairwise comparisons, then under a crude
/// binomial approximation:
///
/// \[
/// P(0) \approx (1 - p_2)^{\binom{n}{2}}.
/// \]
///
/// Solve for an upper bound `p_2` such that `P(0) = alpha`, using
/// \(p_2 \le 1 - \alpha^{1/N} \approx -\ln(\alpha)/N\).
pub fn p2_upper_bound_zero_collisions(n: u64, alpha: f64) -> Option<f64> {
    let np = n_pairs(n);
    if np == 0 {
        return None;
    }
    if !(0.0 < alpha && alpha < 1.0) {
        return None;
    }
    Some(-alpha.ln() / (np as f64))
}

/// Extract the `k` most frequent outcomes, descending by count (tie-break by string).
pub fn top_k_counts(counts: &HashMap<String, u32>, k: usize) -> Vec<(String, u32)> {
    let mut v: Vec<(String, u32)> = counts.iter().map(|(s, &c)| (s.clone(), c)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v.truncate(k);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn map_from_counts(cs: &[u32]) -> HashMap<String, u32> {
        let mut m = HashMap::new();
        for (i, &c) in cs.iter().enumerate() {
            if c == 0 {
                continue;
            }
            m.insert(format!("k{i}"), c);
        }
        m
    }

    proptest! {
        #[test]
        fn entropy_invariants_hold(counts in prop::collection::vec(1u32..50u32, 1..20)) {
            let m = map_from_counts(&counts);
            let n: u64 = m.values().map(|&c| c as u64).sum();
            let (h1, h2, hinf, s_p2) = entropy_from_counts(&m, n);

            // Basic bounds.
            prop_assert!(h1.is_finite() && h2.is_finite() && hinf.is_finite());
            prop_assert!(h1 >= -1e-9);
            prop_assert!(h2 >= -1e-9);
            prop_assert!(hinf >= -1e-9);
            prop_assert!(s_p2 >= -1e-12);
            prop_assert!(s_p2 <= 1.0 + 1e-12);

            // Standard inequality chain: H1 >= H2 >= Hinf for any distribution.
            prop_assert!(h1 + 1e-9 >= h2);
            prop_assert!(h2 + 1e-9 >= hinf);

            // Upper bound: H1 <= log2(support_size).
            let support = m.len().max(1) as f64;
            prop_assert!(h1 <= log2_f64(support) + 1e-6);
        }

        #[test]
        fn collision_pairs_bounds(counts in prop::collection::vec(0u32..50u32, 0..50)) {
            let m = map_from_counts(&counts);
            let n: u64 = m.values().map(|&c| c as u64).sum();
            let cp = collision_pairs(&m);
            prop_assert!(cp <= n_pairs(n));
        }

        #[test]
        fn p2_upper_bound_decreases_with_more_samples(
            n1 in 2u64..500u64,
            delta in 1u64..500u64,
        ) {
            let n2 = n1.saturating_add(delta);
            let a = 0.05;
            let p1 = p2_upper_bound_zero_collisions(n1, a).unwrap();
            let p2 = p2_upper_bound_zero_collisions(n2, a).unwrap();
            prop_assert!(p2 <= p1 + 1e-18);
        }
    }
}
