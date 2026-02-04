use std::collections::{HashMap, HashSet};

use crate::data::Row;
use crate::model::{digraph_key, DigraphModel};
use crate::score::graphemes_normalized;

#[derive(Debug, Clone)]
pub struct AdaptConfig {
    /// Pseudo-count strength for the prior (base model) per digraph.
    pub prior_count: f32,
    /// Minimum count for a *new* digraph (present in user data but absent in base) to be added.
    pub min_new_count: usize,
}

impl Default for AdaptConfig {
    fn default() -> Self {
        Self {
            prior_count: 50.0,
            min_new_count: 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptStats {
    pub user_rows: usize,
    pub user_digraph_obs: usize,
    pub base_digraphs: usize,
    pub user_distinct_digraphs: usize,
    pub added_new_digraphs: usize,
    pub tuned_digraphs: usize,
    pub prior_count: f32,
}

/// Adapt a base digraph model to a user by Bayesian-style mean updating.
///
/// The base model provides a prior mean per digraph and a global fallback mean.
/// User rows provide observations of digraph dt(ms). We compute posterior means:
///
/// \[
/// \mu' = \frac{c_0 \mu_0 + \sum dt}{c_0 + n}
/// \]
pub fn adapt_digraph_model(
    base: &DigraphModel,
    user_rows: &[Row],
    cfg: AdaptConfig,
) -> (DigraphModel, AdaptStats) {
    let mut sum: HashMap<String, f32> = HashMap::new();
    let mut cnt: HashMap<String, usize> = HashMap::new();
    let mut global_sum = 0.0f32;
    let mut global_cnt = 0usize;

    for row in user_rows {
        let grams = graphemes_normalized(&row.phrase);
        let n = grams.len();
        if n < 2 {
            continue;
        }
        if row.digraph_dt_ms.len() != n.saturating_sub(1) {
            continue;
        }
        for i in 0..(n - 1) {
            let dt = row.digraph_dt_ms[i];
            let k = digraph_key(&grams[i], &grams[i + 1]);
            *sum.entry(k.clone()).or_insert(0.0) += dt;
            *cnt.entry(k).or_insert(0) += 1;
            global_sum += dt;
            global_cnt += 1;
        }
    }

    let base_global = base.global_mean_ms();
    let prior_c = cfg.prior_count.max(0.0);

    let tuned_global = if global_cnt == 0 {
        base_global
    } else {
        // Treat base as a *prior* with strength `prior_c` by default.
        // If `prior_c==0`, fall back to using actual base counts.
        let n0 = if prior_c > 0.0 {
            prior_c
        } else {
            base.global_count() as f32
        };
        (n0 * base_global + global_sum) / (n0 + (global_cnt as f32))
    };

    let base_keys: HashSet<String> = base.mean_map().keys().cloned().collect();
    let user_keys: HashSet<String> = cnt.keys().cloned().collect();

    let mut out_means: HashMap<String, f32> = HashMap::new();
    let mut out_counts: HashMap<String, u32> = HashMap::new();

    // Always keep base keys.
    for (k, &mu0) in base.mean_map().iter() {
        let n = cnt.get(k).copied().unwrap_or(0);
        if n == 0 {
            out_means.insert(k.clone(), mu0);
            // Preserve base counts if present.
            let c0 = base.count_for_key(k);
            if c0 > 0 {
                out_counts.insert(k.clone(), c0);
            }
            continue;
        }
        let s = sum.get(k).copied().unwrap_or(0.0);
        // Same convention as global: `prior_c` is the default prior strength.
        let n0 = if prior_c > 0.0 {
            prior_c
        } else {
            base.count_for_key(k) as f32
        };
        let mu = (n0 * mu0 + s) / (n0 + (n as f32));
        out_means.insert(k.clone(), mu);
        // Posterior count: real base count (not pseudo) + user count (clamped to u32).
        let c0 = base.count_for_key(k) as u64;
        let c1 = n as u64;
        let c = (c0 + c1).min(u32::MAX as u64) as u32;
        out_counts.insert(k.clone(), c);
    }

    // Add user-only keys if they have enough evidence.
    let mut added = 0usize;
    for k in user_keys.difference(&base_keys) {
        let n = cnt.get(k).copied().unwrap_or(0);
        if n < cfg.min_new_count {
            continue;
        }
        let s = sum.get(k).copied().unwrap_or(0.0);
        let mu0 = base_global;
        let mu = (prior_c * mu0 + s) / (prior_c + (n as f32));
        out_means.insert(k.clone(), mu);
        out_counts.insert(k.clone(), (n.min(u32::MAX as usize)) as u32);
        added += 1;
    }

    (
        DigraphModel::from_parts_with_counts(
            out_means,
            tuned_global,
            out_counts,
            base.global_count().saturating_add(global_cnt as u64),
        ),
        AdaptStats {
            user_rows: user_rows.len(),
            user_digraph_obs: global_cnt,
            base_digraphs: base_keys.len(),
            user_distinct_digraphs: user_keys.len(),
            added_new_digraphs: added,
            tuned_digraphs: base_keys.len() + added,
            prior_count: prior_c,
        },
    )
}
