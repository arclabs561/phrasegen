use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::data::Row;

/// A simple digraph timing model: expected time for each adjacent grapheme pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigraphModel {
    /// Mean dt (ms) per digraph key "a|b".
    mean_ms: HashMap<String, f32>,
    /// Global fallback mean for unseen digraphs.
    global_mean_ms: f32,
    /// Observation count per digraph (best-effort; used for adaptation weighting).
    #[serde(default)]
    count: HashMap<String, u32>,
    /// Total digraph observations used to compute `global_mean_ms`.
    #[serde(default)]
    global_count: u64,
}

impl DigraphModel {
    pub fn mean_map(&self) -> &HashMap<String, f32> {
        &self.mean_ms
    }

    pub fn count_map(&self) -> &HashMap<String, u32> {
        &self.count
    }

    pub fn from_parts(mean_ms: HashMap<String, f32>, global_mean_ms: f32) -> Self {
        Self {
            mean_ms,
            global_mean_ms,
            count: HashMap::new(),
            global_count: 0,
        }
    }

    pub fn from_parts_with_counts(
        mean_ms: HashMap<String, f32>,
        global_mean_ms: f32,
        count: HashMap<String, u32>,
        global_count: u64,
    ) -> Self {
        Self {
            mean_ms,
            global_mean_ms,
            count,
            global_count,
        }
    }

    pub fn global_mean_ms(&self) -> f32 {
        self.global_mean_ms
    }

    pub fn global_count(&self) -> u64 {
        self.global_count
    }

    pub fn count_for_key(&self, key: &str) -> u32 {
        self.count.get(key).copied().unwrap_or(0)
    }

    pub fn mean_ms_for(&self, a: &str, b: &str) -> f32 {
        let k = digraph_key(a, b);
        self.mean_ms.get(&k).copied().unwrap_or(self.global_mean_ms)
    }

    pub fn has_digraph(&self, a: &str, b: &str) -> bool {
        let k = digraph_key(a, b);
        self.mean_ms.contains_key(&k)
    }

    pub fn save_json(&self, path: &Path) -> anyhow::Result<()> {
        let bytes = serde_json::to_vec_pretty(self)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load_json(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        Ok(serde_json::from_slice(&bytes)?)
    }
}

#[derive(Debug, Clone, Default)]
pub struct FitConfig {
    /// Minimum number of observations required to keep a digraph-specific mean.
    pub min_count: usize,
    /// If set, clamp each observed dt(ms) to at most this value (winsorize).
    pub clamp_dt_ms: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct FitStats {
    pub rows: usize,
    pub total_digraph_obs: usize,
    pub distinct_digraphs: usize,
    pub kept_digraphs: usize,
    pub global_mean_ms: f32,
}

pub fn fit_digraph_model(rows: &[Row], cfg: FitConfig) -> (DigraphModel, FitStats) {
    let mut sum: HashMap<String, f32> = HashMap::new();
    let mut cnt: HashMap<String, usize> = HashMap::new();
    let mut global_sum = 0.0f32;
    let mut global_cnt = 0usize;

    for row in rows {
        let grams = crate::score::graphemes_normalized(&row.phrase);
        let n = grams.len();
        if n < 2 {
            continue;
        }
        if row.digraph_dt_ms.len() != n.saturating_sub(1) {
            // Best-effort: skip inconsistent row rather than failing the whole fit.
            continue;
        }
        // Reject rows with invalid timings (negative or non-finite).
        if !row.digraph_dt_ms.iter().all(|x| x.is_finite() && *x >= 0.0) {
            continue;
        }
        for i in 0..(n - 1) {
            let mut dt = row.digraph_dt_ms[i];
            if let Some(c) = cfg.clamp_dt_ms {
                if dt.is_finite() && c.is_finite() {
                    dt = dt.min(c);
                }
            }
            let k = digraph_key(&grams[i], &grams[i + 1]);
            *sum.entry(k.clone()).or_insert(0.0) += dt;
            *cnt.entry(k).or_insert(0) += 1;
            global_sum += dt;
            global_cnt += 1;
        }
    }

    let global_mean_ms = if global_cnt == 0 {
        0.0
    } else {
        global_sum / (global_cnt as f32)
    };

    let distinct_digraphs = cnt.len();
    let mut mean_ms = HashMap::new();
    let mut count_u32 = HashMap::new();
    let mut kept = 0usize;
    for (k, c) in cnt {
        if c >= cfg.min_count {
            let s = sum.get(&k).copied().unwrap_or(0.0);
            mean_ms.insert(k.clone(), s / (c as f32));
            count_u32.insert(k, c.min(u32::MAX as usize) as u32);
            kept += 1;
        }
    }

    (
        DigraphModel::from_parts_with_counts(mean_ms, global_mean_ms, count_u32, global_cnt as u64),
        FitStats {
            rows: rows.len(),
            total_digraph_obs: global_cnt,
            distinct_digraphs,
            kept_digraphs: kept,
            global_mean_ms,
        },
    )
}

pub(crate) fn digraph_key(a: &str, b: &str) -> String {
    // Use an explicit separator to avoid ambiguity.
    let mut s = String::with_capacity(a.len() + b.len() + 1);
    s.push_str(a);
    s.push('|');
    s.push_str(b);
    s
}
