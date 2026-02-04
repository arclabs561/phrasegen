use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::model::DigraphModel;
use crate::{adapt, data, score};

/// Common interface for models that can predict digraph dt(ms).
pub trait TimingModel {
    fn mean_ms_for(&self, a: &str, b: &str) -> f32;
    fn global_mean_ms(&self) -> f32;
    fn has_digraph(&self, a: &str, b: &str) -> bool;
}

impl TimingModel for DigraphModel {
    fn mean_ms_for(&self, a: &str, b: &str) -> f32 {
        DigraphModel::mean_ms_for(self, a, b)
    }
    fn global_mean_ms(&self) -> f32 {
        DigraphModel::global_mean_ms(self)
    }
    fn has_digraph(&self, a: &str, b: &str) -> bool {
        DigraphModel::has_digraph(self, a, b)
    }
}

/// Lightweight QWERTY-ish hand classification for ASCII letters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Hand {
    Left,
    Right,
    Other,
}

pub fn hand_of_grapheme(g: &str) -> Hand {
    let b = g.as_bytes();
    if b.len() != 1 {
        return Hand::Other;
    }
    let c = b[0] as char;
    let c = c.to_ascii_lowercase();
    // Very rough US QWERTY split.
    let left = "qwertasdfgzxcvb";
    let right = "yuiophjklnm";
    if left.contains(c) {
        Hand::Left
    } else if right.contains(c) {
        Hand::Right
    } else {
        Hand::Other
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CharClass {
    Vowel,
    Consonant,
    Other,
}

pub fn class_of_grapheme(g: &str) -> CharClass {
    let b = g.as_bytes();
    if b.len() != 1 {
        return CharClass::Other;
    }
    let c = (b[0] as char).to_ascii_lowercase();
    if matches!(c, 'a' | 'e' | 'i' | 'o' | 'u') {
        CharClass::Vowel
    } else if matches!(c, 'a'..='z') {
        CharClass::Consonant
    } else {
        CharClass::Other
    }
}

/// A hybrid personalized model: base digraph means + user-adjustments with backoff.
///
/// Intended behavior with small user data:
/// - if user has evidence for the digraph, use user posterior mean
/// - else back off to per-key (outgoing/incoming) deltas, then group deltas, then base
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizedModel {
    pub base: DigraphModel,
    pub tuned: DigraphModel,
    /// Mean dt(ms) for user outgoing edges keyed by grapheme.
    pub user_out_mean: std::collections::HashMap<String, f32>,
    pub user_out_count: std::collections::HashMap<String, u32>,
    /// Mean dt(ms) for user incoming edges keyed by grapheme.
    pub user_in_mean: std::collections::HashMap<String, f32>,
    pub user_in_count: std::collections::HashMap<String, u32>,
    /// Mean dt(ms) by (hand(a), hand(b)) group.
    pub user_hand: Vec<HandPairStat>,
    /// Mean dt(ms) by (class(a), class(b)) group.
    pub user_class: Vec<ClassPairStat>,
    /// Baselines from the base model for deltas (optional but improves calibration).
    #[serde(default)]
    pub base_out_mean: std::collections::HashMap<String, f32>,
    #[serde(default)]
    pub base_in_mean: std::collections::HashMap<String, f32>,
    #[serde(default)]
    pub base_hand: Vec<HandPairStat>,
    #[serde(default)]
    pub base_class: Vec<ClassPairStat>,
    /// Minimum user count to trust per-key/group effects.
    pub min_backoff_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandPairStat {
    pub a: Hand,
    pub b: Hand,
    pub mean_ms: f32,
    pub count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassPairStat {
    pub a: CharClass,
    pub b: CharClass,
    pub mean_ms: f32,
    pub count: u32,
}

impl PersonalizedModel {
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

#[derive(Debug, Clone)]
pub struct BuildPersonalizedStats {
    pub user_rows_used: usize,
    pub user_digraph_obs_used: usize,
    pub user_distinct_out: usize,
    pub user_distinct_in: usize,
    pub user_hand_groups: usize,
    pub user_class_groups: usize,
}

pub fn build_personalized_model(
    base: &DigraphModel,
    user_rows: &[data::Row],
    adapt_cfg: adapt::AdaptConfig,
    min_backoff_count: u32,
) -> (PersonalizedModel, adapt::AdaptStats, BuildPersonalizedStats) {
    let (tuned, adapt_stats) = adapt::adapt_digraph_model(base, user_rows, adapt_cfg);

    let mut out_sum: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut out_cnt: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut in_sum: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut in_cnt: std::collections::HashMap<String, u32> = std::collections::HashMap::new();

    let mut hand_sum: std::collections::HashMap<(Hand, Hand), f64> =
        std::collections::HashMap::new();
    let mut hand_cnt: std::collections::HashMap<(Hand, Hand), u32> =
        std::collections::HashMap::new();
    let mut class_sum: std::collections::HashMap<(CharClass, CharClass), f64> =
        std::collections::HashMap::new();
    let mut class_cnt: std::collections::HashMap<(CharClass, CharClass), u32> =
        std::collections::HashMap::new();

    // Compute base baselines from base mean_map (weighted by count_map when available).
    let mut base_out_sum: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut base_out_cnt: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    let mut base_in_sum: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut base_in_cnt: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    let mut base_hand_sum: std::collections::HashMap<(Hand, Hand), f64> =
        std::collections::HashMap::new();
    let mut base_hand_cnt: std::collections::HashMap<(Hand, Hand), u64> =
        std::collections::HashMap::new();
    let mut base_class_sum: std::collections::HashMap<(CharClass, CharClass), f64> =
        std::collections::HashMap::new();
    let mut base_class_cnt: std::collections::HashMap<(CharClass, CharClass), u64> =
        std::collections::HashMap::new();

    for (k, &mu) in base.mean_map().iter() {
        let (a, b) = k.split_once('|').unwrap_or(("", ""));
        if a.is_empty() || b.is_empty() {
            continue;
        }
        let w = base.count_map().get(k).copied().unwrap_or(1) as u64;
        let mu = mu as f64;
        *base_out_sum.entry(a.to_string()).or_insert(0.0) += mu * (w as f64);
        *base_out_cnt.entry(a.to_string()).or_insert(0) += w;
        *base_in_sum.entry(b.to_string()).or_insert(0.0) += mu * (w as f64);
        *base_in_cnt.entry(b.to_string()).or_insert(0) += w;

        let ha = hand_of_grapheme(a);
        let hb = hand_of_grapheme(b);
        *base_hand_sum.entry((ha, hb)).or_insert(0.0) += mu * (w as f64);
        *base_hand_cnt.entry((ha, hb)).or_insert(0) += w;

        let ca = class_of_grapheme(a);
        let cb = class_of_grapheme(b);
        *base_class_sum.entry((ca, cb)).or_insert(0.0) += mu * (w as f64);
        *base_class_cnt.entry((ca, cb)).or_insert(0) += w;
    }
    let base_out_mean = base_out_sum
        .iter()
        .map(|(k, &s)| {
            let c = base_out_cnt.get(k).copied().unwrap_or(0).max(1) as f64;
            (k.clone(), (s / c) as f32)
        })
        .collect();
    let base_in_mean = base_in_sum
        .iter()
        .map(|(k, &s)| {
            let c = base_in_cnt.get(k).copied().unwrap_or(0).max(1) as f64;
            (k.clone(), (s / c) as f32)
        })
        .collect();
    let mut base_hand: Vec<HandPairStat> = Vec::new();
    for (k, &s) in base_hand_sum.iter() {
        let c = base_hand_cnt.get(k).copied().unwrap_or(0).max(1) as f64;
        base_hand.push(HandPairStat {
            a: k.0,
            b: k.1,
            mean_ms: (s / c) as f32,
            count: (base_hand_cnt
                .get(k)
                .copied()
                .unwrap_or(0)
                .min(u32::MAX as u64)) as u32,
        });
    }
    let mut base_class: Vec<ClassPairStat> = Vec::new();
    for (k, &s) in base_class_sum.iter() {
        let c = base_class_cnt.get(k).copied().unwrap_or(0).max(1) as f64;
        base_class.push(ClassPairStat {
            a: k.0,
            b: k.1,
            mean_ms: (s / c) as f32,
            count: (base_class_cnt
                .get(k)
                .copied()
                .unwrap_or(0)
                .min(u32::MAX as u64)) as u32,
        });
    }

    let mut used_rows = 0usize;
    let mut used_obs = 0usize;

    for row in user_rows {
        let grams = score::graphemes_normalized(&row.phrase);
        let n = grams.len();
        if n < 2 {
            continue;
        }
        if row.digraph_dt_ms.len() != n.saturating_sub(1) {
            continue;
        }
        if !row.digraph_dt_ms.iter().all(|x| x.is_finite() && *x >= 0.0) {
            continue;
        }
        used_rows += 1;
        for i in 0..(n - 1) {
            let dt = row.digraph_dt_ms[i] as f64;
            let a = &grams[i];
            let b = &grams[i + 1];

            *out_sum.entry(a.clone()).or_insert(0.0) += dt;
            *out_cnt.entry(a.clone()).or_insert(0) += 1;
            *in_sum.entry(b.clone()).or_insert(0.0) += dt;
            *in_cnt.entry(b.clone()).or_insert(0) += 1;

            let ha = hand_of_grapheme(a);
            let hb = hand_of_grapheme(b);
            *hand_sum.entry((ha, hb)).or_insert(0.0) += dt;
            *hand_cnt.entry((ha, hb)).or_insert(0) += 1;

            let ca = class_of_grapheme(a);
            let cb = class_of_grapheme(b);
            *class_sum.entry((ca, cb)).or_insert(0.0) += dt;
            *class_cnt.entry((ca, cb)).or_insert(0) += 1;

            used_obs += 1;
        }
    }

    let user_out_mean = out_sum
        .iter()
        .map(|(k, &s)| {
            let c = out_cnt.get(k).copied().unwrap_or(0).max(1) as f64;
            (k.clone(), (s / c) as f32)
        })
        .collect();
    let user_in_mean = in_sum
        .iter()
        .map(|(k, &s)| {
            let c = in_cnt.get(k).copied().unwrap_or(0).max(1) as f64;
            (k.clone(), (s / c) as f32)
        })
        .collect();
    let mut user_hand: Vec<HandPairStat> = Vec::new();
    for (k, &s) in hand_sum.iter() {
        let c = hand_cnt.get(k).copied().unwrap_or(0).max(1);
        user_hand.push(HandPairStat {
            a: k.0,
            b: k.1,
            mean_ms: (s / (c as f64)) as f32,
            count: c,
        });
    }
    let mut user_class: Vec<ClassPairStat> = Vec::new();
    for (k, &s) in class_sum.iter() {
        let c = class_cnt.get(k).copied().unwrap_or(0).max(1);
        user_class.push(ClassPairStat {
            a: k.0,
            b: k.1,
            mean_ms: (s / (c as f64)) as f32,
            count: c,
        });
    }

    let p = PersonalizedModel {
        base: base.clone(),
        tuned,
        user_out_mean,
        user_out_count: out_cnt.clone(),
        user_in_mean,
        user_in_count: in_cnt.clone(),
        user_hand,
        user_class,
        base_out_mean,
        base_in_mean,
        base_hand,
        base_class,
        min_backoff_count,
    };

    (
        p,
        adapt_stats,
        BuildPersonalizedStats {
            user_rows_used: used_rows,
            user_digraph_obs_used: used_obs,
            user_distinct_out: out_cnt.len(),
            user_distinct_in: in_cnt.len(),
            user_hand_groups: hand_cnt.len(),
            user_class_groups: class_cnt.len(),
        },
    )
}

impl TimingModel for PersonalizedModel {
    fn mean_ms_for(&self, a: &str, b: &str) -> f32 {
        // 1) If tuned has the digraph explicitly (user or base-kept), use it.
        if self.tuned.has_digraph(a, b) {
            return self.tuned.mean_ms_for(a, b);
        }

        // 2) Base estimate (digraph if present; else global).
        let mut mu = self.base.mean_ms_for(a, b);

        // 3) Add user per-key deltas if available.
        // We compute deltas relative to base per-key means (if known), else base global.
        let mut w = 0.0f32;
        let mut delta = 0.0f32;

        if let Some(&c) = self.user_out_count.get(a) {
            if c >= self.min_backoff_count {
                if let Some(&m) = self.user_out_mean.get(a) {
                    let base_m = self
                        .base_out_mean
                        .get(a)
                        .copied()
                        .unwrap_or(self.base.global_mean_ms());
                    delta += (m - base_m) * 0.5;
                    w += 0.5;
                }
            }
        }
        if let Some(&c) = self.user_in_count.get(b) {
            if c >= self.min_backoff_count {
                if let Some(&m) = self.user_in_mean.get(b) {
                    let base_m = self
                        .base_in_mean
                        .get(b)
                        .copied()
                        .unwrap_or(self.base.global_mean_ms());
                    delta += (m - base_m) * 0.5;
                    w += 0.5;
                }
            }
        }
        if w > 0.0 {
            mu += delta / w.max(1e-6);
        }

        // 4) Hand-group adjustment
        let ha = hand_of_grapheme(a);
        let hb = hand_of_grapheme(b);
        if let Some(s) = self.user_hand.iter().find(|s| s.a == ha && s.b == hb) {
            if s.count >= self.min_backoff_count {
                // Adjust relative to base group mean if available.
                let base_g = self
                    .base_hand
                    .iter()
                    .find(|t| t.a == ha && t.b == hb)
                    .map(|t| t.mean_ms)
                    .unwrap_or(self.base.global_mean_ms());
                let target = mu + (s.mean_ms - base_g);
                mu = 0.8 * mu + 0.2 * target;
            }
        }

        // 5) Class-group adjustment
        let ca = class_of_grapheme(a);
        let cb = class_of_grapheme(b);
        if let Some(s) = self.user_class.iter().find(|s| s.a == ca && s.b == cb) {
            if s.count >= self.min_backoff_count {
                let base_g = self
                    .base_class
                    .iter()
                    .find(|t| t.a == ca && t.b == cb)
                    .map(|t| t.mean_ms)
                    .unwrap_or(self.base.global_mean_ms());
                let target = mu + (s.mean_ms - base_g);
                mu = 0.8 * mu + 0.2 * target;
            }
        }

        mu.max(0.0)
    }

    fn global_mean_ms(&self) -> f32 {
        self.tuned.global_mean_ms()
    }

    fn has_digraph(&self, a: &str, b: &str) -> bool {
        self.tuned.has_digraph(a, b) || self.base.has_digraph(a, b)
    }
}

#[derive(Debug, Clone)]
pub enum AnyTimingModel {
    Digraph(DigraphModel),
    Personalized(PersonalizedModel),
}

impl AnyTimingModel {
    pub fn load_json(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        // Try personalized first (it will fail quickly if fields don't match).
        if let Ok(p) = serde_json::from_slice::<PersonalizedModel>(&bytes) {
            return Ok(AnyTimingModel::Personalized(p));
        }
        let d: DigraphModel = serde_json::from_slice(&bytes)?;
        Ok(AnyTimingModel::Digraph(d))
    }
}

impl TimingModel for AnyTimingModel {
    fn mean_ms_for(&self, a: &str, b: &str) -> f32 {
        match self {
            AnyTimingModel::Digraph(m) => m.mean_ms_for(a, b),
            AnyTimingModel::Personalized(m) => m.mean_ms_for(a, b),
        }
    }
    fn global_mean_ms(&self) -> f32 {
        match self {
            AnyTimingModel::Digraph(m) => m.global_mean_ms(),
            AnyTimingModel::Personalized(m) => m.global_mean_ms(),
        }
    }
    fn has_digraph(&self, a: &str, b: &str) -> bool {
        match self {
            AnyTimingModel::Digraph(m) => m.has_digraph(a, b),
            AnyTimingModel::Personalized(m) => m.has_digraph(a, b),
        }
    }
}
