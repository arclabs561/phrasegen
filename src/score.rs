use unicode_normalization::UnicodeNormalization as _;
use unicode_segmentation::UnicodeSegmentation;

use crate::timing::TimingModel;

/// Normalize a phrase and split into grapheme clusters (best match for “what you type”).
pub fn graphemes_normalized(s: &str) -> Vec<String> {
    // NFC makes composed characters consistent (e.g. é).
    let nfc: String = s.nfc().collect();
    UnicodeSegmentation::graphemes(nfc.as_str(), true)
        .map(|g| g.to_string())
        .collect()
}

#[derive(Debug, Clone)]
pub struct Score {
    /// Predicted total time in ms.
    pub predicted_ms: f32,
    /// Predicted standard deviation (ms) using a naive independence approximation:
    /// \(\mathrm{Var}(\sum dt_i) \approx \sum \mathrm{Var}(dt_i)\).
    ///
    /// This is a diagnostic, not a guarantee.
    pub predicted_std_ms: Option<f32>,
    /// Count of digraphs used to compute the score.
    pub digraphs: usize,
    /// Digraphs that used a specific (non-global) estimate.
    pub hits: usize,
    /// Digraphs that fell back to a global/backoff estimate.
    pub misses: usize,
}

pub fn score_phrase(model: &impl TimingModel, phrase: &str) -> Score {
    let grams = graphemes_normalized(phrase);
    if grams.len() < 2 {
        return Score {
            predicted_ms: 0.0,
            predicted_std_ms: None,
            digraphs: 0,
            hits: 0,
            misses: 0,
        };
    }
    let mut total = 0.0f32;
    let mut var_total = 0.0f32;
    let mut saw_any_var = false;
    let mut hits = 0usize;
    let mut misses = 0usize;
    for i in 0..(grams.len() - 1) {
        total += model.mean_ms_for(&grams[i], &grams[i + 1]);
        if let Some(v) = model.var_ms2_for(&grams[i], &grams[i + 1]) {
            if v.is_finite() && v >= 0.0 {
                // f32 has no saturating_add; just keep it finite-ish.
                var_total += v;
                saw_any_var = true;
            }
        }
        if model.has_digraph(&grams[i], &grams[i + 1]) {
            hits += 1;
        } else {
            misses += 1;
        }
    }
    Score {
        predicted_ms: total,
        predicted_std_ms: if saw_any_var {
            Some(var_total.max(0.0).sqrt())
        } else {
            None
        },
        digraphs: grams.len() - 1,
        hits,
        misses,
    }
}
