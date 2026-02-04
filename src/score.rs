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
            digraphs: 0,
            hits: 0,
            misses: 0,
        };
    }
    let mut total = 0.0f32;
    let mut hits = 0usize;
    let mut misses = 0usize;
    for i in 0..(grams.len() - 1) {
        total += model.mean_ms_for(&grams[i], &grams[i + 1]);
        if model.has_digraph(&grams[i], &grams[i + 1]) {
            hits += 1;
        } else {
            misses += 1;
        }
    }
    Score {
        predicted_ms: total,
        digraphs: grams.len() - 1,
        hits,
        misses,
    }
}
