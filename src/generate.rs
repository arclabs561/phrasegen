use std::path::Path;

use rand::prelude::IndexedRandom as _;
use rand::Rng as _;
use rand::SeedableRng as _;

use crate::score::{score_phrase, Score};
use crate::textprep;
use crate::timing::TimingModel;

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub words: usize,
    pub separator: String,
    pub samples: usize,
    pub top_k: usize,
    pub min_word_len: usize,
    pub max_word_len: usize,
    pub ascii_lower_only: bool,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            words: 4,
            separator: " ".to_string(),
            samples: 10_000,
            top_k: 20,
            min_word_len: 3,
            max_word_len: 12,
            ascii_lower_only: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub phrase: String,
    pub score: Score,
}

#[derive(Debug, Clone)]
pub struct GenerateStats {
    pub total_words_in_file: usize,
    pub usable_words: usize,
    pub effective_entropy_bits: f32,
}

pub struct Wordlist {
    pub total_words_in_file: usize,
    pub words: Vec<String>,
}

pub fn wordlist_from_vec(words: Vec<String>) -> anyhow::Result<Wordlist> {
    if words.is_empty() {
        anyhow::bail!("wordlist is empty");
    }
    Ok(Wordlist {
        total_words_in_file: words.len(),
        words,
    })
}

pub fn load_wordlist(path: &Path, cfg: &GenerateConfig) -> anyhow::Result<Wordlist> {
    let text = std::fs::read_to_string(path)?;
    let mut words = Vec::new();
    let mut total = 0usize;
    for line in text.lines() {
        let w = line.trim();
        if w.is_empty() || w.starts_with('#') {
            continue;
        }
        total += 1;

        // For now: build a “typing-friendly” candidate word by scrubbing (lower + diacritics).
        let w = textprep::scrub(w);

        if w.len() < cfg.min_word_len || w.len() > cfg.max_word_len {
            continue;
        }
        if cfg.ascii_lower_only && !is_ascii_lower_word(&w) {
            continue;
        }
        words.push(w);
    }
    if words.is_empty() {
        anyhow::bail!(
            "no usable words after filtering (total_words_in_file={total}); try relaxing filters"
        );
    }
    Ok(Wordlist {
        total_words_in_file: total,
        words,
    })
}

pub fn generate_top(
    model: &impl TimingModel,
    wordlist: &Wordlist,
    cfg: &GenerateConfig,
    rng: &mut impl rand::RngCore,
) -> (Vec<Candidate>, GenerateStats) {
    let words = wordlist.words.as_slice();
    let effective_entropy_bits = (cfg.words as f32) * (words.len() as f32).log2();
    let stats = GenerateStats {
        total_words_in_file: wordlist.total_words_in_file,
        usable_words: words.len(),
        effective_entropy_bits,
    };

    let mut best: Vec<Candidate> = Vec::new();

    for _ in 0..cfg.samples {
        let mut parts = Vec::with_capacity(cfg.words);
        for _ in 0..cfg.words {
            let Some(w) = words.choose(rng) else {
                // Defensive: the caller promised non-empty.
                break;
            };
            parts.push(w.as_str());
        }
        if parts.len() != cfg.words {
            break;
        }
        let phrase = parts.join(&cfg.separator);
        let score = score_phrase(model, &phrase);

        // Insert into a small top-k list (k is small: O(k) maintenance is fine).
        if best.len() < cfg.top_k {
            best.push(Candidate { phrase, score });
            best.sort_by(|a, b| a.score.predicted_ms.total_cmp(&b.score.predicted_ms));
        } else if let Some(worst) = best.last() {
            if score.predicted_ms < worst.score.predicted_ms {
                best.pop();
                best.push(Candidate { phrase, score });
                best.sort_by(|a, b| a.score.predicted_ms.total_cmp(&b.score.predicted_ms));
            }
        }
    }

    (best, stats)
}

pub fn estimate_avg_phrase_ms(
    model: &impl TimingModel,
    wordlist: &Wordlist,
    cfg: &GenerateConfig,
    samples: usize,
    rng: &mut impl rand::RngCore,
) -> anyhow::Result<(f64, f64)> {
    let words = wordlist.words.as_slice();
    if words.is_empty() {
        anyhow::bail!("wordlist is empty");
    }
    if cfg.words == 0 {
        anyhow::bail!("words must be >= 1");
    }
    let n = samples.max(1);

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;

    for _ in 0..n {
        let mut parts = Vec::with_capacity(cfg.words);
        for _ in 0..cfg.words {
            let Some(w) = words.choose(rng) else {
                break;
            };
            parts.push(w.as_str());
        }
        if parts.len() != cfg.words {
            anyhow::bail!("unexpected empty wordlist during sampling");
        }
        let phrase = parts.join(&cfg.separator);
        let ms = score_phrase(model, &phrase).predicted_ms as f64;
        sum += ms;
        sum_sq += ms * ms;
    }

    let mean = sum / (n as f64);
    let var = (sum_sq / (n as f64)) - mean * mean;
    let std = var.max(0.0).sqrt();
    Ok((mean, std))
}

fn is_ascii_lower_word(s: &str) -> bool {
    // Allow a-z only (no spaces/hyphens) for wordlist elements.
    s.bytes().all(|b| matches!(b, b'a'..=b'z'))
}

/// Convenience for deterministic generation from a seed.
pub fn rng_from_seed(seed: Option<u64>) -> impl rand::RngCore {
    match seed {
        // Use a conservative CSPRNG variant (ChaCha20) for passphrase generation.
        //
        // `--seed` is for reproducible demos/experiments; for real passwords, omit it.
        Some(seed) => rand_chacha::ChaCha20Rng::seed_from_u64(seed),
        None => {
            let mut seed_bytes = [0u8; 32];
            rand::rng().fill(&mut seed_bytes);
            rand_chacha::ChaCha20Rng::from_seed(seed_bytes)
        }
    }
}
