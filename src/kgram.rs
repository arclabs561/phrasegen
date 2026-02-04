use std::collections::{HashMap, HashSet};

/// A simple character k-gram transition model (Markov model over Unicode scalar values).
///
/// - `k=1` means an i.i.d. character model.
/// - `k=2` means bigram transitions.
/// - `k=3` means trigram transitions, etc.
#[derive(Debug, Clone)]
pub struct KGramModel {
    k: usize,
    alpha: f64,
    // context (length k-1) -> next_char -> count
    counts: HashMap<Vec<char>, HashMap<char, u64>>,
    // set of observed next-chars (for smoothing vocabulary)
    vocab: HashSet<char>,
}

impl KGramModel {
    pub fn new(k: usize, alpha: f64) -> anyhow::Result<Self> {
        if k == 0 {
            anyhow::bail!("k must be >= 1");
        }
        if !alpha.is_finite() || alpha < 0.0 {
            anyhow::bail!("alpha must be finite and >= 0");
        }
        Ok(Self {
            k,
            alpha,
            counts: HashMap::new(),
            vocab: HashSet::new(),
        })
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Train from a corpus of `(token, weight)` pairs.
    pub fn train<I>(&mut self, corpus: I)
    where
        I: IntoIterator<Item = (String, u64)>,
    {
        for (token, w) in corpus {
            if w == 0 {
                continue;
            }
            self.observe_token(&token, w);
        }
    }

    /// Surprisal in bits: \(-\log_2 p(token)\).
    pub fn surprisal_bits(&self, token: &str) -> f64 {
        if token.is_empty() {
            return 0.0;
        }
        let mut ctx = vec![START; self.k.saturating_sub(1)];
        let mut s = 0.0f64;
        for ch in token.chars().chain(std::iter::once(END)) {
            s += -self.log2_p_next(&ctx, ch);
            if self.k > 1 {
                ctx.push(ch);
                let keep = self.k - 1;
                if ctx.len() > keep {
                    let drop = ctx.len() - keep;
                    ctx.drain(0..drop);
                }
            }
        }
        s
    }

    fn observe_token(&mut self, token: &str, weight: u64) {
        let mut ctx = vec![START; self.k.saturating_sub(1)];
        for ch in token.chars().chain(std::iter::once(END)) {
            self.vocab.insert(ch);
            let entry = self.counts.entry(ctx.clone()).or_default();
            *entry.entry(ch).or_insert(0) += weight;

            if self.k > 1 {
                ctx.push(ch);
                let keep = self.k - 1;
                if ctx.len() > keep {
                    let drop = ctx.len() - keep;
                    ctx.drain(0..drop);
                }
            }
        }
    }

    fn log2_p_next(&self, ctx: &[char], next: char) -> f64 {
        // If we have no training, fall back to 0 bits (treat prob=1).
        if self.vocab.is_empty() {
            return 0.0;
        }
        let v = self.vocab.len() as f64;
        let alpha = self.alpha;

        let (num, denom) = match self.counts.get(ctx) {
            Some(m) => {
                let total: u64 = m.values().copied().sum();
                let c = m.get(&next).copied().unwrap_or(0);
                ((c as f64) + alpha, (total as f64) + alpha * v)
            }
            None => (alpha, alpha * v),
        };

        // alpha=0 with unseen context => denom=0; treat as uniform over vocab.
        let p = if denom > 0.0 { num / denom } else { 1.0 / v };
        p.log2()
    }
}

const START: char = '\u{0002}'; // STX
const END: char = '\u{0003}'; // ETX
