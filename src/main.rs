use anyhow::Context as _;
use clap::{Parser, Subcommand, ValueEnum};
use std::io::Write as _;

use fastphrase::adapt::{adapt_digraph_model, AdaptConfig};
use fastphrase::generate::{
    estimate_avg_phrase_ms, generate_top, load_wordlist, rng_from_seed, wordlist_from_vec,
    GenerateConfig,
};
use fastphrase::import::{bksd, cmu_dsl, cmu_laser2012, greyc_web, keyrecs};
use fastphrase::kgram::KGramModel;
use fastphrase::model::{fit_digraph_model, FitConfig};
use fastphrase::record::{append_row_jsonl, record_once, RecordConfig, RecordOutcome};
use fastphrase::timing::AnyTimingModel;
use fastphrase::timing::TimingModel as _;
use rand::prelude::IndexedRandom as _;
use rand::seq::SliceRandom as _;
use rand08::SeedableRng as _;

#[derive(Debug, Parser)]
#[command(name = "fastphrase")]
#[command(about = "Learn typing-timing models and score/generate fast-to-type passphrases")]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

/// How to rank/select words for a wordset.
///
/// - `ms-only`: optimize *typing speed only* (best when you truly sample uniformly from the final wordset).
/// - `ms-per-lm-bit`: optimize \(ms / surprisal_bits(word)\) under a character k-gram LM trained on the corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum WordsetObjective {
    #[value(name = "ms-only")]
    MsOnly,
    #[value(name = "ms-per-lm-bit")]
    MsPerLmBit,
}

/// Word casing transform applied during sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CaseMode {
    #[value(name = "lower")]
    Lower,
    #[value(name = "title")]
    Title,
    #[value(name = "upper")]
    Upper,
    /// Randomly TitleCase each word with probability 1/2.
    #[value(name = "random-title")]
    RandomTitle,
}

/// Convenience presets for `sample-passphrases`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SampleStyle {
    /// Use provided flags (separator/gap_regex/case/prefix/suffix).
    #[value(name = "custom")]
    Custom,
    /// Words separated by spaces, lower-case.
    #[value(name = "spaces")]
    Spaces,
    /// Words separated by hyphens, lower-case (1Password-like).
    #[value(name = "hyphens")]
    Hyphens,
    /// Random digit separators between words.
    #[value(name = "numbers")]
    Numbers,
    /// Random digit or symbol separators between words.
    #[value(name = "numbers-symbols")]
    NumbersSymbols,
    /// TitleCase words concatenated, append 2 digits (login-box friendly).
    #[value(name = "login-title-2digits")]
    LoginTitle2Digits,
    /// TitleCase words concatenated, end with one of `! ? _ -`.
    #[value(name = "login-title-endpunct")]
    LoginTitleEndPunct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum AltMode {
    /// Show alternatives with smaller predicted_ms.
    #[value(name = "faster")]
    Faster,
    /// Show alternatives closest in predicted_ms (absolute difference).
    #[value(name = "similar")]
    Similar,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Smoke command: parse a dataset and print basic stats.
    Inspect {
        /// Path to CSV or JSONL dataset.
        #[arg(long)]
        input: std::path::PathBuf,
    },
    /// Fit a digraph model and print summary stats.
    Fit {
        /// Path to CSV or JSONL dataset.
        #[arg(long)]
        input: std::path::PathBuf,
        /// Minimum observations to keep a digraph-specific mean.
        #[arg(long, default_value_t = 3)]
        min_count: usize,
        /// Optional path to write the fitted model as JSON.
        #[arg(long)]
        output_model: Option<std::path::PathBuf>,
        /// If set, clamp each observed dt(ms) to at most this value (winsorize).
        #[arg(long)]
        clamp_dt_ms: Option<f32>,
    },
    /// Score a phrase using a model fit from a dataset.
    Score {
        /// Path to a previously-saved model JSON.
        #[arg(long)]
        model: std::path::PathBuf,
        /// Phrase to score.
        #[arg(long)]
        phrase: String,
        /// Optional reference wordlist to calibrate against (sample random phrases).
        #[arg(long)]
        ref_wordlist: Option<std::path::PathBuf>,
        /// Words per reference phrase (only used if ref_wordlist is set).
        #[arg(long, default_value_t = 4)]
        ref_words: usize,
        /// Samples for calibration distribution (only used if ref_wordlist is set).
        #[arg(long, default_value_t = 5000)]
        ref_samples: usize,
        /// Optional regex for reference gaps (only used if ref_wordlist is set).
        #[arg(long)]
        ref_gap_regex: Option<String>,
        /// Max characters allowed for reference phrases (optional).
        #[arg(long)]
        ref_max_chars: Option<usize>,
        /// RNG seed for calibration sampling.
        #[arg(long)]
        ref_seed: Option<u64>,
        /// If set, print per-digraph timing breakdown (after normalization).
        #[arg(long, default_value_t = false)]
        explain_digraphs: bool,
    },
    /// Generate passphrases from a wordlist and keep the fastest-to-type ones.
    Generate {
        /// Path to a previously-saved model JSON.
        #[arg(long)]
        model: std::path::PathBuf,
        /// Path to a newline-delimited wordlist (one word per line).
        #[arg(long)]
        wordlist: std::path::PathBuf,
        /// Number of words per passphrase.
        #[arg(long, default_value_t = 4)]
        words: usize,
        /// Separator used between words.
        #[arg(long, default_value = " ")]
        separator: String,
        /// How many random samples to evaluate.
        #[arg(long, default_value_t = 10_000)]
        samples: usize,
        /// How many top candidates to print.
        #[arg(long, default_value_t = 20)]
        top: usize,
        /// Optional RNG seed for reproducible generation.
        #[arg(long)]
        seed: Option<u64>,
        /// Minimum word length to keep from wordlist.
        #[arg(long, default_value_t = 3)]
        min_word_len: usize,
        /// Maximum word length to keep from wordlist.
        #[arg(long, default_value_t = 12)]
        max_word_len: usize,
        /// If set, restrict wordlist words to ASCII a-z after scrubbing.
        #[arg(long, default_value_t = true)]
        ascii_lower_only: bool,
    },
    /// Import a public dataset into `fastphrase` JSONL rows.
    ImportCmuDsl {
        /// CMU DSL-StrongPasswordData.csv URL.
        #[arg(
            long,
            default_value = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"
        )]
        url: String,
        /// Output path for JSONL rows.
        #[arg(long)]
        output: std::path::PathBuf,
    },
    /// Download public datasets into a directory (raw files; no parsing).
    DownloadDatasets {
        /// Output directory (created if needed).
        #[arg(long)]
        out_dir: std::path::PathBuf,
        /// Include BKSD (zips from the BKSD repo).
        #[arg(long, default_value_t = true)]
        bksd: bool,
        /// Include CMU DSL-StrongPasswordData.csv.
        #[arg(long, default_value_t = true)]
        cmu_dsl: bool,
        /// Include GREYC web-based dataset (archived tar.gz).
        #[arg(long, default_value_t = true)]
        greyc_web: bool,
        /// Include CMU LASER-2012 (Free vs Transcribed) zip.
        #[arg(long, default_value_t = true)]
        cmu_laser2012: bool,
        /// Include KeyRecs (Zenodo CC-BY 4.0) free-text and fixed-text CSVs.
        #[arg(long, default_value_t = true)]
        keyrecs: bool,
    },
    /// Download a default corpus (word list / word frequencies).
    ///
    /// Intended for “works out of the box” runs without a custom user corpus.
    DownloadCorpus {
        /// Output path for the corpus text file.
        #[arg(long, default_value = "data/corpus/hybrid_wordfreq25k_dwyl.tsv")]
        output: std::path::PathBuf,
        /// Which corpus to download.
        #[arg(long, default_value = "hybrid_wordfreq25k_plus_dwyl")]
        corpus: String,
    },
    /// Produce a single union JSONL from all downloaded datasets we know how to import.
    ///
    /// This is “best effort”: anything we can’t interpret as character-labeled digraphs
    /// will still be imported as positional digraph rows (BKSD) so it contributes to
    /// global timing statistics.
    UnionDatasets {
        /// Directory used by `download-datasets`.
        #[arg(long, default_value = "data/datasets")]
        datasets_dir: std::path::PathBuf,
        /// Output JSONL path.
        #[arg(long)]
        output: std::path::PathBuf,
        /// If true, download missing raw files into `datasets_dir` first.
        #[arg(long, default_value_t = true)]
        fetch_if_missing: bool,
    },
    /// Import the GREYC web-based keystroke dataset tar.gz into JSONL rows.
    ImportGreycWeb {
        /// Input tar.gz path (download with `download-datasets` or pass your own).
        #[arg(long)]
        input: std::path::PathBuf,
        /// Output JSONL path.
        #[arg(long)]
        output: std::path::PathBuf,
        /// Include password samples.
        #[arg(long, default_value_t = true)]
        passwords: bool,
        /// Include passphrase samples.
        #[arg(long, default_value_t = true)]
        passphrases: bool,
        /// Include impostor samples.
        #[arg(long, default_value_t = true)]
        impostor: bool,
        /// Optional max rows (for quick trials).
        #[arg(long)]
        max_rows: Option<usize>,
    },
    /// Adapt a base model using user timing rows (fine-tune).
    AdaptModel {
        /// Base model JSON (e.g. union model).
        #[arg(long)]
        base_model: std::path::PathBuf,
        /// User timing rows (JSONL or CSV in fastphrase format).
        #[arg(long)]
        user_data: std::path::PathBuf,
        /// Output tuned model JSON.
        #[arg(long)]
        output_model: std::path::PathBuf,
        /// Pseudo-count strength for the base prior per digraph.
        #[arg(long, default_value_t = 50.0)]
        prior_count: f32,
        /// Minimum count for new digraphs not present in the base.
        #[arg(long, default_value_t = 3)]
        min_new_count: usize,
    },
    /// Record typing timing data from terminal keypresses and append to a JSONL file.
    ///
    /// This is the simplest “just type stuff” workflow: run repeatedly over time and the
    /// data accumulates. You can optionally auto-adapt a base model after each session.
    RecordSession {
        /// Where to append recorded rows (JSONL).
        #[arg(long, default_value = "data/user/user.jsonl")]
        output: std::path::PathBuf,
        /// Number of samples to record in this session.
        #[arg(long, default_value_t = 20)]
        reps: usize,
        /// Optional target string to type (requires exact match; no backspaces).
        #[arg(long)]
        target: Option<String>,
        /// Optional base model to adapt after recording.
        #[arg(long)]
        base_model: Option<std::path::PathBuf>,
        /// Where to write the adapted model (required if base_model is set).
        #[arg(long)]
        output_model: Option<std::path::PathBuf>,
        /// Pseudo-count strength for the base prior per digraph.
        #[arg(long, default_value_t = 50.0)]
        prior_count: f32,
        /// Minimum count for new digraphs not present in the base.
        #[arg(long, default_value_t = 3)]
        min_new_count: usize,
    },
    /// Estimate enumeration time for the passphrase search space.
    EstimateSearch {
        /// Model JSON (used for ms/phrase estimation).
        #[arg(long)]
        model: std::path::PathBuf,
        /// Wordlist path.
        #[arg(long)]
        wordlist: std::path::PathBuf,
        /// Words per passphrase.
        #[arg(long, default_value_t = 4)]
        words: usize,
        /// Separator between words.
        #[arg(long, default_value = " ")]
        separator: String,
        /// If true, allow repeats (search space = |W|^words). If false, use falling factorial.
        #[arg(long, default_value_t = true)]
        allow_repeats: bool,
        /// Samples used to estimate average ms/phrase.
        #[arg(long, default_value_t = 5000)]
        samples: usize,
        /// Optional RNG seed.
        #[arg(long)]
        seed: Option<u64>,
        /// Minimum word length to keep from wordlist.
        #[arg(long, default_value_t = 3)]
        min_word_len: usize,
        /// Maximum word length to keep from wordlist.
        #[arg(long, default_value_t = 12)]
        max_word_len: usize,
        /// If set, restrict wordlist words to ASCII a-z after scrubbing.
        #[arg(long, default_value_t = true)]
        ascii_lower_only: bool,
    },
    /// Rank words by typing difficulty and k-gram surprisal (entropy).
    ///
    /// Reads a corpus file where each line is either:
    /// - `word`
    /// - `word<TAB>count`  (or any whitespace separator)
    RankWords {
        /// Path to a previously-saved model JSON (for difficulty).
        #[arg(long)]
        model: std::path::PathBuf,
        /// Corpus / word frequency file.
        #[arg(long)]
        corpus: std::path::PathBuf,
        /// k in the k-gram model (k>=1).
        #[arg(long, default_value_t = 3)]
        k: usize,
        /// Add-alpha smoothing (>=0).
        #[arg(long, default_value_t = 0.5)]
        alpha: f64,
        /// Show top N by ms_per_bit (lower is better).
        #[arg(long, default_value_t = 50)]
        top: usize,
        /// Ranking objective.
        #[arg(long, value_enum, default_value_t = WordsetObjective::MsPerLmBit)]
        objective: WordsetObjective,
        /// Apply the same word filters as generation (ASCII a-z only after scrubbing).
        #[arg(long, default_value_t = true)]
        ascii_lower_only: bool,
        /// Minimum word length.
        #[arg(long, default_value_t = 3)]
        min_word_len: usize,
        /// Maximum word length.
        #[arg(long, default_value_t = 12)]
        max_word_len: usize,
    },
    /// Export a filtered/scored wordset (one word per line) optimized for ms/bit.
    ///
    /// This is the practical bridge from “ranking” to “generation”: you can export a
    /// good wordlist and then run `generate` / `estimate-search` against it.
    ExportWordset {
        /// Path to a previously-saved model JSON (for difficulty).
        #[arg(long)]
        model: std::path::PathBuf,
        /// Corpus / word frequency file.
        #[arg(long)]
        corpus: std::path::PathBuf,
        /// Output wordlist path (one word per line by default).
        #[arg(long)]
        output: std::path::PathBuf,
        /// k in the k-gram model (k>=1).
        #[arg(long, default_value_t = 3)]
        k: usize,
        /// Add-alpha smoothing (>=0).
        #[arg(long, default_value_t = 0.5)]
        alpha: f64,
        /// Keep top N words by ms/bit (lower is better).
        #[arg(long, default_value_t = 5000)]
        top: usize,
        /// Ranking objective.
        #[arg(long, value_enum, default_value_t = WordsetObjective::MsOnly)]
        objective: WordsetObjective,
        /// Minimum per-word digraph hit fraction required for inclusion.
        ///
        /// This is a reliability guard: low hit fraction means the timing score depends heavily on
        /// global/backoff fallbacks, so the “easy to type” ranking is less trustworthy.
        #[arg(long, default_value_t = 0.0)]
        min_hit_frac: f64,
        /// Minimum number of vowels (a/e/i/o/u/y) required in a word.
        ///
        /// This is a simple “wordlikeness” guard that helps avoid abbreviation-like tokens
        /// (e.g., `cpl`, `rfs`) that can be fast to type but annoying to remember.
        #[arg(long, default_value_t = 0)]
        min_vowels: usize,
        /// If true, write TSV with columns: word, ms, bits, ms_per_bit.
        #[arg(long, default_value_t = false)]
        tsv: bool,
        /// Apply the same word filters as generation (ASCII a-z only after scrubbing).
        #[arg(long, default_value_t = true)]
        ascii_lower_only: bool,
        /// Minimum word length.
        #[arg(long, default_value_t = 3)]
        min_word_len: usize,
        /// Maximum word length.
        #[arg(long, default_value_t = 12)]
        max_word_len: usize,
    },
    /// Plan a passphrase wordset for a target entropy budget, using base model + corpus.
    ///
    /// - Picks the top-N words by ms/bit (difficulty per surprisal-bit).
    /// - Writes `output` (one word per line).
    /// - Estimates avg ms/phrase by sampling random phrases from the planned wordset.
    PlanPassphrase {
        /// Model JSON (difficulty).
        #[arg(long)]
        model: std::path::PathBuf,
        /// Corpus / word frequency file (entropy model).
        #[arg(long)]
        corpus: std::path::PathBuf,
        /// Output wordset path (one word per line).
        #[arg(long)]
        output: std::path::PathBuf,
        /// Words per passphrase.
        #[arg(long, default_value_t = 4)]
        words: usize,
        /// Target entropy in bits (approx; assumes uniform sampling over wordset).
        #[arg(long, default_value_t = 60.0)]
        target_bits: f64,
        /// If true, allow repeats (search space ≈ |W|^words). If false, use falling factorial.
        #[arg(long, default_value_t = true)]
        allow_repeats: bool,
        /// Separator between words.
        #[arg(long, default_value = " ")]
        separator: String,
        /// k in the k-gram model (k>=1).
        #[arg(long, default_value_t = 3)]
        k: usize,
        /// Add-alpha smoothing (>=0).
        #[arg(long, default_value_t = 0.5)]
        alpha: f64,
        /// Ranking objective for selecting the wordset.
        #[arg(long, value_enum, default_value_t = WordsetObjective::MsOnly)]
        objective: WordsetObjective,
        /// Minimum per-word digraph hit fraction required for inclusion.
        #[arg(long, default_value_t = 0.0)]
        min_hit_frac: f64,
        /// Minimum number of vowels (a/e/i/o/u/y) required in a word.
        #[arg(long, default_value_t = 0)]
        min_vowels: usize,
        /// Samples for avg ms/phrase estimation.
        #[arg(long, default_value_t = 5000)]
        samples: usize,
        /// Optional RNG seed.
        #[arg(long)]
        seed: Option<u64>,
        /// Apply the same word filters as generation (ASCII a-z only after scrubbing).
        #[arg(long, default_value_t = true)]
        ascii_lower_only: bool,
        /// Minimum word length.
        #[arg(long, default_value_t = 3)]
        min_word_len: usize,
        /// Maximum word length.
        #[arg(long, default_value_t = 12)]
        max_word_len: usize,
    },
    /// One-shot base pipeline (no user data, no custom corpus):
    ///
    /// - download public typing datasets (CMU + BKSD)
    /// - union them → `union.jsonl`
    /// - fit base model → `model_union.json`
    /// - download a default corpus (dwyl words)
    /// - plan a wordset for target entropy
    /// - generate examples and estimate enumeration time
    BasePipeline {
        /// Working directory for artifacts.
        #[arg(long, default_value = "data/base")]
        out_dir: std::path::PathBuf,
        /// Words per passphrase.
        #[arg(long, default_value_t = 4)]
        words: usize,
        /// Target entropy bits for the planned wordset.
        #[arg(long, default_value_t = 60.0)]
        target_bits: f64,
        /// Allow word repeats in passphrases.
        #[arg(long, default_value_t = true)]
        allow_repeats: bool,
        /// Wordset selection objective (default: typing speed only).
        #[arg(long, value_enum, default_value_t = WordsetObjective::MsOnly)]
        objective: WordsetObjective,
        /// Minimum per-word digraph hit fraction (default: 0.95).
        #[arg(long, default_value_t = 0.95)]
        min_hit_frac: f64,
        /// Minimum vowels per word (default: 1).
        #[arg(long, default_value_t = 1)]
        min_vowels: usize,
        /// Reuse existing artifacts in `out_dir` if present.
        #[arg(long, default_value_t = true)]
        reuse_existing: bool,
        /// Print stage/progress information.
        #[arg(long, default_value_t = false)]
        debug: bool,
        /// Generation samples.
        #[arg(long, default_value_t = 50_000)]
        samples: usize,
        /// Print top K generated examples.
        #[arg(long, default_value_t = 20)]
        top: usize,
        /// Optional RNG seed.
        #[arg(long)]
        seed: Option<u64>,
    },
    /// One-shot personalization pipeline (reuses a base pipeline + accumulated user data):
    ///
    /// - adapts base model with `user.jsonl`
    /// - plans a personalized wordset
    /// - generates examples + estimates enumeration time
    PersonalizePipeline {
        /// Base directory from a prior `base-pipeline` run (must contain model + corpus).
        #[arg(long, default_value = "data/base")]
        base_dir: std::path::PathBuf,
        /// Accumulated user timing rows (JSONL).
        #[arg(long, default_value = "data/user/user.jsonl")]
        user_data: std::path::PathBuf,
        /// Output directory for personalized artifacts.
        #[arg(long, default_value = "data/user")]
        out_dir: std::path::PathBuf,
        /// Words per passphrase.
        #[arg(long, default_value_t = 4)]
        words: usize,
        /// Target entropy bits for the planned wordset.
        #[arg(long, default_value_t = 60.0)]
        target_bits: f64,
        /// Allow word repeats in passphrases.
        #[arg(long, default_value_t = true)]
        allow_repeats: bool,
        /// Wordset selection objective (default: typing speed only).
        #[arg(long, value_enum, default_value_t = WordsetObjective::MsOnly)]
        objective: WordsetObjective,
        /// Minimum per-word digraph hit fraction (default: 0.95).
        #[arg(long, default_value_t = 0.95)]
        min_hit_frac: f64,
        /// Minimum vowels per word (default: 1).
        #[arg(long, default_value_t = 1)]
        min_vowels: usize,
        /// Reuse existing artifacts in `out_dir` if present.
        #[arg(long, default_value_t = true)]
        reuse_existing: bool,
        /// Print stage/progress information.
        #[arg(long, default_value_t = false)]
        debug: bool,
        /// Generation samples.
        #[arg(long, default_value_t = 50_000)]
        samples: usize,
        /// Print top K generated examples.
        #[arg(long, default_value_t = 20)]
        top: usize,
        /// Optional RNG seed.
        #[arg(long)]
        seed: Option<u64>,
        /// Pseudo-count strength for the base prior per digraph.
        #[arg(long, default_value_t = 50.0)]
        prior_count: f32,
        /// Minimum count for new digraphs not present in the base.
        #[arg(long, default_value_t = 3)]
        min_new_count: usize,
    },
    /// Evaluate a model by fitting on a train split and scoring held-out data.
    ///
    /// Reports digraph-level errors (MAE/RMSE), phrase-level errors, and breakdowns by source.
    EvalModel {
        /// Input dataset (usually a union JSONL).
        #[arg(long)]
        input: std::path::PathBuf,
        /// Minimum observations to keep a digraph-specific mean in the fitted model.
        #[arg(long, default_value_t = 3)]
        min_count: usize,
        /// Fraction of rows to allocate to the test set.
        #[arg(long, default_value_t = 0.2)]
        test_frac: f64,
        /// Optional RNG seed (for stable splits).
        #[arg(long)]
        seed: Option<u64>,
        /// If set, only evaluate rows whose `source` starts with this prefix (e.g. "cmu_dsl").
        #[arg(long)]
        source_prefix: Option<String>,
        /// If true, print a small sample of worst-predicted phrases.
        #[arg(long, default_value_t = true)]
        show_worst: bool,
        /// Number of worst phrases to show.
        #[arg(long, default_value_t = 10)]
        worst_k: usize,
        /// If set, clamp each observed dt(ms) to at most this value (winsorize) during fit
        /// and when computing test-set “true” metrics (robust evaluation).
        #[arg(long)]
        clamp_dt_ms: Option<f32>,
    },
    /// Print dataset timing statistics (quantiles, outlier counts).
    DatasetStats {
        /// Input dataset (CSV or JSONL).
        #[arg(long)]
        input: std::path::PathBuf,
        /// If set, only include rows whose `source` starts with this prefix.
        #[arg(long)]
        source_prefix: Option<String>,
        /// If set, only include rows with digraph length >= this.
        #[arg(long, default_value_t = 1)]
        min_digraphs: usize,
    },
    /// Build a personalized timing model that supports imputation/backoff for unseen digraphs.
    ///
    /// Output is a `PersonalizedModel` JSON which can be used anywhere a model is accepted.
    BuildPersonalizedModel {
        /// Base model JSON (e.g. `data/base/model_union.json`).
        #[arg(long)]
        base_model: std::path::PathBuf,
        /// User timing rows (JSONL or CSV in fastphrase format).
        #[arg(long)]
        user_data: std::path::PathBuf,
        /// Output personalized model JSON.
        #[arg(long)]
        output_model: std::path::PathBuf,
        /// Pseudo-count strength for the base prior per digraph.
        #[arg(long, default_value_t = 50.0)]
        prior_count: f32,
        /// Minimum count for new digraphs not present in the base.
        #[arg(long, default_value_t = 3)]
        min_new_count: usize,
        /// Minimum count for per-key/group backoff effects to activate.
        #[arg(long, default_value_t = 5)]
        min_backoff_count: u32,
    },
    /// Sample random passphrases subject to constraints.
    ///
    /// Useful for the “final step”: emit K random passphrases, optionally inserting
    /// regex-generated gap strings, and enforcing a max character length.
    SamplePassphrases {
        /// Model JSON (digraph or personalized).
        #[arg(long)]
        model: std::path::PathBuf,
        /// Wordlist path (one word per line).
        #[arg(long)]
        wordlist: std::path::PathBuf,
        /// Number of passphrases to output.
        #[arg(long, default_value_t = 10)]
        count: usize,
        /// Words per passphrase.
        #[arg(long, default_value_t = 4)]
        words: usize,
        /// If true, allow word repeats.
        #[arg(long, default_value_t = true)]
        allow_repeats: bool,
        /// Default separator if `gap_regex` is not set.
        #[arg(long, default_value = " ")]
        separator: String,
        /// Optional regex for each inter-word “gap” (replaces separator with random match).
        ///
        /// Example: `[-_\\.]{1,2}[0-9]{0,2}`
        #[arg(long)]
        gap_regex: Option<String>,
        /// Max total characters allowed for the final passphrase.
        #[arg(long)]
        max_chars: Option<usize>,
        /// Min total characters allowed for the final passphrase.
        #[arg(long)]
        min_chars: Option<usize>,
        /// RNG seed.
        #[arg(long)]
        seed: Option<u64>,
        /// Formatting preset (optional convenience).
        #[arg(long, value_enum, default_value_t = SampleStyle::Custom)]
        style: SampleStyle,
        /// Apply a casing transform to each sampled word.
        ///
        /// Useful for “login password” style outputs (no spaces, mixed case).
        #[arg(long, value_enum, default_value_t = CaseMode::Lower)]
        case: CaseMode,
        /// Optional regex to prepend to each sampled passphrase.
        #[arg(long)]
        prefix_regex: Option<String>,
        /// Optional regex to append to each sampled passphrase.
        #[arg(long)]
        suffix_regex: Option<String>,
        /// Show per-sample alternatives found by local random search (0 disables).
        ///
        /// Note: choosing among alternatives manually can reduce effective entropy unless the choice
        /// is itself randomized.
        #[arg(long, default_value_t = 0)]
        alternatives: usize,
        /// Random replacement tries per word-position when searching alternatives.
        #[arg(long, default_value_t = 200)]
        alt_tries: usize,
        /// How to rank alternatives.
        #[arg(long, value_enum, default_value_t = AltMode::Faster)]
        alt_mode: AltMode,
        /// If true, print extra per-sample metadata (chars, hit_frac).
        #[arg(long, default_value_t = false)]
        meta: bool,
        /// Max attempts per output (avoid infinite loops under tight constraints).
        #[arg(long, default_value_t = 50_000)]
        max_tries: usize,
    },
    /// Analyze a concrete generator distribution: typing-time stats + effective entropy estimates.
    ///
    /// This is meant to answer “what is optimal actually?” for the *real* distribution you use:
    /// - uniform word choice + separators/suffix/prefix randomness
    /// - rejection sampling under char limits
    /// - and optional “pick the fastest of M” (models manual choice among alternatives)
    AnalyzeGenerator {
        /// Model JSON (digraph or personalized).
        #[arg(long)]
        model: std::path::PathBuf,
        /// Wordlist path (one word per line).
        #[arg(long)]
        wordlist: std::path::PathBuf,
        /// Optional corpus counts file (word<TAB>count) for IDF/commonality diagnostics.
        ///
        /// If not set, we’ll try `corpus.txt` next to the wordlist.
        #[arg(long)]
        corpus: Option<std::path::PathBuf>,
        /// Monte Carlo samples.
        #[arg(long, default_value_t = 20_000)]
        samples: usize,
        /// If >1, simulate choosing the fastest among this many random draws.
        ///
        /// This approximates the “entropy penalty” of showing alternatives and then manually picking.
        #[arg(long, default_value_t = 1)]
        pick_best_of: usize,
        /// Words per passphrase.
        #[arg(long, default_value_t = 4)]
        words: usize,
        /// If true, allow word repeats.
        #[arg(long, default_value_t = true)]
        allow_repeats: bool,
        /// Default separator if `gap_regex` is not set.
        #[arg(long, default_value = " ")]
        separator: String,
        /// Optional regex for each inter-word “gap” (replaces separator with random match).
        #[arg(long)]
        gap_regex: Option<String>,
        /// Max total characters allowed for the final passphrase.
        #[arg(long)]
        max_chars: Option<usize>,
        /// Min total characters allowed for the final passphrase.
        #[arg(long)]
        min_chars: Option<usize>,
        /// RNG seed.
        #[arg(long)]
        seed: Option<u64>,
        /// Formatting preset (optional convenience).
        #[arg(long, value_enum, default_value_t = SampleStyle::Custom)]
        style: SampleStyle,
        /// Apply a casing transform to each sampled word.
        #[arg(long, value_enum, default_value_t = CaseMode::Lower)]
        case: CaseMode,
        /// Optional regex to prepend to each sampled passphrase.
        #[arg(long)]
        prefix_regex: Option<String>,
        /// Optional regex to append to each sampled passphrase.
        #[arg(long)]
        suffix_regex: Option<String>,
        /// Max total sampling attempts (across all outputs).
        #[arg(long, default_value_t = 5_000_000)]
        max_tries_total: usize,
        /// Show the most frequent sampled outputs (detects surprising concentration).
        #[arg(long, default_value_t = 10)]
        show_top: usize,
    },
    /// Audit a model’s digraph coverage and normalized speed on a word corpus.
    ///
    /// This answers: “Is this model actually informed for typical words, or mostly fallback?”
    AuditModel {
        /// Model JSON (digraph or personalized).
        #[arg(long)]
        model: std::path::PathBuf,
        /// Corpus / word list file (same format as `rank-words`: `word` or `word<TAB>count`).
        #[arg(long)]
        corpus: std::path::PathBuf,
        /// If true, restrict to ASCII a-z after scrubbing.
        #[arg(long, default_value_t = true)]
        ascii_lower_only: bool,
        /// Min word length.
        #[arg(long, default_value_t = 3)]
        min_word_len: usize,
        /// Max word length.
        #[arg(long, default_value_t = 12)]
        max_word_len: usize,
        /// Number of words to sample from corpus (bounded).
        #[arg(long, default_value_t = 50_000)]
        sample_words: usize,
        /// RNG seed for sampling.
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Compute a Pareto frontier of “security bits vs typing time” for 1Password-like styles.
    ///
    /// This evaluates multiple (style, wordset_size) configurations by Monte Carlo sampling:
    /// - objective 1: minimize expected typing time (ms)
    /// - objective 2: maximize estimated entropy bits (from uniform word choice + separator/suffix randomness)
    ParetoStyles {
        /// Model JSON (digraph or personalized).
        #[arg(long)]
        model: std::path::PathBuf,
        /// Wordset path (one word per line, ordered by “easiest first”).
        #[arg(long)]
        wordlist: std::path::PathBuf,
        /// Optional corpus counts (word<TAB>count), for “commonality/IDF” diagnostics.
        ///
        /// If not provided, we’ll try `corpus.txt` next to the wordlist.
        #[arg(long)]
        corpus: Option<std::path::PathBuf>,
        /// Words per password/passphrase.
        #[arg(long, default_value_t = 4)]
        words: usize,
        /// If true, allow repeats (≈ N^k). If false, use falling factorial.
        #[arg(long, default_value_t = true)]
        allow_repeats: bool,
        /// Candidate wordset sizes N to evaluate (uses first N words from wordlist).
        #[arg(long, num_args = 1.., value_delimiter = ',')]
        n: Vec<usize>,
        /// Styles to evaluate (repeated or comma-separated).
        #[arg(long, value_enum, num_args = 1.., value_delimiter = ',')]
        style: Vec<SampleStyle>,
        /// Samples per configuration.
        #[arg(long, default_value_t = 5000)]
        samples: usize,
        /// RNG seed for reproducibility.
        #[arg(long)]
        seed: Option<u64>,
        /// If set, print a “best under constraints” recommendation.
        #[arg(long, default_value_t = false)]
        recommend: bool,
        /// Minimum acceptable entropy bits (used only for `--recommend`).
        #[arg(long)]
        target_bits: Option<f64>,
        /// Minimum acceptable digraph hit fraction (used only for `--recommend`).
        #[arg(long)]
        min_hit_frac: Option<f64>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Inspect { input } => {
            let rows = fastphrase::data::load_rows(&input)
                .with_context(|| format!("loading dataset: {}", input.display()))?;
            println!("rows: {}", rows.len());
            let (min_len, max_len) = rows.iter().fold((usize::MAX, 0usize), |acc, r| {
                (
                    acc.0.min(r.phrase.chars().count()),
                    acc.1.max(r.phrase.chars().count()),
                )
            });
            if rows.is_empty() {
                println!("min_chars: 0");
                println!("max_chars: 0");
            } else {
                println!("min_chars: {min_len}");
                println!("max_chars: {max_len}");
            }
        }
        Command::Fit {
            input,
            min_count,
            output_model,
            clamp_dt_ms,
        } => {
            let rows = fastphrase::data::load_rows(&input)
                .with_context(|| format!("loading dataset: {}", input.display()))?;
            let (model, stats) = fit_digraph_model(
                &rows,
                FitConfig {
                    min_count,
                    clamp_dt_ms,
                },
            );
            println!("rows: {}", stats.rows);
            println!("total_digraph_obs: {}", stats.total_digraph_obs);
            println!("distinct_digraphs: {}", stats.distinct_digraphs);
            println!("kept_digraphs: {}", stats.kept_digraphs);
            println!("global_mean_ms: {:.3}", stats.global_mean_ms);

            if let Some(p) = output_model {
                model
                    .save_json(&p)
                    .with_context(|| format!("writing model: {}", p.display()))?;
                println!("model_written: {}", p.display());
            }
        }
        Command::Score {
            model,
            phrase,
            ref_wordlist,
            ref_words,
            ref_samples,
            ref_gap_regex,
            ref_max_chars,
            ref_seed,
            explain_digraphs,
        } => {
            let model = AnyTimingModel::load_json(&model)
                .with_context(|| format!("loading model: {}", model.display()))?;
            let s = fastphrase::score::score_phrase(&model, &phrase);
            let per_digraph = if s.digraphs == 0 {
                0.0
            } else {
                (s.predicted_ms as f64) / (s.digraphs as f64)
            };
            let denom = (s.digraphs as f64) * (model.global_mean_ms() as f64);
            let norm_ratio = if denom > 0.0 {
                (s.predicted_ms as f64) / denom
            } else {
                0.0
            };
            println!("phrase: {phrase}");
            println!("predicted_ms: {:.3}", s.predicted_ms);
            println!("digraphs: {}", s.digraphs);
            println!("digraph_hits: {}", s.hits);
            println!("digraph_misses: {}", s.misses);
            println!(
                "digraph_hit_frac: {:.6}",
                if s.digraphs == 0 {
                    0.0
                } else {
                    (s.hits as f64) / (s.digraphs as f64)
                }
            );
            println!("ms_per_digraph: {:.3}", per_digraph);
            println!(
                "normalized_vs_global: {:.6} ( <1 faster than avg digraph )",
                norm_ratio
            );

            if explain_digraphs {
                let grams = fastphrase::score::graphemes_normalized(&phrase);
                println!();
                println!("digraph_breakdown (normalized graphemes):");
                if grams.len() < 2 {
                    println!("  (no digraphs)");
                } else {
                    let mut sum = 0.0f64;
                    for i in 0..(grams.len() - 1) {
                        let a = &grams[i];
                        let b = &grams[i + 1];
                        let ms = model.mean_ms_for(a, b) as f64;
                        let hit = model.has_digraph(a, b);
                        sum += ms;
                        println!(
                            "  {:>2}. {:>3} | {:<3}  {:>8.3} ms  {}",
                            i + 1,
                            a,
                            b,
                            ms,
                            if hit { "hit" } else { "miss" }
                        );
                    }
                    println!("  sum_digraph_ms: {:.3}", sum);
                }
            }

            if let Some(ref_wordlist) = ref_wordlist {
                // Calibrate by sampling random phrases from ref_wordlist and computing mean/std of predicted_ms.
                let mut gcfg = GenerateConfig::default();
                gcfg.words = ref_words;
                gcfg.separator = " ".to_string();
                gcfg.ascii_lower_only = false;
                gcfg.min_word_len = 1;
                gcfg.max_word_len = usize::MAX;
                let wl = load_wordlist(&ref_wordlist, &gcfg)
                    .with_context(|| format!("loading ref wordlist: {}", ref_wordlist.display()))?;

                let gap_dist = if let Some(pat) = ref_gap_regex.as_deref() {
                    Some(
                        rand_regex::Regex::compile(pat, 256)
                            .with_context(|| format!("compiling ref_gap_regex: {pat}"))?,
                    )
                } else {
                    None
                };
                let mut rng = rng_from_seed(ref_seed);
                let mut rng_gap = match ref_seed {
                    Some(s) => rand08::rngs::StdRng::seed_from_u64(s ^ 0x9e37_79b9_7f4a_7c15),
                    None => rand08::rngs::StdRng::from_entropy(),
                };

                let mut sum = 0.0f64;
                let mut sum_sq = 0.0f64;
                let mut kept = 0usize;
                let mut vals: Vec<f64> = Vec::with_capacity(ref_samples.min(100_000));
                let mut tries = 0usize;
                let max_tries = ref_samples.saturating_mul(50).max(1000);
                while kept < ref_samples && tries < max_tries {
                    tries += 1;
                    // Sample a phrase.
                    let mut parts: Vec<&str> = Vec::with_capacity(ref_words);
                    for _ in 0..ref_words {
                        let Some(w) = wl.words.choose(&mut rng) else {
                            break;
                        };
                        parts.push(w.as_str());
                    }
                    if parts.len() != ref_words {
                        break;
                    }
                    let mut p = String::new();
                    for (i, w) in parts.iter().enumerate() {
                        if i > 0 {
                            if let Some(dist) = &gap_dist {
                                let gap: String =
                                    rand08::distributions::Distribution::sample(dist, &mut rng_gap);
                                p.push_str(&gap);
                            } else {
                                p.push(' ');
                            }
                        }
                        p.push_str(w);
                    }
                    if let Some(m) = ref_max_chars {
                        if p.chars().count() > m {
                            continue;
                        }
                    }
                    let ms = fastphrase::score::score_phrase(&model, &p).predicted_ms as f64;
                    sum += ms;
                    sum_sq += ms * ms;
                    vals.push(ms);
                    kept += 1;
                }

                if kept >= 10 {
                    vals.sort_by(|a, b| a.total_cmp(b));
                    let mean = sum / (kept as f64);
                    let var = (sum_sq / (kept as f64)) - mean * mean;
                    let std = var.max(0.0).sqrt();
                    let z = if std > 0.0 {
                        ((s.predicted_ms as f64) - mean) / std
                    } else {
                        0.0
                    };
                    // Empirical percentile (lower is faster).
                    let phrase_ms = s.predicted_ms as f64;
                    let le = vals.partition_point(|x| *x <= phrase_ms);
                    let pct_emp = (le as f64) / (kept as f64);

                    let at = |p: f64| -> f64 {
                        let idx = ((p * ((kept - 1) as f64)).round() as usize).min(kept - 1);
                        vals[idx]
                    };
                    let q10 = at(0.10);
                    let q50 = at(0.50);
                    let q90 = at(0.90);
                    let q95 = at(0.95);
                    let q99 = at(0.99);

                    println!("calibration.ref_wordlist: {}", ref_wordlist.display());
                    println!("calibration.samples_used: {kept}");
                    println!("calibration.mean_ms: {:.3}", mean);
                    println!("calibration.std_ms: {:.3}", std);
                    println!("calibration.z: {:.6}", z);
                    println!(
                        "calibration.quantiles_ms(p10/p50/p90/p95/p99): {:.3}/{:.3}/{:.3}/{:.3}/{:.3}",
                        q10, q50, q90, q95, q99
                    );
                    println!(
                        "calibration.percentile_empirical: {:.6} (lower is faster)",
                        pct_emp
                    );
                    println!(
                        "calibration.faster_than_frac: {:.6}",
                        (1.0 - pct_emp).max(0.0)
                    );
                } else {
                    println!("calibration: insufficient_samples (used={kept})");
                }
            }
        }
        Command::Generate {
            model,
            wordlist,
            words,
            separator,
            samples,
            top,
            seed,
            min_word_len,
            max_word_len,
            ascii_lower_only,
        } => {
            let model = AnyTimingModel::load_json(&model)
                .with_context(|| format!("loading model: {}", model.display()))?;

            let mut gcfg = GenerateConfig::default();
            gcfg.words = words;
            gcfg.separator = separator;
            gcfg.samples = samples;
            gcfg.top_k = top;
            gcfg.min_word_len = min_word_len;
            gcfg.max_word_len = max_word_len;
            gcfg.ascii_lower_only = ascii_lower_only;

            let wl = load_wordlist(&wordlist, &gcfg)
                .with_context(|| format!("loading wordlist: {}", wordlist.display()))?;

            let mut rng = rng_from_seed(seed);
            let (best, gen_stats) = generate_top(&model, &wl, &gcfg, &mut rng);

            println!("model.global_mean_ms: {:.3}", model.global_mean_ms());
            println!(
                "wordlist.total_words_in_file: {}",
                gen_stats.total_words_in_file
            );
            println!("wordlist.usable_words: {}", gen_stats.usable_words);
            println!(
                "entropy_bits_approx: {:.2} (assuming uniform sampling, words={}, sep={:?})",
                gen_stats.effective_entropy_bits, gcfg.words, gcfg.separator
            );
            println!("samples: {}", gcfg.samples);
            println!();

            for (i, c) in best.iter().enumerate() {
                println!(
                    "{:>2}. {:>8.3} ms  {}",
                    i + 1,
                    c.score.predicted_ms,
                    c.phrase
                );
            }
        }
        Command::ImportCmuDsl { url, output } => {
            let bytes =
                cmu_dsl::download_csv(&url).with_context(|| format!("downloading: {url}"))?;
            let rows = cmu_dsl::parse_csv_bytes(&bytes).context("parsing CMU DSL CSV")?;
            cmu_dsl::write_jsonl(&rows, &output)
                .with_context(|| format!("writing jsonl: {}", output.display()))?;
            println!("rows_written: {}", rows.len());
            println!("phrase: {}", cmu_dsl::PHRASE);
            println!("output: {}", output.display());
        }
        Command::ImportGreycWeb {
            input,
            output,
            passwords,
            passphrases,
            impostor,
            max_rows,
        } => {
            let wrote = greyc_web::write_jsonl_from_tar_gz_path(
                &input,
                &output,
                greyc_web::ImportConfig {
                    include_passwords: passwords,
                    include_passphrases: passphrases,
                    include_impostor: impostor,
                    max_rows,
                },
            )
            .with_context(|| format!("importing GREYC web dataset from {}", input.display()))?;
            println!("rows_written: {wrote}");
            println!("output: {}", output.display());
        }
        Command::DownloadDatasets {
            out_dir,
            bksd,
            cmu_dsl,
            greyc_web,
            cmu_laser2012,
            keyrecs,
        } => {
            std::fs::create_dir_all(&out_dir)
                .with_context(|| format!("creating out_dir: {}", out_dir.display()))?;

            if cmu_dsl {
                let p = out_dir.join("cmu").join("DSL-StrongPasswordData.csv");
                let url = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv";
                fastphrase::import::download_to_file(url, &p)
                    .with_context(|| format!("downloading CMU DSL CSV: {url}"))?;
                println!("downloaded: {}", p.display());
            }

            if bksd {
                // BKSD repo uses MIT; dataset is packaged as zips in-repo.
                let base = "https://raw.githubusercontent.com/ntwaijry/BKSD/eb0b5907a090a9ba06d47f98cd63cf1c7f0e3339/dataset";
                for name in [
                    "Password-Arabic.zip",
                    "Password-English.zip",
                    "Phrase-Arabic.zip",
                    "Phrase-English.zip",
                ] {
                    let url = format!("{base}/{name}");
                    let p = out_dir.join("bksd").join(name);
                    fastphrase::import::download_to_file(&url, &p)
                        .with_context(|| format!("downloading BKSD file: {url}"))?;
                    println!("downloaded: {}", p.display());
                }
            }

            if greyc_web {
                let p = out_dir.join("greyc_web").join("webkeystroke.tar.gz");
                let url = greyc_web::default_url();
                fastphrase::import::download_to_file(url, &p)
                    .with_context(|| format!("downloading GREYC web dataset: {url}"))?;
                println!("downloaded: {}", p.display());
            }

            if cmu_laser2012 {
                let p = out_dir
                    .join("cmu_laser2012")
                    .join("DSL-Free-vs-Transcribed.zip");
                let url = cmu_laser2012::default_url();
                fastphrase::import::download_to_file(url, &p)
                    .with_context(|| format!("downloading CMU LASER-2012 zip: {url}"))?;
                println!("downloaded: {}", p.display());
            }

            if keyrecs {
                let p1 = out_dir.join("keyrecs").join("free-text.csv");
                let url1 = keyrecs::free_text_url();
                fastphrase::import::download_to_file(url1, &p1)
                    .with_context(|| format!("downloading KeyRecs free-text: {url1}"))?;
                println!("downloaded: {}", p1.display());

                let p2 = out_dir.join("keyrecs").join("fixed-text.csv");
                let url2 = keyrecs::fixed_text_url();
                fastphrase::import::download_to_file(url2, &p2)
                    .with_context(|| format!("downloading KeyRecs fixed-text: {url2}"))?;
                println!("downloaded: {}", p2.display());
            }
        }
        Command::DownloadCorpus { output, corpus } => {
            match corpus.as_str() {
                "dwyl_words_alpha" => {
                    let url = "https://raw.githubusercontent.com/dwyl/english-words/20f5cc9b3f0ccc8ce45d814c532b7c2031bba31c/words_alpha.txt";
                    fastphrase::import::download_to_file(url, &output)
                        .with_context(|| format!("downloading corpus from {url}"))?;
                    println!("downloaded: {}", output.display());
                    println!("corpus: {corpus}");
                    println!("url: {url}");
                }
                "pdwl_5000_more_common" => {
                    let url = "https://raw.githubusercontent.com/MichaelWehar/Public-Domain-Word-Lists/e2fe9d32eeb86179b464aba0e91fa96904b8e7e8/5000-more-common.txt";
                    fastphrase::import::download_to_file(url, &output)
                        .with_context(|| format!("downloading corpus from {url}"))?;
                    println!("downloaded: {}", output.display());
                    println!("corpus: {corpus}");
                    println!("url: {url}");
                }
                "aparrish_wordfreq_en_25k" => {
                    // JSON list of [word, ln(freq)] where freq is a fraction.
                    let url = "https://raw.githubusercontent.com/aparrish/wordfreq-en-25000/master/wordfreq-en-25000-log.json";
                    let tmp = std::env::temp_dir().join("fastphrase_wordfreq_en_25k.json");
                    fastphrase::import::download_to_file(url, &tmp)
                        .with_context(|| format!("downloading corpus json from {url}"))?;
                    let bytes = std::fs::read(&tmp)?;
                    let rows: Vec<(String, f64)> = serde_json::from_slice(&bytes)
                        .context("parsing wordfreq-en-25000-log.json")?;
                    if let Some(parent) = output.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    let mut f = std::io::BufWriter::new(std::fs::File::create(&output)?);
                    writeln!(
                        f,
                        "# wordfreq-en-25000 (word\\tcount), derived from `wordfreq` via aparrish/wordfreq-en-25000"
                    )?;
                    // Scale fractional frequencies to integer counts. Relative weights matter more than absolute.
                    let scale = 1e9_f64;
                    for (w, ln_freq) in rows {
                        let freq = ln_freq.exp();
                        if !freq.is_finite() || freq <= 0.0 {
                            continue;
                        }
                        let count = (freq * scale).round().max(1.0) as u64;
                        writeln!(f, "{w}\t{count}")?;
                    }
                    f.flush()?;
                    let _ = std::fs::remove_file(&tmp);
                    println!("downloaded: {}", output.display());
                    println!("corpus: {corpus}");
                    println!("url: {url}");
                }
                "hybrid_wordfreq25k_plus_dwyl" => {
                    // Merge:
                    // - aparrish/wordfreq-en-25000 log-frequencies (scaled to counts)
                    // - dwyl/english-words words_alpha.txt (count=1 for tail coverage)
                    //
                    // This gives us a large vocab (entropy budget) with sane weighting near the head.
                    let url_wordfreq = "https://raw.githubusercontent.com/aparrish/wordfreq-en-25000/master/wordfreq-en-25000-log.json";
                    let url_dwyl = "https://raw.githubusercontent.com/dwyl/english-words/20f5cc9b3f0ccc8ce45d814c532b7c2031bba31c/words_alpha.txt";
                    let tmp1 = std::env::temp_dir().join("fastphrase_wordfreq_en_25k.json");
                    let tmp2 = std::env::temp_dir().join("fastphrase_dwyl_words_alpha.txt");
                    fastphrase::import::download_to_file(url_wordfreq, &tmp1)?;
                    fastphrase::import::download_to_file(url_dwyl, &tmp2)?;

                    let bytes = std::fs::read(&tmp1)?;
                    let rows: Vec<(String, f64)> = serde_json::from_slice(&bytes)
                        .context("parsing wordfreq-en-25000-log.json")?;
                    let mut map: std::collections::HashMap<String, u64> =
                        std::collections::HashMap::new();
                    let scale = 1e9_f64;
                    for (w, ln_freq) in rows {
                        let freq = ln_freq.exp();
                        if !freq.is_finite() || freq <= 0.0 {
                            continue;
                        }
                        let count = (freq * scale).round().max(1.0) as u64;
                        map.insert(w, count);
                    }

                    let dwyl = std::fs::read_to_string(&tmp2)?;
                    for line in dwyl.lines() {
                        let w = line.trim();
                        if w.is_empty() || w.starts_with('#') {
                            continue;
                        }
                        map.entry(w.to_string()).or_insert(1);
                    }

                    if let Some(parent) = output.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    let mut items: Vec<(String, u64)> = map.into_iter().collect();
                    items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                    let mut f = std::io::BufWriter::new(std::fs::File::create(&output)?);
                    writeln!(f, "# hybrid: aparrish/wordfreq-en-25000 (scaled counts) + dwyl/words_alpha tail (count=1)")?;
                    for (w, c) in items {
                        writeln!(f, "{w}\t{c}")?;
                    }
                    f.flush()?;

                    let _ = std::fs::remove_file(&tmp1);
                    let _ = std::fs::remove_file(&tmp2);
                    println!("downloaded: {}", output.display());
                    println!("corpus: {corpus}");
                    println!("url.wordfreq: {url_wordfreq}");
                    println!("url.dwyl: {url_dwyl}");
                }
                "hybrid_wordfreq25k_plus_eff_large" => {
                    // Merge:
                    // - aparrish/wordfreq-en-25000 log-frequencies (scaled to counts)
                    // - EFF large wordlist (7,776 words; count=1 tail add)
                    //
                    // This gives us ~32,776 candidate words that are more “passphrase-usable”
                    // than raw dictionary tails, while still hitting 60 bits with 4 words.
                    let url_wordfreq = "https://raw.githubusercontent.com/aparrish/wordfreq-en-25000/master/wordfreq-en-25000-log.json";
                    let url_eff = "https://www.eff.org/files/2016/07/18/eff_large_wordlist.txt";
                    let tmp1 = std::env::temp_dir().join("fastphrase_wordfreq_en_25k.json");
                    let tmp2 = std::env::temp_dir().join("fastphrase_eff_large_wordlist.txt");
                    fastphrase::import::download_to_file(url_wordfreq, &tmp1)?;
                    fastphrase::import::download_to_file(url_eff, &tmp2)?;

                    let bytes = std::fs::read(&tmp1)?;
                    let rows: Vec<(String, f64)> = serde_json::from_slice(&bytes)
                        .context("parsing wordfreq-en-25000-log.json")?;
                    let mut map: std::collections::HashMap<String, u64> =
                        std::collections::HashMap::new();
                    let scale = 1e9_f64;
                    for (w, ln_freq) in rows {
                        let freq = ln_freq.exp();
                        if !freq.is_finite() || freq <= 0.0 {
                            continue;
                        }
                        let count = (freq * scale).round().max(1.0) as u64;
                        map.insert(w, count);
                    }

                    let eff = std::fs::read_to_string(&tmp2)?;
                    for line in eff.lines() {
                        let line = line.trim();
                        if line.is_empty() || line.starts_with('#') {
                            continue;
                        }
                        // Format: "11111 abacus"
                        let mut it = line.split_whitespace();
                        let _code = it.next();
                        let Some(word) = it.next() else { continue };
                        map.entry(word.to_string()).or_insert(1);
                    }

                    if let Some(parent) = output.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    let mut items: Vec<(String, u64)> = map.into_iter().collect();
                    items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                    let mut f = std::io::BufWriter::new(std::fs::File::create(&output)?);
                    writeln!(
                        f,
                        "# hybrid: aparrish/wordfreq-en-25000 (scaled counts) + eff_large_wordlist tail (count=1)"
                    )?;
                    for (w, c) in items {
                        writeln!(f, "{w}\t{c}")?;
                    }
                    f.flush()?;

                    let _ = std::fs::remove_file(&tmp1);
                    let _ = std::fs::remove_file(&tmp2);
                    println!("downloaded: {}", output.display());
                    println!("corpus: {corpus}");
                    println!("url.wordfreq: {url_wordfreq}");
                    println!("url.eff: {url_eff}");
                }
                _ => anyhow::bail!(
                    "unknown corpus: {corpus} (try aparrish_wordfreq_en_25k or dwyl_words_alpha)"
                ),
            }
        }
        Command::UnionDatasets {
            datasets_dir,
            output,
            fetch_if_missing,
        } => {
            // Ensure raw files exist (optionally download).
            let cmu_path = datasets_dir.join("cmu").join("DSL-StrongPasswordData.csv");
            let bksd_dir = datasets_dir.join("bksd");
            let greyc_path = datasets_dir.join("greyc_web").join("webkeystroke.tar.gz");
            let laser_path = datasets_dir
                .join("cmu_laser2012")
                .join("DSL-Free-vs-Transcribed.zip");
            let keyrecs_free = datasets_dir.join("keyrecs").join("free-text.csv");
            let bksd_zips = [
                ("bksd_password_ar", "Password-Arabic.zip"),
                ("bksd_password_en", "Password-English.zip"),
                ("bksd_phrase_ar", "Phrase-Arabic.zip"),
                ("bksd_phrase_en", "Phrase-English.zip"),
            ];

            if fetch_if_missing {
                std::fs::create_dir_all(&datasets_dir)?;
                if !cmu_path.exists() {
                    let url = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv";
                    fastphrase::import::download_to_file(url, &cmu_path)
                        .with_context(|| format!("downloading CMU DSL CSV: {url}"))?;
                }
                for (_tag, name) in bksd_zips {
                    let p = bksd_dir.join(name);
                    if !p.exists() {
                        let base = "https://raw.githubusercontent.com/ntwaijry/BKSD/eb0b5907a090a9ba06d47f98cd63cf1c7f0e3339/dataset";
                        let url = format!("{base}/{name}");
                        fastphrase::import::download_to_file(&url, &p)
                            .with_context(|| format!("downloading BKSD file: {url}"))?;
                    }
                }
                if !greyc_path.exists() {
                    let url = greyc_web::default_url();
                    fastphrase::import::download_to_file(url, &greyc_path)
                        .with_context(|| format!("downloading GREYC web dataset: {url}"))?;
                }
                if !laser_path.exists() {
                    let url = cmu_laser2012::default_url();
                    fastphrase::import::download_to_file(url, &laser_path)
                        .with_context(|| format!("downloading CMU LASER-2012 zip: {url}"))?;
                }
                if !keyrecs_free.exists() {
                    let url = keyrecs::free_text_url();
                    fastphrase::import::download_to_file(url, &keyrecs_free)
                        .with_context(|| format!("downloading KeyRecs free-text: {url}"))?;
                }
            }

            // Build union.
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut f = std::io::BufWriter::new(std::fs::File::create(&output)?);

            let mut total = 0usize;
            let mut per_source: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();

            // CMU (character-labeled digraphs for a fixed password).
            if cmu_path.exists() {
                let bytes = std::fs::read(&cmu_path)
                    .with_context(|| format!("reading CMU CSV: {}", cmu_path.display()))?;
                let rows = cmu_dsl::parse_csv_bytes(&bytes)?;
                for row in rows {
                    serde_json::to_writer(&mut f, &row)?;
                    f.write_all(b"\n")?;
                    total += 1;
                    *per_source
                        .entry(row.source.clone().unwrap_or_else(|| "unknown".to_string()))
                        .or_insert(0) += 1;
                }
            }

            // BKSD (positional DD.*_*) from zips.
            for (tag, name) in bksd_zips {
                let p = bksd_dir.join(name);
                if !p.exists() {
                    continue;
                }
                let bytes = std::fs::read(&p)
                    .with_context(|| format!("reading BKSD zip: {}", p.display()))?;
                let rows = bksd::parse_zip_bytes(&bytes, tag)
                    .with_context(|| format!("parsing BKSD zip: {}", p.display()))?;
                for row in rows {
                    serde_json::to_writer(&mut f, &row)?;
                    f.write_all(b"\n")?;
                    total += 1;
                    *per_source
                        .entry(row.source.clone().unwrap_or_else(|| "unknown".to_string()))
                        .or_insert(0) += 1;
                }
            }

            // GREYC webkeystroke (character-labeled; diverse passwords/passphrases).
            if greyc_path.exists() {
                let tmp = std::env::temp_dir().join("fastphrase_greyc_web_import.jsonl");
                let wrote = greyc_web::write_jsonl_from_tar_gz_path(
                    &greyc_path,
                    &tmp,
                    greyc_web::ImportConfig {
                        include_passwords: true,
                        include_passphrases: true,
                        include_impostor: true,
                        max_rows: None,
                    },
                )
                .with_context(|| {
                    format!("importing GREYC web dataset: {}", greyc_path.display())
                })?;
                let imported = fastphrase::data::load_rows(&tmp)?;
                for row in imported {
                    serde_json::to_writer(&mut f, &row)?;
                    f.write_all(b"\n")?;
                    total += 1;
                    *per_source
                        .entry(row.source.clone().unwrap_or_else(|| "unknown".to_string()))
                        .or_insert(0) += 1;
                }
                println!("greyc_web_rows_appended: {wrote}");
                // Best-effort cleanup.
                let _ = std::fs::remove_file(&tmp);
            }

            // CMU LASER-2012 free/transcribed DD digraph features.
            if laser_path.exists() {
                let bytes = std::fs::read(&laser_path)
                    .with_context(|| format!("reading LASER-2012 zip: {}", laser_path.display()))?;
                let rows = cmu_laser2012::parse_zip_bytes(&bytes)
                    .with_context(|| format!("parsing LASER-2012 zip: {}", laser_path.display()))?;
                for row in rows {
                    serde_json::to_writer(&mut f, &row)?;
                    f.write_all(b"\n")?;
                    total += 1;
                    *per_source
                        .entry(row.source.clone().unwrap_or_else(|| "unknown".to_string()))
                        .or_insert(0) += 1;
                }
            }

            // KeyRecs free-text digraph features (CC-BY 4.0).
            if keyrecs_free.exists() {
                let bytes = std::fs::read(&keyrecs_free).with_context(|| {
                    format!("reading KeyRecs free-text: {}", keyrecs_free.display())
                })?;
                let rows = keyrecs::parse_free_text_csv_bytes(&bytes).with_context(|| {
                    format!("parsing KeyRecs free-text: {}", keyrecs_free.display())
                })?;
                for row in rows {
                    serde_json::to_writer(&mut f, &row)?;
                    f.write_all(b"\n")?;
                    total += 1;
                    *per_source
                        .entry(row.source.clone().unwrap_or_else(|| "unknown".to_string()))
                        .or_insert(0) += 1;
                }
            }

            f.flush()?;
            println!("rows_written: {total}");
            println!("output: {}", output.display());
            let mut ks: Vec<_> = per_source.into_iter().collect();
            ks.sort_by(|a, b| a.0.cmp(&b.0));
            for (k, v) in ks {
                println!("source.{k}: {v}");
            }
        }
        Command::AdaptModel {
            base_model,
            user_data,
            output_model,
            prior_count,
            min_new_count,
        } => {
            let base = fastphrase::model::DigraphModel::load_json(&base_model)
                .with_context(|| format!("loading base model: {}", base_model.display()))?;
            let user_rows = fastphrase::data::load_rows(&user_data)
                .with_context(|| format!("loading user data: {}", user_data.display()))?;
            let (tuned, stats) = adapt_digraph_model(
                &base,
                &user_rows,
                AdaptConfig {
                    prior_count,
                    min_new_count,
                },
            );
            tuned
                .save_json(&output_model)
                .with_context(|| format!("writing tuned model: {}", output_model.display()))?;
            println!("user_rows: {}", stats.user_rows);
            println!("user_digraph_obs: {}", stats.user_digraph_obs);
            println!("base_digraphs: {}", stats.base_digraphs);
            println!("user_distinct_digraphs: {}", stats.user_distinct_digraphs);
            println!("added_new_digraphs: {}", stats.added_new_digraphs);
            println!("tuned_digraphs: {}", stats.tuned_digraphs);
            println!("prior_count: {:.3}", stats.prior_count);
            println!("output_model: {}", output_model.display());
        }
        Command::RecordSession {
            output,
            reps,
            target,
            base_model,
            output_model,
            prior_count,
            min_new_count,
        } => {
            if reps == 0 {
                anyhow::bail!("reps must be >= 1");
            }
            if base_model.is_some() && output_model.is_none() {
                anyhow::bail!("when --base-model is set, you must also set --output-model");
            }

            let cfg = RecordConfig {
                target,
                abort_on_backspace: true,
                max_len: 200,
            };

            let mut recorded = 0usize;
            while recorded < reps {
                match record_once(&cfg)? {
                    RecordOutcome::Recorded(row) => {
                        append_row_jsonl(&output, &row)?;
                        recorded += 1;
                        eprintln!("(saved) {recorded}/{reps}");
                    }
                    RecordOutcome::Aborted => {
                        eprintln!("(skipped) not saved");
                    }
                }
            }

            println!("rows_appended: {}", recorded);
            println!("output: {}", output.display());

            if let (Some(base_model), Some(output_model)) = (base_model, output_model) {
                let base = fastphrase::model::DigraphModel::load_json(&base_model)
                    .with_context(|| format!("loading base model: {}", base_model.display()))?;
                let user_rows = fastphrase::data::load_rows(&output).with_context(|| {
                    format!("loading accumulated user data: {}", output.display())
                })?;
                let (tuned, stats) = adapt_digraph_model(
                    &base,
                    &user_rows,
                    AdaptConfig {
                        prior_count,
                        min_new_count,
                    },
                );
                tuned.save_json(&output_model)?;
                println!("adapted_model_written: {}", output_model.display());
                println!("adapt.user_rows_total: {}", stats.user_rows);
                println!("adapt.user_digraph_obs_total: {}", stats.user_digraph_obs);
                println!("adapt.prior_count: {:.3}", stats.prior_count);
            }
        }
        Command::EstimateSearch {
            model,
            wordlist,
            words,
            separator,
            allow_repeats,
            samples,
            seed,
            min_word_len,
            max_word_len,
            ascii_lower_only,
        } => {
            let model = AnyTimingModel::load_json(&model)
                .with_context(|| format!("loading model: {}", model.display()))?;
            let mut gcfg = GenerateConfig::default();
            gcfg.words = words;
            gcfg.separator = separator;
            gcfg.min_word_len = min_word_len;
            gcfg.max_word_len = max_word_len;
            gcfg.ascii_lower_only = ascii_lower_only;
            let wl = load_wordlist(&wordlist, &gcfg)
                .with_context(|| format!("loading wordlist: {}", wordlist.display()))?;

            let mut rng = rng_from_seed(seed);
            let (mean_ms, std_ms) = fastphrase::generate::estimate_avg_phrase_ms(
                &model, &wl, &gcfg, samples, &mut rng,
            )?;

            let m = wl.words.len() as f64;
            let n = words as u64;
            let combos = if allow_repeats {
                m.powi(words as i32)
            } else {
                // falling factorial m*(m-1)*...*(m-n+1)
                let mut p = 1.0f64;
                let mut k = 0u64;
                while k < n {
                    let term = m - (k as f64);
                    if term <= 0.0 {
                        p = 0.0;
                        break;
                    }
                    p *= term;
                    k += 1;
                }
                p
            };
            let bits = if combos > 0.0 { combos.log2() } else { 0.0 };
            let seconds = combos * (mean_ms / 1000.0);

            println!("usable_words: {}", wl.words.len());
            println!("words_per_phrase: {}", words);
            println!("allow_repeats: {}", allow_repeats);
            println!("search_space: {:.0}", combos);
            println!("entropy_bits: {:.3}", bits);
            println!("avg_ms_per_phrase: {:.3}", mean_ms);
            println!("std_ms_per_phrase: {:.3}", std_ms);
            println!("expected_seconds_to_enumerate: {:.3}", seconds);
            println!("expected_hours_to_enumerate: {:.3}", seconds / 3600.0);
            println!("expected_days_to_enumerate: {:.3}", seconds / 86400.0);
            println!("expected_years_to_enumerate: {:.6}", seconds / 31_557_600.0);
        }
        Command::RankWords {
            model,
            corpus,
            k,
            alpha,
            top,
            objective,
            ascii_lower_only,
            min_word_len,
            max_word_len,
        } => {
            let model = AnyTimingModel::load_json(&model)
                .with_context(|| format!("loading model: {}", model.display()))?;

            let counts = load_corpus_counts(&corpus, ascii_lower_only, min_word_len, max_word_len)
                .with_context(|| format!("loading corpus counts: {}", corpus.display()))?;

            let mut km = KGramModel::new(k, alpha)?;
            if objective == WordsetObjective::MsPerLmBit {
                km.train(counts.iter().map(|(w, &c)| (w.clone(), c)));
            }

            #[derive(Debug)]
            struct RowOut {
                word: String,
                ms: f64,
                bits: f64,
                ms_per_bit: f64,
                ms_times_bits: f64,
            }

            let mut rows = Vec::new();
            for w in counts.keys() {
                let ms = fastphrase::score::score_phrase(&model, w).predicted_ms as f64;
                let bits = if objective == WordsetObjective::MsPerLmBit {
                    km.surprisal_bits(w)
                } else {
                    0.0
                };
                let ms_per_bit = if bits > 0.0 { ms / bits } else { f64::INFINITY };
                rows.push(RowOut {
                    word: w.clone(),
                    ms,
                    bits,
                    ms_per_bit,
                    ms_times_bits: ms * bits,
                });
            }

            match objective {
                WordsetObjective::MsOnly => rows.sort_by(|a, b| a.ms.total_cmp(&b.ms)),
                WordsetObjective::MsPerLmBit => {
                    rows.sort_by(|a, b| a.ms_per_bit.total_cmp(&b.ms_per_bit))
                }
            }

            println!("objective: {:?}", objective);
            if objective == WordsetObjective::MsPerLmBit {
                println!("k: {k}");
                println!("alpha: {alpha}");
                println!("vocab_size: {}", km.vocab_size());
            }
            println!("unique_words: {}", rows.len());
            println!();
            println!("top_words:");
            for r in rows.into_iter().take(top) {
                match objective {
                    WordsetObjective::MsOnly => {
                        println!("{:>8.3} ms  {}", r.ms, r.word);
                    }
                    WordsetObjective::MsPerLmBit => {
                        println!(
                            "{:>8.3} ms/bit  {:>8.3} ms  {:>8.3} bits  {:>10.3} ms*bits  {}",
                            r.ms_per_bit, r.ms, r.bits, r.ms_times_bits, r.word
                        );
                    }
                }
            }
        }
        Command::ExportWordset {
            model,
            corpus,
            output,
            k,
            alpha,
            top,
            objective,
            min_hit_frac,
            min_vowels,
            tsv,
            ascii_lower_only,
            min_word_len,
            max_word_len,
        } => {
            let model = AnyTimingModel::load_json(&model)
                .with_context(|| format!("loading model: {}", model.display()))?;
            let counts = load_corpus_counts(&corpus, ascii_lower_only, min_word_len, max_word_len)
                .with_context(|| format!("loading corpus counts: {}", corpus.display()))?;

            let mut km = KGramModel::new(k, alpha)?;
            if objective == WordsetObjective::MsPerLmBit {
                km.train(counts.iter().map(|(w, &c)| (w.clone(), c)));
            }

            struct RowOut {
                word: String,
                ms: f64,
                bits: f64,
                ms_per_bit: f64,
            }

            let mut rows = Vec::new();
            let mut skipped_low_hit = 0u64;
            let mut skipped_low_vowels = 0u64;
            for w in counts.keys() {
                if min_vowels > 0 && count_vowels(w) < min_vowels {
                    skipped_low_vowels += 1;
                    continue;
                }
                let sc = fastphrase::score::score_phrase(&model, w);
                if sc.digraphs == 0 {
                    continue;
                }
                let hit_frac = (sc.hits as f64) / (sc.digraphs as f64);
                if min_hit_frac > 0.0 && hit_frac + 1e-12 < min_hit_frac {
                    skipped_low_hit += 1;
                    continue;
                }
                let ms = sc.predicted_ms as f64;
                let bits = if objective == WordsetObjective::MsPerLmBit {
                    km.surprisal_bits(w)
                } else {
                    0.0
                };
                let ms_per_bit = if bits > 0.0 { ms / bits } else { f64::INFINITY };
                rows.push(RowOut {
                    word: w.clone(),
                    ms,
                    bits,
                    ms_per_bit,
                });
            }
            match objective {
                WordsetObjective::MsOnly => rows.sort_by(|a, b| a.ms.total_cmp(&b.ms)),
                WordsetObjective::MsPerLmBit => {
                    rows.sort_by(|a, b| a.ms_per_bit.total_cmp(&b.ms_per_bit))
                }
            }

            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut f = std::io::BufWriter::new(std::fs::File::create(&output)?);
            if tsv {
                writeln!(f, "word\tms\tbits\tms_per_bit")?;
                for r in rows.into_iter().take(top) {
                    writeln!(
                        f,
                        "{}\t{:.6}\t{:.6}\t{:.6}",
                        r.word, r.ms, r.bits, r.ms_per_bit
                    )?;
                }
            } else {
                for r in rows.into_iter().take(top) {
                    writeln!(f, "{}", r.word)?;
                }
            }
            f.flush()?;

            println!("output: {}", output.display());
            println!("objective: {:?}", objective);
            println!("min_hit_frac: {:.3}", min_hit_frac);
            println!("min_vowels: {}", min_vowels);
            if min_hit_frac > 0.0 {
                println!("skipped_low_hit_frac: {}", skipped_low_hit);
            }
            if min_vowels > 0 {
                println!("skipped_low_vowels: {}", skipped_low_vowels);
            }
            println!("k: {k}");
            println!("alpha: {alpha}");
            println!("kept_words: {}", top.min(counts.len()));
            println!("vocab_size: {}", km.vocab_size());
        }
        Command::PlanPassphrase {
            model,
            corpus,
            output,
            words,
            target_bits,
            allow_repeats,
            separator,
            k,
            alpha,
            objective,
            min_hit_frac,
            min_vowels,
            samples,
            seed,
            ascii_lower_only,
            min_word_len,
            max_word_len,
        } => {
            if words == 0 {
                anyhow::bail!("words must be >= 1");
            }
            if !target_bits.is_finite() || target_bits <= 0.0 {
                anyhow::bail!("target_bits must be finite and > 0");
            }
            let model = AnyTimingModel::load_json(&model)
                .with_context(|| format!("loading model: {}", model.display()))?;

            let counts = load_corpus_counts(&corpus, ascii_lower_only, min_word_len, max_word_len)
                .with_context(|| format!("loading corpus counts: {}", corpus.display()))?;

            // Optional heuristic: a character k-gram “LM surprisal” for words.
            let mut km = KGramModel::new(k, alpha)?;
            if objective == WordsetObjective::MsPerLmBit {
                km.train(counts.iter().map(|(w, &c)| (w.clone(), c)));
            }

            // Score all candidate words and select top-N.
            struct RowOut {
                word: String,
                ms: f64,
                ms_per_bit: f64,
            }

            let mut rows = Vec::new();
            let mut skipped_low_hit = 0u64;
            let mut skipped_low_vowels = 0u64;
            for w in counts.keys() {
                if min_vowels > 0 && count_vowels(w) < min_vowels {
                    skipped_low_vowels += 1;
                    continue;
                }
                let sc = fastphrase::score::score_phrase(&model, w);
                if sc.digraphs == 0 {
                    continue;
                }
                let hit_frac = (sc.hits as f64) / (sc.digraphs as f64);
                if min_hit_frac > 0.0 && hit_frac + 1e-12 < min_hit_frac {
                    skipped_low_hit += 1;
                    continue;
                }
                let ms = sc.predicted_ms as f64;
                let ms_per_bit = if objective == WordsetObjective::MsPerLmBit {
                    let bits = km.surprisal_bits(w);
                    if bits > 0.0 {
                        ms / bits
                    } else {
                        f64::INFINITY
                    }
                } else {
                    // Not used for sorting in ms-only mode.
                    f64::INFINITY
                };
                rows.push(RowOut {
                    word: w.clone(),
                    ms,
                    ms_per_bit,
                });
            }
            if rows.is_empty() {
                anyhow::bail!(
                    "no candidate words after filtering (min_hit_frac={min_hit_frac}); try lowering it"
                );
            }
            match objective {
                WordsetObjective::MsOnly => rows.sort_by(|a, b| a.ms.total_cmp(&b.ms)),
                WordsetObjective::MsPerLmBit => {
                    rows.sort_by(|a, b| a.ms_per_bit.total_cmp(&b.ms_per_bit))
                }
            }

            // Minimum N needed for target entropy (uniform assumption).
            let needed_n = (2f64).powf(target_bits / (words as f64)).ceil() as usize;
            let n = needed_n.min(rows.len()).max(1);
            let chosen: Vec<String> = rows.into_iter().take(n).map(|r| r.word).collect();

            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            {
                let mut f = std::io::BufWriter::new(std::fs::File::create(&output)?);
                for w in &chosen {
                    writeln!(f, "{w}")?;
                }
                f.flush()?;
            }

            // Estimate avg ms/phrase for this planned wordset.
            let wl = wordlist_from_vec(chosen)?;
            let mut gcfg = GenerateConfig::default();
            gcfg.words = words;
            gcfg.separator = separator;
            // Wordlist already filtered; these only affect downstream printing.
            gcfg.ascii_lower_only = ascii_lower_only;
            gcfg.min_word_len = min_word_len;
            gcfg.max_word_len = max_word_len;

            let mut rng = rng_from_seed(seed);
            let (mean_ms, std_ms) = estimate_avg_phrase_ms(&model, &wl, &gcfg, samples, &mut rng)?;

            let m = wl.words.len() as f64;
            let combos = if allow_repeats {
                m.powi(words as i32)
            } else {
                let mut p = 1.0f64;
                for i in 0..(words as u64) {
                    let term = m - (i as f64);
                    if term <= 0.0 {
                        p = 0.0;
                        break;
                    }
                    p *= term;
                }
                p
            };
            let bits = if combos > 0.0 { combos.log2() } else { 0.0 };
            let seconds = combos * (mean_ms / 1000.0);

            println!("output: {}", output.display());
            println!("objective: {:?}", objective);
            println!("min_hit_frac: {:.3}", min_hit_frac);
            println!("min_vowels: {}", min_vowels);
            if min_hit_frac > 0.0 {
                println!("skipped_low_hit_frac: {}", skipped_low_hit);
            }
            if min_vowels > 0 {
                println!("skipped_low_vowels: {}", skipped_low_vowels);
            }
            println!("words_per_phrase: {}", words);
            println!("allow_repeats: {}", allow_repeats);
            println!("target_bits: {:.3}", target_bits);
            println!("planned_wordset_size: {}", wl.words.len());
            println!("achieved_entropy_bits: {:.3}", bits);
            if bits + 1e-9 < target_bits {
                println!("warning: target_bits_not_achievable_with_current_corpus_size");
                println!("hint: increase --words, relax filters, or use a larger corpus");
            }
            println!("avg_ms_per_phrase: {:.3}", mean_ms);
            println!("std_ms_per_phrase: {:.3}", std_ms);
            println!("expected_hours_to_enumerate: {:.3}", seconds / 3600.0);
            println!("expected_days_to_enumerate: {:.3}", seconds / 86400.0);
            if objective == WordsetObjective::MsPerLmBit {
                println!("k: {k}");
                println!("alpha: {alpha}");
                println!("vocab_size: {}", km.vocab_size());
            }
            println!("limits: uniform-wordset entropy assumption; ms/phrase estimated by sampling");
        }
        Command::BasePipeline {
            out_dir,
            words,
            target_bits,
            allow_repeats,
            objective,
            min_hit_frac,
            min_vowels,
            reuse_existing,
            debug,
            samples,
            top,
            seed,
        } => {
            std::fs::create_dir_all(&out_dir)?;
            let datasets_dir = out_dir.join("datasets");
            let union_jsonl = out_dir.join("union.jsonl");
            let model_json = out_dir.join("model_union.json");
            let corpus_path = out_dir.join("corpus.txt");
            let wordset_path = out_dir.join("wordset.txt");

            // 1) union datasets (downloads if missing)
            // (call our own subcommand implementation inline)
            {
                if reuse_existing && union_jsonl.exists() {
                    if debug {
                        println!(
                            "base-pipeline: reusing union_jsonl: {}",
                            union_jsonl.display()
                        );
                    }
                } else {
                    if debug {
                        println!(
                            "base-pipeline: building union_jsonl: {}",
                            union_jsonl.display()
                        );
                    }
                    let t0 = std::time::Instant::now();
                    // reuse logic: just call UnionDatasets path directly by duplicating minimal code
                    let cmu_path = datasets_dir.join("cmu").join("DSL-StrongPasswordData.csv");
                    let bksd_dir = datasets_dir.join("bksd");
                    let greyc_path = datasets_dir.join("greyc_web").join("webkeystroke.tar.gz");
                    let laser_path = datasets_dir
                        .join("cmu_laser2012")
                        .join("DSL-Free-vs-Transcribed.zip");
                    let keyrecs_free = datasets_dir.join("keyrecs").join("free-text.csv");
                    let bksd_zips = [
                        ("bksd_password_ar", "Password-Arabic.zip"),
                        ("bksd_password_en", "Password-English.zip"),
                        ("bksd_phrase_ar", "Phrase-Arabic.zip"),
                        ("bksd_phrase_en", "Phrase-English.zip"),
                    ];
                    std::fs::create_dir_all(&datasets_dir)?;
                    if !cmu_path.exists() {
                        let url = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv";
                        fastphrase::import::download_to_file(url, &cmu_path)?;
                    }
                    for (_tag, name) in bksd_zips {
                        let p = bksd_dir.join(name);
                        if !p.exists() {
                            let base = "https://raw.githubusercontent.com/ntwaijry/BKSD/eb0b5907a090a9ba06d47f98cd63cf1c7f0e3339/dataset";
                            let url = format!("{base}/{name}");
                            fastphrase::import::download_to_file(&url, &p)?;
                        }
                    }
                    if !greyc_path.exists() {
                        let url = greyc_web::default_url();
                        fastphrase::import::download_to_file(url, &greyc_path)?;
                    }
                    if !laser_path.exists() {
                        let url = cmu_laser2012::default_url();
                        fastphrase::import::download_to_file(url, &laser_path)?;
                    }
                    if !keyrecs_free.exists() {
                        let url = keyrecs::free_text_url();
                        fastphrase::import::download_to_file(url, &keyrecs_free)?;
                    }
                    let mut f = std::io::BufWriter::new(std::fs::File::create(&union_jsonl)?);
                    let bytes = std::fs::read(&cmu_path)?;
                    let rows = cmu_dsl::parse_csv_bytes(&bytes)?;
                    for row in rows {
                        serde_json::to_writer(&mut f, &row)?;
                        f.write_all(b"\n")?;
                    }
                    for (tag, name) in bksd_zips {
                        let p = bksd_dir.join(name);
                        let bytes = std::fs::read(&p)?;
                        let rows = bksd::parse_zip_bytes(&bytes, tag)?;
                        for row in rows {
                            serde_json::to_writer(&mut f, &row)?;
                            f.write_all(b"\n")?;
                        }
                    }
                    // GREYC webkeystroke
                    if greyc_path.exists() {
                        let tmp =
                            std::env::temp_dir().join("fastphrase_greyc_web_basepipeline.jsonl");
                        let _wrote = greyc_web::write_jsonl_from_tar_gz_path(
                            &greyc_path,
                            &tmp,
                            greyc_web::ImportConfig::default(),
                        )?;
                        let imported = fastphrase::data::load_rows(&tmp)?;
                        for row in imported {
                            serde_json::to_writer(&mut f, &row)?;
                            f.write_all(b"\n")?;
                        }
                        let _ = std::fs::remove_file(&tmp);
                    }
                    // LASER-2012
                    if laser_path.exists() {
                        let bytes = std::fs::read(&laser_path)?;
                        let rows = cmu_laser2012::parse_zip_bytes(&bytes)?;
                        for row in rows {
                            serde_json::to_writer(&mut f, &row)?;
                            f.write_all(b"\n")?;
                        }
                    }
                    // KeyRecs free-text
                    if keyrecs_free.exists() {
                        let bytes = std::fs::read(&keyrecs_free)?;
                        let rows = keyrecs::parse_free_text_csv_bytes(&bytes)?;
                        for row in rows {
                            serde_json::to_writer(&mut f, &row)?;
                            f.write_all(b"\n")?;
                        }
                    }
                    f.flush()?;
                    if debug {
                        println!("base-pipeline: union_jsonl done in {:.2?}", t0.elapsed());
                    }
                }
            }

            // 2) fit base model
            {
                if reuse_existing && model_json.exists() {
                    if debug {
                        println!("base-pipeline: reusing model: {}", model_json.display());
                    }
                } else {
                    if debug {
                        println!("base-pipeline: fitting model: {}", model_json.display());
                    }
                    let t0 = std::time::Instant::now();
                    let rows = fastphrase::data::load_rows(&union_jsonl)?;
                    if debug {
                        println!("base-pipeline: loaded rows: {}", rows.len());
                    }
                    let (m, _stats) = fit_digraph_model(
                        &rows,
                        FitConfig {
                            min_count: 3,
                            // Robust default for base: clamp pauses/outliers.
                            clamp_dt_ms: Some(2000.0),
                        },
                    );
                    m.save_json(&model_json)?;
                    if debug {
                        println!("base-pipeline: fit done in {:.2?}", t0.elapsed());
                    }
                }
            }

            // 3) download corpus (hybrid: frequency-weighted head + long tail vocab)
            {
                if reuse_existing && corpus_path.exists() {
                    if debug {
                        println!("base-pipeline: reusing corpus: {}", corpus_path.display());
                    }
                } else {
                    if debug {
                        println!("base-pipeline: building corpus: {}", corpus_path.display());
                    }
                    let t0 = std::time::Instant::now();
                    // Inline the `hybrid_wordfreq25k_plus_dwyl` logic.
                    let url_wordfreq = "https://raw.githubusercontent.com/aparrish/wordfreq-en-25000/master/wordfreq-en-25000-log.json";
                    let url_dwyl = "https://raw.githubusercontent.com/dwyl/english-words/20f5cc9b3f0ccc8ce45d814c532b7c2031bba31c/words_alpha.txt";
                    let tmp1 =
                        std::env::temp_dir().join("fastphrase_wordfreq_en_25k_basepipeline.json");
                    let tmp2 = std::env::temp_dir().join("fastphrase_words_alpha_basepipeline.txt");
                    fastphrase::import::download_to_file(url_wordfreq, &tmp1)?;
                    fastphrase::import::download_to_file(url_dwyl, &tmp2)?;

                    let bytes = std::fs::read(&tmp1)?;
                    let rows: Vec<(String, f64)> = serde_json::from_slice(&bytes)?;
                    let mut map: std::collections::HashMap<String, u64> =
                        std::collections::HashMap::new();
                    let scale = 1e9_f64;
                    for (w, ln_freq) in rows {
                        let freq = ln_freq.exp();
                        if !freq.is_finite() || freq <= 0.0 {
                            continue;
                        }
                        let count = (freq * scale).round().max(1.0) as u64;
                        map.insert(w, count);
                    }
                    let dwyl = std::fs::read_to_string(&tmp2)?;
                    for line in dwyl.lines() {
                        let w = line.trim();
                        if w.is_empty() || w.starts_with('#') {
                            continue;
                        }
                        map.entry(w.to_string()).or_insert(1);
                    }
                    let mut items: Vec<(String, u64)> = map.into_iter().collect();
                    items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

                    let mut f = std::io::BufWriter::new(std::fs::File::create(&corpus_path)?);
                    writeln!(f, "# hybrid: aparrish/wordfreq-en-25000 (scaled counts) + dwyl/words_alpha tail (count=1)")?;
                    for (w, c) in items {
                        writeln!(f, "{w}\t{c}")?;
                    }
                    f.flush()?;

                    let _ = std::fs::remove_file(&tmp1);
                    let _ = std::fs::remove_file(&tmp2);
                    if debug {
                        println!("base-pipeline: corpus done in {:.2?}", t0.elapsed());
                    }
                }
            }

            // 4) plan a wordset for target entropy
            {
                if reuse_existing && wordset_path.exists() {
                    if debug {
                        println!("base-pipeline: reusing wordset: {}", wordset_path.display());
                    }
                } else {
                    if debug {
                        println!(
                            "base-pipeline: planning wordset: {}",
                            wordset_path.display()
                        );
                    }
                    let t0 = std::time::Instant::now();
                    let cmd = Command::PlanPassphrase {
                        model: model_json.clone(),
                        corpus: corpus_path.clone(),
                        output: wordset_path.clone(),
                        words,
                        target_bits,
                        allow_repeats,
                        separator: " ".to_string(),
                        k: 3,
                        alpha: 0.5,
                        objective,
                        min_hit_frac,
                        min_vowels,
                        samples: 5000,
                        seed,
                        ascii_lower_only: true,
                        min_word_len: 3,
                        max_word_len: 12,
                    };
                    // execute by tail-calling the match arm logic: simplest is to recursively run a small helper.
                    run_plan_passphrase(cmd)?;
                    if debug {
                        println!("base-pipeline: plan done in {:.2?}", t0.elapsed());
                    }
                }
            }

            // 5) generate examples + estimate enumeration time using the planned wordset
            {
                let model = AnyTimingModel::load_json(&model_json)?;
                let mut gcfg = GenerateConfig::default();
                gcfg.words = words;
                gcfg.separator = " ".to_string();
                gcfg.samples = samples;
                gcfg.top_k = top;
                gcfg.ascii_lower_only = true;
                gcfg.min_word_len = 3;
                gcfg.max_word_len = 12;

                let wl = load_wordlist(&wordset_path, &gcfg)?;
                let mut rng = rng_from_seed(seed);
                let (best, _gen_stats) = generate_top(&model, &wl, &gcfg, &mut rng);
                let (mean_ms, std_ms) = estimate_avg_phrase_ms(&model, &wl, &gcfg, 5000, &mut rng)?;

                let m = wl.words.len() as f64;
                let combos = if allow_repeats {
                    m.powi(words as i32)
                } else {
                    let mut p = 1.0f64;
                    for i in 0..(words as u64) {
                        let term = m - (i as f64);
                        if term <= 0.0 {
                            p = 0.0;
                            break;
                        }
                        p *= term;
                    }
                    p
                };
                let bits = if combos > 0.0 { combos.log2() } else { 0.0 };
                let seconds = combos * (mean_ms / 1000.0);

                println!("out_dir: {}", out_dir.display());
                println!("union_jsonl: {}", union_jsonl.display());
                println!("model: {}", model_json.display());
                println!("corpus: {}", corpus_path.display());
                println!("wordset: {}", wordset_path.display());
                println!("planned_wordset_size: {}", wl.words.len());
                println!("achieved_entropy_bits: {:.3}", bits);
                println!("avg_ms_per_phrase: {:.3}", mean_ms);
                println!("std_ms_per_phrase: {:.3}", std_ms);
                println!("expected_days_to_enumerate: {:.3}", seconds / 86400.0);
                println!();
                println!("top_generated:");
                for (i, c) in best.iter().enumerate() {
                    println!(
                        "{:>2}. {:>8.3} ms  {}",
                        i + 1,
                        c.score.predicted_ms,
                        c.phrase
                    );
                }
            }
        }
        Command::PersonalizePipeline {
            base_dir,
            user_data,
            out_dir,
            words,
            target_bits,
            allow_repeats,
            objective,
            min_hit_frac,
            min_vowels,
            reuse_existing,
            debug,
            samples,
            top,
            seed,
            prior_count,
            min_new_count,
        } => {
            if words == 0 {
                anyhow::bail!("words must be >= 1");
            }
            if !target_bits.is_finite() || target_bits <= 0.0 {
                anyhow::bail!("target_bits must be finite and > 0");
            }

            // Inputs produced by base-pipeline.
            let base_model = base_dir.join("model_union.json");
            let corpus = base_dir.join("corpus.txt");
            if !base_model.exists() {
                anyhow::bail!(
                    "missing base model: {} (run base-pipeline first)",
                    base_model.display()
                );
            }
            if !corpus.exists() {
                anyhow::bail!(
                    "missing corpus: {} (run base-pipeline first)",
                    corpus.display()
                );
            }
            if !user_data.exists() {
                anyhow::bail!(
                    "missing user data: {} (run record-session first)",
                    user_data.display()
                );
            }

            std::fs::create_dir_all(&out_dir)?;
            let tuned_model_path = out_dir.join("model_user.json");
            let wordset_path = out_dir.join("wordset_user.txt");

            // 1) adapt base model with accumulated user rows
            let base = fastphrase::model::DigraphModel::load_json(&base_model)
                .with_context(|| format!("loading base model: {}", base_model.display()))?;
            let user_rows = fastphrase::data::load_rows(&user_data)
                .with_context(|| format!("loading user data: {}", user_data.display()))?;
            let (tuned, stats) = adapt_digraph_model(
                &base,
                &user_rows,
                AdaptConfig {
                    prior_count,
                    min_new_count,
                },
            );
            if reuse_existing && tuned_model_path.exists() {
                if debug {
                    println!(
                        "personalize-pipeline: reusing tuned model: {}",
                        tuned_model_path.display()
                    );
                }
            } else {
                tuned.save_json(&tuned_model_path)?;
            }
            // Also build a personalized model that can impute unseen digraphs via backoff.
            let personalized_path = out_dir.join("model_personalized.json");
            let (pm, _adapt_stats2, _bstats) = fastphrase::timing::build_personalized_model(
                &base,
                &user_rows,
                AdaptConfig {
                    prior_count,
                    min_new_count,
                },
                5,
            );
            if reuse_existing && personalized_path.exists() {
                if debug {
                    println!(
                        "personalize-pipeline: reusing personalized model: {}",
                        personalized_path.display()
                    );
                }
            } else {
                pm.save_json(&personalized_path)?;
            }

            // 2) plan a personalized wordset
            {
                if reuse_existing && wordset_path.exists() {
                    if debug {
                        println!(
                            "personalize-pipeline: reusing wordset: {}",
                            wordset_path.display()
                        );
                    }
                } else {
                    let cmd = Command::PlanPassphrase {
                        model: personalized_path.clone(),
                        corpus: corpus.clone(),
                        output: wordset_path.clone(),
                        words,
                        target_bits,
                        allow_repeats,
                        separator: " ".to_string(),
                        k: 3,
                        alpha: 0.5,
                        objective,
                        min_hit_frac,
                        min_vowels,
                        samples: 5000,
                        seed,
                        ascii_lower_only: true,
                        min_word_len: 3,
                        max_word_len: 12,
                    };
                    run_plan_passphrase(cmd)?;
                }
            }

            // 3) generate examples + estimate enumeration time using the planned wordset
            {
                let model = AnyTimingModel::load_json(&personalized_path)?;
                let mut gcfg = GenerateConfig::default();
                gcfg.words = words;
                gcfg.separator = " ".to_string();
                gcfg.samples = samples;
                gcfg.top_k = top;
                gcfg.ascii_lower_only = true;
                gcfg.min_word_len = 3;
                gcfg.max_word_len = 12;

                let wl = load_wordlist(&wordset_path, &gcfg)?;
                let mut rng = rng_from_seed(seed);
                let (best, _gen_stats) = generate_top(&model, &wl, &gcfg, &mut rng);
                let (mean_ms, std_ms) = estimate_avg_phrase_ms(&model, &wl, &gcfg, 5000, &mut rng)?;

                let m = wl.words.len() as f64;
                let combos = if allow_repeats {
                    m.powi(words as i32)
                } else {
                    let mut p = 1.0f64;
                    for i in 0..(words as u64) {
                        let term = m - (i as f64);
                        if term <= 0.0 {
                            p = 0.0;
                            break;
                        }
                        p *= term;
                    }
                    p
                };
                let bits = if combos > 0.0 { combos.log2() } else { 0.0 };
                let seconds = combos * (mean_ms / 1000.0);

                println!("base_dir: {}", base_dir.display());
                println!("user_data: {}", user_data.display());
                println!("out_dir: {}", out_dir.display());
                println!("tuned_model: {}", tuned_model_path.display());
                println!("personalized_model: {}", personalized_path.display());
                println!("wordset: {}", wordset_path.display());
                println!("adapt.user_rows_total: {}", stats.user_rows);
                println!("adapt.user_digraph_obs_total: {}", stats.user_digraph_obs);
                println!("adapt.prior_count: {:.3}", stats.prior_count);
                println!("planned_wordset_size: {}", wl.words.len());
                println!("achieved_entropy_bits: {:.3}", bits);
                println!("avg_ms_per_phrase: {:.3}", mean_ms);
                println!("std_ms_per_phrase: {:.3}", std_ms);
                println!("expected_days_to_enumerate: {:.3}", seconds / 86400.0);
                println!();
                println!("top_generated:");
                for (i, c) in best.iter().enumerate() {
                    println!(
                        "{:>2}. {:>8.3} ms  {}",
                        i + 1,
                        c.score.predicted_ms,
                        c.phrase
                    );
                }
            }
        }
        Command::EvalModel {
            input,
            min_count,
            test_frac,
            seed,
            source_prefix,
            show_worst,
            worst_k,
            clamp_dt_ms,
        } => {
            fn quantiles(mut v: Vec<f64>) -> (usize, f64, f64, f64, f64, f64, f64) {
                if v.is_empty() {
                    return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                }
                v.sort_by(|a, b| a.total_cmp(b));
                let n = v.len();
                let at = |p: f64| -> f64 {
                    let idx = ((p * ((n - 1) as f64)).round() as usize).min(n - 1);
                    v[idx]
                };
                (n, v[0], at(0.50), at(0.90), at(0.95), at(0.99), v[n - 1])
            }

            if !(0.0..1.0).contains(&test_frac) {
                anyhow::bail!("test_frac must be in (0,1)");
            }

            let mut rows = fastphrase::data::load_rows(&input)
                .with_context(|| format!("loading dataset: {}", input.display()))?;
            if let Some(pfx) = &source_prefix {
                rows.retain(|r| r.source.as_deref().unwrap_or("").starts_with(pfx));
            }
            if rows.len() < 5 {
                anyhow::bail!(
                    "not enough rows to evaluate after filtering: {}",
                    rows.len()
                );
            }

            let mut rng = rng_from_seed(seed);
            // Shuffle then split.
            {
                use rand::seq::SliceRandom as _;
                rows.shuffle(&mut rng);
            }
            let n_total = rows.len();
            let n_test = ((n_total as f64) * test_frac)
                .round()
                .clamp(1.0, (n_total - 1) as f64) as usize;
            let (test_rows, train_rows) = rows.split_at(n_test);

            let (model, fit_stats) = fit_digraph_model(
                train_rows,
                FitConfig {
                    min_count,
                    clamp_dt_ms,
                },
            );

            #[derive(Default, Debug, Clone)]
            struct Agg {
                n: u64,
                sum_abs: f64,
                sum_sq: f64,
                sum_y: f64,
                sum_y2: f64,
                sum_yhat: f64,
                sum_yhat2: f64,
                sum_y_yhat: f64,
                seen: u64,
                unseen: u64,
            }
            impl Agg {
                fn push(&mut self, y: f64, yhat: f64, seen: bool) {
                    let e = yhat - y;
                    self.n += 1;
                    self.sum_abs += e.abs();
                    self.sum_sq += e * e;
                    self.sum_y += y;
                    self.sum_y2 += y * y;
                    self.sum_yhat += yhat;
                    self.sum_yhat2 += yhat * yhat;
                    self.sum_y_yhat += y * yhat;
                    if seen {
                        self.seen += 1;
                    } else {
                        self.unseen += 1;
                    }
                }
                fn mae(&self) -> f64 {
                    if self.n == 0 {
                        0.0
                    } else {
                        self.sum_abs / (self.n as f64)
                    }
                }
                fn rmse(&self) -> f64 {
                    if self.n == 0 {
                        0.0
                    } else {
                        (self.sum_sq / (self.n as f64)).sqrt()
                    }
                }
                fn corr(&self) -> f64 {
                    // Pearson correlation between y and yhat.
                    let n = self.n as f64;
                    if n < 2.0 {
                        return 0.0;
                    }
                    let cov = (self.sum_y_yhat / n) - (self.sum_y / n) * (self.sum_yhat / n);
                    let vy = (self.sum_y2 / n) - (self.sum_y / n).powi(2);
                    let vyh = (self.sum_yhat2 / n) - (self.sum_yhat / n).powi(2);
                    if vy <= 0.0 || vyh <= 0.0 {
                        return 0.0;
                    }
                    cov / (vy.sqrt() * vyh.sqrt())
                }
            }

            let mut dig_all = Agg::default();
            let mut dig_seen = Agg::default();
            let mut dig_unseen = Agg::default();
            let mut phrase_all = Agg::default();
            let mut abs_err_digraph: Vec<f64> = Vec::new();
            let mut abs_err_phrase: Vec<f64> = Vec::new();

            let mut by_source_digraph: std::collections::HashMap<String, Agg> =
                std::collections::HashMap::new();
            let mut by_source_phrase: std::collections::HashMap<String, Agg> =
                std::collections::HashMap::new();

            #[derive(Debug, Clone)]
            struct WorstPhrase {
                abs_err_ms: f64,
                true_ms: f64,
                pred_ms: f64,
                source: String,
                phrase: String,
            }
            let mut worst: Vec<WorstPhrase> = Vec::new();
            let mut skipped_invalid = 0u64;

            for row in test_rows {
                let grams = fastphrase::score::graphemes_normalized(&row.phrase);
                let n = grams.len();
                if n < 2 {
                    continue;
                }
                if row.digraph_dt_ms.len() != n.saturating_sub(1) {
                    continue;
                }
                if !row.digraph_dt_ms.iter().all(|x| x.is_finite() && *x >= 0.0) {
                    skipped_invalid += 1;
                    continue;
                }

                let mut true_total = 0.0f64;
                let mut pred_total = 0.0f64;
                let src = row.source.clone().unwrap_or_else(|| "unknown".to_string());

                for i in 0..(n - 1) {
                    let mut y = row.digraph_dt_ms[i] as f64;
                    if let Some(c) = clamp_dt_ms {
                        y = y.min(c as f64);
                    }
                    let yhat = model.mean_ms_for(&grams[i], &grams[i + 1]) as f64;
                    let seen = model.has_digraph(&grams[i], &grams[i + 1]);
                    abs_err_digraph.push((yhat - y).abs());

                    dig_all.push(y, yhat, seen);
                    if seen {
                        dig_seen.push(y, yhat, true);
                    } else {
                        dig_unseen.push(y, yhat, false);
                    }
                    by_source_digraph
                        .entry(src.clone())
                        .or_default()
                        .push(y, yhat, seen);

                    true_total += y;
                    pred_total += yhat;
                }

                phrase_all.push(true_total, pred_total, true);
                by_source_phrase
                    .entry(src.clone())
                    .or_default()
                    .push(true_total, pred_total, true);
                abs_err_phrase.push((pred_total - true_total).abs());

                if show_worst {
                    let abs_err = (pred_total - true_total).abs();
                    worst.push(WorstPhrase {
                        abs_err_ms: abs_err,
                        true_ms: true_total,
                        pred_ms: pred_total,
                        source: src,
                        phrase: row.phrase.clone(),
                    });
                }
            }

            if show_worst {
                worst.sort_by(|a, b| b.abs_err_ms.total_cmp(&a.abs_err_ms));
            }

            println!("input: {}", input.display());
            if let Some(pfx) = source_prefix {
                println!("source_prefix: {pfx}");
            }
            println!("rows.total: {n_total}");
            println!("rows.train: {}", train_rows.len());
            println!("rows.test: {}", test_rows.len());
            println!("rows.test_skipped_invalid: {}", skipped_invalid);
            println!("fit.min_count: {min_count}");
            println!("fit.global_mean_ms: {:.3}", fit_stats.global_mean_ms);
            println!("fit.kept_digraphs: {}", fit_stats.kept_digraphs);
            println!("fit.total_digraph_obs: {}", fit_stats.total_digraph_obs);
            if let Some(c) = clamp_dt_ms {
                println!("fit.clamp_dt_ms: {:.3}", c);
            }
            println!();

            println!("digraph_eval (ms):");
            println!("  n: {}", dig_all.n);
            println!("  mae: {:.3}", dig_all.mae());
            println!("  rmse: {:.3}", dig_all.rmse());
            println!("  corr: {:.3}", dig_all.corr());
            println!(
                "  mae_over_global_mean: {:.6}",
                if fit_stats.global_mean_ms > 0.0 {
                    dig_all.mae() / (fit_stats.global_mean_ms as f64)
                } else {
                    0.0
                }
            );
            println!("  seen_digraph_n: {}", dig_all.seen);
            println!("  unseen_digraph_n: {}", dig_all.unseen);
            println!(
                "  seen_digraph_frac: {:.6}",
                if dig_all.n == 0 {
                    0.0
                } else {
                    (dig_all.seen as f64) / (dig_all.n as f64)
                }
            );
            println!(
                "  unseen_digraph_frac: {:.6}",
                if dig_all.n == 0 {
                    0.0
                } else {
                    (dig_all.unseen as f64) / (dig_all.n as f64)
                }
            );
            if dig_seen.n > 0 {
                println!("  seen.mae: {:.3}", dig_seen.mae());
                println!("  seen.rmse: {:.3}", dig_seen.rmse());
            }
            if dig_unseen.n > 0 {
                println!("  unseen.n: {}", dig_unseen.n);
                println!("  unseen.mae: {:.3}", dig_unseen.mae());
                println!("  unseen.rmse: {:.3}", dig_unseen.rmse());
            }
            println!();

            println!("phrase_eval (sum ms):");
            println!("  n: {}", phrase_all.n);
            println!("  mae: {:.3}", phrase_all.mae());
            println!("  rmse: {:.3}", phrase_all.rmse());
            println!("  corr: {:.3}", phrase_all.corr());
            println!();

            let (_en, emin, e50, e90, e95, e99, emax) = quantiles(abs_err_digraph);
            println!(
                "digraph_abs_err_quantiles(min/p50/p90/p95/p99/max): {:.3}/{:.3}/{:.3}/{:.3}/{:.3}/{:.3}",
                emin, e50, e90, e95, e99, emax
            );
            let (_pn, pmin, p50, p90, p95, p99, pmax) = quantiles(abs_err_phrase);
            println!(
                "phrase_abs_err_quantiles(min/p50/p90/p95/p99/max): {:.3}/{:.3}/{:.3}/{:.3}/{:.3}/{:.3}",
                pmin, p50, p90, p95, p99, pmax
            );
            println!();

            // Source breakdown (top few sources by sample size).
            let mut sources: Vec<(String, u64)> = by_source_digraph
                .iter()
                .map(|(k, a)| (k.clone(), a.n))
                .collect();
            sources.sort_by(|a, b| b.1.cmp(&a.1));

            println!("by_source (digraph mae/rmse, phrase mae/rmse):");
            for (k, _n) in sources.iter().take(12) {
                let da = by_source_digraph.get(k).unwrap();
                let (phr_n, phr_mae, phr_rmse) = by_source_phrase
                    .get(k)
                    .map(|pa| (pa.n, pa.mae(), pa.rmse()))
                    .unwrap_or((0, 0.0, 0.0));
                println!(
                    "  {k}: dig_n={} dig_mae={:.3} dig_rmse={:.3} | phr_n={} phr_mae={:.3} phr_rmse={:.3}",
                    da.n,
                    da.mae(),
                    da.rmse(),
                    phr_n,
                    phr_mae,
                    phr_rmse
                );
            }

            if show_worst {
                println!();
                println!("worst_phrases_by_abs_error (top {}):", worst_k);
                for w in worst.into_iter().take(worst_k) {
                    println!(
                        "  abs_err_ms={:>9.3}  true={:>9.3}  pred={:>9.3}  src={}  phrase={}",
                        w.abs_err_ms, w.true_ms, w.pred_ms, w.source, w.phrase
                    );
                }
            }
        }
        Command::DatasetStats {
            input,
            source_prefix,
            min_digraphs,
        } => {
            let mut rows = fastphrase::data::load_rows(&input)
                .with_context(|| format!("loading dataset: {}", input.display()))?;
            if let Some(pfx) = &source_prefix {
                rows.retain(|r| r.source.as_deref().unwrap_or("").starts_with(pfx));
            }

            let mut dt_all: Vec<f32> = Vec::new();
            let mut phrase_all: Vec<f32> = Vec::new();
            let mut per_source: std::collections::HashMap<String, (Vec<f32>, Vec<f32>)> =
                std::collections::HashMap::new();

            let mut kept_rows = 0usize;
            let mut skipped_rows = 0usize;

            for r in rows {
                let n_d = r.digraph_dt_ms.len();
                if n_d < min_digraphs {
                    skipped_rows += 1;
                    continue;
                }
                if !r.digraph_dt_ms.iter().all(|x| x.is_finite() && *x >= 0.0) {
                    skipped_rows += 1;
                    continue;
                }
                let total: f32 = r.digraph_dt_ms.iter().copied().sum();
                dt_all.extend(r.digraph_dt_ms.iter().copied());
                phrase_all.push(total);
                let src = r.source.unwrap_or_else(|| "unknown".to_string());
                let (dts, totals) = per_source
                    .entry(src)
                    .or_insert_with(|| (Vec::new(), Vec::new()));
                dts.extend(r.digraph_dt_ms);
                totals.push(total);
                kept_rows += 1;
            }

            fn quantiles(mut v: Vec<f32>) -> (usize, f32, f32, f32, f32, f32, f32) {
                if v.is_empty() {
                    return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                }
                v.sort_by(|a, b| a.total_cmp(b));
                let n = v.len();
                let at = |p: f32| -> f32 {
                    let idx = ((p * ((n - 1) as f32)).round() as usize).min(n - 1);
                    v[idx]
                };
                (n, v[0], at(0.50), at(0.90), at(0.95), at(0.99), v[n - 1])
            }

            fn count_gt(v: &[f32], thr: f32) -> usize {
                v.iter().filter(|x| **x > thr).count()
            }

            println!("input: {}", input.display());
            if let Some(pfx) = source_prefix {
                println!("source_prefix: {pfx}");
            }
            println!("min_digraphs: {min_digraphs}");
            println!("rows.kept: {kept_rows}");
            println!("rows.skipped: {skipped_rows}");
            println!();

            let (n, min, p50, p90, p95, p99, max) = quantiles(dt_all.clone());
            println!("digraph_dt_ms overall:");
            println!("  n: {n}");
            println!("  min/p50/p90/p95/p99/max: {min:.3} / {p50:.3} / {p90:.3} / {p95:.3} / {p99:.3} / {max:.3}");
            println!("  count_gt_1000ms: {}", count_gt(&dt_all, 1000.0));
            println!("  count_gt_2000ms: {}", count_gt(&dt_all, 2000.0));
            println!("  count_gt_5000ms: {}", count_gt(&dt_all, 5000.0));
            println!();

            let (n2, min2, p50_2, p90_2, p95_2, p99_2, max2) = quantiles(phrase_all.clone());
            println!("phrase_total_ms overall:");
            println!("  n: {n2}");
            println!("  min/p50/p90/p95/p99/max: {min2:.3} / {p50_2:.3} / {p90_2:.3} / {p95_2:.3} / {p99_2:.3} / {max2:.3}");
            println!("  count_gt_10000ms: {}", count_gt(&phrase_all, 10_000.0));
            println!("  count_gt_60000ms: {}", count_gt(&phrase_all, 60_000.0));
            println!("  count_gt_300000ms: {}", count_gt(&phrase_all, 300_000.0));
            println!();

            let mut keys: Vec<String> = per_source.keys().cloned().collect();
            keys.sort();
            println!("per_source:");
            for k in keys {
                let (dts, totals) = per_source.remove(&k).unwrap();
                let (dn, dmin, dp50, dp90, dp95, dp99, dmax) = quantiles(dts.clone());
                let (tn, tmin, tp50, tp90, tp95, tp99, tmax) = quantiles(totals.clone());
                println!("  {k}:");
                println!("    digraph_dt_ms: n={dn} min/p50/p90/p95/p99/max={dmin:.3}/{dp50:.3}/{dp90:.3}/{dp95:.3}/{dp99:.3}/{dmax:.3} gt2000ms={}", count_gt(&dts, 2000.0));
                println!("    phrase_total_ms: n={tn} min/p50/p90/p95/p99/max={tmin:.3}/{tp50:.3}/{tp90:.3}/{tp95:.3}/{tp99:.3}/{tmax:.3} gt60000ms={}", count_gt(&totals, 60_000.0));
            }
        }
        Command::BuildPersonalizedModel {
            base_model,
            user_data,
            output_model,
            prior_count,
            min_new_count,
            min_backoff_count,
        } => {
            let base = fastphrase::model::DigraphModel::load_json(&base_model)
                .with_context(|| format!("loading base model: {}", base_model.display()))?;
            let user_rows = fastphrase::data::load_rows(&user_data)
                .with_context(|| format!("loading user data: {}", user_data.display()))?;
            let (p, adapt_stats, bstats) = fastphrase::timing::build_personalized_model(
                &base,
                &user_rows,
                AdaptConfig {
                    prior_count,
                    min_new_count,
                },
                min_backoff_count,
            );
            if let Some(parent) = output_model.parent() {
                std::fs::create_dir_all(parent)?;
            }
            p.save_json(&output_model)?;
            println!("output_model: {}", output_model.display());
            println!("adapt.user_rows_total: {}", adapt_stats.user_rows);
            println!(
                "adapt.user_digraph_obs_total: {}",
                adapt_stats.user_digraph_obs
            );
            println!("backoff.min_backoff_count: {}", min_backoff_count);
            println!("backoff.user_rows_used: {}", bstats.user_rows_used);
            println!(
                "backoff.user_digraph_obs_used: {}",
                bstats.user_digraph_obs_used
            );
            println!("backoff.user_distinct_out: {}", bstats.user_distinct_out);
            println!("backoff.user_distinct_in: {}", bstats.user_distinct_in);
            println!("backoff.user_hand_groups: {}", bstats.user_hand_groups);
            println!("backoff.user_class_groups: {}", bstats.user_class_groups);
        }
        Command::SamplePassphrases {
            model,
            wordlist,
            count,
            words,
            allow_repeats,
            separator,
            gap_regex,
            max_chars,
            min_chars,
            seed,
            style,
            case,
            prefix_regex,
            suffix_regex,
            alternatives,
            alt_tries,
            alt_mode,
            meta,
            max_tries,
        } => {
            if count == 0 {
                anyhow::bail!("count must be >= 1");
            }
            if words == 0 {
                anyhow::bail!("words must be >= 1");
            }
            if let Some(maxc) = max_chars {
                if maxc == 0 {
                    anyhow::bail!("max_chars must be >= 1");
                }
            }
            if let (Some(minc), Some(maxc)) = (min_chars, max_chars) {
                if minc > maxc {
                    anyhow::bail!("min_chars cannot exceed max_chars");
                }
            }

            let model = AnyTimingModel::load_json(&model)
                .with_context(|| format!("loading model: {}", model.display()))?;
            let mut gcfg = GenerateConfig::default();
            gcfg.words = words;
            gcfg.separator = separator.clone();
            // We intentionally do not filter here; wordlist should already be curated.
            gcfg.ascii_lower_only = false;
            gcfg.min_word_len = 1;
            gcfg.max_word_len = usize::MAX;
            let wl = load_wordlist(&wordlist, &gcfg)
                .with_context(|| format!("loading wordlist: {}", wordlist.display()))?;

            // Apply style preset as defaults (still allows explicit overrides via flags).
            let (mut separator, mut gap_regex, mut case, mut prefix_regex, mut suffix_regex) =
                (separator, gap_regex, case, prefix_regex, suffix_regex);
            match style {
                SampleStyle::Custom => {}
                SampleStyle::Spaces => {
                    if gap_regex.is_none() {
                        separator = " ".to_string();
                    }
                    if prefix_regex.is_none() {
                        prefix_regex = None;
                    }
                    if suffix_regex.is_none() {
                        suffix_regex = None;
                    }
                    case = CaseMode::Lower;
                }
                SampleStyle::Hyphens => {
                    if gap_regex.is_none() {
                        separator = "-".to_string();
                    }
                    case = CaseMode::Lower;
                }
                SampleStyle::Numbers => {
                    if gap_regex.is_none() {
                        gap_regex = Some("[0-9]".to_string());
                    }
                    case = CaseMode::Lower;
                }
                SampleStyle::NumbersSymbols => {
                    if gap_regex.is_none() {
                        gap_regex = Some(r"[0-9!@#$%^&*_\-]".to_string());
                    }
                    case = CaseMode::Lower;
                }
                SampleStyle::LoginTitle2Digits => {
                    if gap_regex.is_none() {
                        separator = "".to_string();
                    }
                    case = CaseMode::Title;
                    if suffix_regex.is_none() {
                        suffix_regex = Some(r"[0-9]{2}".to_string());
                    }
                }
                SampleStyle::LoginTitleEndPunct => {
                    if gap_regex.is_none() {
                        separator = "".to_string();
                    }
                    case = CaseMode::Title;
                    if suffix_regex.is_none() {
                        suffix_regex = Some(r"[!?_\-]".to_string());
                    }
                }
            }

            let gap_dist = if let Some(pat) = gap_regex.as_deref() {
                Some(
                    rand_regex::Regex::compile(pat, 256)
                        .with_context(|| format!("compiling gap_regex: {pat}"))?,
                )
            } else {
                None
            };
            let prefix_dist = if let Some(pat) = prefix_regex.as_deref() {
                Some(
                    rand_regex::Regex::compile(pat, 256)
                        .with_context(|| format!("compiling prefix_regex: {pat}"))?,
                )
            } else {
                None
            };
            let suffix_dist = if let Some(pat) = suffix_regex.as_deref() {
                Some(
                    rand_regex::Regex::compile(pat, 256)
                        .with_context(|| format!("compiling suffix_regex: {pat}"))?,
                )
            } else {
                None
            };

            let mut rng = rng_from_seed(seed);
            // Separate RNG for regex generation, because `rand_regex` currently uses rand 0.8.
            let mut rng_gap = match seed {
                Some(s) => rand08::rngs::StdRng::seed_from_u64(s ^ 0x9e37_79b9_7f4a_7c15),
                None => rand08::rngs::StdRng::from_entropy(),
            };
            let mut out = 0usize;

            #[derive(Debug, Clone)]
            struct Parts {
                prefix: String,
                words: Vec<String>,
                gaps: Vec<String>, // len = words-1
                suffix: String,
            }

            fn assemble(parts: &Parts) -> String {
                let mut s = String::new();
                s.push_str(&parts.prefix);
                for (i, w) in parts.words.iter().enumerate() {
                    if i > 0 {
                        if let Some(g) = parts.gaps.get(i - 1) {
                            s.push_str(g);
                        }
                    }
                    s.push_str(w);
                }
                s.push_str(&parts.suffix);
                s
            }

            // Helper to build one phrase attempt (stable parts -> alternatives can reuse formatting).
            let make_one = |rng: &mut dyn rand::RngCore,
                            rng_gap: &mut rand08::rngs::StdRng|
             -> anyhow::Result<Parts> {
                let mut picked: Vec<String> = Vec::with_capacity(words);
                if allow_repeats {
                    for _ in 0..words {
                        let Some(w) = wl.words.choose(rng) else {
                            anyhow::bail!("unexpected empty wordlist");
                        };
                        picked.push(apply_case(w, case, rng));
                    }
                } else {
                    // Sample without replacement by shuffling indices.
                    let mut idx: Vec<usize> = (0..wl.words.len()).collect();
                    idx.shuffle(rng);
                    if idx.len() < words {
                        anyhow::bail!(
                            "wordlist too small for allow_repeats=false and words={words}"
                        );
                    }
                    for &i in idx.iter().take(words) {
                        picked.push(apply_case(&wl.words[i], case, rng));
                    }
                }

                let prefix = if let Some(dist) = &prefix_dist {
                    rand08::distributions::Distribution::sample(dist, rng_gap)
                } else {
                    String::new()
                };
                let suffix = if let Some(dist) = &suffix_dist {
                    rand08::distributions::Distribution::sample(dist, rng_gap)
                } else {
                    String::new()
                };

                let mut gaps: Vec<String> = Vec::new();
                if words >= 2 {
                    gaps.reserve(words - 1);
                    for _ in 0..(words - 1) {
                        let g = if let Some(dist) = &gap_dist {
                            rand08::distributions::Distribution::sample(dist, rng_gap)
                        } else {
                            separator.clone()
                        };
                        gaps.push(g);
                    }
                }

                Ok(Parts {
                    prefix,
                    words: picked,
                    gaps,
                    suffix,
                })
            };

            if alternatives > 0 {
                println!(
                    "note: alternatives_enabled; choosing manually reduces effective entropy unless randomized"
                );
            }

            // Rejection sample until we print count phrases.
            let mut tries = 0usize;
            while out < count {
                tries += 1;
                if tries > max_tries {
                    anyhow::bail!(
                        "failed to sample enough phrases under constraints: produced {out}/{count} within {max_tries} tries"
                    );
                }
                let parts = make_one(&mut rng, &mut rng_gap)?;
                let phrase = assemble(&parts);
                let clen = phrase.chars().count();
                if let Some(maxc) = max_chars {
                    if clen > maxc {
                        continue;
                    }
                }
                if let Some(minc) = min_chars {
                    if clen < minc {
                        continue;
                    }
                }

                let sc0 = fastphrase::score::score_phrase(&model, &phrase);
                let ms0 = sc0.predicted_ms as f64;
                if meta {
                    let hit_frac = if sc0.digraphs == 0 {
                        0.0
                    } else {
                        (sc0.hits as f64) / (sc0.digraphs as f64)
                    };
                    println!(
                        "{:>9.3} ms  hit_frac={:.3}  chars={}  {}",
                        ms0, hit_frac, clen, phrase
                    );
                } else {
                    println!("{:>9.3} ms  {}", ms0, phrase);
                }

                if alternatives > 0 && !parts.words.is_empty() {
                    #[derive(Debug, Clone)]
                    struct Alt {
                        ms: f64,
                        phrase: String,
                        pos: usize,
                        word: String,
                    }
                    let mut cand: Vec<Alt> = Vec::new();
                    // Deterministic-ish alternatives when seed is set; otherwise random.
                    let mut rng_alt = rng_from_seed(seed.map(|s| s ^ 0x51d7_6e3a_9f03_17c1));
                    for pos in 0..parts.words.len() {
                        for _ in 0..alt_tries.max(1) {
                            let Some(w) = wl.words.choose(&mut rng_alt) else {
                                break;
                            };
                            let new_word = apply_case(w, case, &mut rng_alt);
                            if new_word == parts.words[pos] {
                                continue;
                            }
                            let mut p2 = parts.clone();
                            p2.words[pos] = new_word.clone();
                            let s2 = assemble(&p2);
                            let clen2 = s2.chars().count();
                            if let Some(maxc) = max_chars {
                                if clen2 > maxc {
                                    continue;
                                }
                            }
                            if let Some(minc) = min_chars {
                                if clen2 < minc {
                                    continue;
                                }
                            }
                            let ms =
                                fastphrase::score::score_phrase(&model, &s2).predicted_ms as f64;
                            cand.push(Alt {
                                ms,
                                phrase: s2,
                                pos,
                                word: new_word,
                            });
                        }
                    }
                    cand.sort_by(|a, b| a.phrase.cmp(&b.phrase));
                    cand.dedup_by(|a, b| a.phrase == b.phrase);
                    match alt_mode {
                        AltMode::Faster => cand.sort_by(|a, b| a.ms.total_cmp(&b.ms)),
                        AltMode::Similar => cand.sort_by(|a, b| {
                            let da = (a.ms - ms0).abs();
                            let db = (b.ms - ms0).abs();
                            da.total_cmp(&db).then_with(|| a.ms.total_cmp(&b.ms))
                        }),
                    }
                    let take = alternatives.min(cand.len());
                    if take > 0 {
                        println!("  alternatives ({:?}):", alt_mode);
                        for a in cand.into_iter().take(take) {
                            println!(
                                "   - {:>9.3} ms  (Δ{:+.3})  swap_pos={} -> {}  {}",
                                a.ms,
                                a.ms - ms0,
                                a.pos + 1,
                                a.word,
                                a.phrase
                            );
                        }
                    }
                }

                out += 1;
            }
            println!("(done) sampled {out}/{count}");
        }
        Command::AnalyzeGenerator {
            model,
            wordlist,
            corpus,
            samples,
            pick_best_of,
            words,
            allow_repeats,
            separator,
            gap_regex,
            max_chars,
            min_chars,
            seed,
            style,
            case,
            prefix_regex,
            suffix_regex,
            max_tries_total,
            show_top,
        } => {
            use std::collections::HashMap;

            if samples == 0 {
                anyhow::bail!("samples must be >= 1");
            }
            if pick_best_of == 0 {
                anyhow::bail!("pick_best_of must be >= 1");
            }
            if words == 0 {
                anyhow::bail!("words must be >= 1");
            }
            if let Some(maxc) = max_chars {
                if maxc == 0 {
                    anyhow::bail!("max_chars must be >= 1");
                }
            }
            if let (Some(minc), Some(maxc)) = (min_chars, max_chars) {
                if minc > maxc {
                    anyhow::bail!("min_chars cannot exceed max_chars");
                }
            }
            if max_tries_total == 0 {
                anyhow::bail!("max_tries_total must be >= 1");
            }

            fn log2_f64(x: f64) -> f64 {
                x.ln() / 2f64.ln()
            }
            fn quantile(mut v: Vec<f64>, p: f64) -> f64 {
                if v.is_empty() {
                    return 0.0;
                }
                v.sort_by(|a, b| a.total_cmp(b));
                let n = v.len();
                let idx = ((p.clamp(0.0, 1.0) * ((n - 1) as f64)).round() as usize).min(n - 1);
                v[idx]
            }
            fn mean_std(v: &[f64]) -> (f64, f64) {
                if v.is_empty() {
                    return (0.0, 0.0);
                }
                let n = v.len() as f64;
                let mean = v.iter().sum::<f64>() / n;
                let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
                (mean, var.max(0.0).sqrt())
            }
            fn is_shift_us(ch: char) -> bool {
                ch.is_ascii_uppercase()
                    || matches!(
                        ch,
                        '!' | '@'
                            | '#'
                            | '$'
                            | '%'
                            | '^'
                            | '&'
                            | '*'
                            | '('
                            | ')'
                            | '_'
                            | '+'
                            | '{'
                            | '}'
                            | '|'
                            | ':'
                            | '"'
                            | '<'
                            | '>'
                            | '?'
                    )
            }
            fn shift_frac_us(s: &str) -> f64 {
                let mut n = 0usize;
                let mut sh = 0usize;
                for ch in s.chars() {
                    n += 1;
                    if is_shift_us(ch) {
                        sh += 1;
                    }
                }
                if n == 0 {
                    0.0
                } else {
                    (sh as f64) / (n as f64)
                }
            }
            fn idf_phrase(
                words: &[String],
                counts: &std::collections::HashMap<String, u64>,
                total: u64,
            ) -> f64 {
                if total == 0 {
                    return 0.0;
                }
                let tot = total as f64;
                let mut sum = 0.0f64;
                for w in words {
                    let c = counts.get(w).copied().unwrap_or(1).max(1) as f64;
                    sum += log2_f64(tot / c);
                }
                sum / (words.len().max(1) as f64)
            }
            fn entropy_from_counts(counts: &HashMap<String, u32>, n: u64) -> (f64, f64, f64, f64) {
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
            fn collision_pairs(counts: &HashMap<String, u32>) -> u64 {
                // number of equal pairs among n draws: sum_x C(c_x, 2)
                counts
                    .values()
                    .map(|&c| {
                        let c = c as u64;
                        c.saturating_mul(c.saturating_sub(1)) / 2
                    })
                    .sum()
            }
            fn n_pairs(n: u64) -> u64 {
                n.saturating_mul(n.saturating_sub(1)) / 2
            }
            fn p2_upper_bound_zero_collisions(n: u64, alpha: f64) -> Option<f64> {
                // If we observe 0 collisions among Npairs pairwise comparisons, then under
                // a crude (but useful) binomial approximation, P(0) = (1 - p2)^Npairs.
                // Solve for an upper bound p2 s.t. P(0) = alpha:
                //   p2 <= 1 - alpha^(1/Npairs) ≈ -ln(alpha) / Npairs
                let np = n_pairs(n);
                if np == 0 {
                    return None;
                }
                if !(0.0 < alpha && alpha < 1.0) {
                    return None;
                }
                Some(-alpha.ln() / (np as f64))
            }
            fn top_k_counts(counts: &HashMap<String, u32>, k: usize) -> Vec<(String, u32)> {
                let mut v: Vec<(String, u32)> =
                    counts.iter().map(|(s, &c)| (s.clone(), c)).collect();
                v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                v.truncate(k);
                v
            }

            let model = AnyTimingModel::load_json(&model)
                .with_context(|| format!("loading model: {}", model.display()))?;
            let mut gcfg = GenerateConfig::default();
            gcfg.words = words;
            gcfg.separator = separator.clone();
            gcfg.ascii_lower_only = false;
            gcfg.min_word_len = 1;
            gcfg.max_word_len = usize::MAX;
            let wl = load_wordlist(&wordlist, &gcfg)
                .with_context(|| format!("loading wordlist: {}", wordlist.display()))?;

            let inferred_corpus = wordlist
                .parent()
                .map(|p| p.join("corpus.txt"))
                .filter(|p| p.exists());
            let corpus = corpus.or(inferred_corpus);
            let corpus_counts = corpus
                .as_ref()
                .map(|p| {
                    load_corpus_counts(p, true, 1, usize::MAX)
                        .with_context(|| format!("loading corpus counts: {}", p.display()))
                })
                .transpose()?;
            let corpus_total: u64 = corpus_counts
                .as_ref()
                .map(|m| m.values().copied().sum())
                .unwrap_or(0);

            // Apply style preset as defaults (same logic as sample-passphrases).
            let (mut separator, mut gap_regex, mut case, prefix_regex, mut suffix_regex) =
                (separator, gap_regex, case, prefix_regex, suffix_regex);
            match style {
                SampleStyle::Custom => {}
                SampleStyle::Spaces => {
                    if gap_regex.is_none() {
                        separator = " ".to_string();
                    }
                    case = CaseMode::Lower;
                }
                SampleStyle::Hyphens => {
                    if gap_regex.is_none() {
                        separator = "-".to_string();
                    }
                    case = CaseMode::Lower;
                }
                SampleStyle::Numbers => {
                    if gap_regex.is_none() {
                        gap_regex = Some("[0-9]".to_string());
                    }
                    case = CaseMode::Lower;
                }
                SampleStyle::NumbersSymbols => {
                    if gap_regex.is_none() {
                        gap_regex = Some(r"[0-9!@#$%^&*_\-]".to_string());
                    }
                    case = CaseMode::Lower;
                }
                SampleStyle::LoginTitle2Digits => {
                    if gap_regex.is_none() {
                        separator = "".to_string();
                    }
                    case = CaseMode::Title;
                    if suffix_regex.is_none() {
                        suffix_regex = Some(r"[0-9]{2}".to_string());
                    }
                }
                SampleStyle::LoginTitleEndPunct => {
                    if gap_regex.is_none() {
                        separator = "".to_string();
                    }
                    case = CaseMode::Title;
                    if suffix_regex.is_none() {
                        suffix_regex = Some(r"[!?_\-]".to_string());
                    }
                }
            }

            let gap_dist = if let Some(pat) = gap_regex.as_deref() {
                Some(
                    rand_regex::Regex::compile(pat, 256)
                        .with_context(|| format!("compiling gap_regex: {pat}"))?,
                )
            } else {
                None
            };
            let prefix_dist = if let Some(pat) = prefix_regex.as_deref() {
                Some(
                    rand_regex::Regex::compile(pat, 256)
                        .with_context(|| format!("compiling prefix_regex: {pat}"))?,
                )
            } else {
                None
            };
            let suffix_dist = if let Some(pat) = suffix_regex.as_deref() {
                Some(
                    rand_regex::Regex::compile(pat, 256)
                        .with_context(|| format!("compiling suffix_regex: {pat}"))?,
                )
            } else {
                None
            };

            // “Nominal” bits (upper-ish bound) for a few known patterns.
            fn nominal_bits_for_regex(pat: &str) -> Option<f64> {
                match pat {
                    "[0-9]" => Some(log2_f64(10.0)),
                    r"[0-9]{2}" => Some(log2_f64(100.0)),
                    r"[!?_\-]" => Some(log2_f64(4.0)),
                    r"[0-9!@#$%^&*_\-]" => Some(log2_f64(20.0)),
                    _ => None,
                }
            }
            fn bits_words(n: usize, k: usize, allow_repeats: bool) -> f64 {
                let n = n as f64;
                if allow_repeats {
                    (k as f64) * log2_f64(n)
                } else {
                    let mut s = 0.0f64;
                    for i in 0..k {
                        let term = n - (i as f64);
                        if term <= 0.0 {
                            return 0.0;
                        }
                        s += log2_f64(term);
                    }
                    s
                }
            }

            let mut rng = rng_from_seed(seed);
            let mut rng_gap = match seed {
                Some(s) => rand08::rngs::StdRng::seed_from_u64(s ^ 0x9e37_79b9_7f4a_7c15),
                None => rand08::rngs::StdRng::from_entropy(),
            };

            #[derive(Debug, Clone)]
            struct Parts {
                prefix: String,
                words: Vec<String>,
                gaps: Vec<String>,
                suffix: String,
            }
            fn assemble(parts: &Parts) -> String {
                let mut s = String::new();
                s.push_str(&parts.prefix);
                for (i, w) in parts.words.iter().enumerate() {
                    if i > 0 {
                        if let Some(g) = parts.gaps.get(i - 1) {
                            s.push_str(g);
                        }
                    }
                    s.push_str(w);
                }
                s.push_str(&parts.suffix);
                s
            }
            let make_one = |rng: &mut dyn rand::RngCore,
                            rng_gap: &mut rand08::rngs::StdRng|
             -> anyhow::Result<Parts> {
                let mut picked: Vec<String> = Vec::with_capacity(words);
                if allow_repeats {
                    for _ in 0..words {
                        let Some(w) = wl.words.choose(rng) else {
                            anyhow::bail!("unexpected empty wordlist");
                        };
                        picked.push(apply_case(w, case, rng));
                    }
                } else {
                    let mut idx: Vec<usize> = (0..wl.words.len()).collect();
                    idx.shuffle(rng);
                    if idx.len() < words {
                        anyhow::bail!(
                            "wordlist too small for allow_repeats=false and words={words}"
                        );
                    }
                    for &i in idx.iter().take(words) {
                        picked.push(apply_case(&wl.words[i], case, rng));
                    }
                }

                let prefix = if let Some(dist) = &prefix_dist {
                    rand08::distributions::Distribution::sample(dist, rng_gap)
                } else {
                    String::new()
                };
                let suffix = if let Some(dist) = &suffix_dist {
                    rand08::distributions::Distribution::sample(dist, rng_gap)
                } else {
                    String::new()
                };

                let mut gaps: Vec<String> = Vec::new();
                if words >= 2 {
                    gaps.reserve(words - 1);
                    for _ in 0..(words - 1) {
                        let g = if let Some(dist) = &gap_dist {
                            rand08::distributions::Distribution::sample(dist, rng_gap)
                        } else {
                            separator.clone()
                        };
                        gaps.push(g);
                    }
                }
                Ok(Parts {
                    prefix,
                    words: picked,
                    gaps,
                    suffix,
                })
            };

            let mut tries_total = 0usize;
            let mut freq_base: HashMap<String, u32> = HashMap::new();
            let mut freq_picked: HashMap<String, u32> = HashMap::new();
            let mut ms_base: Vec<f64> = Vec::with_capacity(samples);
            let mut ms_picked: Vec<f64> = Vec::with_capacity(samples);
            let mut hit_base: Vec<f64> = Vec::with_capacity(samples);
            let mut hit_picked: Vec<f64> = Vec::with_capacity(samples);
            let mut shift_base: Vec<f64> = Vec::with_capacity(samples);
            let mut shift_picked: Vec<f64> = Vec::with_capacity(samples);
            let mut chars_base: Vec<f64> = Vec::with_capacity(samples);
            let mut chars_picked: Vec<f64> = Vec::with_capacity(samples);
            let mut idf_base: Vec<f64> = Vec::with_capacity(samples);
            let mut idf_picked: Vec<f64> = Vec::with_capacity(samples);

            let mut wordpos_base: Vec<HashMap<String, u32>> =
                (0..words).map(|_| HashMap::new()).collect();
            let mut wordpos_picked: Vec<HashMap<String, u32>> =
                (0..words).map(|_| HashMap::new()).collect();
            let mut wordpairs_base: Vec<HashMap<String, u32>> = (0..words.saturating_sub(1))
                .map(|_| HashMap::new())
                .collect();
            let mut wordpairs_picked: Vec<HashMap<String, u32>> = (0..words.saturating_sub(1))
                .map(|_| HashMap::new())
                .collect();
            let mut wordtuple_base: HashMap<String, u32> = HashMap::new();
            let mut wordtuple_picked: HashMap<String, u32> = HashMap::new();

            let mut sample_valid = |rng: &mut dyn rand::RngCore,
                                    rng_gap: &mut rand08::rngs::StdRng|
             -> anyhow::Result<(String, Vec<String>)> {
                loop {
                    tries_total += 1;
                    if tries_total > max_tries_total {
                        anyhow::bail!(
                            "exceeded max_tries_total={max_tries_total} before collecting all samples; loosen constraints"
                        );
                    }
                    let parts = make_one(rng, rng_gap)?;
                    let phrase = assemble(&parts);
                    let clen = phrase.chars().count();
                    if let Some(maxc) = max_chars {
                        if clen > maxc {
                            continue;
                        }
                    }
                    if let Some(minc) = min_chars {
                        if clen < minc {
                            continue;
                        }
                    }
                    let scrubbed_words: Vec<String> = parts
                        .words
                        .iter()
                        .map(|w| fastphrase::textprep::scrub(w))
                        .collect();
                    return Ok((phrase, scrubbed_words));
                }
            };

            for _ in 0..samples {
                let mut candidates: Vec<(String, Vec<String>, f64, f64)> =
                    Vec::with_capacity(pick_best_of);
                for _ in 0..pick_best_of {
                    let (phrase, scrubbed_words) = sample_valid(&mut rng, &mut rng_gap)?;
                    let sc = fastphrase::score::score_phrase(&model, &phrase);
                    let ms = sc.predicted_ms as f64;
                    let hit = if sc.digraphs == 0 {
                        0.0
                    } else {
                        (sc.hits as f64) / (sc.digraphs as f64)
                    };
                    candidates.push((phrase, scrubbed_words, ms, hit));
                }
                let (base_phrase, base_words, base_ms0, base_hit0) = candidates[0].clone();
                let (best_phrase, best_words, best_ms0, best_hit0) = candidates
                    .into_iter()
                    .min_by(|a, b| a.2.total_cmp(&b.2))
                    .unwrap();

                *freq_base.entry(base_phrase.clone()).or_insert(0) += 1;
                *freq_picked.entry(best_phrase.clone()).or_insert(0) += 1;

                ms_base.push(base_ms0);
                ms_picked.push(best_ms0);
                hit_base.push(base_hit0);
                hit_picked.push(best_hit0);
                shift_base.push(shift_frac_us(&base_phrase));
                shift_picked.push(shift_frac_us(&best_phrase));
                chars_base.push(base_phrase.chars().count() as f64);
                chars_picked.push(best_phrase.chars().count() as f64);

                for (i, w) in base_words.iter().enumerate().take(words) {
                    *wordpos_base[i].entry(w.clone()).or_insert(0) += 1;
                }
                for (i, w) in best_words.iter().enumerate().take(words) {
                    *wordpos_picked[i].entry(w.clone()).or_insert(0) += 1;
                }
                if words >= 2 {
                    for i in 0..(words - 1) {
                        let k = format!("{}\t{}", base_words[i], base_words[i + 1]);
                        *wordpairs_base[i].entry(k).or_insert(0) += 1;
                        let k2 = format!("{}\t{}", best_words[i], best_words[i + 1]);
                        *wordpairs_picked[i].entry(k2).or_insert(0) += 1;
                    }
                }
                *wordtuple_base.entry(base_words.join("\t")).or_insert(0) += 1;
                *wordtuple_picked.entry(best_words.join("\t")).or_insert(0) += 1;

                if let Some(counts) = corpus_counts.as_ref() {
                    idf_base.push(idf_phrase(&base_words, counts, corpus_total));
                    idf_picked.push(idf_phrase(&best_words, counts, corpus_total));
                }
            }

            let n = samples as u64;
            let (h1_base, h2_base, hinf_base, p2_base) = entropy_from_counts(&freq_base, n);
            let (h1_pick, h2_pick, hinf_pick, p2_pick) = entropy_from_counts(&freq_picked, n);

            let (ms_mean_base, _ms_std_base) = mean_std(&ms_base);
            let (ms_mean_pick, _ms_std_pick) = mean_std(&ms_picked);

            let nominal_bits = {
                let bw = bits_words(wl.words.len(), words, allow_repeats);
                let mut extra = 0.0f64;
                if let Some(p) = gap_regex.as_deref() {
                    if let Some(b) = nominal_bits_for_regex(p) {
                        extra += (words.saturating_sub(1) as f64) * b;
                    }
                }
                if let Some(p) = prefix_regex.as_deref() {
                    if let Some(b) = nominal_bits_for_regex(p) {
                        extra += b;
                    }
                }
                if let Some(p) = suffix_regex.as_deref() {
                    if let Some(b) = nominal_bits_for_regex(p) {
                        extra += b;
                    }
                }
                bw + extra
            };

            println!("model.global_mean_ms: {:.3}", model.global_mean_ms());
            println!("wordlist: {}", wordlist.display());
            println!("wordset_size: {}", wl.words.len());
            println!(
                "corpus: {}",
                corpus
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "(none)".to_string())
            );
            println!("style: {:?}", style);
            println!("words: {}", words);
            println!("allow_repeats: {}", allow_repeats);
            println!("separator: {:?}", separator);
            println!("gap_regex: {:?}", gap_regex);
            println!("prefix_regex: {:?}", prefix_regex);
            println!("suffix_regex: {:?}", suffix_regex);
            println!("max_chars: {:?}", max_chars);
            println!("min_chars: {:?}", min_chars);
            println!("samples: {}", samples);
            println!("pick_best_of: {}", pick_best_of);
            println!("tries_total: {}", tries_total);
            let draws_total: u64 = (samples as u64) * (pick_best_of as u64);
            let accept_rate = if tries_total == 0 {
                0.0
            } else {
                (draws_total as f64) / (tries_total as f64)
            };
            println!("draws_total: {}", draws_total);
            println!("accept_rate: {:.6}", accept_rate);
            println!(
                "nominal_bits_upperish (ignores rejection + non-uniform regex): {:.3}",
                nominal_bits
            );
            println!();

            let report = |label: &str,
                          ms: &Vec<f64>,
                          hit: &Vec<f64>,
                          shift: &Vec<f64>,
                          chars: &Vec<f64>,
                          idf: &Vec<f64>,
                          counts: &HashMap<String, u32>,
                          h1: f64,
                          h2: f64,
                          hinf: f64,
                          p2: f64| {
                let (mmean, mstd) = mean_std(ms);
                let ms_p50 = quantile(ms.clone(), 0.50);
                let ms_p95 = quantile(ms.clone(), 0.95);
                let ms_p99 = quantile(ms.clone(), 0.99);
                let hit_mean = mean_std(hit).0;
                let hit_p05 = quantile(hit.clone(), 0.05);
                let shift_mean = mean_std(shift).0;
                let shift_p95 = quantile(shift.clone(), 0.95);
                let chars_mean = mean_std(chars).0;
                let chars_p95 = quantile(chars.clone(), 0.95);

                println!("{label}:");
                println!("  unique_outputs: {}", counts.len());
                let repeats = (n as i64) - (counts.len() as i64);
                println!("  observed_repeats: {}", repeats.max(0));
                let cpairs = collision_pairs(counts);
                let npairs = n_pairs(n);
                println!("  observed_collision_pairs: {} / {}", cpairs, npairs);
                println!(
                    "  ms: mean={:.1} std={:.1} p50={:.1} p95={:.1} p99={:.1}",
                    mmean, mstd, ms_p50, ms_p95, ms_p99
                );
                println!("  hit: mean={:.3} p05={:.3}", hit_mean, hit_p05);
                println!(
                    "  shift_frac_us: mean={:.3} p95={:.3}",
                    shift_mean, shift_p95
                );
                println!("  chars: mean={:.1} p95={:.1}", chars_mean, chars_p95);
                if !idf.is_empty() {
                    let idf_p50 = quantile(idf.clone(), 0.50);
                    let idf_p05 = quantile(idf.clone(), 0.05);
                    println!("  idf_bits_per_word: p50={:.2} p05={:.2}", idf_p50, idf_p05);
                }
                let lb_unique = log2_f64((counts.len().max(1)) as f64);
                println!(
                    "  output_entropy_bits (plugin): H1={:.3}  H2={:.3}  Hinf={:.3}  (lower_bound_log2_unique={:.3})",
                    h1, h2, hinf, lb_unique
                );
                if counts.len() == n as usize {
                    println!("  note: no repeats observed -> output-entropy plugin estimate is sample-size-limited (true entropy likely much larger)");
                }
                println!("  collision_prob_est: sum_p2≈{:.6}  (so H2≈{:.3})", p2, h2);
                if cpairs == 0 {
                    if let Some(p2_ub) = p2_upper_bound_zero_collisions(n, 0.05) {
                        let h2_lb = if p2_ub > 0.0 { -log2_f64(p2_ub) } else { 0.0 };
                        println!(
                            "  collision_bound_95pct: p2 <= {:.6e}  =>  H2 >= {:.3}  (binomial approx, 0 collisions)",
                            p2_ub, h2_lb
                        );
                    }
                }
                if h1 > 0.0 {
                    println!("  ms_per_H1_bit (mean): {:.3}", mmean / h1);
                }
                if h2 > 0.0 {
                    println!("  ms_per_H2_bit (mean): {:.3}", mmean / h2);
                }
                println!();
            };

            report(
                "baseline (pick_best_of=1 semantics)",
                &ms_base,
                &hit_base,
                &shift_base,
                &chars_base,
                &idf_base,
                &freq_base,
                h1_base,
                h2_base,
                hinf_base,
                p2_base,
            );
            if pick_best_of > 1 {
                report(
                    &format!("picked_fastest_of_{pick_best_of} (models manual choice)"),
                    &ms_picked,
                    &hit_picked,
                    &shift_picked,
                    &chars_picked,
                    &idf_picked,
                    &freq_picked,
                    h1_pick,
                    h2_pick,
                    hinf_pick,
                    p2_pick,
                );

                let mut sum_h1_base = 0.0f64;
                let mut sum_h1_pick = 0.0f64;
                let mut sum_h2_base = 0.0f64;
                let mut sum_h2_pick = 0.0f64;
                let mut sum_hinf_base = 0.0f64;
                let mut sum_hinf_pick = 0.0f64;
                for i in 0..words {
                    let (h1b, h2b, hinfb, _) = entropy_from_counts(&wordpos_base[i], n);
                    let (h1p, h2p, hinfp, _) = entropy_from_counts(&wordpos_picked[i], n);
                    sum_h1_base += h1b;
                    sum_h1_pick += h1p;
                    sum_h2_base += h2b;
                    sum_h2_pick += h2p;
                    sum_hinf_base += hinfb;
                    sum_hinf_pick += hinfp;
                }
                println!("word_marginal_entropy_bits (upper bound on joint entropy):");
                println!(
                    "  baseline.sum_positions: H1={:.3} H2={:.3} Hinf={:.3}",
                    sum_h1_base, sum_h2_base, sum_hinf_base
                );
                println!(
                    "  picked.sum_positions:   H1={:.3} H2={:.3} Hinf={:.3}",
                    sum_h1_pick, sum_h2_pick, sum_hinf_pick
                );
                println!(
                    "  Δ(sum_positions) (picked - baseline): ΔH1={:+.3} ΔH2={:+.3} ΔHinf={:+.3}",
                    sum_h1_pick - sum_h1_base,
                    sum_h2_pick - sum_h2_base,
                    sum_hinf_pick - sum_hinf_base
                );
                println!();

                let (tw1b, tw2b, twinfb, _) = entropy_from_counts(&wordtuple_base, n);
                let (tw1p, tw2p, twinfp, _) = entropy_from_counts(&wordtuple_picked, n);
                println!("word_tuple_entropy_bits (words only; ignores separators/prefix/suffix):");
                println!(
                    "  baseline: H1={:.3} H2={:.3} Hinf={:.3}",
                    tw1b, tw2b, twinfb
                );
                println!(
                    "  picked:   H1={:.3} H2={:.3} Hinf={:.3}",
                    tw1p, tw2p, twinfp
                );
                println!(
                    "  Δ(picked-baseline): ΔH1={:+.3} ΔH2={:+.3} ΔHinf={:+.3}",
                    tw1p - tw1b,
                    tw2p - tw2b,
                    twinfp - twinfb
                );
                println!();

                if words >= 2 {
                    println!("adjacent_pair_entropy_bits (per position-pair; words only):");
                    for i in 0..(words - 1) {
                        let (p1b, p2b, pinfb, _) = entropy_from_counts(&wordpairs_base[i], n);
                        let (p1p, p2p, pinfp, _) = entropy_from_counts(&wordpairs_picked[i], n);
                        println!(
                            "  pair({},{}) baseline: H1={:.3} H2={:.3} Hinf={:.3}",
                            i + 1,
                            i + 2,
                            p1b,
                            p2b,
                            pinfb
                        );
                        println!(
                            "  pair({},{}) picked:   H1={:.3} H2={:.3} Hinf={:.3}  ΔH1={:+.3}",
                            i + 1,
                            i + 2,
                            p1p,
                            p2p,
                            pinfp,
                            p1p - p1b
                        );
                    }
                    println!();
                }

                println!(
                    "entropy_penalty_bits (picked - baseline): ΔH1={:+.3}  ΔH2={:+.3}  ΔHinf={:+.3}",
                    h1_pick - h1_base,
                    h2_pick - h2_base,
                    hinf_pick - hinf_base
                );
                println!(
                    "typing_gain_ms (picked - baseline): Δmean={:+.1}  Δp95={:+.1}",
                    ms_mean_pick - ms_mean_base,
                    quantile(ms_picked.clone(), 0.95) - quantile(ms_base.clone(), 0.95)
                );
                println!();
            }

            let top = show_top.min(50);
            if top > 0 {
                println!("top_outputs_baseline:");
                for (s, c) in top_k_counts(&freq_base, top) {
                    println!("  {:>7}  {}", c, s);
                }
                println!();
                if pick_best_of > 1 {
                    println!("top_outputs_picked:");
                    for (s, c) in top_k_counts(&freq_picked, top) {
                        println!("  {:>7}  {}", c, s);
                    }
                    println!();
                }
            }

            println!("limits: entropy estimates are Monte Carlo over observed outputs; small-sample bias exists, especially for large true entropy; nominal_bits is only an upper-ish bound when rejection/regex bias exists; shift_frac_us is a US-layout heuristic");
        }
        Command::AuditModel {
            model,
            corpus,
            ascii_lower_only,
            min_word_len,
            max_word_len,
            sample_words,
            seed,
        } => {
            fn quantiles(mut v: Vec<f64>) -> (usize, f64, f64, f64, f64, f64, f64) {
                if v.is_empty() {
                    return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                }
                v.sort_by(|a, b| a.total_cmp(b));
                let n = v.len();
                let at = |p: f64| -> f64 {
                    let idx = ((p * ((n - 1) as f64)).round() as usize).min(n - 1);
                    v[idx]
                };
                (n, v[0], at(0.50), at(0.90), at(0.95), at(0.99), v[n - 1])
            }

            let model = AnyTimingModel::load_json(&model)
                .with_context(|| format!("loading model: {}", model.display()))?;
            let counts = load_corpus_counts(&corpus, ascii_lower_only, min_word_len, max_word_len)
                .with_context(|| format!("loading corpus counts: {}", corpus.display()))?;
            if counts.is_empty() {
                anyhow::bail!("empty corpus after filtering");
            }

            let mut words: Vec<String> = counts.keys().cloned().collect();
            let mut rng = rng_from_seed(seed);
            words.shuffle(&mut rng);
            let take = sample_words.min(words.len()).max(1);
            words.truncate(take);

            let mut n_words = 0u64;
            let mut n_zero_digraphs = 0u64;
            let mut sum_norm = 0.0f64;
            let mut sum_norm_sq = 0.0f64;
            let mut sum_hit_frac = 0.0f64;
            let mut sum_digraphs = 0u64;
            let mut sum_hits = 0u64;
            let mut norms: Vec<f64> = Vec::new();
            let mut hit_fracs: Vec<f64> = Vec::new();

            // Track worst (slowest normalized) words for inspection.
            #[derive(Debug, Clone)]
            struct Worst {
                norm: f64,
                ms: f64,
                hits: usize,
                digraphs: usize,
                word: String,
            }
            let mut worst: Vec<Worst> = Vec::new();

            for w in words.iter() {
                let sc = fastphrase::score::score_phrase(&model, w);
                if sc.digraphs == 0 {
                    n_zero_digraphs += 1;
                    continue;
                }
                let denom = (sc.digraphs as f64) * (model.global_mean_ms() as f64);
                if denom <= 0.0 {
                    continue;
                }
                let norm = (sc.predicted_ms as f64) / denom;
                let hit_frac = (sc.hits as f64) / (sc.digraphs as f64);

                n_words += 1;
                sum_norm += norm;
                sum_norm_sq += norm * norm;
                sum_hit_frac += hit_frac;
                sum_digraphs += sc.digraphs as u64;
                sum_hits += sc.hits as u64;
                norms.push(norm);
                hit_fracs.push(hit_frac);

                // maintain top 15 slowest normalized
                let item = Worst {
                    norm,
                    ms: sc.predicted_ms as f64,
                    hits: sc.hits,
                    digraphs: sc.digraphs,
                    word: w.clone(),
                };
                if worst.len() < 15 {
                    worst.push(item);
                    worst.sort_by(|a, b| b.norm.total_cmp(&a.norm));
                } else if let Some(last) = worst.last() {
                    if norm > last.norm {
                        worst.pop();
                        worst.push(item);
                        worst.sort_by(|a, b| b.norm.total_cmp(&a.norm));
                    }
                }
            }

            let n = n_words.max(1) as f64;
            let mean = sum_norm / n;
            let var = (sum_norm_sq / n) - mean * mean;
            let std = var.max(0.0).sqrt();
            let mean_hit = sum_hit_frac / n;
            let global_hit = if sum_digraphs == 0 {
                0.0
            } else {
                (sum_hits as f64) / (sum_digraphs as f64)
            };

            println!("model.global_mean_ms: {:.3}", model.global_mean_ms());
            println!("corpus: {}", corpus.display());
            println!("words_considered: {}", take);
            println!("words_scored: {}", n_words);
            println!("words_with_zero_digraphs: {}", n_zero_digraphs);
            println!("normalized_vs_global.mean: {:.6}", mean);
            println!("normalized_vs_global.std: {:.6}", std);
            let (_qn, qmin, q50, q90, q95, q99, qmax) = quantiles(norms);
            println!(
                "normalized_vs_global.quantiles(min/p50/p90/p95/p99/max): {:.6}/{:.6}/{:.6}/{:.6}/{:.6}/{:.6}",
                qmin, q50, q90, q95, q99, qmax
            );
            println!("digraph_hit_frac.mean_by_word: {:.6}", mean_hit);
            println!("digraph_hit_frac.global: {:.6}", global_hit);
            let (_hn, hmin, h50, h90, h95, h99, hmax) = quantiles(hit_fracs);
            println!(
                "digraph_hit_frac.quantiles(min/p50/p90/p95/p99/max): {:.6}/{:.6}/{:.6}/{:.6}/{:.6}/{:.6}",
                hmin, h50, h90, h95, h99, hmax
            );
            println!();
            println!("slowest_words_by_normalized_vs_global (top 15):");
            for w in worst {
                println!(
                    "  norm={:.3}  ms={:.1}  hit={}/{}  {}",
                    w.norm, w.ms, w.hits, w.digraphs, w.word
                );
            }
        }
        Command::ParetoStyles {
            model,
            wordlist,
            corpus,
            words,
            allow_repeats,
            mut n,
            mut style,
            samples,
            seed,
            recommend,
            target_bits,
            min_hit_frac,
        } => {
            use pare::{Direction, ParetoFrontier};

            if words == 0 {
                anyhow::bail!("words must be >= 1");
            }
            if samples == 0 {
                anyhow::bail!("samples must be >= 1");
            }

            let model = AnyTimingModel::load_json(&model)
                .with_context(|| format!("loading model: {}", model.display()))?;

            // Load the full ordered wordset (one per line).
            let mut gcfg = GenerateConfig::default();
            gcfg.words = words;
            gcfg.separator = " ".to_string();
            gcfg.ascii_lower_only = false;
            gcfg.min_word_len = 1;
            gcfg.max_word_len = usize::MAX;
            let wl = load_wordlist(&wordlist, &gcfg)
                .with_context(|| format!("loading wordlist: {}", wordlist.display()))?;
            if wl.words.is_empty() {
                anyhow::bail!("wordlist is empty");
            }

            let inferred_corpus = wordlist
                .parent()
                .map(|p| p.join("corpus.txt"))
                .filter(|p| p.exists());
            let corpus = corpus.or(inferred_corpus);
            let corpus_counts = corpus
                .as_ref()
                .map(|p| {
                    load_corpus_counts(p, true, 1, usize::MAX)
                        .with_context(|| format!("loading corpus counts: {}", p.display()))
                })
                .transpose()?;
            let corpus_total: u64 = corpus_counts
                .as_ref()
                .map(|m| m.values().copied().sum())
                .unwrap_or(0);

            if n.is_empty() {
                n = vec![1024, 2048, 4096, 8192, 16384, 32768];
            }
            // Clamp and unique-sort.
            n.retain(|&x| x >= 2);
            for x in n.iter_mut() {
                *x = (*x).min(wl.words.len());
            }
            n.sort_unstable();
            n.dedup();

            if style.is_empty() {
                style = vec![
                    SampleStyle::Hyphens,
                    SampleStyle::Numbers,
                    SampleStyle::NumbersSymbols,
                    SampleStyle::LoginTitle2Digits,
                ];
            }

            // Restrict to styles with known entropy accounting.
            for &st in style.iter() {
                if matches!(st, SampleStyle::Custom) {
                    anyhow::bail!(
                        "ParetoStyles does not support style=custom (entropy accounting undefined)"
                    );
                }
            }

            #[derive(Debug, Clone)]
            struct Row {
                style: SampleStyle,
                n: usize,
                bits: f64,
                ms_p50: f64,
                ms_p95: f64,
                ms_p99: f64,
                hit_frac_mean: f64,
                hit_frac_p05: f64,
                chars_mean: f64,
                chars_p95: f64,
                shift_frac_mean: f64,
                shift_frac_p95: f64,
                idf_mean: Option<f64>,
                idf_p05: Option<f64>,
            }

            fn log2_f64(x: f64) -> f64 {
                x.ln() / 2f64.ln()
            }

            fn bits_words(n: usize, k: usize, allow_repeats: bool) -> f64 {
                let n = n as f64;
                if allow_repeats {
                    (k as f64) * log2_f64(n)
                } else {
                    // log2 of falling factorial: n*(n-1)*...*(n-k+1)
                    let mut s = 0.0f64;
                    for i in 0..k {
                        let term = n - (i as f64);
                        if term <= 0.0 {
                            return 0.0;
                        }
                        s += log2_f64(term);
                    }
                    s
                }
            }

            fn bits_separators(style: SampleStyle, k: usize) -> f64 {
                if k <= 1 {
                    return 0.0;
                }
                let gaps = (k - 1) as f64;
                match style {
                    SampleStyle::Numbers => gaps * log2_f64(10.0),
                    SampleStyle::NumbersSymbols => gaps * log2_f64(20.0), // 10 digits + 10 symbols in our preset
                    _ => 0.0,
                }
            }

            fn bits_suffix(style: SampleStyle) -> f64 {
                match style {
                    SampleStyle::LoginTitle2Digits => log2_f64(100.0),
                    SampleStyle::LoginTitleEndPunct => log2_f64(4.0),
                    _ => 0.0,
                }
            }

            fn style_case(style: SampleStyle) -> CaseMode {
                match style {
                    SampleStyle::LoginTitle2Digits | SampleStyle::LoginTitleEndPunct => {
                        CaseMode::Title
                    }
                    _ => CaseMode::Lower,
                }
            }

            // Uniform separators for entropy-aligned evaluation.
            fn rand_sep(style: SampleStyle, rng: &mut dyn rand::RngCore) -> Option<char> {
                match style {
                    SampleStyle::Hyphens => Some('-'),
                    SampleStyle::Spaces => Some(' '),
                    SampleStyle::Numbers => {
                        let d = (rng.next_u32() % 10) as u8;
                        Some((b'0' + d) as char)
                    }
                    SampleStyle::NumbersSymbols => {
                        const SYM: &[u8] = b"!@#$%^&*_-";
                        let r = (rng.next_u32() % 20) as u8;
                        if r < 10 {
                            Some((b'0' + r) as char)
                        } else {
                            Some(SYM[(r - 10) as usize] as char)
                        }
                    }
                    SampleStyle::LoginTitle2Digits | SampleStyle::LoginTitleEndPunct => None,
                    SampleStyle::Custom => None,
                }
            }

            fn rand_suffix(style: SampleStyle, rng: &mut dyn rand::RngCore) -> String {
                match style {
                    SampleStyle::LoginTitle2Digits => {
                        let x = (rng.next_u32() % 100) as u8;
                        format!("{:02}", x)
                    }
                    SampleStyle::LoginTitleEndPunct => {
                        const END: &[u8] = b"!?_-";
                        let i = (rng.next_u32() as usize) % END.len();
                        (END[i] as char).to_string()
                    }
                    _ => String::new(),
                }
            }

            fn mean_std(v: &[f64]) -> (f64, f64) {
                if v.is_empty() {
                    return (0.0, 0.0);
                }
                let n = v.len() as f64;
                let mean = v.iter().sum::<f64>() / n;
                let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
                (mean, var.max(0.0).sqrt())
            }

            fn quantile(mut v: Vec<f64>, p: f64) -> f64 {
                if v.is_empty() {
                    return 0.0;
                }
                v.sort_by(|a, b| a.total_cmp(b));
                let n = v.len();
                let idx = ((p.clamp(0.0, 1.0) * ((n - 1) as f64)).round() as usize).min(n - 1);
                v[idx]
            }

            fn is_shift_us(ch: char) -> bool {
                // Approximate “needs shift” on US keyboards:
                // - uppercase ASCII letters
                // - common shifted punctuation
                ch.is_ascii_uppercase()
                    || matches!(
                        ch,
                        '!' | '@'
                            | '#'
                            | '$'
                            | '%'
                            | '^'
                            | '&'
                            | '*'
                            | '('
                            | ')'
                            | '_'
                            | '+'
                            | '{'
                            | '}'
                            | '|'
                            | ':'
                            | '"'
                            | '<'
                            | '>'
                            | '?'
                    )
            }

            fn shift_frac_us(s: &str) -> f64 {
                let mut n = 0usize;
                let mut sh = 0usize;
                for ch in s.chars() {
                    n += 1;
                    if is_shift_us(ch) {
                        sh += 1;
                    }
                }
                if n == 0 {
                    0.0
                } else {
                    (sh as f64) / (n as f64)
                }
            }

            fn idf_phrase(
                words: &[String],
                counts: &std::collections::HashMap<String, u64>,
                total: u64,
            ) -> f64 {
                if total == 0 {
                    return 0.0;
                }
                let tot = total as f64;
                let mut sum = 0.0f64;
                for w in words {
                    let c = counts.get(w).copied().unwrap_or(1).max(1) as f64;
                    sum += log2_f64(tot / c);
                }
                sum / (words.len().max(1) as f64)
            }

            let mut rows: Vec<Row> = Vec::new();
            for &st in &style {
                for &n0 in &n {
                    let n0 = n0.min(wl.words.len()).max(2);
                    let wsub = &wl.words[..n0];
                    let mut rng =
                        rng_from_seed(seed.map(|s| s ^ ((n0 as u64) << 1) ^ ((st as u64) << 48)));

                    let case = style_case(st);
                    let mut ms_samples: Vec<f64> = Vec::with_capacity(samples);
                    let mut hit_fracs: Vec<f64> = Vec::with_capacity(samples);
                    let mut chars: Vec<f64> = Vec::with_capacity(samples);
                    let mut shift_fracs: Vec<f64> = Vec::with_capacity(samples);
                    let mut idfs: Vec<f64> = Vec::with_capacity(samples);

                    for _ in 0..samples {
                        // pick words
                        let mut chosen: Vec<&str> = Vec::with_capacity(words);
                        if allow_repeats {
                            for _ in 0..words {
                                chosen.push(wsub.choose(&mut rng).unwrap().as_str());
                            }
                        } else {
                            // sample without replacement by shuffling indices
                            let mut idx: Vec<usize> = (0..wsub.len()).collect();
                            idx.shuffle(&mut rng);
                            for &i in idx.iter().take(words) {
                                chosen.push(wsub[i].as_str());
                            }
                        }

                        // build phrase
                        let mut s = String::new();
                        let mut chosen_words: Vec<String> = Vec::with_capacity(words);
                        for (i, w) in chosen.iter().enumerate() {
                            if i > 0 {
                                if let Some(sep) = rand_sep(st, &mut rng) {
                                    s.push(sep);
                                }
                            }
                            let ww = apply_case(w, case, &mut rng);
                            chosen_words.push(fastphrase::textprep::scrub(&ww));
                            s.push_str(&ww);
                        }
                        s.push_str(&rand_suffix(st, &mut rng));

                        let sc = fastphrase::score::score_phrase(&model, &s);
                        let ms = sc.predicted_ms as f64;
                        let hit_frac = if sc.digraphs == 0 {
                            0.0
                        } else {
                            (sc.hits as f64) / (sc.digraphs as f64)
                        };
                        ms_samples.push(ms);
                        hit_fracs.push(hit_frac);
                        chars.push(s.chars().count() as f64);
                        shift_fracs.push(shift_frac_us(&s));
                        if let Some(counts) = corpus_counts.as_ref() {
                            idfs.push(idf_phrase(&chosen_words, counts, corpus_total));
                        }
                    }

                    let (_ms_mean, _ms_std) = mean_std(&ms_samples);
                    let (hit_mean, _hit_std) = mean_std(&hit_fracs);
                    let (chars_mean, _chars_std) = mean_std(&chars);
                    let (shift_mean, _shift_std) = mean_std(&shift_fracs);
                    let ms_p50 = quantile(ms_samples.clone(), 0.50);
                    let ms_p95 = quantile(ms_samples.clone(), 0.95);
                    let ms_p99 = quantile(ms_samples.clone(), 0.99);
                    let hit_p05 = quantile(hit_fracs.clone(), 0.05);
                    let chars_p95 = quantile(chars.clone(), 0.95);
                    let shift_p95 = quantile(shift_fracs.clone(), 0.95);
                    let (idf_mean, idf_p05) = if corpus_counts.is_some() {
                        (
                            Some(quantile(idfs.clone(), 0.50)),
                            Some(quantile(idfs.clone(), 0.05)),
                        )
                    } else {
                        (None, None)
                    };
                    let bits = bits_words(n0, words, allow_repeats)
                        + bits_separators(st, words)
                        + bits_suffix(st);

                    rows.push(Row {
                        style: st,
                        n: n0,
                        bits,
                        ms_p50,
                        ms_p95,
                        ms_p99,
                        hit_frac_mean: hit_mean,
                        hit_frac_p05: hit_p05,
                        chars_mean,
                        chars_p95,
                        shift_frac_mean: shift_mean,
                        shift_frac_p95: shift_p95,
                        idf_mean,
                        idf_p05,
                    });
                }
            }

            // Pareto frontier: minimize ms_p95, maximize bits.
            let mut front: ParetoFrontier<Row> =
                ParetoFrontier::new(vec![Direction::Minimize, Direction::Maximize])
                    .with_labels(vec!["ms_p95".to_string(), "bits".to_string()]);
            for r in rows.clone() {
                let _ = front.push(vec![r.ms_p95, r.bits], r);
            }

            println!("model.global_mean_ms: {:.3}", model.global_mean_ms());
            println!("wordlist: {}", wordlist.display());
            if let Some(p) = corpus.as_ref() {
                println!("corpus: {}", p.display());
            } else {
                println!("corpus: (none)");
            }
            println!("words: {}", words);
            println!("allow_repeats: {}", allow_repeats);
            println!("samples_per_config: {}", samples);
            println!("styles: {:?}", style);
            println!("n_values: {:?}", n);
            println!();
            println!("all_configs:");
            for r in rows.iter() {
                println!(
                    "  style={:?}  N={:<6}  bits={:>7.3}  ms_p50={:>8.1}  ms_p95={:>8.1}  ms_p99={:>8.1}  hit_mean={:>5.3}  hit_p05={:>5.3}  shift_mean={:>5.3}  shift_p95={:>5.3}  chars_mean={:>5.1}  chars_p95={:>5.1}{}",
                    r.style,
                    r.n,
                    r.bits,
                    r.ms_p50,
                    r.ms_p95,
                    r.ms_p99,
                    r.hit_frac_mean,
                    r.hit_frac_p05,
                    r.shift_frac_mean,
                    r.shift_frac_p95,
                    r.chars_mean,
                    r.chars_p95,
                    match (r.idf_mean, r.idf_p05) {
                        (Some(m), Some(p05)) => format!("  idf_p50={:>6.2}  idf_p05={:>6.2}", m, p05),
                        _ => "".to_string(),
                    }
                );
            }
            println!();
            println!("pareto_frontier (min ms_p95, max bits):");
            let mut pts = front.points().to_vec();
            pts.sort_by(|a, b| {
                b.values[1]
                    .total_cmp(&a.values[1])
                    .then_with(|| a.values[0].total_cmp(&b.values[0]))
            });
            for p in pts {
                let r = p.data;
                println!(
                    "  style={:?}  N={:<6}  bits={:>7.3}  ms_p95={:>8.1}  ms_p50={:>8.1}  hit_mean={:>5.3}  shift_mean={:>5.3}",
                    r.style, r.n, r.bits, r.ms_p95, r.ms_p50, r.hit_frac_mean, r.shift_frac_mean
                );
            }

            println!();
            // 3D frontier makes “coverage reliability” first-class rather than only a constraint.
            let mut front3: ParetoFrontier<Row> = ParetoFrontier::new(vec![
                Direction::Minimize,
                Direction::Maximize,
                Direction::Maximize,
            ])
            .with_labels(vec![
                "ms_p95".to_string(),
                "bits".to_string(),
                "hit_mean".to_string(),
            ]);
            for r in rows.clone() {
                let _ = front3.push(vec![r.ms_p95, r.bits, r.hit_frac_mean], r);
            }
            println!(
                "pareto_frontier_3d (min ms_p95, max bits, max hit_mean) (size={}):",
                front3.len()
            );
            let mut pts3 = front3.points().to_vec();
            pts3.sort_by(|a, b| {
                b.values[1]
                    .total_cmp(&a.values[1])
                    .then_with(|| b.values[2].total_cmp(&a.values[2]))
                    .then_with(|| a.values[0].total_cmp(&b.values[0]))
            });
            for p in pts3 {
                let r = p.data;
                println!(
                    "  style={:?}  N={:<6}  bits={:>7.3}  ms_p95={:>8.1}  hit_mean={:>5.3}  hit_p05={:>5.3}",
                    r.style, r.n, r.bits, r.ms_p95, r.hit_frac_mean, r.hit_frac_p05
                );
            }

            println!();
            if recommend {
                let tb = target_bits.unwrap_or(60.0);
                let mh = min_hit_frac.unwrap_or(0.70);
                let mut best: Option<&Row> = None;
                for r in rows.iter() {
                    if r.bits + 1e-9 < tb {
                        continue;
                    }
                    if r.hit_frac_mean + 1e-9 < mh {
                        continue;
                    }
                    match best {
                        None => best = Some(r),
                        Some(b) => {
                            if r.ms_p95 < b.ms_p95 {
                                best = Some(r);
                            }
                        }
                    }
                }
                println!("recommendation (constraints: bits>={tb:.3}, hit_mean>={mh:.3}, objective=min ms_p95):");
                if let Some(r) = best {
                    println!(
                        "  style={:?}  N={}  bits={:.3}  ms_p95={:.1}  hit_mean={:.3}  shift_mean={:.3}",
                        r.style, r.n, r.bits, r.ms_p95, r.hit_frac_mean, r.shift_frac_mean
                    );
                } else {
                    println!("  (none) no config met constraints");
                }
                println!();
            }

            println!("limits: bits assume uniform choice over first N words; separators/suffix counted only for built-in styles; hit_* warns when digits/symbols are mostly fallback; shift_* is a US-layout heuristic (not from timing data)");
        }
    }
    Ok(())
}

fn run_plan_passphrase(cmd: Command) -> anyhow::Result<()> {
    // Reuse the same logic as the main match arm for PlanPassphrase.
    let Command::PlanPassphrase {
        model,
        corpus,
        output,
        words,
        target_bits,
        allow_repeats,
        separator,
        k,
        alpha,
        objective,
        min_hit_frac,
        min_vowels,
        samples,
        seed,
        ascii_lower_only,
        min_word_len,
        max_word_len,
    } = cmd
    else {
        anyhow::bail!("internal error: expected PlanPassphrase");
    };

    let model = AnyTimingModel::load_json(&model)
        .with_context(|| format!("loading model: {}", model.display()))?;
    let counts = load_corpus_counts(&corpus, ascii_lower_only, min_word_len, max_word_len)?;

    let mut km = KGramModel::new(k, alpha)?;
    if objective == WordsetObjective::MsPerLmBit {
        km.train(counts.iter().map(|(w, &c)| (w.clone(), c)));
    }

    struct RowOut {
        word: String,
        ms: f64,
        ms_per_bit: f64,
    }
    let mut rows = Vec::new();
    let mut skipped_low_hit = 0u64;
    let mut skipped_low_vowels = 0u64;
    for w in counts.keys() {
        if min_vowels > 0 && count_vowels(w) < min_vowels {
            skipped_low_vowels += 1;
            continue;
        }
        let sc = fastphrase::score::score_phrase(&model, w);
        if sc.digraphs == 0 {
            continue;
        }
        let hit_frac = (sc.hits as f64) / (sc.digraphs as f64);
        if min_hit_frac > 0.0 && hit_frac + 1e-12 < min_hit_frac {
            skipped_low_hit += 1;
            continue;
        }
        let ms = sc.predicted_ms as f64;
        let ms_per_bit = if objective == WordsetObjective::MsPerLmBit {
            let bits = km.surprisal_bits(w);
            if bits > 0.0 {
                ms / bits
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        };
        rows.push(RowOut {
            word: w.clone(),
            ms,
            ms_per_bit,
        });
    }
    if rows.is_empty() {
        anyhow::bail!(
            "no candidate words after filtering (min_hit_frac={min_hit_frac}); try lowering it"
        );
    }
    match objective {
        WordsetObjective::MsOnly => rows.sort_by(|a, b| a.ms.total_cmp(&b.ms)),
        WordsetObjective::MsPerLmBit => rows.sort_by(|a, b| a.ms_per_bit.total_cmp(&b.ms_per_bit)),
    }

    let needed_n = (2f64).powf(target_bits / (words as f64)).ceil() as usize;
    let n = needed_n.min(rows.len()).max(1);
    let chosen: Vec<String> = rows.into_iter().take(n).map(|r| r.word).collect();

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&output)?);
        for w in &chosen {
            writeln!(f, "{w}")?;
        }
        f.flush()?;
    }

    let wl = wordlist_from_vec(chosen)?;
    let mut gcfg = GenerateConfig::default();
    gcfg.words = words;
    gcfg.separator = separator;
    gcfg.ascii_lower_only = ascii_lower_only;
    gcfg.min_word_len = min_word_len;
    gcfg.max_word_len = max_word_len;
    let mut rng = rng_from_seed(seed);
    let (mean_ms, std_ms) = estimate_avg_phrase_ms(&model, &wl, &gcfg, samples, &mut rng)?;

    let m = wl.words.len() as f64;
    let combos = if allow_repeats {
        m.powi(words as i32)
    } else {
        let mut p = 1.0f64;
        for i in 0..(words as u64) {
            let term = m - (i as f64);
            if term <= 0.0 {
                p = 0.0;
                break;
            }
            p *= term;
        }
        p
    };
    let bits = if combos > 0.0 { combos.log2() } else { 0.0 };
    let seconds = combos * (mean_ms / 1000.0);

    println!("output: {}", output.display());
    println!("objective: {:?}", objective);
    println!("min_hit_frac: {:.3}", min_hit_frac);
    println!("min_vowels: {}", min_vowels);
    if min_hit_frac > 0.0 {
        println!("skipped_low_hit_frac: {}", skipped_low_hit);
    }
    if min_vowels > 0 {
        println!("skipped_low_vowels: {}", skipped_low_vowels);
    }
    println!("planned_wordset_size: {}", wl.words.len());
    println!("target_bits: {:.3}", target_bits);
    println!("achieved_entropy_bits: {:.3}", bits);
    if bits + 1e-9 < target_bits {
        println!("warning: target_bits_not_achievable_with_current_corpus_size");
        println!("hint: increase --words, relax filters, or use a larger corpus");
    }
    println!("avg_ms_per_phrase: {:.3}", mean_ms);
    println!("std_ms_per_phrase: {:.3}", std_ms);
    println!("expected_days_to_enumerate: {:.3}", seconds / 86400.0);
    Ok(())
}

fn load_corpus_counts(
    corpus: &std::path::Path,
    ascii_lower_only: bool,
    min_word_len: usize,
    max_word_len: usize,
) -> anyhow::Result<std::collections::HashMap<String, u64>> {
    let text = std::fs::read_to_string(corpus)?;
    let mut counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let Some(word_raw) = it.next() else { continue };
        let count: u64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(1);

        let w = fastphrase::textprep::scrub(word_raw);
        if w.len() < min_word_len || w.len() > max_word_len {
            continue;
        }
        if ascii_lower_only && !w.bytes().all(|b| matches!(b, b'a'..=b'z')) {
            continue;
        }
        *counts.entry(w).or_insert(0) += count;
    }
    Ok(counts)
}

fn count_vowels(s: &str) -> usize {
    // After scrub+ascii-lower filtering, this is stable and cheap.
    s.bytes()
        .filter(|b| matches!(b, b'a' | b'e' | b'i' | b'o' | b'u' | b'y'))
        .count()
}

fn apply_case(word: &str, mode: CaseMode, rng: &mut dyn rand::RngCore) -> String {
    match mode {
        CaseMode::Lower => word.to_string(),
        CaseMode::Upper => word.to_ascii_uppercase(),
        CaseMode::Title => title_case_ascii(word),
        CaseMode::RandomTitle => {
            if rng.next_u32() & 1 == 1 {
                title_case_ascii(word)
            } else {
                word.to_string()
            }
        }
    }
}

fn title_case_ascii(word: &str) -> String {
    let mut out = String::with_capacity(word.len());
    let mut first = true;
    for ch in word.chars() {
        if first {
            out.extend(ch.to_uppercase());
            first = false;
        } else {
            out.push(ch);
        }
    }
    out
}
