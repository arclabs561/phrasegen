# Experiments (refining speed/entropy claims)

This page is a small set of **reproducible experiments** that stress-test the assumptions in `docs/math.md`.
The goal is not “prove security,” but to produce **auditable numbers** for:

- how much speed you gain by biasing toward “easy to type”
- how much entropy you lose (or at least how the distribution concentrates)
- when “nominal bits” stop being a clean statement (constraints, best-of-\(M\), manual choice)

All commands are intended to run from the repo root.

## 0) Build a base model + corpus + wordset (one-time)

This creates a local artifact directory with a trained model, a corpus counts file, and an ordered wordset.

```bash
cargo run -- base-pipeline \
  --out-dir data/_exp/base \
  --words 4 \
  --target-bits 52 \
  --seed 1 \
  --samples 20000 \
  --top 10
```

Artifacts (paths printed by the command):

- `data/_exp/base/model_union.json` (timing model)
- `data/_exp/base/corpus.txt` (corpus counts: `word<TAB>count`)
- `data/_exp/base/wordset.txt` (one word per line; ordered “easiest first” under the chosen objective)

## 0b) “Checked henceforth”: small offline experiment suite

CI runs `cargo test`, and this repo includes a **small, offline** (no downloads) experiment suite that sanity-checks the
speed/entropy tradeoffs across **all built-in styles** using only committed fixtures:

- `data/example.csv` (tiny timing dataset → fitted model)
- `data/wordlist.txt` (tiny wordset)

See `tests/cli_experiments.rs` for the exact assertions.

## 1) Measure “pick fastest of M” (models manual choice)

This answers: “If I look at 10 candidates and pick the fastest-looking one, what’s the entropy penalty?”

```bash
cargo run -- analyze-generator \
  --model data/_exp/base/model_union.json \
  --wordlist data/_exp/base/wordset.txt \
  --style numbers-symbols \
  --words 4 \
  --samples 50000 \
  --pick-best-of 10 \
  --seed 1 \
  --show-top 0
```

How to read it:

- `typing_gain_ms`: direct speed gain from “best-of-\(M\)”.
- `entropy_penalty_bits`: **full-output plugin estimate**, which is often **sample-size-limited** in high-entropy regimes
  (e.g., when `unique_outputs == samples`).
- `entropy_penalty_bits_word_marginals`: a more sensitive signal when the full outputs are all unique; it measures how
  much the **per-position word distribution** concentrates under best-of-\(M\).

Example excerpt (this exact config, seed=1):

- `typing_gain_ms (picked - baseline): Δmean=-428.2  Δp95=-814.2`
- `entropy_penalty_bits_word_marginals (Δsum_positions upper bound): ΔH1=-1.271  ΔH2=-2.117  ΔHinf=-3.642`

## 1b) Run the same analysis across *all* built-in styles (and scale it up)

The simplest way to do “all of them” is a loop over style presets. You can scale `SAMPLES` up (e.g. 20k → 200k) to reduce
Monte Carlo noise in the timing quantiles and in the marginal-entropy deltas.

```bash
MODEL="data/_exp/base/model_union.json"
WORDLIST="data/_exp/base/wordset.txt"

SAMPLES=50000
PICK=10
SEED=1

for STYLE in spaces hyphens numbers numbers-symbols login-title-2digits login-title-endpunct; do
  # Style-specific char limits (typical “login box” limits).
  # Tune these; if accept_rate becomes tiny, raise MAX_CHARS or lower words/style complexity.
  case "$STYLE" in
    login-title-2digits)   MAX_CHARS=28 ;;
    login-title-endpunct)  MAX_CHARS=28 ;;
    *)                     MAX_CHARS=32 ;;
  esac

  echo
  echo "=== $STYLE  (max_chars=$MAX_CHARS) ==="
  cargo run -- analyze-generator \
    --model "$MODEL" \
    --wordlist "$WORDLIST" \
    --style "$STYLE" \
    --words 4 \
    --samples "$SAMPLES" \
    --pick-best-of "$PICK" \
    --max-chars "$MAX_CHARS" \
    --seed "$SEED" \
    --show-top 0
done
```

## 2) Quantify rejection-sampling bias under `--max-chars`

This answers: “What happens when my constraints reject most draws?”

```bash
cargo run -- analyze-generator \
  --model data/_exp/base/model_union.json \
  --wordlist data/_exp/base/wordset.txt \
  --style numbers-symbols \
  --words 4 \
  --samples 10000 \
  --pick-best-of 10 \
  --max-chars 16 \
  --seed 1 \
  --show-top 0
```

Key signals:

- `accept_rate`: when this is \(\ll 1\), the generator is doing a lot of rejection sampling; “nominal bits” becomes
  an optimistic upper-ish bound.
- `chars` and `ms` quantiles: tighter limits often shorten strings (faster) while concentrating the distribution.

Example excerpt (this exact config, seed=1):

- `accept_rate: 0.041043`
- `typing_gain_ms (picked - baseline): Δmean=-322.3  Δp95=-744.1`
- `entropy_penalty_bits_word_marginals (Δsum_positions upper bound): ΔH1=-1.356  ΔH2=-1.704  ΔHinf=-2.597`

## 3) Compare built-in styles and wordset sizes (bits vs ms)

This answers: “Which style preset gives me the best ms_p95 at a target bits budget, under the uniform-wordset assumption?”

```bash
cargo run -- pareto-styles \
  --model data/_exp/base/model_union.json \
  --wordlist data/_exp/base/wordset.txt \
  --words 4 \
  --n 512,1024,2048,4096,8192 \
  --style spaces,hyphens,numbers,numbers-symbols,login-title-2digits,login-title-endpunct \
  --samples 8000 \
  --seed 1 \
  --recommend \
  --target-bits 52 \
  --min-hit-frac 0.70
```

How to read it:

- `bits`: “nominal bits” under **uniform choice over the first N words** (plus explicit randomness counted for built-in
  styles). This is clean only if your actual procedure is close to that.
- `ms_p95`: a “pessimistic but not worst-case” timing target for login UX.
- `hit_mean` / `hit_p05`: a reliability signal; styles that introduce digits/symbols often reduce hit rate because those
  digraphs may be missing and fall back to a global prior.

## 4) Audit model coverage (how often are we falling back?)

This answers: “Is the model informed on typical words, or mostly using global/backoff?”

```bash
cargo run -- audit-model \
  --model data/_exp/base/model_union.json \
  --corpus data/_exp/base/corpus.txt \
  --seed 1
```

