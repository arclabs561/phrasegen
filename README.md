# fastphrase

Learn a simple digraph (adjacent grapheme-pair) timing model from typing-timing data, then score phrases by predicted total entry time.

## Dataset formats

### CSV

Headers:
- `phrase` (string)
- `digraph_dt_ms_json` (string containing a JSON array of numbers, milliseconds)

Example: `data/example.csv`

### JSONL

Each line is:

```json
{"phrase":"hello","digraph_dt_ms":[120,90,110,130]}
```

## Quick start

From `fastphrase/`:

```bash
just base data/base
cargo run -- score --model data/base/model_union.json --phrase "fastphrase" --explain-digraphs
just login-sample data/base data/user  # after `just personalize` + some `just record`
```

Repo note: generated artifacts live under `data/` and are ignored by default (see `.gitignore`).

## Golden path (DX-first)

Run these in order:

```bash
# 1) Build a strong base (public typing data + default corpus + planned wordset)
just base data/base

# 2) Record your typing over time (appends to data/user/user.jsonl)
just record 20

# 3) Build a personalized model + personalized wordset
just personalize data/base data/user

# 4) Sample “login password” style outputs (no spaces, TitleCase, 2 digits)
just login-sample data/user/model_personalized.json data/user/wordset_user.txt 10
```

## “Login password” style (no spaces, caps + numbers)

If you want a passphrase-like password for a typical login box (often no spaces), sample from your curated wordset
but **change formatting at the end**:

```bash
# TitleCase each word, concatenate (no separator), and append 2 random digits:
cargo run -- sample-passphrases \
  --model data/base12/model_union.json \
  --wordlist data/base12/wordset.txt \
  --count 12 --words 4 \
  --style login-title-2digits \
  --max-chars 28 \
  --seed 2
```

This keeps the “word randomness” story intact, while producing a password string that matches common policy constraints.

## Showing alternatives (UX)

If you want “a few close options” per generated password (e.g., “same shape but faster to type”), you can ask
`sample-passphrases` to do a small local search:

```bash
cargo run -- sample-passphrases \
  --model data/base12/model_union.json \
  --wordlist data/base12/wordset.txt \
  --style login-title-2digits \
  --count 2 \
  --alternatives 4 \
  --alt-tries 120 \
  --alt-mode faster \
  --meta \
  --seed 1
```

Security note: choosing among alternatives manually introduces user bias. Treat alternatives as a **debugging/tuning tool**
unless you also randomize the final choice.

### Fastness percentile (interpretability)

To make “how fast is this?” easier to interpret, `sample-passphrases` can optionally calibrate a **fastness percentile**
against the same generator settings (style/case/regex/constraints) using Monte Carlo reference sampling:

```bash
cargo run -- sample-passphrases \
  --model data/base12/model_union.json \
  --wordlist data/base12/wordset.txt \
  --style numbers-symbols \
  --words 4 \
  --count 3 \
  --max-chars 28 \
  --alternatives 3 \
  --alt-tries 120 \
  --percentile \
  --percentile-samples 5000 \
  --meta \
  --seed 1
```

It prints `fast_pct` where **higher is faster** (e.g. `fast_pct=90` means “faster than ~90% of reference samples”).

## 1Password-style “Memorable Password” separators

1Password’s generator offers a “Memorable” password type with options including **hyphen separators**, and (per 1Password staff) separators that are **Numbers** or **Numbers and Symbols** instead of hyphens.
See:
- `https://1password.com/password-generator/` (Memorable + “Capitalize the first letter”, “Numbers”, “Symbols” options shown)
- `https://www.1password.community/discussions/1password/feature-request---include-number-in-memorable-passwords/167106/replies/167111` (staff note about separators “Hyphens” vs “Numbers” vs “Numbers and Symbols”)

You can reproduce those formatting styles with `fastphrase`:

```bash
# Hyphens:
cargo run -- sample-passphrases --model data/base12/model_union.json --wordlist data/base12/wordset.txt \
  --count 12 --words 4 --style hyphens --seed 3

# Numbers as separators:
cargo run -- sample-passphrases --model data/base12/model_union.json --wordlist data/base12/wordset.txt \
  --count 12 --words 4 --style numbers --seed 3

# Numbers+symbols as separators:
cargo run -- sample-passphrases --model data/base12/model_union.json --wordlist data/base12/wordset.txt \
  --count 12 --words 4 --style numbers-symbols --seed 3
```

For “ends with `!`/`?`/`_`/`-`”, use `--suffix-regex`:

```bash
cargo run -- sample-passphrases --model data/base12/model_union.json --wordlist data/base12/wordset.txt \
  --count 12 --words 4 --style login-title-endpunct --seed 4
```

## Pareto optimization: security vs typing time (uses `pare`)

To compare 1Password-like formatting styles and wordset sizes on a **Pareto frontier**
(\(\uparrow\) bits, \(\downarrow\) expected ms), run:

```bash
cargo run -- pareto-styles \
  --model data/base12/model_union.json \
  --wordlist data/base12/wordset.txt \
  --words 4 \
  --samples 5000 \
  --n 1024,4096,16384,32768 \
  --style hyphens,numbers,numbers-symbols,login-title-2digits \
  --seed 1
```

It prints:
- ms quantiles (`ms_p50`, `ms_p95`, `ms_p99`) rather than only the mean (robustness matters)
- `hit_mean` and `hit_p05` (coverage diagnostics: low hit rate means more fallback-to-global for digits/symbols)
- `shift_mean` / `shift_p95` (US-layout heuristic for “modifier pain”; not derived from timing data)
- optional `idf_*` if `corpus.txt` is available next to the wordset

### Math: what those numbers mean

#### Typing-time objective

Given a phrase \(s\) with \(m\) digraphs, the model predicts a per-digraph latency \(\mu_{g_i}\) for each adjacent grapheme-pair
\(g_i\), and reports:

- \( \widehat{T}(s) = \sum_{i=1}^{m} \mu_{g_i} \) (predicted milliseconds)

When a digraph \(g_i\) is unseen, scoring falls back (base global mean / per-key backoff in `PersonalizedModel`), which is why
`hit_*` matters.

#### Security bits (uniform sampling)

If you sample \(k\) words uniformly from a wordset of size \(N\):

- **with repeats**: \(H = k \log_2 N\)
- **without repeats**: \(H = \log_2 \bigl(N (N-1)\cdots(N-k+1)\bigr)\)

For style presets that add randomness:
- digits as separators add \((k-1)\log_2 10\)
- digits+symbols separators add \((k-1)\log_2 20\) (10 digits + 10 symbols)
- `login-title-2digits` adds \(\log_2 100\) from the suffix

This is a **clean guarantee** only when the generator is close to uniform over its output distribution and users don’t bias
selection (e.g. “pick the prettiest”).

#### Reliability, shift, and IDF (diagnostics, not guarantees)

- `hit_mean`/`hit_p05`: fraction of digraphs that were scored using learned digraph means. Low hit rate means more fallback-to-global.
- `shift_*`: fraction of characters that likely require shift on a US layout (heuristic; not measured by timing data).
- `idf_*`: average \(\log_2(\text{total\_count}/\text{count}(w))\) from `corpus.txt`; this is a **commonality/bias-risk signal**, not entropy.

To get a single “best under constraints” suggestion:

```bash
cargo run -- pareto-styles \
  --model data/base12/model_union.json \
  --wordlist data/base12/wordset.txt \
  --words 4 \
  --samples 5000 \
  --n 1024,4096,16384,32768 \
  --style hyphens,numbers,numbers-symbols,login-title-2digits \
  --recommend \
  --target-bits 60 \
  --min-hit-frac 0.70 \
  --seed 1
```

## Analyze the *actual* generator distribution (`analyze-generator`)

When you care about “maximally secure” in practice, the generator details matter:
- rejection sampling from `--max-chars/--min-chars`
- regex gaps/prefix/suffix (which may be non-uniform)
- and especially “pick the best-looking / fastest among M alternatives”, which can reduce entropy

Use:

```bash
cargo run -- analyze-generator \
  --model data/base12/model_union.json \
  --wordlist data/base12/wordset.txt \
  --style numbers-symbols \
  --words 4 \
  --samples 20000 \
  --pick-best-of 5 \
  --max-chars 28 \
  --seed 1
```

It reports ms quantiles, hit/shift diagnostics, and **empirical entropy estimates** over sampled outputs, plus a
word-marginal entropy diagnostic that’s sensitive to “pick best of M” bias even when full-output collisions are rare.

### Math: effective entropy, rejection, and “pick best of M”

`analyze-generator` estimates several entropies from sampled output frequencies:

- \(H_1\) (Shannon / “average surprise”)
- \(H_2\) (Rényi-2 / collision entropy)
- \(H_\infty\) (min-entropy / worst-case surprise)

These are estimated by the plugin (empirical) counts of observed outputs. When the space is enormous, you may see **no repeats**
at 20k–200k samples; in that regime, the plugin estimator is **sample-size-limited**, and you should treat it as a lower bound.

In addition, when there are **0 observed collisions** among \(n\) samples, we print a simple 95% upper bound on collision
probability \(p_2\) (hence a lower bound on \(H_2\)) using a binomial approximation over the \(\binom{n}{2}\) sample pairs:
\(p_2 \lesssim -\ln(0.05) / \binom{n}{2}\).

### Canonical generator policy (recommended)

For a “maximally secure *and* easy typing” default that stays close to the clean entropy story:

- **generator**: `sample-passphrases --style numbers-symbols --words 4 --max-chars 28`
- **sampling**: generate **one** uniformly random output (no manual picking among alternatives)
- **wordset**: use a planned wordset that meets target bits under uniform sampling (`plan-passphrase --target-bits 60 ...`)

If you display alternatives and manually pick, you are **changing the distribution**. `--pick-best-of M` in `analyze-generator`
is our crude model of that bias.

### Real output example (50k samples, pick-best-of 10)

This command:

```bash
cargo run -- analyze-generator \
  --model data/base12/model_union.json \
  --wordlist data/base12/wordset.txt \
  --style numbers-symbols \
  --words 4 \
  --samples 50000 \
  --pick-best-of 10 \
  --max-chars 28 \
  --seed 1 \
  --show-top 0
```

Produced (excerpt):

```text
accept_rate: 1.000000
nominal_bits_upperish (ignores rejection + non-uniform regex): 72.966

baseline (pick_best_of=1 semantics):
  observed_collision_pairs: 0 / 1249975000
  ms: mean=5005.7 ... p95=5727.6
  hit: mean=0.760 p05=0.700
  collision_bound_95pct: p2 <= 2.396634e-9  =>  H2 >= 28.636  (binomial approx, 0 collisions)

picked_fastest_of_10 (models manual choice):
  ms: mean=4394.6 ... p95=4738.8
  hit: mean=0.708 p05=0.667

word_marginal_entropy_bits (upper bound on joint entropy):
  Δ(sum_positions) (picked - baseline): ΔH1=-1.271 ΔH2=-1.731 ΔHinf=-2.860
typing_gain_ms (picked - baseline): Δmean=-611.1  Δp95=-988.8
```

Interpretation:
- The **collision bound** gives a conservative, repeat-free **lower bound** on collision entropy \(H_2\) of the *output distribution*.
- The **word-marginal deltas** are a practical “bias meter”: picking the fastest of 10 improves typing time, but it measurably
  concentrates the distribution (especially in \(H_\infty\)).

Two mechanisms reduce entropy in practice:

- **Rejection sampling**: `--max-chars`/`--min-chars` filters proposals. The resulting distribution is the proposal distribution
  conditioned on acceptance; `accept_rate` is printed to show how strong that conditioning is.
- **Selection bias**: `--pick-best-of M` simulates generating \(M\) candidates and choosing the fastest. This changes the distribution
  even if the underlying generator is uniform. Full-output entropy may still be hard to estimate without repeats, so we also report
  word-marginal and adjacent-pair entropy proxies (words-only) that usually *do* move under selection.

## CI

This repo includes a minimal GitHub Actions workflow at `fastphrase/.github/workflows/ci.yml` that runs:
- `cargo fmt --check`
- `cargo test`

## Seeding from public keystroke datasets

### CMU Keystroke Dynamics (DSL Strong Password)

This dataset provides timing features for 51 subjects typing the fixed password `.tie5Roanl` (see the CMU page: [`cs.cmu.edu/~keystroke/`](https://www.cs.cmu.edu/~keystroke/)).

Import (downloads the public CSV and writes `Row` JSONL):

```bash
cargo run -- import-cmu-dsl --output data/cmu_dsl.jsonl
cargo run -- fit --input data/cmu_dsl.jsonl --output-model model_cmu.json
```

### Bulk dataset download

This downloads raw files into a directory (for later import/parsing):

```bash
cargo run -- download-datasets --out-dir data/datasets
```

### Union of all downloaded datasets

This downloads missing raw files (CMU + BKSD) and writes one JSONL you can fit from:

```bash
cargo run -- union-datasets --datasets-dir data/datasets --output data/union.jsonl
cargo run -- fit --input data/union.jsonl --output-model model_union.json
```

### CMU LASER-2012 (Free vs. Transcribed Text)

`download-datasets` and `union-datasets` also include CMU’s LASER-2012 supplementary dataset:
`DSL-Free-vs-Transcribed.zip` with `data/TimingFeatures-DD.txt` and `data/SessionMap.txt`.

We import DD digraph timing features where both keys can be mapped to a single character (letters, space, some punctuation).
Rows are tagged as `cmu_laser2012_free` or `cmu_laser2012_trans` when possible.

### KeyRecs (Zenodo, CC-BY 4.0)

`download-datasets` can also fetch the KeyRecs dataset (`free-text.csv`, `fixed-text.csv`) from Zenodo.
We currently import the **free-text** file’s `DD.key1.key2` digraph latencies (skipping negative or unmappable key pairs),
tagged as `keyrecs_free_text`.

### GREYC web-based dataset (archived)

`download-datasets` now also fetches the GREYC web-based dataset archive, and `union-datasets` includes it automatically.
This dataset includes many different passwords/passphrases plus keypress timestamps; we import:
- `password.txt` / `passphrase.txt` as the phrase
- `p_pp.txt` as digraph down-down timings (ms)

You can also import it directly:

```bash
cargo run -- import-greyc-web --input data/datasets/greyc_web/webkeystroke.tar.gz --output data/greyc_web.jsonl
```

## Personalization (fine-tune after base)

You can adapt a base model (e.g. `model_union.json`) to a particular user’s timing rows:

```bash
# user.jsonl is in the same Row JSONL format (`phrase`, `digraph_dt_ms`)
cargo run -- adapt-model --base-model model_union.json --user-data user.jsonl --output-model model_user.json
```

### Recording your own data over time (just type)

This appends new timing rows to `data/user/user.jsonl` as you do sessions:

```bash
cargo run -- record-session --reps 20
```

To record a fixed target string (for consistency):

```bash
cargo run -- record-session --reps 20 --target ".tie5Roanl"
```

To auto-update a personalized model after each session:

```bash
cargo run -- record-session --reps 20 --target ".tie5Roanl" \
  --base-model model_union.json \
  --output-model model_user.json
```

### One-shot personalization pipeline (base → user → wordset → examples)

After you’ve run `base-pipeline` at least once, and you’ve accumulated some user rows with `record-session`,
you can regenerate everything personalized in one command:

```bash
cargo run -- personalize-pipeline \
  --base-dir data/base \
  --user-data data/user/user.jsonl \
  --out-dir data/user \
  --target-bits 60 \
  --words 4
```

### Personalized model with imputation/backoff (recommended for small user data)

This builds a `PersonalizedModel` JSON that can impute unseen digraphs using your per-key tendencies:

```bash
cargo run -- build-personalized-model \
  --base-model data/base/model_union.json \
  --user-data data/user/user.jsonl \
  --output-model data/user/model_personalized.json
```

You can then use it anywhere a model is accepted (e.g. `score`, `plan-passphrase`, `generate`, `estimate-search`).

## Search-space enumeration time

Estimate how long it would take to enumerate the full passphrase space for a wordlist:

```bash
cargo run -- estimate-search --model model_union.json --wordlist data/wordlist.txt --words 4 --allow-repeats true
```

## Word difficulty × entropy (k-gram transitions)

You can score words by combining:
- **difficulty**: predicted typing time from a timing model (ms)
- **entropy**: k-gram surprisal in bits from a corpus model

```bash
# Example corpus can be `data/wordlist.txt` (no counts => weight=1 per line)
cargo run -- rank-words --model model_cmu.json --corpus data/wordlist.txt --k 3 --top 20
```

### Export an optimized wordset (then generate)

```bash
# 1) Build a base model (union of public datasets)
cargo run -- union-datasets --datasets-dir data/datasets --output data/union.jsonl
cargo run -- fit --input data/union.jsonl --output-model model_union.json

# 2) Export a “fast-to-type” wordlist from a corpus (for uniform sampling security)
cargo run -- export-wordset --model model_union.json --corpus data/wordlist.txt --output data/wordset.txt --top 5000 --objective ms-only --min-hit-frac 0.95 --min-vowels 1

# 3) Generate using that optimized wordset
cargo run -- generate --model model_union.json --wordlist data/wordset.txt --samples 20000 --top 20
```

### Plan a wordset for a target entropy (base-only)

This picks the smallest wordset size needed (roughly) to hit `target-bits` for `words` words, chooses the best words by a selected objective, writes the wordset, and reports the expected enumeration time.

Security note: if you will **sample uniformly at random from the resulting wordset**, then the clean objective is `--objective ms-only` (typing speed only). Entropy comes from the wordset size \(N\) and number of words \(k\): \(k \log_2 N\).

```bash
cargo run -- plan-passphrase \
  --model model_union.json \
  --corpus data/wordlist.txt \
  --output data/wordset_planned.txt \
  --words 4 \
  --target-bits 60 \
  --allow-repeats \
  --samples 5000 \
  --objective ms-only \
  --min-hit-frac 0.95 \
  --min-vowels 1
```

## Base pipeline (no custom corpus)

This tries to “just work” by using only public online sources:
- typing data: CMU DSL + BKSD + CMU LASER-2012 + GREYC web + KeyRecs (union)
- corpus: `aparrish/wordfreq-en-25000` + `dwyl/english-words` tail (to reach a large vocabulary for entropy planning; apply `--min-vowels` / `--min-hit-frac` to avoid abbreviation-like tokens)

```bash
cargo run -- base-pipeline --target-bits 60 --words 4
```

## Evaluating the base model

To measure predictive quality on held-out data (train/test split), run:

```bash
cargo run -- eval-model --input data/base/union.jsonl --seed 1
```

You can also focus on a single source:

```bash
cargo run -- eval-model --input data/base/union.jsonl --source-prefix cmu_dsl --seed 1
```

## Interpreting metrics (what is “good” vs “bad”)

### `eval-model`

These are held-out prediction metrics (train/test split) on the dataset you provide.

- **`digraph_eval.mae`**: typical absolute error per digraph timing in ms.
  - A rough sanity check is the ratio **`mae_over_global_mean`**:
    - **\< 0.5**: usually decent signal
    - **0.5–1.0**: weak-to-moderate (often “some structure + lots of noise/outliers”)
    - **\> 1.0**: something is off (bad parse, unit mismatch, or huge pauses/outliers)
- **`digraph_abs_err_quantiles`**: makes outliers visible. If p99 is enormous, clamp or filter.
- **`phrase_eval`** sums digraph errors across a phrase; correlations are more meaningful when phrases vary.

Important bounds/assumptions:
- `--clamp-dt-ms` winsorizes “pause” events; it improves robustness but also changes the target being evaluated.
- Rows with any negative/non-finite timings are skipped (reported as `rows.test_skipped_invalid`).

### `audit-model`

This is a downstream “product sanity” audit: it answers whether the model covers typical words.

- **`digraph_hit_frac`**: fraction of digraphs in scored words that used a learned digraph mean (vs fallback/backoff).
  - **\> 0.7**: model is usually informative for typical words
  - **\> 0.9**: very good coverage
  - **\< 0.3**: the model is mostly a length/global-mean model for that corpus
- **`normalized_vs_global`**: \( \text{predicted\_ms} / (\text{digraphs} \cdot \text{global\_mean\_ms}) \).
  - **1.0** means “average digraph speed”.
  - Values \<1 and \>1 aren’t “good/bad” universally; use the **distribution** (quantiles) and compare across models.

### `score --ref-wordlist ...`

Calibration samples a reference distribution and reports:
- empirical percentile of your phrase’s predicted ms among sampled phrases
- reference quantiles (p10/p50/p90/…)

Lower percentile means “faster than most reference phrases”.

## Sampling final passphrases (constraints + optional regex gaps)

Emit K random passphrases (with predicted typing time), optionally inserting regex-generated gap strings and enforcing length constraints:

```bash
cargo run -- sample-passphrases \
  --model data/user/model_personalized.json \
  --wordlist data/user/wordset_user.txt \
  --count 10 \
  --words 4 \
  --gap-regex '[-_.]{1,2}[0-9]{0,2}' \
  --max-chars 32 \
  --seed 1
```

