# Guide (day-to-day usage)

This is the “what do I actually run?” page. The root `README.md` stays short; this file keeps the practical details.

## Cheat sheet (5 commands)

```bash
# base model + wordset
cargo run -- base-pipeline --out-dir data/base --target-bits 60 --words 4 --seed 1

# append typing data
cargo run -- record-session --reps 20 --output data/user/user.jsonl

# personalize (writes model + wordset under data/user/)
cargo run -- personalize-pipeline --base-dir data/base --user-data data/user/user.jsonl --out-dir data/user --target-bits 60 --words 4 --seed 1

# sample passphrases
cargo run -- sample-passphrases --model data/user/model_personalized.json --wordlist data/user/wordset_user.txt --words 4 --count 10 --max-chars 32 --seed 1

# score an arbitrary string
cargo run -- score --model data/user/model_personalized.json --phrase "correct-horse-battery-staple"
```

If you use `just`, the same workflows are wrapped in `justfile` recipes: run `just --list`.

## Typical workflow

### 1) Build a base model and wordset (public-only)

```bash
cargo run -- base-pipeline --out-dir data/base --target-bits 60 --words 4 --seed 1
```

Artifacts written under `data/base/`:
- `model_union.json`: timing model fit from unioned public typing data
- `corpus.txt`: frequency-ish corpus used for planning
- `wordset.txt`: planned wordset for the requested bits/words

### 2) Record your typing (optional, accumulates)

```bash
cargo run -- record-session --reps 20 --output data/user/user.jsonl
```

### 3) Personalize (optional)

```bash
cargo run -- personalize-pipeline \
  --base-dir data/base \
  --user-data data/user/user.jsonl \
  --out-dir data/user \
  --target-bits 60 \
  --words 4 \
  --seed 1
```

### 4) Sample passphrases

```bash
cargo run -- sample-passphrases \
  --model data/user/model_personalized.json \
  --wordlist data/user/wordset_user.txt \
  --words 4 \
  --count 10 \
  --max-chars 32 \
  --seed 1
```

## “Login password” formatting (no spaces, caps + numbers)

Use the built-in style presets:

```bash
cargo run -- sample-passphrases \
  --model data/user/model_personalized.json \
  --wordlist data/user/wordset_user.txt \
  --style login-title-2digits \
  --words 4 \
  --count 10 \
  --max-chars 28 \
  --seed 2
```

## Constraints and regex gaps/prefix/suffix

`sample-passphrases` supports:
- `--min-chars`, `--max-chars`
- `--gap-regex` (between words)
- `--prefix-regex`, `--suffix-regex`

Example:

```bash
cargo run -- sample-passphrases \
  --model data/user/model_personalized.json \
  --wordlist data/user/wordset_user.txt \
  --words 4 \
  --count 10 \
  --gap-regex '[-_.]{1,2}[0-9]{0,2}' \
  --suffix-regex '[!?]' \
  --max-chars 32 \
  --seed 1
```

## Debug/diagnostics switches

- `--meta`: print `norm`, `ms/dg`, `shift`, `hit_frac` per sample
- `--percentile`: calibrate a “fastness percentile” against the same generator settings
- `--alternatives N`: show near-by alternatives (note: manual choice introduces bias)

## Common pitfalls

- If `sampling.accept_rate` is very low, your constraints are too tight (e.g. `--max-chars` too small for your chosen `--style`/`--words`).
- Alternatives are great for tuning, but manual selection changes the distribution (and therefore effective entropy).

