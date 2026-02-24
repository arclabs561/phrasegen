# phrasegen

Generate and score passphrase-style strings using a learned **typing-timing model** (digraph latencies), so you can trade off **security bits** vs **predicted entry time**.

This repo is designed to “just work” from public datasets, and optionally adapt to your own typing over time.
The intent is practical: make the speed/security tradeoff explicit, measurable, and repeatable.

## Requirements

- Rust toolchain (see `Cargo.toml` `rust-version`)
- Optional: [`just`](https://github.com/casey/just) for the convenience recipes in `justfile`

## Install

From the repo root:

```bash
cargo install --path .
```

## Quick start (what to run)

### Option A — Memorable passphrases right now (recommended starting point)

Uses the [EFF large wordlist](https://www.eff.org/dice) (7776 real English words) with the base timing model.
5 words → ~65 bits. No recording required.

```bash
# 1) Build base timing model from public keystroke data
just base data/base

# 2) Fetch the EFF wordlist (one-time, ~50 KB)
just eff-fetch

# 3) Sample — hyphens style
just eff-sample
# e.g.  jinx-ligament-banter-jokester-glory
#       bunny-snooper-cork-reconvene-constant
#       willow-footrest-cranium-strode-uninstall

# Login-box style (TitleCase + 2 digits)
just eff-sample-login
# e.g.  JinxLigamentBanterJokesterGlory42

# Classic Diceware (space-separated)
just eff-sample-diceware
# e.g.  jinx ligament banter jokester glory
```

### Option B — Personalised model (better timing predictions for your hands)

```bash
# 1) Record your typing (20 samples, appends to data/user/user.jsonl)
just record 20

# 2) Personalise (writes data/user/model_personalized.json + data/user/wordset_user.txt)
just personalize data/base data/user

# 3) Sample with the EFF wordlist + your model
cargo run -- sample-passphrases \
  --model data/user/model_personalized.json \
  --wordlist data/eff_wordlist.txt \
  --words 5 --style hyphens
```

Notes:
- Generated artifacts live under `data/` and are ignored by default (see `.gitignore`).
- Run `cargo run -- --help` to see the full CLI, or `just --list` for recipes.
- For real passwords, avoid `--seed` (it is for reproducible demos/experiments).

## Why EFF over the default wordset?

The default `plan-passphrase` command builds a wordset by **typing speed only**, which produces short fast-to-type fragments like "nba", "usee", "mma" — good entropy, but obscure and hard to recall. The EFF large wordlist contains 7776 common English words chosen to be unambiguous when spoken aloud or typed from memory. 5 EFF words give ~65 bits; 4 give ~52 bits.

| Wordset | Bits (4w) | Bits (5w) | Example |
|---|---|---|---|
| Default (speed-opt.) | 60 | 75 | hebete-nidus-bunny-sarus |
| **EFF (recommended)** | **52** | **65** | **jinx-ligament-banter-jokester-glory** |

The ~8 bit tradeoff (52 vs 60 with 4 words) is worth it for memorability; use 5 EFF words to recover the bits.

## Deterministic demo output (seed=42)

If you want stable, copy/pastable examples for docs/issues, use the built-in demo recipe:

```bash
# Default: numbers-symbols, seed=42
just demo

# “Best-of-N” demo: for each printed sample, draw N candidates and keep the fastest one.
# (This intentionally biases the distribution; it’s for showcasing faster examples.)
just demo numbers-symbols 42 10 32 10
```

Behind the scenes this runs:

```bash
cargo run -- sample-passphrases \
  --model data/user/model_personalized.json \
  --wordlist data/user/wordset_user.txt \
  --style numbers-symbols \
  --seed 42 \
  --pick-best-of 10
```

## Docs (next stop)

Start at `docs/README.md`, or jump directly:
- `docs/guide.md`: how to run it day-to-day (constraints, styles, diagnostics)
- `docs/datasets.md`: what we download and how it becomes rows
- `docs/math.md`: equations + interpretation notes
- `docs/experiments.md`: reproducible experiments + how to scale them up
- `docs/security.md`: attacker model + what “bits” means here

## Public data sources (base model)

The default base pipeline unions multiple public keystroke datasets (downloaded on demand) and fits a robust digraph model.
See the CLI (`download-datasets`, `union-datasets`) and the code in `src/import.rs` for exact formats/parsers.

## Development

- **Recipes**: `just --list` (see `justfile`)
- **Tests**: `cargo test`
- **CI**: `.github/workflows/ci.yml`

