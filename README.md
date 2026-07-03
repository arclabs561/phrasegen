# phrasegen

Generate fast-to-type passphrases from keystroke timing models.

Fits digraph timing models from public keystroke datasets, then samples passphrases that minimize typing time for a given entropy budget. Supports personalization from your own recordings.

## Requirements

- Rust toolchain (see `Cargo.toml` `rust-version`)
- Optional: [`just`](https://github.com/casey/just) for the convenience recipes in `justfile`

## Install

This CLI is not published to crates.io. Install from a checkout:

```bash
cargo install --path .
```

## Quick start (what to run)

### Option A: EFF wordlist

Uses the [EFF large wordlist](https://www.eff.org/dice) (7776 real English words) with the base timing model.
5 words → ~65 bits. No recording required.

```bash
# 1) Build base timing model from public keystroke data
just base data/base

# 2) Fetch the EFF wordlist (one-time, ~50 KB)
just eff-fetch

# 3) Sample (hyphens style)
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

### Option B: Personalised model

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

The default `plan-passphrase` command builds a wordset by **typing speed only**, which produces short fast-to-type fragments like "nba", "usee", "mma": good entropy, but obscure and hard to recall. The EFF large wordlist contains 7776 common English words chosen to be unambiguous when spoken aloud or typed from memory. 5 EFF words give ~65 bits; 4 give ~52 bits.

| Wordset | Bits (4w) | Bits (5w) | Example |
|---|---|---|---|
| Default (speed-opt.) | 60 | 75 | hebete-nidus-bunny-sarus |
| **EFF (recommended)** | **52** | **65** | **jinx-ligament-banter-jokester-glory** |

The ~8 bit tradeoff (52 vs 60 with 4 words) is worth it for memorability; use 5 EFF words to recover the bits.

Deterministic output for docs/issues: `just demo` (uses `--seed 42`). See `docs/` for the guide, dataset details, math, experiments, and security model.

## Public data sources (base model)

The default base pipeline unions multiple public keystroke datasets (downloaded on demand) and fits a robust digraph model.
See the CLI (`download-datasets`, `union-datasets`) and the code in `src/import.rs` for exact formats/parsers.

## Development

- **Recipes**: `just --list` (see `justfile`)
- **Tests**: `cargo test`
- **CI**: `.github/workflows/ci.yml`
