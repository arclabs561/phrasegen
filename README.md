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

From the repo root:

```bash
# 1) Build a base model + corpus + wordset (public typing data + public corpus)
just base data/base

# 2) Record your typing (appends to data/user/user.jsonl)
just record 20

# 3) Personalize (writes data/user/model_personalized.json + data/user/wordset_user.txt)
just personalize data/base data/user

# 4) Sample outputs (predicted ms + passphrase text)
just sample data/user/model_personalized.json data/user/wordset_user.txt 10
```

Notes:
- Generated artifacts live under `data/` and are ignored by default (see `.gitignore`).
- Run `cargo run -- --help` to see the full CLI, or `just --list` for recipes.
- For real passwords, avoid `--seed` (it is for reproducible demos/experiments).

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

