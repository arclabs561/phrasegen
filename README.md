# phrasegen

Generate and score passphrase-style strings using a learned **typing-timing model** (digraph latencies), so you can trade off **security bits** vs **predicted entry time**.

This repo is designed to “just work” from public datasets, and optionally adapt to your own typing over time.

## Requirements

- Rust toolchain (see `Cargo.toml` `rust-version`)
- Optional: [`just`](https://github.com/casey/just) for the convenience recipes in `justfile`

## Quick start (golden path)

From the repo root:

```bash
# 1) Build a base model + corpus + planned wordset (public typing data + public corpus)
just base data/base

# 2) Record your typing over time (appends to data/user/user.jsonl)
just record 20

# 3) Build a personalized model + personalized wordset
just personalize data/base data/user

# 4) Sample outputs (with predicted ms)
just sample data/user/model_personalized.json data/user/wordset_user.txt 10
```

Notes:
- Generated artifacts live under `data/` and are ignored by default (see `.gitignore`).
- Run `cargo run -- --help` to see the full CLI.

## Docs (appendix)

Start at `docs/README.md`:
- `docs/guide.md`: day-to-day workflows (including constraints, styles, and diagnostics)
- `docs/datasets.md`: what we download and how it becomes rows
- `docs/math.md`: math + interpretation notes

For details on constraints/styles/diagnostics, read `docs/guide.md`.

## Public data sources (base model)

The default base pipeline unions multiple public keystroke datasets (downloaded on demand) and fits a robust digraph model.
See the CLI (`download-datasets`, `union-datasets`) and the code in `src/import.rs` for exact formats/parsers.

## Development

- **Recipes**: `just --list` (see `justfile`)
- **Tests**: `cargo test`
- **CI**: `.github/workflows/ci.yml`

