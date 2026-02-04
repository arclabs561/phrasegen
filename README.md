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

## Core commands you’ll actually use

- **Build base artifacts**:

```bash
cargo run -- base-pipeline --out-dir data/base --target-bits 60 --words 4 --seed 1
```

- **Record typing** (interactive; appends JSONL):

```bash
cargo run -- record-session --reps 20 --output data/user/user.jsonl
```

- **Personalize** (base → personalized model + planned wordset):

```bash
cargo run -- personalize-pipeline --base-dir data/base --user-data data/user/user.jsonl --out-dir data/user --target-bits 60 --words 4 --seed 1
```

- **Sample passphrases** (constraints + optional regex gaps/prefix/suffix):

```bash
cargo run -- sample-passphrases \
  --model data/user/model_personalized.json \
  --wordlist data/user/wordset_user.txt \
  --words 4 \
  --count 10 \
  --gap-regex '[-_.]{1,2}[0-9]{0,2}' \
  --max-chars 32 \
  --seed 1
```

## What the output fields mean

- **`predicted_ms`**: predicted total entry time under the timing model.
- **`digraph_hit_frac` / `hit_frac`**: fraction of digraphs scored using learned digraph means (vs fallback/backoff).
- **`normalized_vs_global` / `norm`**: normalized speed vs the model’s global mean digraph speed.
- **`shift`**: heuristic fraction of characters that likely require shift on a US layout.
- **`sampling.accept_rate`**: how tight your constraints are (rejection sampling rate).

For the math and interpretation, see `docs/math.md`.

## Public data sources (base model)

The default base pipeline unions multiple public keystroke datasets (downloaded on demand) and fits a robust digraph model.
See the CLI (`download-datasets`, `union-datasets`) and the code in `src/import.rs` for exact formats/parsers.

## Development

- **Recipes**: `just --list` (see `justfile`)
- **Tests**: `cargo test`
- **CI**: `.github/workflows/ci.yml`

