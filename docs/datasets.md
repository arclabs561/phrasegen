# Datasets (what we download and how we use it)

The base pipeline unions multiple public keystroke datasets and fits a single digraph timing model.

## Policy: what we store and commit

- Raw downloads and generated unions/models live under `data/` and are **ignored by git** by default.
- The repo commits only small, deterministic examples (e.g. `data/example.csv`, `data/wordlist.txt`) so tests and formats have something stable.

## Download + union

The CLI can download raw files and then build a single `union.jsonl`:

```bash
cargo run -- download-datasets --out-dir data/base/datasets
cargo run -- union-datasets --datasets-dir data/base/datasets --output data/base/union.jsonl
```

The one-shot `base-pipeline` runs equivalent steps for you.

## Which datasets?

The concrete download URLs and parsers are defined in code (see `src/import.rs` and the `download-datasets` command).
If you need an audit trail (exact URLs, file names, parse assumptions), treat `src/import.rs` as the source of truth.

## What “rows” look like

Internally, we work with rows like:

```json
{"phrase":"hello","digraph_dt_ms":[120,90,110,130]}
```

The `phrase` is the text, and `digraph_dt_ms[i]` is the latency between grapheme i and i+1.

## Implementation pointers

- The dataset-specific parsers live under `src/import.rs`.
- The unified row type is `Row` in `src/data.rs`.

