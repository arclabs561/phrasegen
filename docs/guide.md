# Guide (day-to-day usage)

This is the “what do I actually run?” page. The root `README.md` stays short; this file keeps the practical details.

## Cheat sheet (5 commands)

```bash
# base model + wordset
cargo run -- base-pipeline --out-dir data/base --target-bits 60 --words 4

# append typing data
cargo run -- record-session --reps 20 --output data/user/user.jsonl

# (optional) summarize your accumulated rows (corrections, total_ms, etc.)
cargo run -- analyze-user-data --input data/user/user.jsonl --model data/base/model_union.json

# personalize (writes model + wordset under data/user/)
cargo run -- personalize-pipeline --base-dir data/base --user-data data/user/user.jsonl --out-dir data/user --target-bits 60 --words 4

# sample passphrases
cargo run -- sample-passphrases --model data/user/model_personalized.json --wordlist data/user/wordset_user.txt --words 4 --count 10 --max-chars 32

# score an arbitrary string
cargo run -- score --model data/user/model_personalized.json --phrase "correct-horse-battery-staple"
```

Notes:
- Add `--seed` when you want **reproducible demos/experiments**.
- Omit `--seed` for real passwords.

If you use `just`, the same workflows are wrapped in `justfile` recipes: run `just --list`.

## Defaults that work well

- **Passphrase field (password manager / paste allowed)**: use `--style spaces` or `--style hyphens` and a comfortable `--max-chars` (often 32).
- **Tight login box**: use `--style login-title-2digits` and set `--max-chars` to the site limit (often 28).
- **Clean “bits” accounting**: keep `--pick-best-of 1` and avoid manual choice among `--alternatives`.
- **If you do choose from a menu**: quantify the penalty with `analyze-generator` (see `docs/experiments.md`).

## “Login password” formatting (no spaces, caps + numbers)

Use the built-in style presets:

```bash
cargo run -- sample-passphrases \
  --model data/user/model_personalized.json \
  --wordlist data/user/wordset_user.txt \
  --style login-title-2digits \
  --words 4 \
  --count 10 \
  --max-chars 28
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
  --max-chars 32
```

## Debug/diagnostics switches

- `--meta`: print `norm`, `ms/dg`, `shift`, `hit_frac` per sample
- `--percentile`: calibrate a “fastness percentile” against the same generator settings
- `--alternatives N`: show near-by alternatives (note: manual choice introduces bias)

## Quantifying bias / effective entropy

When you use constraints (`--max-chars`, regex gaps/prefix/suffix) or any “menu choice” workflow (alternatives, best-of-\(M\)),
the output distribution is no longer uniform. Use `analyze-generator` and `pareto-styles` to quantify the tradeoffs.
See `docs/experiments.md` for reproducible runs and how to interpret the output.

## Common pitfalls

- If `sampling.accept_rate` is very low, your constraints are too tight (e.g. `--max-chars` too small for your chosen `--style`/`--words`).
- Alternatives are great for tuning, but manual selection changes the distribution (and therefore effective entropy).

## Correction-aware recording (backspace)

By default, `record-session` treats **backspace as an abort** so the accumulated rows represent “clean” uninterrupted typing.
This keeps the fitted timing model closer to “motor latency per digraph” rather than “motor latency + correction behavior”.

If you want to **measure correction behavior**, you can allow backspace:

```bash
cargo run -- record-session --reps 20 --output data/user/user.jsonl --allow-backspace
```

Rows recorded this way include:

- `backspaces`: number of backspace keypresses observed
- `total_ms`: wall-clock time from first keypress to Enter

Methodology note: rows with `backspaces > 0` are **excluded by default** from fitting/adaptation; they’re meant for
diagnostics and future “time including corrections” models.

You can summarize and sanity-check this data (and optionally fit a simple correction model) with:

```bash
cargo run -- analyze-user-data --input data/user/user.jsonl --model data/user/model_personalized.json
```

And for a single phrase, you can print an **estimated total time for clean entry** using a fitted correction model:

```bash
cargo run -- score --model data/user/model_personalized.json --phrase "correct-horse-battery-staple" --corrections-from data/user/user.jsonl
```