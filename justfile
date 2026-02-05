set dotenv-load := false

# phrasegen: handy automation for common workflows
#
# Examples:
#   just base data/base4
#   just eval data/base4
#   just record 20 ".tie5Roanl"
#   just personalize data/base4 data/user
#   just sample data/user/model_personalized.json data/user/wordset_user.txt 10

default:
  @just --list

# Build the full public-data base pipeline into OUT_DIR.
base OUT_DIR="data/base":
  cargo run -- base-pipeline --out-dir {{OUT_DIR}} --target-bits 60 --words 4 --samples 20000 --top 10

# Demo: repeatable sampling with seed=42 (default style: numbers-symbols).
demo STYLE="numbers-symbols" SEED="42" COUNT="10" MAX_CHARS="32" PICK="1":
  cargo run -- sample-passphrases --model data/user/model_personalized.json --wordlist data/user/wordset_user.txt --style {{STYLE}} --words 4 --count {{COUNT}} --max-chars {{MAX_CHARS}} --seed {{SEED}} --pick-best-of {{PICK}}

# Download all supported raw datasets into OUT_DIR/datasets.
download OUT_DIR="data/base":
  cargo run -- download-datasets --out-dir {{OUT_DIR}}/datasets

# Union all datasets (downloads missing raw files into DATASETS_DIR).
union DATASETS_DIR="data/datasets" OUT_JSONL="data/union.jsonl":
  cargo run -- union-datasets --datasets-dir {{DATASETS_DIR}} --output {{OUT_JSONL}}

# Fit a timing model from a union file (robust by default).
fit INPUT="data/union.jsonl" OUT_MODEL="model_union.json" CLAMP_MS="2000":
  cargo run -- fit --input {{INPUT}} --min-count 3 --clamp-dt-ms {{CLAMP_MS}} --output-model {{OUT_MODEL}}

# Evaluate a union dataset with a robust clamp.
eval OUT_DIR="data/base" CLAMP_MS="2000":
  cargo run -- eval-model --input {{OUT_DIR}}/union.jsonl --seed 1 --clamp-dt-ms {{CLAMP_MS}}

# Audit model coverage on its corpus.
audit OUT_DIR="data/base":
  cargo run -- audit-model --model {{OUT_DIR}}/model_union.json --corpus {{OUT_DIR}}/corpus.txt --seed 1

# Pareto compare 1Password-like styles (bits vs ms_p95).
pareto OUT_DIR="data/base":
  cargo run -- pareto-styles --model {{OUT_DIR}}/model_union.json --wordlist {{OUT_DIR}}/wordset.txt --words 4 --samples 5000 --n 1024,4096,16384,32768 --style spaces,hyphens,numbers,numbers-symbols,login-title-2digits,login-title-endpunct --seed 1 --recommend --target-bits 60 --min-hit-frac 0.70

# Analyze the actual generator distribution (incl. “pick best of M” entropy penalty).
# Usage:
#   just analyze data/base12 numbers-symbols 10 28
analyze OUT_DIR="data/base" STYLE="numbers-symbols" PICK="5" MAX_CHARS="28":
  cargo run -- analyze-generator --model {{OUT_DIR}}/model_union.json --wordlist {{OUT_DIR}}/wordset.txt --style {{STYLE}} --words 4 --samples 20000 --pick-best-of {{PICK}} --max-chars {{MAX_CHARS}} --seed 1

# Run “everything” on a base directory (pareto + generator analyses).
doitall OUT_DIR="data/base" SAMPLES="20000" PICK="10" MAX_CHARS="28":
  just pareto {{OUT_DIR}}
  cargo run -- analyze-generator --model {{OUT_DIR}}/model_union.json --wordlist {{OUT_DIR}}/wordset.txt --style spaces --words 4 --samples {{SAMPLES}} --pick-best-of 1 --max-chars {{MAX_CHARS}} --seed 1 --show-top 5
  cargo run -- analyze-generator --model {{OUT_DIR}}/model_union.json --wordlist {{OUT_DIR}}/wordset.txt --style spaces --words 4 --samples {{SAMPLES}} --pick-best-of {{PICK}} --max-chars {{MAX_CHARS}} --seed 1 --show-top 5
  cargo run -- analyze-generator --model {{OUT_DIR}}/model_union.json --wordlist {{OUT_DIR}}/wordset.txt --style hyphens --words 4 --samples {{SAMPLES}} --pick-best-of 1 --max-chars {{MAX_CHARS}} --seed 1 --show-top 5
  cargo run -- analyze-generator --model {{OUT_DIR}}/model_union.json --wordlist {{OUT_DIR}}/wordset.txt --style hyphens --words 4 --samples {{SAMPLES}} --pick-best-of {{PICK}} --max-chars {{MAX_CHARS}} --seed 1 --show-top 5
  cargo run -- analyze-generator --model {{OUT_DIR}}/model_union.json --wordlist {{OUT_DIR}}/wordset.txt --style numbers --words 4 --samples {{SAMPLES}} --pick-best-of 1 --max-chars {{MAX_CHARS}} --seed 1 --show-top 5
  cargo run -- analyze-generator --model {{OUT_DIR}}/model_union.json --wordlist {{OUT_DIR}}/wordset.txt --style numbers --words 4 --samples {{SAMPLES}} --pick-best-of {{PICK}} --max-chars {{MAX_CHARS}} --seed 1 --show-top 5
  cargo run -- analyze-generator --model {{OUT_DIR}}/model_union.json --wordlist {{OUT_DIR}}/wordset.txt --style numbers-symbols --words 4 --samples {{SAMPLES}} --pick-best-of 1 --max-chars {{MAX_CHARS}} --seed 1 --show-top 5
  cargo run -- analyze-generator --model {{OUT_DIR}}/model_union.json --wordlist {{OUT_DIR}}/wordset.txt --style numbers-symbols --words 4 --samples {{SAMPLES}} --pick-best-of {{PICK}} --max-chars {{MAX_CHARS}} --seed 1 --show-top 5
  cargo run -- analyze-generator --model {{OUT_DIR}}/model_union.json --wordlist {{OUT_DIR}}/wordset.txt --style login-title-2digits --words 4 --samples {{SAMPLES}} --pick-best-of 1 --max-chars {{MAX_CHARS}} --seed 1 --show-top 5
  cargo run -- analyze-generator --model {{OUT_DIR}}/model_union.json --wordlist {{OUT_DIR}}/wordset.txt --style login-title-2digits --words 4 --samples {{SAMPLES}} --pick-best-of {{PICK}} --max-chars {{MAX_CHARS}} --seed 1 --show-top 5
  cargo run -- analyze-generator --model {{OUT_DIR}}/model_union.json --wordlist {{OUT_DIR}}/wordset.txt --style login-title-endpunct --words 4 --samples {{SAMPLES}} --pick-best-of 1 --max-chars {{MAX_CHARS}} --seed 1 --show-top 5
  cargo run -- analyze-generator --model {{OUT_DIR}}/model_union.json --wordlist {{OUT_DIR}}/wordset.txt --style login-title-endpunct --words 4 --samples {{SAMPLES}} --pick-best-of {{PICK}} --max-chars {{MAX_CHARS}} --seed 1 --show-top 5

# Show dataset stats/outliers.
stats INPUT="data/union.jsonl":
  cargo run -- dataset-stats --input {{INPUT}}

# Plan a wordset for entropy target (uses corpus downloaded by base-pipeline).
plan OUT_DIR="data/base":
  cargo run -- plan-passphrase --model {{OUT_DIR}}/model_union.json --corpus {{OUT_DIR}}/corpus.txt --output {{OUT_DIR}}/wordset.txt --words 4 --target-bits 60 --allow-repeats --samples 5000 --seed 1

# Record a quick typing session into the default user file.
record REPS="20" TARGET="":
  @if [ -n "{{TARGET}}" ]; then \
    cargo run -- record-session --reps {{REPS}} --target "{{TARGET}}"; \
  else \
    cargo run -- record-session --reps {{REPS}}; \
  fi

# Build a personalized model with imputation/backoff.
build-personalized BASE_DIR="data/base" USER_JSONL="data/user/user.jsonl" OUT_MODEL="data/user/model_personalized.json":
  cargo run -- build-personalized-model --base-model {{BASE_DIR}}/model_union.json --user-data {{USER_JSONL}} --output-model {{OUT_MODEL}}

# One-shot personalize pipeline (also writes model_personalized.json in OUT_DIR).
personalize BASE_DIR="data/base" OUT_DIR="data/user":
  cargo run -- personalize-pipeline --base-dir {{BASE_DIR}} --user-data {{OUT_DIR}}/user.jsonl --out-dir {{OUT_DIR}} --target-bits 60 --words 4

# Sample final passphrases with optional regex gaps and max chars.
sample MODEL="data/user/model_personalized.json" WORDSET="data/user/wordset_user.txt" COUNT="10" GAP_REGEX="[-_.]{1,2}[0-9]{0,2}" MAX_CHARS="32":
  @if [ -n "{{GAP_REGEX}}" ]; then \
    cargo run -- sample-passphrases --model {{MODEL}} --wordlist {{WORDSET}} --count {{COUNT}} --words 4 --gap-regex '{{GAP_REGEX}}' --max-chars {{MAX_CHARS}}; \
  else \
    cargo run -- sample-passphrases --model {{MODEL}} --wordlist {{WORDSET}} --count {{COUNT}} --words 4; \
  fi

# “Login password” style: no spaces, TitleCase each word, append 2 digits.
login-sample MODEL="data/user/model_personalized.json" WORDSET="data/user/wordset_user.txt" COUNT="10" MAX_CHARS="28":
  cargo run -- sample-passphrases --model {{MODEL}} --wordlist {{WORDSET}} --count {{COUNT}} --words 4 --style login-title-2digits --max-chars {{MAX_CHARS}}

# 1Password-like “Memorable password” styles.
# - hyphens: classic correct-horse style, but with our typing model.
onepass-hyphens MODEL="data/user/model_personalized.json" WORDSET="data/user/wordset_user.txt" COUNT="10" MAX_CHARS="32":
  cargo run -- sample-passphrases --model {{MODEL}} --wordlist {{WORDSET}} --count {{COUNT}} --words 4 --style hyphens --max-chars {{MAX_CHARS}}

# - numbers: use a random digit as the separator between words.
onepass-numbers MODEL="data/user/model_personalized.json" WORDSET="data/user/wordset_user.txt" COUNT="10" MAX_CHARS="32":
  cargo run -- sample-passphrases --model {{MODEL}} --wordlist {{WORDSET}} --count {{COUNT}} --words 4 --style numbers --max-chars {{MAX_CHARS}}

# - numbers+symbols: random digit or common symbol separator between words.
onepass-numsym MODEL="data/user/model_personalized.json" WORDSET="data/user/wordset_user.txt" COUNT="10" MAX_CHARS="32":
  cargo run -- sample-passphrases --model {{MODEL}} --wordlist {{WORDSET}} --count {{COUNT}} --words 4 --style numbers-symbols --max-chars {{MAX_CHARS}}

# Score the repo name itself.
score-phrasegen MODEL="data/base/model_union.json":
  cargo run -- score --model {{MODEL}} --phrase phrasegen

