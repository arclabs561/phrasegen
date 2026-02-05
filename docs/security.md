# Security notes (what “bits” means here)

This project treats “security” as **offline guessing resistance**: how many candidate strings an attacker must try when they
know your generator and your constraints.

Typing-time prediction is a ranking tool; the security story is about the **distribution over outputs**.

If you remember one thing:

- **Don’t use `--seed` for real passwords**, and don’t treat “bits” as clean unless your sampling is close to uniform.

## 1) Attacker model (what we do and don’t claim)

We do **not** claim cryptographic proofs. We do try to make “bits” statements **auditable** under explicit assumptions:

- The attacker knows you used `phrasegen`, your `style`, wordset size, and constraints like `--max-chars`.
- The attacker does **not** know your internal randomness (the RNG draws).
- The attacker may also know you used “menu choice” workflows (best-of-\(M\), alternatives).

Under that model, the meaningful quantity is:

- **Search space size / entropy of the generator distribution** (especially min-entropy if you care about worst-case).

## 2) “Clean bits” vs “effective bits”

### Clean bits (uniform sampling)

Statements like \(k\log_2 N\) are clean only when you are close to **uniform** over a defined set of outputs and you do not
manually choose among candidates.

This is the “honest” mode:

- `--pick-best-of 1`
- `--alternatives 0`
- avoid tight rejection sampling unless you account for it

### Effective bits (biased distributions)

If you do any of the following, the output distribution changes:

- `--pick-best-of > 1` (choose the fastest among multiple draws)
- show `--alternatives` and pick “the nicest”
- tight constraints (e.g. `--max-chars`) that create heavy rejection sampling
- nontrivial regex gaps/prefix/suffix with non-uniform match distributions

In those regimes, use `analyze-generator` to measure how concentrated the distribution becomes. When the full output strings
are all unique in your Monte Carlo sample, prefer the **word-marginal concentration** signal it prints:
`entropy_penalty_bits_word_marginals (Δsum_positions upper bound)`.

See `docs/experiments.md` for reproducible runs.

## 3) Randomness (don’t sabotage yourself)

- For real passwords: **do not pass `--seed`**. Seeding is for reproducible demos/experiments only.
- `phrasegen` uses OS randomness to seed a CSPRNG (ChaCha20) for generation when no seed is provided.

