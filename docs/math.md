# Math notes (what we compute, and what it *doesn’t* guarantee)

This project uses a simple **digraph (adjacent-pair) timing model** and reports three categories of numbers:

- **Typing-time estimates** from the timing model (useful for ranking; not a physics model of typing)
- **Security / entropy accounting** from combinatorics under *explicit* sampling assumptions
- **Diagnostics** (`hit_frac`, `shift_frac`, IDF-ish commonality) that are **not guarantees**

The high-level tension to keep in mind:
**anything that biases generation toward “easy to type” also biases the output distribution**, so “bits” must be stated
against the *actual* generator distribution (or against a truly uniform sampling scheme).

For the attacker model and how to interpret “bits” claims, see `docs/security.md`.

---

## 1) Typing-time objective (digraph timing model)

### 1.1 What is a “digraph” here?

We normalize a phrase and then split it into Unicode grapheme clusters:

1. Normalize to NFC (so composed forms like “é” are consistent).
2. Split into graphemes (`unicode-segmentation`), producing a sequence
   \(\,x_1,\dots,x_L\).

We define **digraphs** as adjacent grapheme pairs:
\[
g_i = (x_i, x_{i+1}) \quad \text{for } i=1,\dots,m,\ \text{where } m=L-1.
\]

Practical note: the built-in `record-session` captures **key-press → key-press** deltas and stores them as a vector
`digraph_dt_ms` of length \(L-1\). Rows whose timing length does not match the grapheme count are skipped during fitting.
So most of the “works out of the box” behavior is calibrated for **ASCII-ish** input (where “one key press ≈ one grapheme”).

### 1.2 Predicted entry time

The timing model stores a mean latency \(\mu_{a,b}\) (ms) per digraph \((a,b)\) and a global fallback mean \(\mu_{\text{global}}\).

The predicted total time is:
\[
\widehat{T}(s) \;=\; \sum_{i=1}^{m} \mu_{x_i,x_{i+1}}.
\]

If a digraph is unseen, scoring falls back (global mean in the base model; structured backoff in `PersonalizedModel`),
which is why `hit_frac` is a key reliability diagnostic.

### 1.3 Normalizations reported in `--meta`

Most comparisons are easier if you separate “length” from “per-transition awkwardness”:

- **Digraph count**: \(m=L-1\)
- **ms per digraph**: \(\widehat{T}(s)/m\) (when \(m>0\))
- **Normalized speed ratio**:
\[
\text{norm}(s)=\frac{\widehat{T}(s)}{m\cdot \mu_{\text{global}}}.
\]

Interpretation: `norm < 1` means “faster than the model’s average digraph,” relative to *that model*.

### 1.4 What you can and cannot claim from this model

What it’s good for:

- **Ranking** phrases under a fixed layout/model (“pick faster-looking candidates”).
- **Personalization**: small user data can move means in the right direction and make rankings feel more “you.”

What it does *not* model well (and where more data helps):

- **Error rate / corrections**: time lost to backspace and retyping can dominate.
- **Higher-order context**: some speed effects are trigraph/word-level chunking, not pairwise additive.
- **Uncertainty**: the digraph model now stores a best-effort per-digraph variance (ms^2) when fitting from rows, which can
  be propagated into a rough phrase-level \(\sigma\) under an independence approximation.
  This is still a diagnostic (and older saved models may not include variance).

---

## 2) “Bits” when the generator is *uniform* over a defined set

These statements are **clean only when the generator is close to uniform** over its outputs and the user does not manually
select among alternatives.

### 2.1 Uniform word sampling from a wordset

If you sample \(k\) words uniformly from a wordset of size \(N\):

- With repeats:
  \[
  H = k \log_2 N.
  \]
- Without repeats:
  \[
  H = \log_2\!\bigl(N(N-1)\cdots(N-k+1)\bigr).
  \]

### 2.2 Extra randomness from formatting (when it is truly random)

Style presets may add additional random choices (e.g. random gap characters from a known set, suffix digits).
Those contribute additive bits **only if** the randomness is explicit and close to uniform.

Important: constraints like `--max-chars`, complex regex gaps/prefix/suffix, and rejection sampling can make the output
distribution *non-uniform*. In that case, “nominal bits” are at best an **upper-ish bound**.

---

## 3) Effective entropy diagnostics (generator distribution)

`analyze-generator` empirically samples from the *actual* generator and estimates entropies from observed output frequencies.
This is useful for answering: “how many bits do we lose by biasing toward fast-to-type phrases or by showing alternatives?”

### 3.1 Plugin entropy estimates from sampled outputs

Let \(S\) be the generator output (a whole passphrase string, including separators/prefix/suffix).
From \(n\) samples, we estimate the distribution \(\hat p(s)=c_s/n\) where \(c_s\) is the observed count of output \(s\).

We report:

- **Shannon entropy** (plugin):
  \[
  \widehat{H}_1 = -\sum_s \hat p(s)\log_2 \hat p(s).
  \]
- **Rényi-2 / collision entropy**:
  \[
  \widehat{H}_2 = -\log_2\!\left(\sum_s \hat p(s)^2\right).
  \]
- **Min-entropy**:
  \[
  \widehat{H}_\infty = -\log_2\!\left(\max_s \hat p(s)\right).
  \]

These are **diagnostics**: for large true entropy, \(n\) is often far too small to accurately estimate tails.

Practical warning: if you see `unique_outputs == samples` (no repeats), then the full-output plugin estimates
\(\widehat{H}_1,\widehat{H}_2,\widehat{H}_\infty\) will typically collapse to \(\log_2 n\) and are dominated by sample size.
In that regime, treat them as **lower bounds** only. For bias detection (e.g. best-of-\(M\)), prefer the per-position
word marginals that `analyze-generator` reports as `word_marginal_entropy_bits`, or run far more samples.

For reproducible runs and how to interpret the printed diagnostics, see `docs/experiments.md`.

### 3.2 “Zero collisions” collision-probability bound (rule of thumb)

Let \(p_2 = \sum_s p(s)^2\) be the collision probability of two independent generator draws.
If we observe **0 collisions** among \(n\) samples, we have \(\binom{n}{2}\) sample pairs.

Under a crude binomial/Poisson-style approximation,
\[
\Pr(\text{0 collisions}) \approx (1-p_2)^{\binom{n}{2}}.
\]
Solving \((1-p_2)^{\binom{n}{2}}=\alpha\) gives
\[
p_2 \le 1-\alpha^{1/\binom{n}{2}} \;\approx\; \frac{-\ln(\alpha)}{\binom{n}{2}}.
\]
With \(\alpha=0.05\), this yields a **95%** upper bound heuristic for \(p_2\), hence a lower-bound heuristic for \(H_2\).

Limit: sample pairs are not independent (pairs share samples), so treat this as a **useful approximation**, not a theorem.

### 3.3 Selection bias: “pick best-of \(M\)” is not uniform

If you show `--alternatives M` and then a human picks “the nicest / fastest,” the effective distribution is no longer the
baseline generator distribution. `analyze-generator` includes a `--pick-best-of` mode that approximates this kind of bias
so you can quantify the **entropy penalty** of “menu choice.”

---

## 4) Diagnostics (useful signals, not guarantees)

These metrics help you understand “why a phrase scored the way it did” and how brittle the result is.

- **`hit_frac`**: fraction of digraphs with a specific estimate (vs fallback/backoff). Low hit fraction means
  \(\widehat{T}\) depends heavily on generic priors.
- **`shift_frac`**: fraction of characters that are Shift-requiring under a US-ASCII heuristic (roughly:
  uppercase or common shifted symbols). This is not layout-universal.
- **IDF-ish commonality**: if a corpus is provided, we compute an average per-word
  \(\log_2(\text{total}/\text{count(word)})\).
  This is *not* a security claim; it’s a “how common are these words in the corpus?” signal.

---

## 5) Where to use existing data better (and what new data would help most)

### 5.1 Existing data you already have

- **Public datasets** (unioned in the base pipeline): give broad coverage of common digraph timings and a decent global mean.
- **Your user JSONL** from `record-session`: gives personalized means (and structured backoff stats) for what *you* type.
- **Corpus counts** (optional): support the “IDF-ish” commonality diagnostics and k-gram surprisal models.
- **Generator Monte Carlo** (`analyze-generator`): directly measures the output distribution under your constraints/styles.

### 5.2 High-ROI additions (small changes, big payoff)

- **Store variance / uncertainty per digraph**: keep \(\sum dt\), \(\sum dt^2\), and count so you can report
  confidence intervals and propagate uncertainty into phrase scores (not just a point estimate).
- **Measure and model errors**: add a recording mode that *allows* backspace and records correction events;
  report an “expected time including corrections” metric and a “typo risk” diagnostic.
  (We now record `backspaces` and `total_ms` in `record-session --allow-backspace`, but we still exclude those rows from
  timing-model fitting by default. You can inspect this data via `analyze-user-data`, and fit a simple
  `total_ms ~= a + b*predicted_ms + c*backspaces` diagnostic model.)
- **Active data collection**: use the model’s low-count/high-impact digraphs to choose what the user should type next
  (coverage-focused sessions instead of random text).
- **Layout personalization**: current hand/shift heuristics are US-QWERTY-ish; if you care about other layouts,
  record and parameterize that mapping.

### 5.3 Honest security mode (when you need a clean bits claim)

If you want “bits” to be *clean and defensible*:

- Define a filtered set of allowed outputs (or a filtered wordset) and **sample uniformly** from it.
- Then the security claim reduces to the combinatorics in §2, and `analyze-generator` becomes a sanity check rather than
  the only truth source.

