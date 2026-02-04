# Math notes

This project uses a simple **digraph timing model** and reports a mix of:

- **Typing-time predictions** (from the timing model)
- **Security/entropy calculations** (from combinatorics, under explicit sampling assumptions)
- **Diagnostics** (hit rate, shift fraction, IDF-ish commonality) that are **not guarantees**

## Typing-time objective

Let a phrase be a sequence of graphemes with adjacent digraphs $g_1, \\dots, g_m$.

- For each digraph $g_i$, the model stores/estimates a mean latency $\\mu_{g_i}$ in ms.
- The predicted total entry time is:

$$
\\widehat{T}(s) = \\sum_{i=1}^{m} \\mu_{g_i}.
$$

If a digraph $g_i$ is unseen, scoring falls back (base global mean / backoff in `PersonalizedModel`), which is why `hit_*`
is an important reliability diagnostic.

### Normalized speed

`score` and `sample-passphrases --meta` also report a normalized ratio:

$$
\\text{norm}(s) = \\frac{\\widehat{T}(s)}{m \\cdot \\mu_{global}},
$$

where $\\mu_{global}$ is the global mean digraph latency learned by the model. Values \< 1.0 mean “faster than the model’s
average digraph”.

## Security bits (uniform sampling)

If you sample $k$ words uniformly from a wordset of size $N$:

- With repeats: $H = k \\log_2 N$
- Without repeats: $H = \\log_2 \\bigl(N (N-1)\\cdots(N-k+1)\\bigr)$

Style presets that add randomness (e.g. random separators) add additional bits from those random choices. These are clean
bits-only statements **only** when your generator is close to uniform over outputs and users don’t introduce selection bias
(e.g. “pick the prettiest from a menu”).

## Effective entropy diagnostics (generator distribution)

`analyze-generator` estimates several entropies from sampled output frequencies:

- $H_1$ (Shannon)
- $H_2$ (Rényi-2 / collision entropy)
- $H_\\infty$ (min-entropy)

When there are **0 observed collisions** among $n$ samples, it also prints a simple 95% upper bound on collision probability
$p_2$ (hence a lower bound on $H_2$) using a binomial approximation over the $\\binom{n}{2}$ sample pairs:

$$
p_2 \\lesssim -\\ln(0.05) / \\binom{n}{2}.
$$

