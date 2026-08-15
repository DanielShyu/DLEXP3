# strategy3_sudden_switch — Design Notes

## Design Logic

A variant of strategy 1's "hidden leaf in a bad branch" structure, with a
time dynamic added: instead of being good from the start, the hidden leaf
(`arm_000`) starts out BAD (matching all 15 of its branch-0 siblings, no
exception) for the first half of the horizon, then suddenly becomes the
single best leaf in the tree at `t = T/2`. The other 3 branches stay
uniformly moderate throughout, unchanged.

Intent: during phase 1, a policy has every reason to thoroughly write off
branch_0 (ALL 16 of its leaves look bad, with no hint of the future
upside) — by the time phase 2 starts, `pi_root(branch_0)` may already be
decayed close to its exploration floor. The question is whether a policy
can "wake up" and rediscover branch_0 once the payoff structure flips,
starting from a position of having already heavily discounted it — LINEXP3
does this discovery through a doubly-compromised channel (low ancestor
`rho` AND a local-only estimator that doesn't correct for it).

Same context-independent cost convention and context distribution as
strategies 1 and 2.

## Spec

- Tree: H=3, K=4 (64 leaves)
- Context: d=4, 4 context points, Sigma condition number ≈ 275
- Cost range (normalized): **min = 1.000000, max = 2.000000** (exact)
- Structure: 1 branch (16 leaves) uniformly bad in phase 1 (`t<T/2`); at
  `t=T/2` one specific leaf in that branch flips to the single best leaf in
  the tree while its 15 siblings stay bad; 3 branches (48 leaves) constant/
  moderate throughout both phases

## Empirical Result (T = 1e4 .. 3e6, 3 seeds, LINEXP3/DLEXP3/DLEXP3-SE/Uniform)

| Policy | log-log slope | avg regret @ T=1e4 | avg regret @ T=3e6 |
|---|---|---|---|
| LINEXP3 | 0.911 | 0.057 | 0.034 |
| DLEXP3 | 0.907 | 0.057 | 0.033 |
| DLEXP3-SE | 0.843 | 0.056 | 0.023 |
| Uniform | 1.001 | 0.188 | 0.190 |

**Did not achieve LINEXP3-specific separation either.** LINEXP3 and DLEXP3
again track each other closely (slopes 0.911 vs 0.907), suggesting the
"rediscover after being written off" difficulty hits both policies about
equally, rather than exploiting LINEXP3's specific denominator bug more
than DLEXP3's properly-weighted one. Kept as a second negative-result
reference; strategy 1 (no switch, just a persistently hidden leaf) remains
the only one of the three that shows real LINEXP3-vs-DLEXP3 separation.
