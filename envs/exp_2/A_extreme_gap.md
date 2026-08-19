# A_extreme_gap — Design Notes

## Design Logic

Direction 1 only: same single-hidden-leaf-in-branch_0 structure as
`strategy1_hidden_stationary`, but with the good/bad cost gap pushed much
further apart (hidden leaf 0.01, branch_0's other 15 leaves 0.98, vs the
original 0.05 / 0.90). Branches 1-3 stay uniformly moderate (0.30),
unchanged. Context-independent costs, same shared 4-point context
distribution as the strategy1/2/3 family.

## Spec

- Tree: H=3, K=4 (64 leaves), unchanged from strategy1-3
- Cost range (normalized): **min = 1.000000, max = 2.000000** (exact)
- Structure: 1 branch (16 leaves) mostly-bad with 1 hidden-best leaf
  (extreme gap); 3 branches (48 leaves) moderate

## Empirical Result (T = 1e4 .. 3e6, 3 seeds)

| Policy | log-log slope | avg regret @ T=1e4 | avg regret @ T=3e6 |
|---|---|---|---|
| LINEXP3 | 0.987 | 0.319 | 0.296 |
| DLEXP3 | 0.930 | 0.317 | 0.202 |
| DLEXP3-SE | 0.578 | 0.318 | 0.034 |
| Uniform | 1.000 | 0.454 | 0.454 |

Widening the gap alone modestly improves separation over baseline strategy1
(LINEXP3 vs DLEXP3 slope gap: 0.057 vs 0.048). Uniform regret ≈ 45%,
comfortably above the 20% floor.
