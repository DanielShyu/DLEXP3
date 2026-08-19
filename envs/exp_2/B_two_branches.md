# B_two_branches — Design Notes

## Design Logic

Direction 2 only: repeats the hidden-leaf-in-bad-branch structure across
TWO root branches instead of one. Branch_0's hidden leaf (cost 0.05) is
the global best; branch_1's hidden leaf (cost 0.10) is a close second-best,
so there's still a unique global optimum. Both branches' other 15 leaves
each are bad (~0.90). Branches 2-3 stay uniformly moderate (0.30). Same
context-independent convention and shared context distribution.

## Spec

- Tree: H=3, K=4 (64 leaves), unchanged
- Cost range (normalized): **min = 1.000000, max = 2.000000** (exact)
- Structure: 2 branches (32 leaves) each mostly-bad with 1 hidden-good
  leaf; 2 branches (32 leaves) moderate

## Empirical Result (T = 1e4 .. 3e6, 3 seeds)

| Policy | log-log slope | avg regret @ T=1e4 | avg regret @ T=3e6 |
|---|---|---|---|
| LINEXP3 | 0.971 | 0.341 | 0.288 |
| DLEXP3 | 0.917 | 0.341 | 0.202 |
| DLEXP3-SE | 0.575 | 0.338 | 0.034 |
| Uniform | 1.000 | 0.609 | 0.607 |

Repeating the structure across a second branch (more of the 64 leaves are
now "bad") pushes Uniform's regret up to ~61% while giving a similar
LINEXP3/DLEXP3 separation to env A. Comfortably above the 20% floor.
