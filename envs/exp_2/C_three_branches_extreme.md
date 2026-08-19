# C_three_branches_extreme — Design Notes

## Design Logic

Combines both directions: the hidden-leaf-in-bad-branch structure is
repeated across THREE of the four root branches (only branch_3 stays
moderate), AND the good/bad gap within each is pushed to the extreme
values from env A (hidden leaves at 0.01/0.03/0.05, siblings at ~0.98).
Each branch's hidden leaf is slightly different so there remains a single
unique global optimum (`arm_000`, cost 0.01). Same context-independent
convention and shared context distribution as the rest of the family.

## Spec

- Tree: H=3, K=4 (64 leaves), unchanged
- Cost range (normalized): **min = 1.000000, max = 2.000000** (exact)
- Structure: 3 branches (48 leaves) each mostly-bad (extreme gap) with 1
  hidden-good leaf; 1 branch (16 leaves) moderate

## Empirical Result (T = 1e4 .. 3e6, 3 seeds)

| Policy | log-log slope | avg regret @ T=1e4 | avg regret @ T=3e6 |
|---|---|---|---|
| LINEXP3 | 0.954 | 0.397 | 0.303 |
| DLEXP3 | 0.878 | 0.392 | 0.194 |
| DLEXP3-SE | 0.574 | 0.356 | 0.034 |
| Uniform | 1.000 | 0.772 | 0.772 |

**Best separation of the three v2 candidates.** Combining both directions
compounds the effect: LINEXP3/DLEXP3 slope gap widens to 0.076 (vs 0.048 in
the original strategy1 baseline), and the avg-regret gap at T=3e6 widens to
0.109 (vs 0.079 baseline). DLEXP3's slope (0.878) is now noticeably below
its counterparts in envs A/B, suggesting even DLEXP3's properly-weighted
estimator starts to strain when the "trap" is repeated across most of the
tree. Uniform regret ≈ 77%, well above the 20% floor.
