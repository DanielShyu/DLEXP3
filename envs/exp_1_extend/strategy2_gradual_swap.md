# strategy2_gradual_swap — Design Notes

## Design Logic

Tests a different, more generic hypothesis: gradual concept drift combined
with LINEXP3's estimator bias. Two ordinary leaves (`arm_100`, `arm_300`,
not hidden behind any bad branch) swap ranks over 5 equally-spaced,
piecewise-constant phases (`t = 0, T/5, 2T/5, 3T/5, 4T/5`): leaf A starts
best (cost 0.05) and drifts to worst (cost 0.90) while leaf B mirrors it in
reverse. All other 62 leaves stay at a constant moderate cost throughout.

Unlike strategy 1, this does NOT specifically exploit the local-vs-global
routing-probability mismatch — it's a general "can the policy track a
slowly moving target" stress test that should affect any policy with slow
adaptation, not something unique to LINEXP3's denominator bug.

Costs are context-independent (theta = [0,0,0,cost]), same rationale as
strategy 1. Same 4-point context distribution reused for the estimator
machinery.

## Spec

- Tree: H=3, K=4 (64 leaves)
- Context: d=4, 4 context points, Sigma condition number ≈ 275
- Cost range (normalized): **min = 1.000000, max = 2.000000** (exact)
- Structure: 2 leaves undergo a 5-phase mirrored linear ramp (piecewise-
  constant between breakpoints); 62 leaves constant/moderate throughout

## Empirical Result (T = 1e4 .. 3e6, 3 seeds, LINEXP3/DLEXP3/DLEXP3-SE/Uniform)

| Policy | log-log slope | avg regret @ T=1e4 | avg regret @ T=3e6 |
|---|---|---|---|
| LINEXP3 | 0.950 | 0.035 | 0.026 |
| DLEXP3 | 0.951 | 0.034 | 0.026 |
| DLEXP3-SE | 0.724 | 0.031 | 0.005 |
| Uniform | 1.002 | 0.037 | 0.037 |

**Did not achieve LINEXP3-specific separation.** LINEXP3 and DLEXP3 have
essentially identical slopes and regret trajectories throughout — both
struggle with the gradual drift equally, confirming this mechanism is
generic rather than targeted at LINEXP3's local-denominator bug. Only
DLEXP3-SE (thanks to its synchronized exploration mechanism) does clearly
better. Kept as a negative-result reference / control condition rather
than a promising direction to iterate on further.
