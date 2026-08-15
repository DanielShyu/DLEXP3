# strategy1_hidden_stationary — Design Notes

## Design Logic

Targets LINEXP3's core structural bug directly: its per-node estimator
divides by the LOCAL selection probability `pi^i_t(k|x)` instead of the
full ancestor-to-node path probability `rho_t(j|X_t)`. Since
`rho = rho_parent * pi`, this means LINEXP3's estimator expectation is off
by a factor of `rho_parent` — the more rarely the ancestor chain is
actually taken, the worse the mismatch.

Construction: one root-level branch (`branch_0`, all 16 leaves under the
first child of the root) is entirely BAD except for exactly ONE hidden
leaf (`arm_000`) which is the single best leaf in the whole tree. The
other three branches are uniformly MODERATE (worse than the hidden leaf,
but much better than branch_0's other 15 leaves).

The intended failure mode: branch_0's *average* performance looks bad
early on (15 bad leaves dominate 1 good one), so any policy naturally
de-prioritizes routing to branch_0 at the root. Once `pi_root(branch_0)`
shrinks, reaching anything inside branch_0 — including the hidden good
leaf — requires a correspondingly small `rho`. DLEXP3 accounts for this
correctly (its estimator divides by the true compounded `rho`); LINEXP3
does not, so it under-corrects and struggles to ever discover/exploit the
hidden leaf.

All leaf costs are **context-independent** (theta = [0,0,0,cost], i.e. only
the bias term is nonzero) — this isolates the pure tree-structural failure
mode from any context-conditioned routing behavior. The context
distribution (4 points) exists only so the estimator machinery (Sigma,
Sigma_inv) is well-defined; it does not affect which leaf is optimal.

Stationary (no time-varying schedule).

## Spec

- Tree: H=3, K=4 (64 leaves)
- Context: d=4, 4 context points, Sigma condition number ≈ 275
- Cost range (normalized): **min = 1.000000, max = 2.000000** (exact, by
  construction — see the project's standard [1,2] min-max normalization)
- Structure: 1 branch (16 leaves) mostly-bad with 1 hidden-best leaf; 3
  branches (48 leaves) uniformly moderate

## Empirical Result (T = 1e4 .. 3e6, 3 seeds, LINEXP3/DLEXP3/DLEXP3-SE/Uniform)

| Policy | log-log slope | avg regret @ T=1e4 | avg regret @ T=3e6 |
|---|---|---|---|
| LINEXP3 | 0.983 | 0.305 | 0.276 (barely decaying) |
| DLEXP3 | 0.935 | 0.303 | 0.197 |
| DLEXP3-SE | 0.589 | 0.302 | 0.034 |
| Uniform | 1.000 | 0.438 | 0.439 (no learning, as expected) |

The most promising of the three candidates tried: LINEXP3's slope is close
to the linear-regret ceiling (1.0) and its average regret is barely
decaying, unlike DLEXP3's. The separation from DLEXP3 is real but not yet
extreme — DLEXP3 is also somewhat affected by this environment, just less
severely. Next step under consideration: make the bad/good cost gap more
extreme and/or repeat the hidden-leaf structure across multiple branches to
widen the DLEXP3/LINEXP3 gap further.
