# DLExp3

Simulation framework for distributed linear contextual bandit routing
(DLExp3 / DLExp3-SE) in a tree-structured multi-hop network

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/DanielShyu/DLEXP3.git
cd dlexp3
uv sync
```

`uv sync` installs everything from `pyproject.toml` / `uv.lock` into a
local `.venv`. Run any script with `uv run`, e.g.:

```bash
uv run python main.py envs/env_H4K2_skewed.json configs/standard.json
```

(or `uv run --with matplotlib python ...` if you hit a missing-package
error in a stripped-down environment — `main.py`'s plotting code imports
matplotlib lazily, so it's only needed if you actually call the plotting
functions.)

## Quick Start

```bash
uv run python main.py <env.json> <config.json>
```

Writes results/plots to `results/<env_stem>/`:

- `<env_stem>_output.json` — regret/cost/misspec, mean+std per policy per T
- `<env_stem>_total.png` — log-log cumulative regret, fitted slope per policy
- `<env_stem>_cost_ratio.png` — time-average regret as % of optimal cost
- `<env_stem>_misspec_ratio.png` — misspecification error as % of total cost
  (only for policies where misspec is defined: DLEXP3, BETA_DLEXP3)

### Config format

```json
{
  "time_points": [10000, 100000, 1000000],
  "num_runs": 5,
  "base_seed": 42,
  "policies": ["DLEXP3", "BETA_DLEXP3", "LINEXP3", "CENTRALIZED_LINEXP3", "DEXP3", "UNIFORM"],
  "alpha": 0.0
}
```

`alpha` is optional (default `0.0`): if set, each round's realized cost is
multiplied by an independent `Uniform[1-alpha, 1+alpha]` draw, for extra
injected randomness. Omit it for old configs / no injected noise.

### Environment format

```json
{
  "metadata": {"L": 3, "K": 4, "d": 4},
  "tree": {"root": {"type": "internal", "children": [...]}, "arm_000": {"type": "leaf"}, ...},
  "arm_parameters": {"arm_000": [[0.0, [a, b, c, e]], [0.5, [...]]], ...},
  "context_distribution": {"vectors": [[n, n2, m, 1], ...], "probabilities": [...]},
  "sigma_inv": [[...]]
}
```

- Tree must be a regular `K`-ary tree of depth `L` (`K^L` leaves).
- `arm_parameters[leaf]` is a piecewise-constant schedule: `[rel_t, theta]`
  pairs, held constant until the next breakpoint (`rel_t` in `[0, 1)`).
- Every `<context, theta>` value should land in `[1, 2]` — see
  `normalize_env` in any `gen_*.py` script for the standard min-max
  normalization used throughout this project (guarantees no negative
  costs, never changes which leaf is best/worst).

## Files

| File | Purpose |
|---|---|
| `main.py` | Core simulation engine + per-env plotting |
| `compare_policies.py` | Re-plot / cross-environment comparison from existing `*_output.json` |
| `plot.py` | Standalone regret/optimal-cost ratio plots |
| `misspec_ratio.py` | Standalone misspecification-ratio plots |
| `envs/` | Environment JSON files |
| `configs/` | Run config JSON files |
| `results/` | Simulation output (one subfolder per environment) |

## Policies

`DLEXP3`, `BETA_DLEXP3` (DLExp3-SE), `LINEXP3` (local LinExp3),
`CENTRALIZED_LINEXP3`, `CE_DLEXP3`, `DEXP3` (ε-EXP3, context-free),
`UNIFORM`.

Pull an evolved result back into this repo's environment format with:

```bash
uv run python extract_environment.py outputs/best_solution.py envs/found_env.json
```
