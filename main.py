#!/usr/bin/env python3
"""
main.py — Simplified Multi-hop Linear Contextual Bandit Simulation

Usage: python main.py <env.json> <config.json>

Policies
  DLEXP3              : depth-dependent rates eta(d)=T^{-(d+2)/(d+3)}, gamma(d)=T^{-1/(d+3)}.
                         Theta update uses the GLOBAL path probability rho^R_t(k)
                         (product of policies along the full root->k path), as in the
                         original report — unchanged.
  DEXP3               : context-free degraded version of DLEXP3. Same depth-dependent
                         eta(d)/gamma(d) and same global path-probability denominator, but
                         the policy and estimator IGNORE the context entirely: Theta is a
                         scalar cumulative-loss estimate per arm (not a vector), the
                         softmax score is -eta*Theta_a (no <x, Theta_a>), and the unbiased
                         estimator is simply (received loss / path probability) — no
                         Sigma_inv, no context.
  CE_DLEXP3           : centralized exploration, single global bit E_t at root, depth-
                         independent eta=T^{-2/3}, gamma=T^{-1/3}. Routing probability
                         rho = gamma/K^l + (1-gamma)*Q^R_t (pure-softmax product) — unchanged.
  LINEXP3             : depth-independent eta=T^{-2/3}, gamma=T^{-1/3}, BUT the importance
                         estimator at each node uses ONLY that node's own LOCAL selection
                         probability pi^i_t(k|x) as the denominator — NOT the global path
                         probability. Each node behaves as if it were solving an isolated
                         one-hop problem, oblivious to the routing context above it. This
                         is the deliberately "naive" baseline.
  CENTRALIZED_LINEXP3 : the whole L-hop tree is flattened into a single one-hop EXP3 problem
                         over K^L leaf arms, chosen directly by the root in one shot, with
                         eta=T^{-2/3}, gamma=T^{-1/3}.
  UNIFORM             : pure random leaf choice (no learning).

Regret (single metric — no LS estimator, no offline_policy.py precomputation needed)
  The optimal/offline policy treats the system as ONE HOP: it sees each leaf's
  TIME-AVERAGED parameter directly, regardless of tree depth, and commits to a FIXED
  per-context choice:
      a*(x) = argmin_l  <x, leaf_avg_l>            (leaf_avg computed in this script)
  Regret per round uses the ACTUAL (segment-current) parameter at that fixed leaf:
      R_T = sum_t [ <X_t, theta_{t,leaf_chosen}>  -  <X_t, theta_{t,a*(X_t)}> ]

Outputs (three plots) in the same folder as <env_stem>_output.json:
  <env_stem>_total.png         log10 cumulative regret vs T, with fitted slope per policy
  <env_stem>_cost_ratio.png    time-average regret as a fraction of optimal cost, per policy
  <env_stem>_misspec_ratio.png misspecification error as a fraction of total cost
                                (only drawn if at least one policy has misspec data)
"""

import json, sys, time
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RESULTS_DIR  = Path("results")

POLICY_META = {
    "DLEXP3":              {"color": "#4a9eff", "marker": "o", "label": "DLExp3"},
    "DEXP3":               {"color": "#fbbf24", "marker": "^", "label": "\u03b5-EXP3"},
    "CE_DLEXP3":           {"color": "#34d399", "marker": "v", "label": "CE-DLExp3"},
    "BETA_DLEXP3":         {"color": "#e879f9", "marker": "*", "label": "DLExp3-SE"},
    "LINEXP3":             {"color": "#ff6b4a", "marker": "s", "label": "LinExp3 (local)"},
    "CENTRALIZED_LINEXP3": {"color": "#f472b6", "marker": "P", "label": "Centralized LinExp3"},
    "UNIFORM":             {"color": "#a78bfa", "marker": "D", "label": "Uniform"},
}


# ── Schedule helpers ───────────────────────────────────────────────────────

def load_schedule(raw):
    return sorted([(e[0], np.array(e[1], dtype=float)) for e in raw], key=lambda e: e[0])

def value_at(schedule, rel_t):
    val = schedule[0][1]
    for t, v in schedule:
        if rel_t >= t: val = v
        else: break
    return val

def time_average(schedule, d):
    avg = np.zeros(d)
    bps = [t for t, _ in schedule] + [1.0]
    for i, (_, v) in enumerate(schedule):
        avg += (bps[i+1] - bps[i]) * v
    return avg


# ── Learning rates ─────────────────────────────────────────────────────────

def node_rates(policy, T, L, K, depth):
    """(eta, gamma) for a node at given depth. Only used by hierarchical policies."""
    if policy == "DLEXP3":
        return T**(-(depth + 2.) / (depth + 3)), T**(-1. / (depth + 3))
    elif policy == "DEXP3":
        # One order lower than DLExp3's schedule (context-free degraded rate).
        return T**(-(depth + 1.) / (depth + 2)), T**(-1. / (depth + 2))
    elif policy == "BETA_DLEXP3":
        # Constant eta=T^-2/3, gamma=0 fixed: beta (drawn separately, T^-1/3) takes
        # over gamma's exploration role entirely, so no per-node gamma floor is
        # added on top of the global partial-depth exploration mechanism.
        return T**(-2. / 3), 0.0
    elif policy in ("CE_DLEXP3", "LINEXP3"):
        return T**(-2. / 3), T**(-1. / 3)
    elif policy == "UNIFORM":
        return 0.0, 1.0
    raise ValueError(f"Unknown policy '{policy}'")


# ── Lookup tables ──────────────────────────────────────────────────────────

def get_leaves_in_order(tree):
    """DFS traversal -> leaves in flat-index order (k0*K^{L-1}+k1*K^{L-2}+...)."""
    leaves = []
    def dfs(n):
        if tree[n]["type"] == "leaf": leaves.append(n)
        else:
            for c in tree[n]["children"]: dfs(c)
    dfs("root")
    return leaves


def build_lookup_tables(tree, arm_params, K, L, d):
    """
    arm_theta_mat : (K^L, n_segs, d)  actual leaf parameters per schedule segment
    leaf_avg_mat  : (K^L, d)          time-averaged leaf parameters (one-hop oracle view)
    all_bps       : sorted schedule breakpoints
    """
    leaves = get_leaves_in_order(tree)
    assert len(leaves) == K**L, f"Expected {K**L} leaves, got {len(leaves)}"

    bps = set()
    for s in arm_params.values():
        bps.update(t for t, _ in s)
    all_bps = sorted(bps)
    n_segs  = len(all_bps)

    arm_theta_mat = np.zeros((K**L, n_segs, d))
    leaf_avg_mat  = np.zeros((K**L, d))
    for i, leaf in enumerate(leaves):
        for si, bp in enumerate(all_bps):
            arm_theta_mat[i, si] = value_at(arm_params[leaf], bp)
        leaf_avg_mat[i] = time_average(arm_params[leaf], d)

    return arm_theta_mat, leaf_avg_mat, all_bps


def segment_dot(theta_mat, flat_idx, seg_b, X_b):
    """<X_b[i], theta_mat[flat_idx[i], seg_b[i], :]> for each i, grouped by segment."""
    out = np.zeros(len(flat_idx))
    for seg in np.unique(seg_b):
        mask     = seg_b == seg
        out[mask] = (X_b[mask] * theta_mat[flat_idx[mask], seg, :]).sum(axis=1)
    return out


def draw_multiplicative_noise(rng, alpha, size):
    """
    a ~ Uniform[1-alpha, 1+alpha], drawn independently per element -- used to
    inject extra per-round, per-arm randomness into realized cost: each arm
    conceptually gets its own independent draw every round, but since bandit
    feedback only ever reveals the CHOSEN arm's outcome, it's equivalent (and
    far cheaper) to draw a single 'a' per round for whichever arm is actually
    realized (online-chosen or offline-optimal), rather than drawing one for
    every arm and discarding all but one. E[a]=1 for any alpha, so this adds
    variance without introducing bias into the unbiased estimator or the
    regret/cost accounting.

    alpha=0 (the default, and what every pre-existing config implicitly
    uses) returns exactly 1.0 WITHOUT drawing from rng at all -- this keeps
    old configs bit-for-bit reproducible (no extra random draws sneak into
    the seeded stream when alpha is left unset).
    """
    if alpha == 0.0:
        return 1.0
    return rng.uniform(1.0 - alpha, 1.0 + alpha, size=size)


# ── Single run ─────────────────────────────────────────────────────────────

def compute_root_G_and_theta_hat(Theta_levels, leaf_theta_seg, vectors, probs, Sigma_inv,
                                 K, L, d, eta_levels, gamma_levels, policy, beta_val=None):
    """
    Exact bottom-up computation of G_i(x) -- the TRUE expected downstream
    cost under the CURRENT (batch-start) policy -- aggregated up to the
    ROOT'S CHILDREN (depth-1 nodes). No linear approximation is used in the
    recursion itself (unlike the old compute_depth1_theta_hat, which chained
    children's theta_hat upward); G is propagated exactly at every depth, and
    theta_hat is fit via population-weighted OLS only at the very end.

    For BETA_DLEXP3, the realized routing mixes pure-mode softmax with the
    beta/y_t partial-depth-exploration mechanism. Marginalizing over y_t ~
    Uniform{0,...,L-1} gives a depth-dependent EFFECTIVE mixing weight:
        gamma_eff(depth) = beta * (L - depth) / L
    (deeper nodes are less likely to be forced uniform), which reduces to
    the same (1-gamma_eff)*softmax + gamma_eff/K form used by the other
    policies -- so a single recursion handles all four hierarchical policies.

    Returns:
      theta_hat_d1 : (K, d)      best linear fit of G at each depth-1 node
      G_d1         : (K, n_ctx)  exact G value at each depth-1 node, per ctx
    """
    n_ctx     = len(vectors)
    G_current = leaf_theta_seg @ vectors.T   # (K^L, n_ctx) -- exact at leaves

    # Aggregate from depth L-1 down to depth 1 (stop AT depth-1; empty range
    # when L==1, since root's children ARE the leaves in that case).
    for depth in reversed(range(1, L)):
        n_nodes  = K**depth
        Theta_nd = Theta_levels[depth].reshape(n_nodes, K, d)
        G_nd     = G_current.reshape(n_nodes, K, n_ctx)

        eta_l  = eta_levels[depth]
        scores = -eta_l * np.einsum("nkd,xd->nkx", Theta_nd, vectors)
        scores -= scores.max(axis=1, keepdims=True)
        w          = np.exp(scores)
        pi_softmax = w / w.sum(axis=1, keepdims=True)

        if policy == "BETA_DLEXP3":
            gamma_eff = beta_val * (L - depth) / L
        else:
            gamma_eff = gamma_levels[depth]

        pi = (1 - gamma_eff) * pi_softmax + gamma_eff / K
        pi /= pi.sum(axis=1, keepdims=True)

        G_current = np.einsum("nkx,nkx->nx", pi, G_nd)   # exact, not an approximation

    G_d1 = G_current   # (K, n_ctx)

    E_xG         = (probs[None, :] * G_d1) @ vectors   # (K, d)
    theta_hat_d1 = E_xG @ Sigma_inv                      # (K, d), Sigma_inv symmetric

    return theta_hat_d1, G_d1


def run_single(T, policy, L, K, d, Sigma_inv,
               vectors, probs,
               arm_theta_mat, leaf_avg_mat, all_bps, seed, batch_size=0, alpha=0.0):
    """
    Returns (cumreg, misspec_cumsum, true_cost_cumsum).

    alpha controls extra injected randomness: each round, the realized cost
    of the arm actually incurred (both the online-chosen arm and, separately
    and independently, the offline-optimal arm used for the regret baseline)
    gets multiplied by an independent a ~ Uniform[1-alpha, 1+alpha] (see
    draw_multiplicative_noise). alpha=0 (default) reproduces the exact old
    behavior with no injected noise -- pass alpha explicitly, or set it in
    the env config, to turn this on.

    misspec_cumsum is the ROOT-LEVEL total misspecification error, summed
    over T rounds: at each round t, letting j_t be the depth-1 node the
    root ACTUALLY routed the job to, misspec contributes
        <X_t, theta_hat_{t,j_t}> - G_{t,j_t}(X_t)
    where theta_hat/G are computed EXACTLY (see compute_root_G_and_theta_hat)
    from batch-start Theta -- not the noisy online estimator theta_tilde.
    Uses ONLY the online policy's own trajectory -- the offline/optimal
    comparator plays no role in this computation. Only defined for DLEXP3
    and BETA_DLEXP3 (None for every other policy, including the other
    hierarchical ones -- CE_DLEXP3, LINEXP3, DEXP3).

    true_cost_cumsum is the policy's raw total realized cost, summed over T
    rounds: sum_t g_t(X_t) = sum_t <X_t, theta_{t,leaf chosen}>. Unlike
    regret, this is NOT offset by the offline/optimal comparator. Tracked
    for EVERY policy (never None). Since regret = cost - cost_optimal and
    cost_optimal is identical across policies at matched (T, seed), the
    offline/optimal comparator's own total cost can be recovered downstream
    as cost - regret without needing a separate accumulator here.
    """
    if batch_size == 0:
        batch_size = max(1, min(2000, T // 100))

    rng   = np.random.default_rng(seed)
    n_ctx = len(vectors)

    ctx_idx = rng.choice(n_ctx, size=T, p=probs)
    X_all   = vectors[ctx_idx]
    SiX_all = X_all @ Sigma_inv

    seg_all = np.searchsorted(all_bps, np.arange(T, dtype=float) / T, side="right") - 1
    seg_all = seg_all.clip(0, len(all_bps) - 1)

    # Fixed offline a*(x): one-hop view using time-averaged leaf params
    a_star_per_ctx = (vectors @ leaf_avg_mat.T).argmin(axis=1)   # (n_ctx,)
    a_star_all     = a_star_per_ctx[ctx_idx]                      # (T,)

    cumreg = 0.0

    # ── Centralized LinExp3: flat one-hop EXP3 over K^L leaves ─────────────
    if policy == "CENTRALIZED_LINEXP3":
        n_leaves    = K**L
        Theta_flat  = np.zeros((n_leaves, d))
        eta, gamma  = T**(-2. / 3), T**(-1. / 3)
        true_cost_cumsum = 0.0

        for t0 in range(0, T, batch_size):
            t1    = min(t0 + batch_size, T)
            Bt    = t1 - t0
            X_b   = X_all[t0:t1]; SiX_b = SiX_all[t0:t1]; seg_b = seg_all[t0:t1]

            scores  = -eta * (X_b @ Theta_flat.T)          # (Bt, n_leaves)
            scores -= scores.max(axis=1, keepdims=True)
            w       = np.exp(scores)
            pi      = (1 - gamma) * w / w.sum(axis=1, keepdims=True) + gamma / n_leaves
            pi      = np.clip(pi / pi.sum(axis=1, keepdims=True), 1e-300, 1.)

            chosen    = (pi.cumsum(axis=1) < rng.random(Bt)[:, None]).sum(axis=1).clip(0, n_leaves - 1)
            pi_chosen = pi[np.arange(Bt), chosen]

            loss = segment_dot(arm_theta_mat, chosen, seg_b, X_b)
            loss = loss * draw_multiplicative_noise(rng, alpha, Bt)
            true_cost_cumsum += loss.sum()
            np.add.at(Theta_flat, chosen, (1. / pi_chosen[:, None]) * SiX_b * loss[:, None])

            offline  = segment_dot(arm_theta_mat, a_star_all[t0:t1], seg_b, X_b)
            offline  = offline * draw_multiplicative_noise(rng, alpha, Bt)
            cumreg  += (loss - offline).sum()

        return cumreg, None, true_cost_cumsum

    # ── Uniform: pure random leaf choice, no learning ───────────────────────
    if policy == "UNIFORM":
        n_leaves = K**L
        true_cost_cumsum = 0.0
        for t0 in range(0, T, batch_size):
            t1    = min(t0 + batch_size, T)
            Bt    = t1 - t0
            X_b   = X_all[t0:t1]; seg_b = seg_all[t0:t1]

            chosen  = rng.integers(0, n_leaves, size=Bt)
            loss    = segment_dot(arm_theta_mat, chosen, seg_b, X_b)
            loss    = loss * draw_multiplicative_noise(rng, alpha, Bt)
            true_cost_cumsum += loss.sum()
            offline = segment_dot(arm_theta_mat, a_star_all[t0:t1], seg_b, X_b)
            offline = offline * draw_multiplicative_noise(rng, alpha, Bt)
            cumreg += (loss - offline).sum()

        return cumreg, None, true_cost_cumsum

    # ── Hierarchical: DLEXP3 / CE_DLEXP3 / LINEXP3 / DEXP3 / BETA_DLEXP3 ──────
    if policy == "DEXP3":
        Theta_levels = [np.zeros(K**(l + 1)) for l in range(L)]          # scalar per arm
    else:
        Theta_levels = [np.zeros((K**(l + 1), d)) for l in range(L)]     # vector per arm
    eta_levels   = [node_rates(policy, T, L, K, l)[0] for l in range(L)]
    gamma_levels = [node_rates(policy, T, L, K, l)[1] for l in range(L)]
    gamma_ce     = gamma_levels[0] if policy == "CE_DLEXP3" else None
    # beta-DLExp3: beta = T^{-1/(L+2)}, same order as DLExp3's root gamma.
    beta_val     = T**(-1. / 3) if policy == "BETA_DLEXP3" else None

    # Root-level misspecification error: restricted to DLExp3 and DLExp3-SE
    # only (see compute_root_G_and_theta_hat). Uses ONLY the online policy's
    # own trajectory -- the offline/optimal comparator plays no role.
    #
    # Raw total cost, in contrast, is tracked for EVERY policy (including
    # DEXP3 here, and CENTRALIZED_LINEXP3/UNIFORM above): sum_t g_t(X_t),
    # NOT offset by the offline/optimal comparator. Since regret_A =
    # cost_A - cost_optimal for every policy A (cost_optimal is identical
    # across policies at matched (T, seed) because the offline term only
    # depends on the realized context sequence, not on the online policy),
    # cost_optimal can be recovered downstream as cost_A - regret_A without
    # needing its own separate accumulator.
    track_misspec    = policy in ("DLEXP3", "BETA_DLEXP3")
    misspec_cumsum   = 0.0 if track_misspec else None
    true_cost_cumsum = 0.0

    for t0 in range(0, T, batch_size):
        t1    = min(t0 + batch_size, T)
        Bt    = t1 - t0
        X_b   = X_all[t0:t1]; SiX_b = SiX_all[t0:t1]; seg_b = seg_all[t0:t1]
        ctx_idx_b = ctx_idx[t0:t1]

        if track_misspec:
            seg0 = int(seg_b[0])
            theta_hat_d1, G_d1 = compute_root_G_and_theta_hat(
                Theta_levels, arm_theta_mat[:, seg0, :],
                vectors, probs, Sigma_inv,
                K, L, d, eta_levels, gamma_levels, policy, beta_val,
            )
            mismatch_mat = (theta_hat_d1 @ vectors.T) - G_d1   # (K, n_ctx)

        node_flat        = np.zeros(Bt, dtype=int)
        rho_b             = np.ones(Bt)   # global path-probability (DLEXP3, DEXP3)
        exploit_rho_b     = np.ones(Bt)   # pure-softmax product Q^R_t (CE_DLEXP3)
        path_flat         = []
        update_denom_list = []            # per-level denominator used for Theta updates
        update_mask_list  = []            # (Bt,) bool: which rounds update at each level

        if policy == "CE_DLEXP3":
            E_b = rng.random(Bt) < gamma_ce   # global exploration bit, shared all levels

        if policy == "BETA_DLEXP3":
            # y_eff = L  ->  pure mode  (1-beta prob)
            # y_eff in {0,...,L-1}  ->  exploration, threshold y_t  (beta prob)
            is_explore_b = rng.random(Bt) < beta_val
            y_b          = rng.integers(0, L, size=Bt)           # threshold y_t
            y_eff_b      = np.where(is_explore_b, y_b, L)        # L = pure-mode sentinel
            P_b          = np.full(Bt, 1.0 - beta_val)           # P_0 = (1-beta)

        for l in range(L):
            child_flat_idx = K * node_flat[:, None] + np.arange(K, dtype=int)[None, :]

            if policy == "DEXP3":
                # Context-free: softmax over cumulative scalar loss estimate per arm,
                # identical for every round at this node regardless of X_b.
                scores  = -eta_levels[l] * Theta_levels[l][child_flat_idx]    # (Bt, K)
                scores -= scores.max(axis=1, keepdims=True)
                w       = np.exp(scores)
                pi      = (1 - gamma_levels[l]) * w / w.sum(axis=1, keepdims=True) \
                          + gamma_levels[l] / K
                pi      = np.clip(pi / pi.sum(axis=1, keepdims=True), 1e-300, 1.)

            elif policy == "CE_DLEXP3":
                child_theta = Theta_levels[l][child_flat_idx]   # (Bt, K, d)
                scores      = -eta_levels[l] * (child_theta * X_b[:, None, :]).sum(axis=2)
                scores     -= scores.max(axis=1, keepdims=True)
                w           = np.exp(scores)
                pi_softmax  = w / w.sum(axis=1, keepdims=True)
                pi          = np.where(E_b[:, None], np.full((Bt, K), 1. / K), pi_softmax)

            elif policy == "BETA_DLEXP3":
                # Node at depth l uses its DLExp3 policy when:
                #   pure mode (y_eff == L)  OR  y_t < l  (depth l > threshold -> policy)
                use_policy_l = (y_eff_b == L) | (y_eff_b < l)          # (Bt,) bool
                child_theta  = Theta_levels[l][child_flat_idx]           # (Bt, K, d)
                scores       = -eta_levels[l] * (child_theta * X_b[:, None, :]).sum(axis=2)
                scores      -= scores.max(axis=1, keepdims=True)
                w            = np.exp(scores)
                pi           = (1 - gamma_levels[l]) * w / w.sum(axis=1, keepdims=True)                                + gamma_levels[l] / K
                pi           = np.clip(pi / pi.sum(axis=1, keepdims=True), 1e-300, 1.)
                # Actual action: policy if use_policy_l, else uniform
                pi_actual    = np.where(use_policy_l[:, None], pi, np.full((Bt, K), 1. / K))

            else:   # DLEXP3, LINEXP3 — standard (1-gamma)softmax + gamma/K mixture
                child_theta = Theta_levels[l][child_flat_idx]   # (Bt, K, d)
                scores  = -eta_levels[l] * (child_theta * X_b[:, None, :]).sum(axis=2)
                scores -= scores.max(axis=1, keepdims=True)
                w       = np.exp(scores)
                pi      = (1 - gamma_levels[l]) * w / w.sum(axis=1, keepdims=True) \
                          + gamma_levels[l] / K
                pi      = np.clip(pi / pi.sum(axis=1, keepdims=True), 1e-300, 1.)

            if policy == "BETA_DLEXP3":
                chosen_k  = (pi_actual.cumsum(axis=1) < rng.random(Bt)[:, None]).sum(axis=1).clip(0, K - 1)
            else:
                chosen_k  = (pi.cumsum(axis=1) < rng.random(Bt)[:, None]).sum(axis=1).clip(0, K - 1)
            pi_chosen  = pi[np.arange(Bt), chosen_k]   # policy prob (always computed)
            child_flat = K * node_flat + chosen_k

            if policy == "CE_DLEXP3":
                exploit_rho_b *= pi_softmax[np.arange(Bt), chosen_k]
                denom_l        = gamma_ce / K**(l + 1) + (1 - gamma_ce) * exploit_rho_b
            elif policy in ("DLEXP3", "DEXP3"):
                # Global path probability — product of selection probs root->leaf
                rho_b   *= pi_chosen
                denom_l  = rho_b.copy()
            elif policy == "BETA_DLEXP3":
                # rho_t(child) = P_l * pi_policy(chosen) + beta/(H*K^{l+1})   (paper, 7/4 rev.)
                # The FULL rho_t(child) — INCLUDING the boundary term — is the correct
                # denominator, and the update indicator is gated on the CHILD's own
                # exploration state (child depth = l+1 is non-exploring iff pure mode
                # OR y_t <= l), not the parent's. This includes the y_t == l boundary
                # round (parent was uniform, but child itself is past threshold),
                # which the previous (6/25) parent-based indicator excluded.
                partial_l   = P_b * pi_chosen                            # rho_t(j)*pi^j(k)
                c_next      = beta_val / (L * K**(l + 1))                # rho^beta_t(j)/|Cj|
                rho_full    = partial_l + c_next                        # rho_t(k), full
                denom_l     = rho_full                                  # use FULL rho as divisor
                P_b         = rho_full.copy()                           # pass full rho onward
                update_mask_l = (y_eff_b == L) | (y_eff_b <= l)         # child's own state
            else:   # LINEXP3 — local-only denominator (pretends one-hop at every node)
                denom_l = pi_chosen.copy()

            path_flat.append(child_flat.copy())
            update_denom_list.append(denom_l.copy())
            if policy == "BETA_DLEXP3":
                update_mask_list.append(update_mask_l.copy())
            node_flat = child_flat

        leaf_flat = node_flat
        loss      = segment_dot(arm_theta_mat, leaf_flat, seg_b, X_b)
        loss      = loss * draw_multiplicative_noise(rng, alpha, Bt)

        true_cost_cumsum += loss.sum()
        if track_misspec:
            j_t = path_flat[0]   # depth-1 node actually visited this round (0..K-1)
            misspec_cumsum += mismatch_mat[j_t, ctx_idx_b].sum()

        for l in range(L):
            if policy == "DEXP3":
                # Scalar unbiased estimator: received loss / path probability,
                # NO context or Sigma_inv — pure context-free EXP3.
                np.add.at(Theta_levels[l], path_flat[l], loss / update_denom_list[l])
            elif policy == "BETA_DLEXP3":
                # Only update when node at depth l used its policy (not uniform).
                mask = update_mask_list[l]
                if mask.any():
                    np.add.at(Theta_levels[l], path_flat[l][mask],
                              (1. / update_denom_list[l][mask, None]) * SiX_b[mask] * loss[mask, None])
            else:
                np.add.at(Theta_levels[l], path_flat[l],
                          (1. / update_denom_list[l][:, None]) * SiX_b * loss[:, None])

        offline  = segment_dot(arm_theta_mat, a_star_all[t0:t1], seg_b, X_b)
        offline  = offline * draw_multiplicative_noise(rng, alpha, Bt)
        cumreg  += (loss - offline).sum()

    return cumreg, misspec_cumsum, true_cost_cumsum


# ── Plot ───────────────────────────────────────────────────────────────────

def loglog_slope(T_vals, means):
    slope, _ = np.polyfit(np.log10(T_vals), np.log10(np.maximum(np.abs(means), 1e-12)), 1)
    return float(slope)


def format_T_label(T):
    """Format a horizon value for axis ticks: 1000 -> '1k', 2_000_000 -> '2M'."""
    T = int(T)
    if T >= 1_000_000:
        val = T / 1_000_000
        return f"{val:g}M"
    if T >= 1_000:
        val = T / 1_000
        return f"{val:g}k"
    return str(T)


def plot_results(results, time_points, policies, out, ylabel, title,
                 log_scale=True, show_title=True,
                 fs_tick=10, fs_label=11, fs_legend=9):
    # White-background theme
    BG    = "#ffffff"; PANEL = "#ffffff"; GRID = "#d9d9d9"
    TEXT  = "#333333"; TITLE = "#111111"

    T_arr = np.array(time_points, dtype=float)
    x_T   = np.log10(T_arr) if log_scale else T_arr

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=fs_tick)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    ax.grid(color=GRID, linestyle="--", linewidth=0.6, alpha=0.8)

    for pol in policies:
        means = np.array(results[pol]["mean"])
        stds  = np.array(results[pol]["std"])
        meta  = POLICY_META[pol]

        if log_scale:
            y_mean  = np.log10(np.maximum(np.abs(means), 1e-12))
            y_upper = np.log10(np.maximum(np.abs(means) + stds, 1e-12))
            # Lower bound: |means|-stds commonly goes non-positive at small T
            # (std >= |mean| is typical for high-variance regret data), where
            # the TRUE interval extends to zero or below -- no log10
            # representation exists there, so clamp to a floor instead of
            # computing a spurious "reflected" value (which would make the
            # lower bound NON-monotonic in std -- see compare_policies.py).
            y_lower = np.log10(np.maximum(np.abs(means) - stds, 1e-3))
            slope   = loglog_slope(T_arr, means)
            label   = f"{meta['label']}  (slope \u2248 {slope:.3f})"
        else:
            y_mean  = means
            y_upper = means + stds
            y_lower = means - stds
            label   = meta["label"]

        ax.plot(x_T, y_mean, color=meta["color"], lw=2.,
                marker=meta["marker"], markersize=6, label=label)
        ax.fill_between(x_T, y_lower, y_upper, color=meta["color"], alpha=0.15)

    if log_scale:
        # Thin x-tick labels to at most 8, evenly spaced by index (always
        # keep first/last), to avoid overlapping labels when many T points
        # are used.
        n_pts    = len(time_points)
        max_tick = 8
        tick_idx = np.unique(np.linspace(0, n_pts - 1, min(n_pts, max_tick)).round().astype(int))
        ax.set_xticks(x_T[tick_idx])
        ax.set_xticklabels([format_T_label(time_points[i]) for i in tick_idx], color=TEXT)
        ax.set_xlabel(r"T  (log$_{10}$ scale)", fontsize=fs_label)
    else:
        # T points are log-spaced, so on a LINEAR axis they cluster near the
        # left edge no matter how few ticks are kept -- numeric labels always
        # overlap. Just turn the numbers off; the T axis label is still shown.
        ax.set_xticks([])
        ax.set_xlabel("T", fontsize=fs_label)

    ax.set_ylabel(ylabel, fontsize=fs_label)
    if show_title:
        ax.set_title(title, fontsize=12, color=TITLE)
    leg = ax.legend(fontsize=fs_legend, framealpha=0.15)
    for txt in leg.get_texts(): txt.set_color(TITLE)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[plot] Saved -> '{out}'")


# ── Ratio plots (ported from the standalone plot.py / misspec_ratio.py) ────
# Each script had its own deliberate styling tweaks (different font sizes,
# different y-axis formatters, and a different BETA_DLEXP3 color in the
# cost-ratio plot specifically) -- kept scoped separately here rather than
# merged into the single POLICY_META above, so neither script's choices get
# silently overwritten by the other's.

COST_RATIO_POLICY_META = {
    "DLEXP3":              {"color": "#4a9eff", "marker": "o", "ls": "-",  "label": "DLEXP3"},
    "CE_DLEXP3":           {"color": "#34d399", "marker": "v", "ls": "-.", "label": "CE-DLEXP3"},
    "BETA_DLEXP3":         {"color": "#21f94c", "marker": "*", "ls": ":",  "label": "DLEXP3-SE"},
    "LINEXP3":             {"color": "#ff6b4a", "marker": "s", "ls": "--", "label": "LinEXP3 (local)"},
    "CENTRALIZED_LINEXP3": {"color": "#f472b6", "marker": "P", "ls": "-.", "label": "Centralized LinEXP3"},
    "DEXP3":               {"color": "#fbbf24", "marker": "^", "ls": "--", "label": "\u03b5-EXP3"},
    "UNIFORM":             {"color": "#a78bfa", "marker": "D", "ls": ":",  "label": "Uniform"},
}
COST_RATIO_FS = {"tick": 17, "label": 22, "legend": 22}

MISSPEC_RATIO_POLICY_META = {
    "DLEXP3":      {"color": "#4a9eff", "marker": "o", "ls": "-",  "label": "DLEXP3"},
    "CE_DLEXP3":   {"color": "#34d399", "marker": "v", "ls": "-.", "label": "CE-DLEXP3"},
    "BETA_DLEXP3": {"color": "#e879f9", "marker": "*", "ls": ":",  "label": "DLEXP3-SE"},
    "LINEXP3":     {"color": "#ff6b4a", "marker": "s", "ls": "--", "label": "LinExp3 (local)"},
}
MISSPEC_RATIO_FS = {"tick": 20, "label": 25, "legend": 25}

# Both source scripts hardcode the same tick indices, so this is shared.
RATIO_TICK_INDICES = [0, 13, 16, 18, 19]


def _ratio_new_axes(fs_tick, y_formatter):
    BG, PANEL, GRID = "#ffffff", "#ffffff", "#d9d9d9"
    TEXT = "#333333"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=fs_tick)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    ax.grid(color=GRID, linestyle="--", linewidth=0.6, alpha=0.8)
    ax.yaxis.set_major_formatter(y_formatter)
    return fig, ax


def _ratio_set_x_ticks(ax, time_points, fs_label):
    n_pts  = len(time_points)
    idx    = [i for i in RATIO_TICK_INDICES if 0 <= i < n_pts]
    x_vals = np.array(time_points, dtype=float)
    ax.set_xticks(x_vals[idx])
    ax.set_xticklabels([format_T_label(time_points[i]) for i in idx], color="#333333")
    ax.set_xlabel("T", fontsize=fs_label)


def _ratio_finish_and_save(fig, ax, out_path, ylabel, fs_label, fs_legend):
    ax.set_ylabel(ylabel, fontsize=fs_label)
    leg = ax.legend(fontsize=fs_legend, framealpha=0.15)
    for txt in leg.get_texts(): txt.set_color("#111111")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"[plot] Saved -> '{out_path}'")


def compute_cost_ratio(results_total, results_true_cost, time_points, cost_policies, min_snr=1.0):
    """
    regret / cost_optimal, where cost_optimal = cost - regret (see plot.py's
    compute_cost_ratios). Works directly off the in-memory results dicts
    already computed in main(), no JSON round-trip needed.
    """
    out = {}
    for pol in cost_policies:
        cost_mean   = np.array(results_true_cost[pol]["mean"])
        cost_std    = np.array(results_true_cost[pol]["std"])
        regret_mean = np.array(results_total[pol]["mean"])
        regret_std  = np.array(results_total[pol]["std"])

        cost_optimal = cost_mean - regret_mean
        credible = cost_optimal >= min_snr * cost_std
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio      = np.where(credible, regret_mean / cost_optimal, np.nan)
            ratio_band = np.where(credible, regret_std / np.abs(cost_optimal), np.nan)
        out[pol] = (time_points, ratio, ratio_band)
    return out


def compute_misspec_ratio(results_misspec, results_true_cost, time_points, misspec_policies, min_snr=1.0):
    """
    misspec / cost (see misspec_ratio.py's compute_ratios). Only ever
    computed for misspec_policies -- policies without misspec data (i.e.
    everything except DLEXP3/BETA_DLEXP3) are never in that list, so this
    naturally skips them; callers should also gate the plot call itself on
    `if misspec_policies:` since not every run has ANY applicable policy.
    """
    out = {}
    for pol in misspec_policies:
        misspec_mean = np.array(results_misspec[pol]["mean"])
        misspec_std  = np.array(results_misspec[pol]["std"])
        cost_mean    = np.array(results_true_cost[pol]["mean"])
        cost_std     = np.array(results_true_cost[pol]["std"])

        credible = np.abs(cost_mean) >= min_snr * cost_std
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio      = np.where(credible, misspec_mean / cost_mean, np.nan)
            ratio_band = np.where(credible, misspec_std / np.abs(cost_mean), np.nan)
        out[pol] = (time_points, ratio, ratio_band)
    return out


def plot_cost_ratio(ratios, out_path):
    """ratios: {policy: (time_points, ratio, ratio_band)} -- see plot.py's plot_per_env."""
    fs = COST_RATIO_FS
    y_formatter = FuncFormatter(lambda x, pos: f"{x:.0%}")
    fig, ax = _ratio_new_axes(fs["tick"], y_formatter)
    time_points_ref = None
    for pol, (time_points, ratio, ratio_band) in ratios.items():
        meta = COST_RATIO_POLICY_META.get(pol, {"color": "#888888", "marker": "o", "ls": "-", "label": pol})
        x = np.array(time_points, dtype=float)
        ax.plot(x, ratio, color=meta["color"], lw=2.2, ls=meta["ls"],
                marker=meta["marker"], markersize=8, label=meta["label"])
        ax.fill_between(x, ratio - ratio_band, ratio + ratio_band, color=meta["color"], alpha=0.15)
        time_points_ref = time_points
    ax.axhline(0, color="#d9d9d9", lw=1.0, zorder=0)
    _ratio_set_x_ticks(ax, time_points_ref, fs["label"])
    _ratio_finish_and_save(fig, ax, out_path, "regret / optimal cost", fs["label"], fs["legend"])


def plot_misspec_ratio(ratios, out_path):
    """ratios: {policy: (time_points, ratio, ratio_band)} -- see misspec_ratio.py's plot_per_env."""
    fs = MISSPEC_RATIO_FS
    y_formatter = FuncFormatter(lambda x, pos: f"{x:.1E}")
    fig, ax = _ratio_new_axes(fs["tick"], y_formatter)
    time_points_ref = None
    for pol, (time_points, ratio, ratio_band) in ratios.items():
        meta = MISSPEC_RATIO_POLICY_META.get(pol, {"color": "#888888", "marker": "o", "ls": "-", "label": pol})
        x = np.array(time_points, dtype=float)
        ax.plot(x, ratio, color=meta["color"], lw=2.2, ls=meta["ls"],
                marker=meta["marker"], markersize=8, label=meta["label"])
        ax.fill_between(x, ratio - ratio_band, ratio + ratio_band, color=meta["color"], alpha=0.15)
        time_points_ref = time_points
    ax.axhline(0, color="#d9d9d9", lw=1.0, zorder=0)
    _ratio_set_x_ticks(ax, time_points_ref, fs["label"])
    _ratio_finish_and_save(fig, ax, out_path, "Misspec Ratio", fs["label"], fs["legend"])


# ── Main ───────────────────────────────────────────────────────────────────

def main(env_path, config_path):
    with open(env_path)    as f: env    = json.load(f)
    with open(config_path) as f: config = json.load(f)

    meta      = env["metadata"]
    L, K, d   = meta["L"], meta["K"], meta["d"]
    tree      = env["tree"]
    vectors   = np.array(env["context_distribution"]["vectors"],      dtype=float)
    probs     = np.array(env["context_distribution"]["probabilities"], dtype=float)
    Sigma_inv = np.array(env["sigma_inv"],                            dtype=float)
    arm_params = {k: load_schedule(v) for k, v in env["arm_parameters"].items()}

    # No offline_policy.py needed — leaf_avg_mat computed directly here.
    arm_theta_mat, leaf_avg_mat, all_bps = build_lookup_tables(tree, arm_params, K, L, d)

    time_points = config["time_points"]
    num_runs    = config["num_runs"]
    base_seed   = config["base_seed"]
    policies    = config["policies"]
    alpha       = config.get("alpha", 0.0)   # extra per-round multiplicative noise; 0 = off (old-config default)

    print(f"[main] {env_path}   L={L}, K={K}, d={d}")
    print(f"[main] Policies : {policies}")
    print(f"[main] T points : {time_points}")
    print(f"[main] Runs     : {num_runs}  =>  "
          f"{len(time_points) * num_runs * len(policies)} total")
    print(f"[main] alpha    : {alpha}"
          + ("  (no extra randomness)" if alpha == 0.0 else "  (cost *= Uniform[1-alpha,1+alpha] per round)"))
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw            = {p: [] for p in policies}
    raw_misspec    = {p: [] for p in policies}   # entries stay [] for non-applicable policies
    raw_true_cost  = {p: [] for p in policies}   # entries stay [] for non-applicable policies
    t_wall = time.time()

    for T_idx, T in enumerate(time_points):
        print(f"-- T = {T:,} {'-'*40}")
        for pol in policies:
            vals, mvals, cvals = [], [], []
            for run_idx in range(num_runs):
                seed = base_seed + T_idx * num_runs + run_idx
                t0   = time.time()
                r, misspec, true_cost = run_single(
                    T, pol, L, K, d, Sigma_inv,
                    vectors, probs,
                    arm_theta_mat, leaf_avg_mat, all_bps, seed,
                    alpha=alpha,
                )
                vals.append(r)
                if misspec is not None:
                    mvals.append(misspec)
                if true_cost is not None:
                    cvals.append(true_cost)
                msg = f"  {pol:<20} run {run_idx+1}/{num_runs}  regret = {r:>12.2f}"
                if misspec is not None:
                    msg += f"   misspec = {misspec:>10.4f}"
                if true_cost is not None:
                    msg += f"   cost = {true_cost:>12.2f}"
                print(msg + f"   ({time.time()-t0:.1f}s)")
            raw[pol].append(vals)
            if mvals:
                raw_misspec[pol].append(mvals)
            if cvals:
                raw_true_cost[pol].append(cvals)

    print(f"\n[main] Total: {time.time()-t_wall:.1f}s")

    misspec_policies = [pol for pol in policies if raw_misspec[pol]]

    # Total regret results
    results_total = {}
    for pol in policies:
        arr = np.array(raw[pol])
        results_total[pol] = {"time_points": time_points,
                              "mean": arr.mean(axis=1).tolist(),
                              "std":  arr.std(axis=1).tolist()}

    # Average regret (= per-run total/T, aggregated)
    results_avg = {}
    for pol in policies:
        arr_T = np.array(raw[pol]) / np.array(time_points, dtype=float)[:, None]  # (n_T, n_runs)
        results_avg[pol] = {"time_points": time_points,
                            "mean": arr_T.mean(axis=1).tolist(),
                            "std":  arr_T.std(axis=1).tolist()}

    # Root-level total misspecification error (only DLEXP3/BETA_DLEXP3)
    results_misspec = {}
    for pol in misspec_policies:
        arr = np.array(raw_misspec[pol])
        results_misspec[pol] = {"time_points": time_points,
                                "mean": arr.mean(axis=1).tolist(),
                                "std":  arr.std(axis=1).tolist()}

    # Policy's own raw total cost, NOT offset by the offline/optimal
    # comparator. Tracked for EVERY policy now.
    cost_policies = [pol for pol in policies if raw_true_cost[pol]]
    results_true_cost = {}
    for pol in cost_policies:
        arr = np.array(raw_true_cost[pol])
        results_true_cost[pol] = {"time_points": time_points,
                                  "mean": arr.mean(axis=1).tolist(),
                                  "std":  arr.std(axis=1).tolist()}

    env_stem = Path(env_path).stem
    base     = Path(config.get("plot_path", str(RESULTS_DIR)))
    plot_dir = (base.parent if "plot_path" in config else RESULTS_DIR) / env_stem
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Per-env comparison-ready output (consumed by compare_policies.py,
    # misspec_ratio.py, and plot.py). Single merged file: "total"/"avg" for
    # every policy, "cost" for every policy, and "misspec" only for the
    # policies where it's defined (DLEXP3, BETA_DLEXP3).
    output_json = {
        "env": env_stem,
        "L": L, "K": K,
        "time_points": time_points,
        "policies": {
            pol: {
                "total": {"mean": results_total[pol]["mean"], "std": results_total[pol]["std"]},
                "avg":   {"mean": results_avg[pol]["mean"],   "std": results_avg[pol]["std"]},
                **({"cost": {"mean": results_true_cost[pol]["mean"],
                            "std":  results_true_cost[pol]["std"]}}
                  if pol in cost_policies else {}),
                **({"misspec": {"mean": results_misspec[pol]["mean"],
                                "std":  results_misspec[pol]["std"]}}
                  if pol in misspec_policies else {}),
            }
            for pol in policies
        },
    }
    output_path = plot_dir / f"{env_stem}_output.json"
    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"[main] Output JSON -> '{output_path}'")

    plot_results(
        results_total, time_points, policies,
        plot_dir / f"{env_stem}_total.png",
        ylabel=r"log$_{10}$  cumulative regret",
        title=env_stem,
        show_title=False,
        fs_tick=17, fs_label=22, fs_legend=22,
    )

    cost_ratio_data = compute_cost_ratio(results_total, results_true_cost, time_points, cost_policies)
    if cost_ratio_data:
        plot_cost_ratio(cost_ratio_data, plot_dir / f"{env_stem}_cost_ratio.png")

    # Not every policy tracks misspec (only DLEXP3/BETA_DLEXP3 do) -- gate
    # both the computation and the plot call on misspec_policies being
    # non-empty, so an environment/config with neither applicable policy
    # simply skips this third plot instead of erroring or drawing an empty one.
    if misspec_policies:
        misspec_ratio_data = compute_misspec_ratio(
            results_misspec, results_true_cost, time_points, misspec_policies
        )
        plot_misspec_ratio(misspec_ratio_data, plot_dir / f"{env_stem}_misspec_ratio.png")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <env.json> <config.json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
