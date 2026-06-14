"""
monotonicity_sim.py
-------------------

Companion simulation for Section 6 ("Monotonicity: When Adding a Site Can
Hurt") of the site-selection paper.

This script answers two empirical questions:

(Q1) Can adding a site STRICTLY INCREASE the worst-case partial-
     identification width over a target region T?  Yes, even in the
     simplified moment formulation Sigma = sum_s c^(s) c^(s)^T.  The
     mechanism is a rank-preserving outcome update y_k that conflicts
     with the existing best fit: the moment-implied affine subspace
     translates within range(Sigma), and the translated slice cuts the
     restriction box R at a less favorable location for some targets.

(Q2) Under what additional conditions on the candidate site is the
     worst-case width *provably* monotone-nonincreasing?  We verify two
     sufficient conditions:

     (M1) RANK EXPANSION.  If the candidate's context c_k is not in
          range(Sigma_S), the new feasible set is a strict subset of the
          old one and width(t) is monotone-nonincreasing for every t.

     (M2) OUTCOME CONSISTENCY.  If the candidate's outcome agrees with
          the existing-pool minimum-norm prediction y_k = c_k^T theta*_S,
          the new feasible set CONTAINS the old one in the rank-preserving
          case (so width is unchanged) and EQUALS old intersect with a
          consistent hyperplane in the rank-expanding case (so width is
          monotone-nonincreasing).

We compare six candidate flavors:

    k1   interior (rank-preserving), outcome from OLS prediction
    k2   boundary (rank-preserving), outcome from OLS prediction
    k3   outside-hull strong (rank-expanding), outcome from OLS prediction
    k4   outside-hull weak (rank-expanding), outcome from OLS prediction
    k5   rank-preserving with CONFLICTING outcome  -- can increase width
    k6   rank-expanding with consistent outcome    -- monotone improvement

Usage:
    python monotonicity_sim.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt


SEED = 7
D = 10                 # site-level context dimension
S = 8                  # existing sites
N_TARGETS = 20
N_REPLICATES = 50
M_BOX = 5.0            # box restriction half-width per coordinate
EFFECTIVE_RANK = 4
BOX_LOPSIDED_SCALE = 1.0
DELTA = 20.0           # outcome shock for k5 (size of "conflict")


# ----------------------------------------------------------------------
# Core routines  (kept aligned with adding_sites_sim.py)
# ----------------------------------------------------------------------

def generate_existing_sites(rng):
    basis = rng.standard_normal((EFFECTIVE_RANK, D))
    basis, _ = np.linalg.qr(basis.T)
    coords = rng.standard_normal((S, EFFECTIVE_RANK))
    contexts = coords @ basis.T
    beta_row = 0.25 * rng.standard_normal(EFFECTIVE_RANK)
    beta_true = basis @ beta_row
    outcomes = contexts @ beta_true + 0.05 * rng.standard_normal(S)
    return contexts, outcomes, beta_true


def moments(contexts, outcomes):
    Sigma = contexts.T @ contexts
    b = -contexts.T @ outcomes
    return Sigma, b


def min_norm_solution(Sigma, b, tol=1e-8):
    """min-norm theta with Sigma theta = -b."""
    U, s, Vt = np.linalg.svd(Sigma, full_matrices=True)
    rank = int((s > tol * max(s.max(), 1)).sum())
    s_inv = np.zeros_like(s); s_inv[:rank] = 1.0 / s[:rank]
    return Vt.T @ (s_inv * (U.T @ (-b))), rank


def in_range(u, Sigma, tol=1e-6):
    """Test whether u lies in range(Sigma) = range(Sigma^T)."""
    U, s, Vt = np.linalg.svd(Sigma, full_matrices=True)
    rank = int((s > tol * max(s.max(), 1)).sum())
    proj = Vt[:rank].T @ (Vt[:rank] @ u)
    return np.linalg.norm(u - proj) <= tol * (1 + np.linalg.norm(u))


def partial_id_width(Sigma, b, t, R_lo, R_hi, tol=1e-6):
    """Width of t^T theta over {Sigma theta = -b} intersect {R_lo<=theta<=R_hi}."""
    d = len(t)
    U, s, Vt = np.linalg.svd(Sigma, full_matrices=True)
    rank = int((s > tol * max(s.max(), 1)).sum())
    s_inv = np.zeros_like(s); s_inv[:rank] = 1.0 / s[:rank]
    theta0 = Vt.T @ (s_inv * (U.T @ (-b)))
    if np.linalg.norm(Sigma @ theta0 + b) > 1e-5 * (1 + np.linalg.norm(b)):
        return np.inf
    N = Vt[rank:].T
    if N.shape[1] == 0:
        if np.all(theta0 >= R_lo - 1e-9) and np.all(theta0 <= R_hi + 1e-9):
            return 0.0
        return np.inf
    cvec = N.T @ t
    A_ub = np.vstack([N, -N])
    b_ub = np.concatenate([R_hi - theta0, theta0 - R_lo])
    bnds = [(None, None)] * N.shape[1]
    r_max = linprog(c=-cvec, A_ub=A_ub, b_ub=b_ub, bounds=bnds, method='highs')
    r_min = linprog(c= cvec, A_ub=A_ub, b_ub=b_ub, bounds=bnds, method='highs')
    if r_max.status != 0 or r_min.status != 0:
        return np.inf
    return (-r_max.fun) - r_min.fun


# ----------------------------------------------------------------------
# Candidate construction
# ----------------------------------------------------------------------

def make_candidates(existing, rng):
    """Six candidates spanning the regimes of interest.

    k1, k2 lie in the affine hull of existing (rank-preserving).
    k3, k4 lie outside the hull (rank-expanding); k3 is far, k4 weak.
    k5    lies in the hull (rank-preserving) but is paired with a
          conflicting outcome to demonstrate (Q1).
    k6    lies outside the hull (rank-expanding) with a consistent
          outcome to demonstrate (Q2-M2).
    """
    mean = existing.mean(axis=0)
    centered = existing - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=True)
    orth_dirs = Vt[EFFECTIVE_RANK:]
    interior = rng.dirichlet(np.ones(S)) @ existing
    boundary = rng.dirichlet(0.3 * np.ones(S)) @ existing
    outside_strong = mean + 2.5 * orth_dirs[0]
    outside_weak   = mean + 0.6 * orth_dirs[min(1, orth_dirs.shape[0]-1)]
    # k5: another interior point (for the conflicting-outcome regime)
    conflict_ctx   = rng.dirichlet(np.ones(S)) @ existing
    # k6: same direction as outside_strong but with consistent outcome
    monotone_ctx   = mean + 2.0 * orth_dirs[0]
    return np.vstack([interior, boundary, outside_strong, outside_weak,
                      conflict_ctx, monotone_ctx])


def candidate_outcomes(candidates, existing, y_exist, beta_true, rng):
    """Outcome assignments per candidate.

    Default rule: y_k = c_k^T beta_hat  (existing-pool OLS prediction).
    Exception:  k5 receives a large conflicting outcome (shifted by a
    'shock' DELTA away from the OLS prediction); k6 gets the OLS
    prediction exactly (consistent).
    """
    beta_hat, *_ = np.linalg.lstsq(existing, y_exist, rcond=None)
    y_cand = candidates @ beta_hat
    y_cand[4] = y_cand[4] + DELTA
    # k6 already gets OLS prediction (consistent); leave unchanged.
    return y_cand


# ----------------------------------------------------------------------
# Asymmetric box  -- needed so the rank-preserving shift can bite.
# ----------------------------------------------------------------------

def make_box(D, rng, M=M_BOX, asymmetry=BOX_LOPSIDED_SCALE):
    """Return (R_lo, R_hi) for an asymmetric box around 0.
    Asymmetry is parameterised by a random shift inside [-M/3, M/3]."""
    centers = asymmetry * rng.uniform(-M/3, M/3, size=D)
    return centers - M, centers + M


# ----------------------------------------------------------------------
# One replicate
# ----------------------------------------------------------------------

def run_replicate(seed):
    rng = np.random.default_rng(seed)
    existing, y_exist, beta_true = generate_existing_sites(rng)
    candidates = make_candidates(existing, rng)
    y_cand = candidate_outcomes(candidates, existing, y_exist, beta_true, rng)
    Sigma_S, b_S = moments(existing, y_exist)
    R_lo, R_hi = make_box(D, rng)

    target_radius = 1.5 * np.max(np.linalg.norm(existing, axis=1))
    targets = rng.uniform(-target_radius, target_radius, size=(N_TARGETS, D))

    out = {"existing": {"widths": [partial_id_width(Sigma_S, b_S, t, R_lo, R_hi)
                                    for t in targets]}}
    for k in range(candidates.shape[0]):
        ctx = candidates[k]
        Sigma_k, b_k = moments(np.vstack([existing, ctx]),
                               np.append(y_exist, y_cand[k]))
        widths = [partial_id_width(Sigma_k, b_k, t, R_lo, R_hi) for t in targets]
        rank_increased = not in_range(ctx, Sigma_S)
        out[f"k{k+1}"] = {"widths": widths,
                          "rank_increased": bool(rank_increased)}
    out["_meta"] = {"R_lo": R_lo, "R_hi": R_hi, "y_cand": y_cand}
    return out


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def summarise(name, widths):
    """Worst-case (sup) width, handling inf gracefully."""
    arr = np.array([w for w in widths if np.isfinite(w)])
    if len(arr) < len(widths):
        return np.inf
    return arr.max()


def main():
    rng_master = np.random.default_rng(SEED)
    seeds = rng_master.integers(0, 2**32 - 1, size=N_REPLICATES)
    keys = ["existing"] + [f"k{i}" for i in range(1, 7)]
    minimax = {k: [] for k in keys}
    nonmonotone_count = 0      # # replicates where adding k5 increased the width
    infeas_count = {k: 0 for k in keys}

    for seed in seeds:
        rep = run_replicate(int(seed))
        for k in keys:
            mx = summarise(k, rep[k]["widths"])
            minimax[k].append(mx)
            if not np.isfinite(mx):
                infeas_count[k] += 1
        if (np.isfinite(minimax["k5"][-1]) and np.isfinite(minimax["existing"][-1])
                and minimax["k5"][-1] > minimax["existing"][-1] + 1e-6):
            nonmonotone_count += 1

    print("=" * 78)
    print(f"Monotonicity simulation  (S={S}, D={D}, {N_REPLICATES} replicates)")
    print("=" * 78)
    print(f"{'design':16s}  {'mean minimax width':>20s}  {'finite reps':>12s}")
    for k in keys:
        ws = np.array([w for w in minimax[k] if np.isfinite(w)])
        if len(ws) > 0:
            print(f"  {k:14s}  {ws.mean():>20.3f}  {len(ws):>12d}")
        else:
            print(f"  {k:14s}  {'inf':>20s}  {0:>12d}")
    print(f"\nReplicates in which k5 increased the worst-case width: "
          f"{nonmonotone_count}/{N_REPLICATES}")
    print(f"Infeasibility counts per design: {infeas_count}")

    # ---- Figure 2: 3-panel layout
    # (A) hand-built 2D counterexample,
    # (B) head-to-head bar chart of worst-case width,
    # (C) per-replicate scatter post-vs-pre.
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.5))

    # ----- Panel A: hand-built 2D counterexample -----
    ax = axes[0]
    # Old feasible: theta_1 + theta_2 = 2 (S=1, c=(1,1), y=2)
    # New feasible: theta_1 + theta_2 = 1 (k_5 with c=(1,1), y_k=0)
    # Asymmetric box R = [0,2] x [-1,1].
    box = plt.Rectangle((0, -1), 2, 2, fill=True, color="lightyellow",
                        ec="black", lw=1.2, alpha=0.7, zorder=1)
    ax.add_patch(box)
    xs = np.linspace(-0.5, 2.5, 200)
    ax.plot(xs, 2 - xs, color="steelblue", lw=2.2,
            label=r"existing: $\theta_1+\theta_2=2$")
    ax.plot(xs, 1 - xs, color="firebrick", lw=2.2, ls="--",
            label=r"+ $k_5$ shock: $\theta_1+\theta_2=1$")
    # Mark feasible segments
    # existing: theta_1 in [1, 2] -> width(t=(1,0)) = 1
    ax.plot([1, 2], [1, 0], color="steelblue", lw=6, alpha=0.6,
            solid_capstyle="round")
    # new:      theta_1 in [0, 2] -> width(t=(1,0)) = 2
    ax.plot([0, 2], [1, -1], color="firebrick", lw=6, alpha=0.6,
            solid_capstyle="round")
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
    ax.set_title("(A) Hand-built 2D counterexample\n"
                 r"width on $t=(1,0)$:  existing $=1$ $\to$  +$k_5$  $=2$")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3); ax.set_aspect("equal", adjustable="box")

    # ----- Panel B: bar chart of mean minimax width -----
    ax = axes[1]
    labels = ["existing", r"$k_1$", r"$k_2$", r"$k_3$", r"$k_4$",
              r"$k_5$ shock", r"$k_6$ rank+consistent"]
    means = []
    for k in keys:
        ws = np.array([w for w in minimax[k] if np.isfinite(w)])
        means.append(ws.mean() if len(ws) > 0 else np.nan)
    colors = ["gray", "tab:orange", "tab:olive", "tab:red", "tab:purple",
              "tab:brown", "tab:green"]
    ax.bar(range(len(labels)), means, color=colors, edgecolor="black", lw=0.7)
    for i, v in enumerate(means):
        if not np.isnan(v):
            ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
    ax.axhline(means[0], ls="--", color="gray", alpha=0.7,
               label="existing-pool minimax width")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(r"$\sup_{t\in\mathcal{T}}\, w_r(t;\Sigma)$  (mean over reps)")
    ax.set_title("(B) Worst-case partial-ID width by candidate")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # ----- Panel C: per-replicate scatter -----
    ax = axes[2]
    base = np.array(minimax["existing"])
    pairs = [(label, np.array(minimax[k])) for label, k in
             [(r"$k_1$ interior",     "k1"),
              (r"$k_2$ boundary",     "k2"),
              (r"$k_3$ outside strong","k3"),
              (r"$k_4$ outside weak", "k4"),
              (r"$k_5$ shock",        "k5"),
              (r"$k_6$ monotone",     "k6")]]
    colors2 = ["tab:orange","tab:olive","tab:red","tab:purple","tab:brown","tab:green"]
    for (lbl, arr), c in zip(pairs, colors2):
        mask = np.isfinite(arr) & np.isfinite(base)
        ax.scatter(base[mask], arr[mask], s=22, alpha=0.65, label=lbl, color=c)
    lo, hi = base[np.isfinite(base)].min(), base[np.isfinite(base)].max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x  (no change)")
    ax.set_xlabel("existing-pool minimax width")
    ax.set_ylabel("post-addition minimax width")
    ax.set_title("(C) Per-replicate effect of adding each candidate\n"
                 r"(points above the diagonal $\Rightarrow$ width INCREASED)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig("figure2.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved to figure2.png")
    return minimax


if __name__ == "__main__":
    main()
