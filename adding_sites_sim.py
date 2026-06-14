"""
adding_sites_sim.py
-------------------

Toy simulation for the site-selection paper "Where to Go Next:
A Minimax Site-Selection Criterion for External Validity".

For a simplified version of the Boyarsky-Egami-Namkoong (2026)
sensitivity framework, we show that:

  (i) the partial-identification width at a target profile t is
      controlled by the null space of the cross-site moment matrix
      Sigma_S, and
 (ii) adding a candidate site whose site-level context is outside
      the convex hull of existing sites shrinks this null space,
      and therefore the width, much more than an interior candidate.

The simulation compares four candidate sites spanning the three
regimes identified in the paper (coverage-expanding, location-
improving, and interior) and reports the minimax width over a
target region T.

Usage:
    python adding_sites_sim.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Simulation configuration
# ----------------------------------------------------------------------

SEED = 42
D = 10                 # site-level context dimension (d_M + d_I)
S = 8                  # number of existing sites
K = 4                  # number of candidates
N_TARGETS = 20         # size of target region T
N_REPLICATES = 50      # Monte Carlo replicates for the results table
M_BOX = 30.0           # box restriction ||theta||_inf <= M_BOX.
                        # Chosen comfortably larger than the OLS solution
                        # to keep the restricted LP feasible while still
                        # compact.
EFFECTIVE_RANK = 4     # existing sites occupy a 4-dim affine subspace
                        # of the 10-dim context space (mimics Meager 2019)


# ----------------------------------------------------------------------
# Core routines
# ----------------------------------------------------------------------

def generate_existing_sites(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Create S existing-site context vectors living in an EFFECTIVE_RANK-
    dimensional affine subspace of R^D, plus synthetic outcomes used to
    form b.

    Returns
    -------
    contexts : (S, D) array of site-level context vectors.
    outcomes : (S,) array of site-level reduced outcomes y_s used to
               build the right-hand side b = -sum_s c^(s) y_s.
    """
    # Basis for the low-rank subspace
    basis = rng.standard_normal((EFFECTIVE_RANK, D))
    basis, _ = np.linalg.qr(basis.T)          # (D, EFFECTIVE_RANK)
    coords = rng.standard_normal((S, EFFECTIVE_RANK))
    contexts = coords @ basis.T              # exactly rank EFFECTIVE_RANK

    # Latent linear treatment effect in context space. We put the
    # treatment effect IN the row space of contexts so that a small
    # least-squares beta exists; this keeps the box restriction
    # ||theta||_inf <= M compatible with Sigma beta = -b.
    #
    # Draw a low-norm beta in the row-space of contexts:
    beta_row = 0.25 * rng.standard_normal(EFFECTIVE_RANK)
    beta_true = basis @ beta_row              # (D,), lies in span(contexts)
    outcomes = contexts @ beta_true + 0.05 * rng.standard_normal(S)
    return contexts, outcomes


def candidate_sites(existing: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Construct K=4 candidates spanning the three regimes:

       k1: interior of conv(existing)          (regime B)
       k2: near boundary of conv(existing)     (regime B)
       k3: far outside conv(existing), in a direction orthogonal
           to the span of existing sites       (regime A, strong)
       k4: weakly outside conv(existing)       (regime A, weak)
    """
    # Project existing sites to their own span to find an orthogonal dir
    mean = existing.mean(axis=0)
    centered = existing - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=True)
    span_dirs = Vt[:EFFECTIVE_RANK]            # spans existing
    orth_dirs = Vt[EFFECTIVE_RANK:]            # orthogonal complement

    # Random convex combinations for interior / boundary candidates
    w = rng.dirichlet(np.ones(S))
    interior = w @ existing

    w2 = rng.dirichlet(0.3 * np.ones(S))       # puts mass on corners
    boundary = w2 @ existing

    # Strongly-outside candidate: push far along an orthogonal direction
    outside_strong = mean + 2.5 * orth_dirs[0]

    # Weakly-outside candidate: push along a DIFFERENT orthogonal dir
    # with a small magnitude. Both candidates reduce the null space of
    # Sigma by one dimension, but k3's resolved direction typically
    # covers a larger share of the target-region variance.
    outside_weak = mean + 0.6 * orth_dirs[1 if D - EFFECTIVE_RANK > 1 else 0]

    return np.vstack([interior, boundary, outside_strong, outside_weak])


def moments(contexts: np.ndarray, outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simplified cross-site moment identity:

        Sigma = sum_s c^(s) (c^(s))^T,    b = - sum_s c^(s) y_s,

    so that Sigma theta + b = 0 is the OLS first-order condition for
    beta = theta (we drop the gamma block and individual-level X for
    simplicity; the block version of (A.9) of Boyarsky et al. 2026 has
    the same rank structure).
    """
    Sigma = contexts.T @ contexts
    b = -contexts.T @ outcomes
    return Sigma, b


def partial_id_width(Sigma: np.ndarray, b: np.ndarray, t: np.ndarray,
                     M: float = M_BOX, tol: float = 1e-4) -> float:
    """Width of the partial-identification interval at target t under
    the box restriction ||theta||_inf <= M:

        w(t; Sigma, b) = max_theta  t^T theta  -  min_theta  t^T theta
              subject to   Sigma theta = -b,    -M <= theta_i <= M.

    We parameterize the affine subspace {Sigma theta = -b} by the
    min-norm particular solution theta_0 plus the null space of Sigma.
    This avoids passing a rank-deficient equality system to scipy's LP
    solver and isolates the "remaining freedom" after the moment
    identity is enforced.
    """
    d = len(t)
    # SVD of Sigma for rank-robust pseudoinverse and null-space basis.
    U, s, Vt = np.linalg.svd(Sigma, full_matrices=True)
    rank = int((s > tol * s.max()).sum())
    # Min-norm particular solution theta_0 satisfying Sigma theta_0 = -b.
    # (If -b not in range(Sigma), this is the closest approximation; we
    # flag infeasibility if the residual exceeds tol.)
    s_inv = np.zeros_like(s)
    s_inv[:rank] = 1.0 / s[:rank]
    theta0 = Vt.T @ (s_inv * (U.T @ (-b)))
    if np.linalg.norm(Sigma @ theta0 + b) > tol * (1 + np.linalg.norm(b)):
        return np.inf
    # Null space of Sigma: columns corresponding to zero singular values.
    N = Vt[rank:].T              # (d, d - rank)
    if N.shape[1] == 0:
        # theta is uniquely determined; width is zero (if theta0 in box).
        if np.all(np.abs(theta0) <= M + tol):
            return 0.0
        return np.inf

    # LP over nu:  max / min  (N^T t)^T nu    s.t.   -M - theta0 <= N nu <= M - theta0
    c_vec = N.T @ t
    # Inequality: N nu <= M - theta0  and  -N nu <= M + theta0
    A_ub = np.vstack([N, -N])
    b_ub = np.concatenate([M - theta0, M + theta0])
    res_max = linprog(c=-c_vec, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * N.shape[1],
                      method="highs")
    res_min = linprog(c=c_vec, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * N.shape[1],
                      method="highs")
    if res_max.status != 0 or res_min.status != 0:
        return np.inf
    return (-res_max.fun) - res_min.fun


def distance_to_affine_hull(point: np.ndarray, contexts: np.ndarray) -> float:
    """Distance from `point` to the affine hull of `contexts`.
    Uses SVD of the centered context matrix to get the orthogonal
    complement of the affine hull."""
    mean = contexts.mean(axis=0)
    centered = contexts - mean
    _, sigmas, Vt = np.linalg.svd(centered, full_matrices=True)
    effective = (sigmas > 1e-8).sum()
    orth = Vt[effective:]           # orthogonal complement of affine hull
    return float(np.linalg.norm(orth @ (point - mean)))


# ----------------------------------------------------------------------
# One replicate of the design comparison
# ----------------------------------------------------------------------

def run_replicate(seed: int, rng: np.random.Generator | None = None) -> dict:
    if rng is None:
        rng = np.random.default_rng(seed)
    existing, y_exist = generate_existing_sites(rng)
    candidates = candidate_sites(existing, rng)
    Sigma_S, b_S = moments(existing, y_exist)

    # Target region: straddles hull, uniformly sampled in a box whose
    # radius matches the spread of existing contexts.
    target_radius = 1.5 * np.max(np.linalg.norm(existing, axis=1))
    targets = rng.uniform(-target_radius, target_radius, size=(N_TARGETS, D))

    # Use existing-pool OLS to predict candidate-site outcomes (for b update)
    beta_hat, *_ = np.linalg.lstsq(existing, y_exist, rcond=None)

    row = {"existing": {
        "widths": [partial_id_width(Sigma_S, b_S, t) for t in targets]
    }}
    for k in range(K):
        c_k = candidates[k]
        y_k = c_k @ beta_hat           # center of the outcome prior
        aug_X = np.vstack([existing, c_k])
        aug_Y = np.append(y_exist, y_k)
        Sigma_k, b_k = moments(aug_X, aug_Y)

        widths_k = [partial_id_width(Sigma_k, b_k, t) for t in targets]
        dhull_k = np.mean([distance_to_affine_hull(t, aug_X) for t in targets])
        row[f"k{k+1}"] = {"widths": widths_k,
                          "dhull": dhull_k,
                          "context": c_k}
    row["_targets"] = targets
    row["_existing"] = existing
    row["_candidates"] = candidates
    return row


# ----------------------------------------------------------------------
# Driver: run many replicates, report minimax widths, save figure
# ----------------------------------------------------------------------

def main():
    rng_master = np.random.default_rng(SEED)
    seeds = rng_master.integers(0, 2**32 - 1, size=N_REPLICATES)

    # Accumulators
    minimax = {key: [] for key in ["existing", "k1", "k2", "k3", "k4"]}
    winners = {f"k{k+1}": 0 for k in range(K)}

    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        rep = run_replicate(int(seed), rng)
        for key in minimax:
            minimax[key].append(max(rep[key]["widths"]))
        # Record which candidate is minimax-optimal
        cand_ws = {k: max(rep[k]["widths"]) for k in ["k1", "k2", "k3", "k4"]}
        winner = min(cand_ws, key=cand_ws.get)
        winners[winner] += 1

    # --- Report ---
    print("=" * 70)
    print(f"Site-selection simulation  (S={S}, K={K}, D={D}, "
          f"{N_REPLICATES} replicates)")
    print("=" * 70)
    print(f"{'design':25s}  {'mean minimax width':>22s}  {'std':>8s}")
    for key in ["existing", "k1", "k2", "k3", "k4"]:
        ws = np.array(minimax[key])
        print(f"  {key:23s}  {ws.mean():>22.3f}  {ws.std():>8.3f}")
    print()
    print("Minimax-optimal candidate frequency:")
    for k, cnt in winners.items():
        print(f"  {k}: {cnt}/{N_REPLICATES}")

    # --- Figure: one representative replicate ---
    rep = run_replicate(int(seeds[0]), np.random.default_rng(int(seeds[0])))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: 2D projection of site-level contexts.
    # We want one PC axis aligned with the existing-site spread, and the
    # other aligned with the direction in which the rank-expanding
    # candidates push outside the hull. Project onto (PC1 of existing
    # sites) x (leading direction of the candidates' orthogonal
    # complement).
    ax = axes[0]
    existing = rep["_existing"]
    candidates = rep["_candidates"]
    mean = existing.mean(axis=0)
    centered = existing - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=True)
    axis_span = Vt[0]                         # PC1 of existing
    # Pick the orthogonal direction best aligned with the candidates
    cand_centered = candidates - mean
    # Project candidates onto the orthogonal complement of the span
    orth_basis = Vt[EFFECTIVE_RANK:]
    cand_orth = cand_centered @ orth_basis.T  # (K, D - rank)
    # Leading direction of variation among the candidates in this space
    if cand_orth.shape[1] == 0 or np.linalg.norm(cand_orth) < 1e-8:
        axis_orth = Vt[1]
    else:
        _, _, Vc = np.linalg.svd(cand_orth, full_matrices=False)
        axis_orth = Vc[0] @ orth_basis        # lift back to R^D
    P = np.stack([axis_span, axis_orth], axis=1)   # (D, 2)
    proj = lambda X: (X - mean) @ P
    exist_2d = proj(existing)
    cand_2d = proj(candidates)
    targ_2d = proj(rep["_targets"])

    if len(exist_2d) >= 3:
        try:
            hull = ConvexHull(exist_2d)
            for simplex in hull.simplices:
                ax.plot(exist_2d[simplex, 0], exist_2d[simplex, 1],
                        color="steelblue", alpha=0.5, lw=1.4)
        except Exception:
            pass  # low-rank projection: skip hull outline

    ax.scatter(exist_2d[:, 0], exist_2d[:, 1], c="steelblue", s=90,
               label="Existing sites", zorder=3, edgecolor="black", lw=0.5)
    ax.scatter(targ_2d[:, 0], targ_2d[:, 1], c="gray", s=18, alpha=0.5,
               label="Target profiles")
    cand_colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]
    cand_names = [r"$k_1$ (interior)", r"$k_2$ (boundary)",
                  r"$k_3$ (outside, strong)", r"$k_4$ (outside, weak)"]
    for k in range(K):
        ax.scatter(cand_2d[k, 0], cand_2d[k, 1], c=cand_colors[k], s=180,
                   marker="*", zorder=4, edgecolor="black", lw=0.7,
                   label=cand_names[k])

    ax.set_xlabel("PC1 of existing sites")
    ax.set_ylabel("Leading orthogonal direction\n(of candidates, from existing hull)")
    ax.set_title("Site-level context space\n(hull-aligned projection)")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    # Right panel: per-candidate minimax width, averaged across replicates
    ax = axes[1]
    labels = ["existing\npool"] + [f"+ $k_{i+1}$" for i in range(K)]
    values = [np.mean(minimax[k]) for k in ["existing", "k1", "k2", "k3", "k4"]]
    bar_colors = ["gray"] + cand_colors
    ax.bar(range(len(labels)), values, color=bar_colors,
           edgecolor="black", lw=0.7)
    for i, v in enumerate(values):
        ax.text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\sup_{t \in \mathcal{T}}\; w_r(t;\Sigma)$")
    ax.set_title(f"Minimax partial-identification width\n"
                 f"(mean across {N_REPLICATES} replicates)")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = "figure1.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {fig_path}")
    return minimax, winners


if __name__ == "__main__":
    main()
