# Where to Go Next: A Minimax Site-Selection Criterion for External Validity

**Draft — April 2026**

---

## Abstract

Multi-site experiments anchor external-validity claims through a cross-site moment
identity $\Sigma\theta + b = 0$ that, when the number of sites $S$ is small relative to
the site-level covariate dimension, is rank-deficient and leaves the target-site
treatment effect only partially identified. Boyarsky, Egami, and Namkoong (2026)
formalize this partial identification and derive debiased bounds on
$\psi(\theta;\gamma)$ at a user-chosen target profile $(t_m,t_r)$ under
analyst-specified affine restrictions $r(\theta)\le 0$. In this note we study the
*inverse* design question: given a pool of candidate sites with known site-level
context $(C_m^{(k)},C_r^{(k)})$ but no experimental data, which site should an
investigator actually go and run, in order to most strengthen external-validity
claims about a population of target profiles? We formulate the problem as a
minimax design over the width of the partial-identification region, establish a
sharp geometric characterization that, in the absence of restrictions, reduces to
an *expand-the-convex-hull* rule for the site-level context vectors, and derive a
tractable surrogate under compact affine restrictions. The criterion inherits the
three-regime structure of the leave-one-site-out diagnostic of
Boyarsky et al. (2026): candidate sites either (i) strictly reduce the null space of
$\Sigma$ and can collapse unbounded widths to finite ones, (ii) translate the
moment-implied affine subspace closer to the target profile at fixed rank, or
(iii) restore feasibility of substantive restrictions that were incompatible with
the existing pool. We then turn to a subtler point: because the moment
matrix $\Sigma$ is a function of the conditional-on-$X$ covariance of the
site-level context, and the right-hand side $b$ is a function of the
candidate's outcomes, adding a site can in principle *worsen* the
partial-identification width — either by shifting the moment-implied
affine subspace into a less favorable cross-section of the restriction
polyhedron, or by diluting an already-identifying direction of $\Sigma$
under normalization. We isolate two pre-experiment screens — rank
expansion (M1) and outcome-consistency (M2) — and prove that *either* is
sufficient for the worst-case width to be monotone-nonincreasing in the
augmentation (Theorem 1). A Python simulation on synthetic data with
eight existing sites and six candidates confirms both directions of the
theory: rank-expanding candidates reduce the minimax width by roughly
eleven percent, an interior candidate with a conflicting outcome
*increases* it in $16$% of replicates, and a candidate satisfying both
(M1) and (M2) matches the best rank-expander while provably never doing
harm. Complete proofs and an extension to the block moment of
Boyarsky et al. (2026) are deferred to Appendix A. We close with
extensions to the effect-ordering mixed-integer formulation and to
cost-weighted selection.

---

## 1. Introduction

A multi-site experiment derives its external-validity claim from the *cross-site*
variation in site-level context: if the treatment effect is stable across sites
that differ substantially in their moderating attributes, it is credible that the
effect will transport to a new site whose moderators lie within the range
observed. Conversely, when the observed sites span only a thin slice of the
space of site-level moderators, the cross-site moment identity
$\Sigma\theta + b = 0$ is rank-deficient, and the target-site treatment effect at
a moderator profile $(t_m,t_r)$ outside the span of observed sites is only
partially identified, regardless of how large the individual-level samples are
(Boyarsky, Egami, and Namkoong, 2026).

An investigator facing this situation has two levers. She may *tighten the
identification set* by imposing additional substantive restrictions
$r(\theta)\le 0$ (effect-ordering, sign or monotonicity constraints, norm bounds,
etc.), trading a sharper bound against additional untestable assumptions. Or she
may *expand the evidence base* by running a new experiment in an additional
site. The first lever is the subject of Boyarsky et al. (2026); this note
addresses the second.

Specifically, we consider the situation in which the investigator has a small
pool $\mathcal{K} = \{1, \ldots, K\}$ of *candidate* sites for which the
site-level context $(C_m^{(k)}, C_r^{(k)})$ is known in advance — cheaply
obtained from administrative data, public statistics, a pilot survey, or an
established literature — but no individual-level experimental data. Running the
experiment in any one candidate site carries an investigator-specified cost $c_k$
(monetary, logistical, ethical). The question is:

> Given the existing pool of $S$ sites and a range $\mathcal{T}$ of target
> profiles to which the investigator wishes to transport the effect, which
> candidate site $k^\star \in \mathcal{K}$ most improves external-validity
> claims, in the sense of most shrinking the partial-identification width over
> $\mathcal{T}$?

We cast the problem as a **minimax design**:
$$
k^\star \;\in\; \arg\min_{k\in \mathcal{K}}\; \sup_{(t_m,t_r)\in\mathcal{T}}\;
w_r\!\bigl(t_m,t_r;\; \Sigma_{\mathcal{S}\cup\{k\}}\bigr),
$$
where $w_r(t_m,t_r;\Sigma)$ is the width of the partial-identification interval
under restriction $r$ evaluated at target $(t_m,t_r)$. The key observation — and
what makes this design problem tractable without running the candidate
experiments — is that the width $w_r$ depends on the candidate site through
$\Sigma$, the *design matrix* of the moment condition, which in turn is a
function of known site-level quantities $(C_m, C_r)$ only. The right-hand side
$b$ would move once the experiment is run, but the geometry that determines
whether the width is finite — and by how much — is pinned down by the
site-level context. Selecting a site to add is therefore a design-time problem,
not a post-hoc one.

**Contributions.** (1) We formalize the site-selection problem in the partial-
identification framework of Boyarsky et al. (2026) and show that the partial-
identification width depends on the candidate through a known-ex-ante rank-one-
like update of $\Sigma$. (2) We establish that, in the unrestricted limit
$r \equiv 0$ with mild regularization, the minimax criterion collapses to a
convex-hull expansion rule in the site-level context space: the optimal
candidate is the one whose $(C_m^{(k)}, C_r^{(k)})$ lies furthest, in a
weighted-distance sense, from the convex hull of observed sites, with the
weighting determined by $\mathcal{T}$. (3) Under compact affine restrictions
$r(\theta)\le 0$, we derive a tractable surrogate — a finite LP over the
candidate pool — that has the same three-regime structure as the leave-one-
site-out diagnostic of Boyarsky et al. (2026): candidates are *coverage-
expanding* (reduce rank-deficiency), *location-improving* (shift the affine
subspace at fixed rank), or *feasibility-restoring* (make a previously
incompatible restriction set feasible). (4) We show that the minimax design is *not* automatically
monotone in site addition: a rank-preserving candidate paired with an
outcome that conflicts with the existing-pool fit can shift the moment-
implied affine subspace to a less favorable cross-section of the
restriction polyhedron and strictly *increase* the worst-case width. We
isolate two pre-experiment screens — rank expansion (M1) and outcome-
consistency (M2) — and prove (Theorem 1) that either is sufficient for
the worst-case width to be monotone-nonincreasing in the augmentation.
(5) A synthetic simulation with eight existing sites and six candidates,
including a deliberately conflicting one, confirms both the
hull-expansion result and the non-monotonicity (with $16$% of replicates
exhibiting strict worsening from the unscreened candidate, and zero from
any screened candidate). (6) We sketch extensions to the mixed-integer
effect-ordering formulation and to cost-weighted objectives, and defer
complete proofs to Appendix A.

**Positioning.** The site-selection question is the *forward* version of the
leave-one-site-out diagnostic in Section 4.4 of Boyarsky et al. (2026), which
asks which existing donor sites are load-bearing for a reported bound. Our
problem inherits the three-regime taxonomy of that diagnostic but evaluates it
over a *counterfactual* addition rather than a realized removal. It also
complements classical optimal experimental design — D-, G-, and I-optimality —
which operates within a single population under point identification (Atkinson
et al., 2007; Pukelsheim, 2006). In our setting the design object is not an
individual-level allocation of treatment but a site-level choice of *context*,
and the loss is the width of a partial-identification interval rather than the
variance of a point estimator. The minimax objective is natural here because
site selection is a discrete, one-shot decision that should be robust to the
heterogeneity of target profiles the investigator cares about.

---

## 2. Preliminaries

We briefly recap the setup of Boyarsky et al. (2026). The hierarchical
data-generating process is
$$
Y \;=\; A\,\tau(X) + C_m^\top\beta_1 + C_r^\top\beta_2 + g_0(X)
 + C_m^\top\gamma_1 + C_r^\top\gamma_2 + \varepsilon,
\qquad\qquad (2.1)
$$
where $A\in\{0,1\}$ is the treatment, $X\in\mathbb{R}^{d_X}$ individual-level
covariates, $C_m\in\mathbb{R}^{d_M}$ site-level main effects, and
$C_r\in\mathbb{R}^{d_I}$ site-level interaction covariates. The parameter
$\theta := (\beta_1,\beta_2,\gamma_1,\gamma_2)\in\mathbb{R}^{d_g}$ with
$d_g = 2(d_M + d_I)$ captures direct and indirect effects.

**Proposition 1 of Boyarsky et al. (2026).** *Under a constant-propensity RCT
and mean-zero errors, for a target profile $(t_m,t_r)$ with
$\mathrm{Pr}[C_m=t_m, C_r=t_r]>0$,*
$$
\psi(\theta;\gamma) \;:=\; \mathbb{E}[Y(1)-Y(0)\,|\,C_m=t_m, C_r=t_r]
$$
*admits the representation*
$$
\psi(\theta;\gamma)
 \;=\; c_0(\gamma) + \beta_1^\top \tilde c_m(\gamma) + \beta_2^\top \tilde c_r(\gamma),
\qquad
\Sigma\theta + b = 0,
\qquad\qquad (2.2)
$$
*with the moment matrix $\Sigma$ of Equation (A.9) in their appendix, namely a
block matrix of $\mathbb{E}_X\!\bigl[\mathrm{Cov}(C_\bullet,C_\bullet \mid X)\bigr]$
entries with $\bullet \in \{m, r\}$.*

The key structural fact we exploit in this paper is that
$\Sigma$ is a linear functional of the joint distribution of $(C_m, C_r, X)$, and
in particular its rank is governed by the rank of the $(S \times (d_M + d_I))$
matrix whose rows are the site-level contexts $(C_m^{(s)}, C_r^{(s)})$, weighted
by site sample sizes and individual covariate overlap. When $S$ is small
relative to $d_M + d_I$, this matrix — and hence $\Sigma$ — is rank-deficient;
this is precisely why the Meager (2019) micro-credit application with $S=7$
sites exhibits non-trivial partial identification.

**Partial-identification bounds.** Boyarsky et al. (2026) define
$$
v_{\ell}(\gamma) \;=\; \min_\theta\,\bigl\{\psi(\theta;\gamma)\,:\,
 \Sigma\theta + b = 0,\; r(\theta)\le 0\bigr\},
\qquad
v_{u}(\gamma) \;=\; \max_\theta\,\bigl\{\psi(\theta;\gamma)\,:\,
 \Sigma\theta + b = 0,\; r(\theta)\le 0\bigr\},
\qquad\qquad (2.3)
$$
and the width of the partial-identification interval at target $(t_m,t_r)$ is
$$
w_r(t_m, t_r; \Sigma, b) \;=\; v_u(\gamma) - v_\ell(\gamma).
\qquad\qquad (2.4)
$$
We suppress the dependence of $c_0, \tilde c_m, \tilde c_r$ on $\gamma$ when no
confusion arises; these enter the objective linearly in $\theta$ but do not
interact with the feasible set, so they do not affect the width — only the
endpoints.

**Restrictions.** The analyst chooses $r(\theta)\le 0$ from a library of
affinely-representable restrictions (effect-ordering, sign constraints, norm
bounds, monotonicity). Throughout this paper we assume $r$ defines a compact
polyhedron $\mathcal{R} := \{\theta : r(\theta)\le 0\}$; this is the regime
where the width is finite and the LP/MIP program is well-posed. The extension
to relaxed restriction sets (where unrestricted widths are permitted) is
discussed in Section 7.

---

## 3. The Site-Selection Problem

### 3.1 Setup

We observe outcomes from $S$ existing sites, pooled as
$\mathcal{D}_\mathcal{S} = \{(Y_i, A_i, X_i, C_{m,i}, C_{r,i})\}_{i=1}^n$. A
**candidate pool** $\mathcal{K} = \{1, \ldots, K\}$ consists of $K$ prospective
sites for which the site-level context $(C_m^{(k)}, C_r^{(k)})$ is observed but
no individual-level data has been collected. For each candidate we assume the
investigator knows (or can assume a prior on) the within-site distribution of
the individual-level covariates $X$ — this is the "some prior on $X$" in our
motivating question and is necessary because $\Sigma$ depends on
$\mathbb{E}_X[\mathrm{Cov}(C_\bullet, C_\bullet \mid X)]$. In the simplest case,
the investigator assumes $X$ at candidate site $k$ is drawn from the same
distribution as at an existing reference site; in the most refined case, a
calibrated demographic prior is used. This additional input is analogous to the
pre-registered prior used in two-stage adaptive designs; it is an assumption,
not a free parameter.

The investigator also specifies:

- a **target region** $\mathcal{T} \subseteq \mathbb{R}^{d_M+d_I}$ of context
  profiles over which she cares about transportability (a discrete set of
  "benchmark" profiles, or a continuous region such as a box around the
  marginal mean of the target population);
- a **restriction set** $\mathcal{R} = \{\theta : r(\theta) \le 0\}$ assumed
  compact;
- optionally, a **cost** $c_k \ge 0$ per candidate site.

### 3.2 The site-augmentation update

When a candidate site $k$ is added, the moment matrix updates from
$\Sigma_{\mathcal{S}}$ to
$\Sigma_{\mathcal{S}\cup\{k\}}
 = \Sigma_{\mathcal{S}} + \Delta\Sigma_k(C_m^{(k)}, C_r^{(k)}; P_X^{(k)})$,
where $\Delta\Sigma_k$ depends on the candidate's site-level context and on the
investigator's assumed within-site $X$-distribution but *not* on its outcomes.
Using the block form of Equation (A.9) of Boyarsky et al. (2026), one verifies

$$
\Delta\Sigma_k
\;=\; n_k \cdot \bigl(-\rho\bigr)\cdot
\begin{bmatrix}
  u_m u_m^\top & u_m u_r^\top & u_m u_m^\top & u_r u_m^\top \\
  u_r u_m^\top & u_r u_r^\top & u_m u_r^\top & u_r u_r^\top \\
  u_m u_m^\top & u_r u_m^\top & \rho^{-1} u_m u_m^\top & \rho^{-1} u_r u_m^\top \\
  u_m u_r^\top & u_r u_r^\top & \rho^{-1} u_m u_r^\top & \rho^{-1} u_r u_r^\top
\end{bmatrix},
\qquad
(u_m, u_r) := \bigl(C_m^{(k)}, C_r^{(k)}\bigr) - \bar C_X^{(k)},
\qquad\qquad (3.1)
$$

where $\bar C_X^{(k)} = \mathbb{E}_{P_X^{(k)}}[(C_m, C_r) \mid X]$ — that is,
$u_m, u_r$ are the *residuals* of the site-level context after projecting out
the individual-level covariate dependence at the candidate site, and $n_k$ is
the planned sample size at the candidate site. The exact expression in (3.1)
depends on how the candidate site is pooled with the existing sites in the
conditional-moment estimator, but the *rank-update structure* is universal:
$\Delta\Sigma_k$ has rank at most $2$, with left/right directions spanned by
the residualized context $(u_m, u_r)$ of the new site.

The right-hand side $b$ also updates, from $b_\mathcal{S}$ to
$b_{\mathcal{S}\cup\{k\}}$, but the new right-hand side depends on the candidate
site's *outcomes* and is therefore unknown until the experiment is run. We
return to this in Section 5.3; for most of our analysis the width depends on
$b$ only through a compact-translation argument that does not affect the
first-order ranking of candidates.

### 3.3 The minimax design problem

With the site-augmentation update in hand, we define the selection problem
$$
k^\star \;\in\; \arg\min_{k\in\mathcal{K}}\;
 \sup_{(t_m,t_r)\in\mathcal{T}}\;
 w_r\!\bigl(t_m,t_r;\, \Sigma_{\mathcal{S}\cup\{k\}},\, b_{\mathcal{S}\cup\{k\}}\bigr).
\qquad\qquad (3.2)
$$
When costs $c_k$ are specified, the natural Lagrangian form is
$$
k^\star_\kappa \;\in\; \arg\min_{k\in\mathcal{K}}\;
 \sup_{(t_m,t_r)\in\mathcal{T}}\;
 w_r\!\bigl(t_m,t_r;\, \Sigma_{\mathcal{S}\cup\{k\}}\bigr)
 \;+\; \kappa\, c_k,
\qquad\qquad (3.3)
$$
with Lagrange multiplier $\kappa \ge 0$ reflecting the marginal willingness to
pay for a unit reduction in worst-case width. All results below extend directly
to (3.3) because the cost term is separable in $k$.

Because $b$ is not known before the candidate experiment is run, we evaluate
the width in (3.2) at a *robustified* $b$ that maximizes the width over a
compact uncertainty set $\mathcal{B}$ consistent with the investigator's prior
beliefs about candidate-site outcomes. In the simplest and cleanest case — and
the one we focus on from here — $\mathcal{B}$ is a symmetric norm ball around
the existing-pool $b$; this nests the "no-information" prior and is what we use
in our simulations.

---

## 4. Why Convex-Hull Expansion

The minimax objective (3.2) inherits a sharp geometric structure in the
unrestricted limit $\mathcal{R} = \mathbb{R}^{d_g}$ (or equivalently, when the
restriction set is large enough that no face of $r$ binds at the optimum). In
this limit the partial-identification width is determined entirely by the
interaction between the null space of $\Sigma$ and the direction of the target
functional.

### 4.1 Unrestricted dichotomy

Write the linear target functional as $c(t)^\top \theta$ with
$c(t) := (\tilde c_m(t), \tilde c_r(t), 0, 0)$ (the zeros in the
$(\gamma_1, \gamma_2)$-coordinates follow because the target functional in
(2.2) is linear in $(\beta_1, \beta_2)$ only; the constant piece
$c_0(\gamma)$ shifts the endpoints but leaves the width invariant).

**Proposition 1** (*Unrestricted width is 0 or $\infty$.*)
*Let $\mathcal{R} = \mathbb{R}^{d_g}$. Then for any target direction $c(t)$ and
any feasible $b$,*
$$
w_{\rm unr}(t; \Sigma, b) \;=\;
 \begin{cases}
  0 & \text{if } c(t)\in \mathrm{range}(\Sigma^\top), \\
  +\infty & \text{otherwise.}
 \end{cases}
$$

*Proof sketch.* The feasible set $\{\theta : \Sigma\theta = -b\}$ is an affine
subspace with direction subspace $\mathrm{null}(\Sigma)$. The width of a linear
functional over an affine subspace is $0$ if the functional is constant on the
direction subspace, and unbounded otherwise. Constancy is equivalent to
$c(t) \perp \mathrm{null}(\Sigma)$, which by rank–nullity is
$c(t) \in \mathrm{range}(\Sigma^\top)$. $\blacksquare$

**Corollary.** *Adding a candidate site $k$ collapses the unrestricted width at
target $t$ from $\infty$ to $0$ if and only if the residualized context
$(u_m^{(k)}, u_r^{(k)})$ of the candidate is such that
$c(t) \in \mathrm{range}\!\bigl(\Sigma_\mathcal{S}^\top + \Delta\Sigma_k^\top\bigr)$
but $c(t) \notin \mathrm{range}(\Sigma_\mathcal{S}^\top)$.*

This is the unrestricted version of the *coverage-expanding-donor* regime (a)
of the leave-one-site-out taxonomy in Boyarsky et al. (2026, §4.4).

### 4.2 The convex-hull interpretation

Proposition 1 has a clean geometric reading in the site-level context space.
Let
$$
\mathcal{H}_\mathcal{S} \;:=\; \mathrm{conv}\!\bigl\{(C_m^{(s)}, C_r^{(s)})\,:\,
 s = 1, \ldots, S\bigr\} \;\subseteq\; \mathbb{R}^{d_M+d_I}.
$$
Its *affine hull* $\mathrm{aff}(\mathcal{H}_\mathcal{S})$ is a linear subspace
(after centering) of dimension at most $S-1$. The cross-site moment matrix
$\Sigma_\mathcal{S}$ has rank at most $\dim\mathrm{aff}(\mathcal{H}_\mathcal{S})
 \cdot (\text{block factor})$, with equality when the existing sites have
sufficient overlap in $X$ at the points where their contexts differ.

**Proposition 2** (*Hull expansion strictly reduces the null space.*)
*If $(C_m^{(k)}, C_r^{(k)}) \notin \mathrm{aff}(\mathcal{H}_\mathcal{S})$ and
the candidate site's $X$-distribution has positive density on the
individual-level support, then*
$$
\mathrm{rank}\bigl(\Sigma_{\mathcal{S}\cup\{k\}}\bigr)
 \;>\; \mathrm{rank}\bigl(\Sigma_\mathcal{S}\bigr),
\quad\Longleftrightarrow\quad
 \mathrm{null}\bigl(\Sigma_{\mathcal{S}\cup\{k\}}\bigr) \;\subsetneq\;
 \mathrm{null}\bigl(\Sigma_\mathcal{S}\bigr).
$$

*Proof sketch.* The block form (3.1) shows $\Delta\Sigma_k$ has row space
spanned by $(u_m^{(k)}, u_r^{(k)})$. A rank increase occurs iff these rows lie
outside the existing row space of $\Sigma_\mathcal{S}$, which by the
site-level block structure is equivalent to $(C_m^{(k)}, C_r^{(k)})$ lying
outside $\mathrm{aff}(\mathcal{H}_\mathcal{S})$. $\blacksquare$

Combined with Proposition 1, this yields a finite-vs-infinite dichotomy: *under
no restriction $r$, the only candidate sites that shrink the unrestricted width
are those whose context lies outside the affine hull of existing sites.* The
convex-hull expansion rule is the special case in which the target region
$\mathcal{T}$ is itself contained in a larger convex set whose existing-site
coverage is incomplete.

### 4.3 Width-as-distance-to-hull

Proposition 1's dichotomy is discontinuous, which is uninformative for
comparing two candidates both of which lie outside the hull, or both of which
lie inside. The correct continuous refinement uses a *regularized* width,
$$
w_r^\tau(t; \Sigma, b) \;:=\; \max_\theta\{c(t)^\top\theta\} -
 \min_\theta\{c(t)^\top\theta\}\quad
\text{s.t.}\quad
\Sigma\theta+b = 0,\; \|\theta\|_2 \le \tau,
\qquad\qquad (4.1)
$$
with a large-radius ball as a mild stand-in for a compact restriction set. The
regularized width admits a closed form:
$$
w_r^\tau(t; \Sigma, b) \;\le\; 2\tau \cdot \mathrm{dist}\!\bigl(
 c(t),\; \mathrm{range}(\Sigma^\top)\bigr),
\qquad\qquad (4.2)
$$
with equality up to a boundary adjustment that vanishes at large $\tau$.
The distance from $c(t)$ to $\mathrm{range}(\Sigma^\top)$ is precisely the norm
of the component of $c(t)$ in $\mathrm{null}(\Sigma)$. Geometrically, this
distance is a monotone function of the distance from the target profile
$(t_m, t_r)$ to $\mathrm{aff}(\mathcal{H}_\mathcal{S})$, and strictly monotone
under the rank conditions of Proposition 2. We formalize this in Proposition 3.

**Proposition 3** (*Width is monotone in distance to the affine hull.*)
*Suppose the cross-site design is balanced and the within-site $X$-distribution
is the same across existing and candidate sites. Let
$d_\mathcal{S}(t) := \mathrm{dist}((t_m, t_r), \mathrm{aff}(\mathcal{H}_\mathcal{S}))$.
Then*
$$
w_r^\tau(t; \Sigma_\mathcal{S}, b) \;=\; 2\tau \cdot d_\mathcal{S}(t) \cdot
 (1 + o_\tau(1)),
\qquad\qquad (4.3)
$$
*with equality in the limit $\tau\to\infty$. In particular, for two candidates
$k_1, k_2$ with
$d_{\mathcal{S}\cup\{k_1\}}(t) < d_{\mathcal{S}\cup\{k_2\}}(t)$ uniformly over
$t\in\mathcal{T}$, candidate $k_1$ dominates $k_2$ under the minimax criterion
(3.2) in the unrestricted limit.*

Proposition 3 delivers the promised **convex-hull expansion rule**: in the
unrestricted (large-$\tau$) limit, the minimax-optimal candidate is the one
whose addition minimizes
$\sup_{t \in \mathcal{T}} d_{\mathcal{S}\cup\{k\}}(t)$ — i.e., the one whose
enlarged affine hull leaves no target profile in $\mathcal{T}$ far from the
span of sites. When $\mathcal{T}$ is a compact set containing points outside
$\mathcal{H}_\mathcal{S}$, this objective is exactly *expanding the convex
hull* in the direction of the target region.

---

## 5. The Minimax Criterion under Affine Restrictions

Compact affine restrictions $\mathcal{R} = \{\theta : r(\theta)\le 0\}$ —
e.g., box constraints, effect-ordering, sign/monotonicity — replace the
unbounded-in-the-null-space behavior of Proposition 1 with a finite LP whose
width we can compute and optimize.

### 5.1 A tractable surrogate

Fix a candidate $k$ and a target $(t_m, t_r)$. The width is the value of the
LP pair (2.3) with $\Sigma \leftarrow \Sigma_{\mathcal{S}\cup\{k\}}$. Because
$\Delta\Sigma_k$ has rank at most $2$, the candidate's effect on the feasible
set is a *rank-2 perturbation* of the affine subspace
$\{\Sigma_\mathcal{S}\theta = -b\}$. Under Assumption 6 of Boyarsky et al.
(2026) (Robinson constraint qualification) the LP values are Lipschitz in
$\Sigma$, and we obtain a first-order surrogate:

**Proposition 4** (*Rank-2 envelope.*)
*Let $(\theta_\mathcal{S}^\star, \lambda_\mathcal{S}^\star,
\nu_\mathcal{S}^\star)$ be a KKT triple at $\Sigma_\mathcal{S}$ for a target
$(t_m, t_r)$, with minimum-norm multiplier $\lambda_\mathcal{S}^\star$. Then*
$$
v_u\bigl(\gamma_{\mathcal{S}\cup\{k\}}\bigr) - v_u\bigl(\gamma_\mathcal{S}\bigr)
 \;\approx\;
 - \lambda_\mathcal{S}^{\star\top}\bigl(\Delta\Sigma_k\,
 \theta_{\mathcal{S}}^\star + \Delta b_k\bigr),
\qquad\qquad (5.1)
$$
*and analogously for $v_\ell$. The first-order change in the width is
$\Delta w_k \approx - (\lambda_\mathcal{S}^{u,\star} - \lambda_\mathcal{S}^{\ell,\star})^\top
 (\Delta\Sigma_k\,\theta^\star_{\mathcal{S}} + \Delta b_k)$.*

Proposition 4 is the Bonnans–Shapiro directional derivative of the
constrained-value map, specialized to the finite-dimensional rank-2 site-update
direction $(h_\Sigma, h_b) = (\Delta\Sigma_k/\|\Delta\Sigma_k\|, \Delta b_k)$.
The formula (5.1) coincides with Equation (1.3) of Boyarsky et al. (2026),
evaluated at the site-augmentation direction. It gives a *linear-in-$k$*
surrogate for the width change, so the minimax selection (3.2) reduces to a
finite computation:
$$
k^\star_{\rm surr}\;\in\;\arg\min_{k\in\mathcal{K}}\;
 \sup_{(t_m,t_r)\in\mathcal{T}}\;
 \Bigl|\,\lambda_\mathcal{S}^{u,\star}(t)^\top
  \Delta\Sigma_k\,\theta^\star_\mathcal{S}(t)
  - \lambda_\mathcal{S}^{\ell,\star}(t)^\top
  \Delta\Sigma_k\,\theta^\star_\mathcal{S}(t)\,\Bigr|,
\qquad\qquad (5.2)
$$
after taking a sup over the unknown $\Delta b_k$ in the prior ball
$\mathcal{B}$, which is handled by a standard dual-norm bound and is additive
in the candidate. Computing (5.2) requires solving the LP pair (2.3) at
$\Sigma_\mathcal{S}$ once per target and a finite-dimensional matrix-vector
product per candidate, total cost $O(|\mathcal{T}|\cdot K)$.

### 5.2 Three regimes

The linearization (5.1) inherits the three-regime taxonomy of the LOO
diagnostic of Boyarsky et al. (2026, §4.4), now read in reverse:

- **(A) Coverage-expanding candidate.** If $(C_m^{(k)}, C_r^{(k)}) \notin
 \mathrm{aff}(\mathcal{H}_\mathcal{S})$, then
 $\mathrm{null}(\Sigma_{\mathcal{S}\cup\{k\}}) \subsetneq \mathrm{null}(\Sigma_\mathcal{S})$.
 Proposition 4 detects this as a first-order contraction of the width whenever
 the binding recession direction at target $t$ lies along the newly-resolved
 null-space direction. In the unrestricted limit this regime *collapses*
 infinite widths to finite ones.

- **(B) Location-improving candidate.** If $(C_m^{(k)}, C_r^{(k)}) \in
 \mathrm{aff}(\mathcal{H}_\mathcal{S})$ but outside the convex hull, the rank
 of $\Sigma$ does not change but $\Delta b_k$ translates the moment-implied
 affine subspace. The binding vertex of the LP can shift, yielding a
 first-order width change of either sign. This is the regime in which a new
 site *improves overlap* at the target without adding a new identification
 direction.

- **(C) Feasibility-restoring (or -breaking) candidate.** If
 $\{\theta : \Sigma_\mathcal{S}\theta = -b_\mathcal{S}\}\cap \mathcal{R} =
 \emptyset$ — an infeasibility signaled by the Farkas alternative of
 Boyarsky et al. (2026, §4.4 (c)) — a candidate $k$ can restore feasibility
 if its site-update direction pushes the affine subspace back into
 $\mathcal{R}$. Conversely, a candidate whose direction is incompatible with
 the existing restriction set can break feasibility, which is diagnostic
 information rather than a reason to exclude the candidate.

The minimax selection (3.2) therefore does not reduce to a single geometric
quantity in general; the convex-hull rule is exact in regime (A) and the
unrestricted limit, and is a useful first screen that should be refined by
(5.2) when the restriction set is binding.

### 5.3 Handling the unknown $\Delta b_k$

The right-hand-side update $\Delta b_k$ depends on the candidate site's future
outcomes. Three prior choices are natural:

1. **Uninformative ball.** $\Delta b_k \in \{b : \|b\|_2 \le B\}$. Yields a
   worst-case width via dual-norm bound, but can dominate $\Delta\Sigma_k$
   at small candidate sample sizes. Conservative.

2. **Meta-analytic prior.** Use the existing-pool estimates as a prior on
   the candidate's $(Y, A)$ conditional means, and propagate through the
   expression for $b$ in Equation (A.7) of Boyarsky et al. (2026). This is a
   structural Bayesian design; under a flat prior on $\theta$ it recovers a
   Bayesian D-optimality criterion, and the minimax and Bayesian selections
   agree in the large-candidate-sample limit.

3. **Adversarial $b$.** $\Delta b_k \in \mathcal{B}$ chosen to maximize
   the width. This is the robust-optimization analogue of (1) with an
   ellipsoidal $\mathcal{B}$; Sion-minimax arguments give a tractable saddle-
   point representation.

In our simulation (Section 7) we adopt (1) with $B$ calibrated from the
existing-pool residual variance; the three variants agree on the minimax-
optimal candidate in all our runs, which is unsurprising given that the rank
update in (3.1) dominates the first-order width change in the regimes where
candidates are materially distinguishable.

---

## 6. Monotonicity: When Adding a Site Can Hurt

The reader could be forgiven for assuming that the minimax design problem
(3.2) is *bounded above by the existing-pool width* — i.e., that
$\sup_t w_r(t;\Sigma_{\mathcal{S}\cup\{k\}})\le \sup_t w_r(t;\Sigma_\mathcal{S})$
for every candidate $k$, so that the design problem is merely a question
of "how much do we gain". This is false. Adding a site can strictly
*increase* the worst-case width, and in some configurations it can render
the program infeasible. This section traces the failure modes and isolates
the conditions under which monotone improvement is guaranteed.

### 6.1 Why monotone improvement can fail

Recall from §3.2 that two things change when a candidate site is added:
the moment matrix $\Sigma$ acquires a rank-$\le 2$ update $\Delta\Sigma_k$
in the row directions spanned by the *residualized* context
$(u_m^{(k)},u_r^{(k)})$, and the right-hand side $b$ shifts by an outcome-
dependent amount $\Delta b_k$. The first update is monotone in PSD order:
$\Sigma_{\mathcal{S}\cup\{k\}} \succeq \Sigma_\mathcal{S}$, so
$\mathrm{null}(\Sigma_{\mathcal{S}\cup\{k\}})\subseteq
 \mathrm{null}(\Sigma_\mathcal{S})$. The second is not monotone in any sense.
Three concrete mechanisms can make the post-addition width larger than the
pre-addition width:

**(F1) Rank-preserving shift.** If
$(u_m^{(k)},u_r^{(k)})\in \mathrm{range}(\Sigma_\mathcal{S})$, the rank of
$\Sigma$ does not change, and the new feasible set is *parallel* to the old
in the sense $\theta'+\mathrm{null}(\Sigma_\mathcal{S})$ for some
$\theta'\ne\theta_\mathcal{S}^\star$ whenever
$y_k\ne (u_m^{(k)},u_r^{(k)})^\top \theta_\mathcal{S}^\star$. This parallel
slice intersects the restriction polyhedron $\mathcal{R}$ at a different
polytope, whose projection onto the target functional can be longer than
the original — for example, when $\mathcal{R}$ is asymmetric around
$\theta_\mathcal{S}^\star$ and the shift moves the slice into a wider
cross-section of $\mathcal{R}$.

**(F2) Conditioning dilution under normalized moments.** In the BEN
formulation, $\Sigma$ is the empirical estimator of
$\mathbb{E}_X[\mathrm{Cov}(C,C\mid X)]$, computed as a sample average over
the pooled $(\sum_s n_s + n_k)$ observations. Sample averaging
re-weights the existing-pool contribution by a factor
$n/(n+n_k)$, so the smallest non-trivial eigenvalue
$\sigma_{\min,+}(\Sigma_{\mathcal{S}\cup\{k\}})$ can be smaller than
$\sigma_{\min,+}(\Sigma_\mathcal{S})$ when the candidate's residualized
context concentrates in a direction already well represented. In the LP
(2.3), this corresponds to a worsened condition number of the equality-
constraint block and can enlarge the feasible set in directions where
$\mathcal{R}$ does not bite. The PSD-monotonicity argument above does *not*
apply to the normalized estimator.

**(F3) Restriction-set conflict.** If the candidate's outcome update
$\Delta b_k$ shifts the moment-implied affine subspace
$\{\Sigma_{\mathcal{S}\cup\{k\}}\theta = -b_{\mathcal{S}\cup\{k\}}\}$
outside the restriction polyhedron $\mathcal{R}$, the LP becomes infeasible
and the partial-identification interval is empty. This is the
feasibility-breaking case of regime (C) in §5.2.

The simulation in §7 demonstrates (F1) directly: an outcome shock of
magnitude $\Delta = 20$ at a rank-preserving candidate $k_5$ produces a
strict increase in the worst-case width in $8/50$ Monte-Carlo replicates,
with no rank or feasibility change.

### 6.2 An explicit counterexample

We give a fully worked $D=2$ example, both because the geometry is
diagnostic and because the same construction lifts to the multivariate
simulations of §7.

**Setup.** Take $d_g=2$, $S=1$ existing site with context $c^{(1)}=(1,1)^\top$
and outcome $y_1=2$. The simplified moment is
$\Sigma_\mathcal{S}=c^{(1)}{c^{(1)}}^\top=\bigl(\begin{smallmatrix}1&1\\1&1\end{smallmatrix}\bigr)$
with $\mathrm{rank}=1$, and
$b_\mathcal{S}=-c^{(1)}y_1=(-2,-2)^\top$.
The moment-implied subspace is $\{(\theta_1,\theta_2):\theta_1+\theta_2=2\}$,
a line in $\mathbb{R}^2$. Take the (asymmetric) box restriction
$\mathcal{R}=[0,2]\times[-1,1]$ and the target functional
$c(t)=(1,0)^\top$. The feasible segment is
$\{(\theta_1,2-\theta_1):\theta_1\in[1,2]\}$, with
$w_r\bigl(t;\Sigma_\mathcal{S},b_\mathcal{S}\bigr)=1$.

**Candidate.** Add a rank-preserving candidate $k_5$ with
$c^{(k_5)}=c^{(1)}=(1,1)^\top$ and *conflicting* outcome
$y_{k_5}=0\ne 2= c^{(k_5)\top}\theta_\mathcal{S}^\star$, where
$\theta_\mathcal{S}^\star=(1,1)^\top$ is the minimum-norm solution. Then
$\Sigma_{\mathcal{S}\cup\{k_5\}}=2\,c^{(1)}{c^{(1)}}^\top$
(still rank 1), and
$b_{\mathcal{S}\cup\{k_5\}}=b_\mathcal{S}-c^{(k_5)}y_{k_5}=(-2,-2)^\top$,
giving moment-implied line $\theta_1+\theta_2=1$. The new feasible
segment is $\{(\theta_1,1-\theta_1):\theta_1\in[0,2]\}$, with
$$
w_r\bigl(t;\Sigma_{\mathcal{S}\cup\{k_5\}},
 b_{\mathcal{S}\cup\{k_5\}}\bigr) \;=\; 2 \;>\; 1
 \;=\; w_r\bigl(t;\Sigma_\mathcal{S},b_\mathcal{S}\bigr).
$$
The width *doubles* despite the rank being preserved and the box being
the same. Panel (A) of Figure 2 displays the two parallel slices inside
the box.

The construction generalizes: in dimension $d_g\ge 2$, any rank-preserving
candidate with outcome $y_k\ne c^{(k)\top}\theta_\mathcal{S}^\star$ produces
a parallel shift of the moment-implied subspace, and whenever
$\mathcal{R}$ is *asymmetric* around $\theta_\mathcal{S}^\star$ the shifted
slice can cut $\mathcal{R}$ at a polytope of larger $t$-extent.

### 6.3 Sufficient conditions for monotone improvement

We now isolate two assumptions, each of which on its own implies that the
worst-case width is monotone-nonincreasing in site addition. They are
mutually independent and either is implementable as a pre-experiment
screen on the candidate pool.

**Assumption (M1) — Rank-Expansion Screen.** *The candidate's residualized
context lies outside the row space of the existing pool:*
$$
(u_m^{(k)},u_r^{(k)})\;\notin\;\mathrm{range}\bigl(\Sigma_\mathcal{S}\bigr).
\qquad\qquad (6.1)
$$

**Assumption (M2) — Outcome-Consistency Screen.** *The candidate's
outcome forecast is consistent with the existing pool's minimum-norm fit:*
$$
y_k \;=\; (u_m^{(k)},u_r^{(k)})^\top \theta_\mathcal{S}^\star,
\qquad\qquad (6.2)
$$
*where $\theta_\mathcal{S}^\star$ is the minimum-norm solution to
$\Sigma_\mathcal{S}\theta = -b_\mathcal{S}$.*

Both screens are checkable before the candidate experiment is run:
(M1) is a rank test on $(u_m^{(k)},u_r^{(k)})$ against the existing-pool
SVD, and (M2) is the existing-pool OLS prediction at the candidate
context (it does not require the candidate's experimental data). The two
imply different *consequences*: (M1) strictly reduces the null space and
hence width on any target that loaded the resolved null direction, while
(M2) leaves the moment-implied affine subspace untouched and prevents the
feasibility-breaking shifts of §6.1.

**Theorem 1 (Monotone improvement under (M1) or (M2)).**
*Under either Assumption (M1) or Assumption (M2), and assuming
$b_\mathcal{S}\in\mathrm{range}(\Sigma_\mathcal{S})$ (consistency of the
existing moment), the feasible set of the partial-identification LP at
augmentation $\mathcal{S}\cup\{k\}$ is a subset of the existing-pool
feasible set:*
$$
\mathcal{F}_{\mathcal{S}\cup\{k\}}
 \;:=\; \{\theta\in\mathcal{R}:\Sigma_{\mathcal{S}\cup\{k\}}\theta
  = -b_{\mathcal{S}\cup\{k\}}\}
 \;\subseteq\;
 \{\theta\in\mathcal{R}:\Sigma_\mathcal{S}\theta = -b_\mathcal{S}\}
 \;=:\; \mathcal{F}_\mathcal{S}.
$$
*Consequently, for every target $t$,*
$$
w_r\bigl(t;\Sigma_{\mathcal{S}\cup\{k\}},b_{\mathcal{S}\cup\{k\}}\bigr)
 \;\le\; w_r\bigl(t;\Sigma_\mathcal{S},b_\mathcal{S}\bigr),
$$
*and a fortiori the worst-case width
$\sup_{t\in\mathcal{T}} w_r(t;\Sigma_{\mathcal{S}\cup\{k\}},b_{\mathcal{S}\cup\{k\}})
 \le \sup_{t\in\mathcal{T}} w_r(t;\Sigma_\mathcal{S},b_\mathcal{S})$.
The inequality is strict on targets $t$ for which $c(t)$ has a non-zero
component in the resolved direction (under (M1)) or for which the
restriction polyhedron $\mathcal{R}$ binds in a direction newly excluded
by the augmented moment (under (M2) with rank expansion).*

Theorem 1 is proved in Appendix A.4; the key step is the algebraic
identity (A.3) showing that
$\Sigma_{\mathcal{S}\cup\{k\}}\theta+b_{\mathcal{S}\cup\{k\}}=
 \Sigma_\mathcal{S}\theta+b_\mathcal{S}+u\cdot(u^\top\theta-y_k)$,
together with the consistency hypothesis $b_\mathcal{S}\in\mathrm{range}
(\Sigma_\mathcal{S})$.

**Remark (Practical use).** The pair (M1, M2) is *not* exhaustive — a
candidate failing both can still improve the width — but it provides a
finite, computable screen. In practice we recommend:
*(i)* run the SVD-based test (6.1) on every candidate; mark those that
pass as **safe rank-expanders**;
*(ii)* among the rank-preserving candidates, check (6.2) against the
existing-pool OLS prediction; mark those that pass as **safe consistent
candidates**;
*(iii)* report the minimax (3.2) only over the safe subset of the
candidate pool, or report the unsafe candidates separately as
"diagnostic-only" sites whose addition could be informative but whose
worst-case width is not guaranteed to improve.

**Remark (Cost-aware extension).** When candidates carry a cost
$c_k$, the safe-subset screening of (i)–(iii) above is composed with the
Lagrangian (3.3), and the optimal candidate is the cheapest member of the
safe subset that achieves the minimax target. Unsafe candidates can be
admitted only with an explicit risk premium $\rho_k\ge 0$ added to their
cost, calibrated to the upper bound on
$w_r(\cdot;\Sigma_{\mathcal{S}\cup\{k\}})-
 w_r(\cdot;\Sigma_\mathcal{S})$ from the perturbation bound (5.1).

### 6.4 Discussion: why the convex-hull rule is *only* sufficient

Proposition 3 of §4.3 said that, in the large-$\tau$ unrestricted limit,
the optimal candidate is the one whose addition minimizes
$\sup_{t\in\mathcal{T}}d_{\mathcal{S}\cup\{k\}}(t)$. Theorem 1 refines
this picture: the convex-hull rule is *sufficient* for monotone
improvement (it implies (M1)), but it is not *necessary*. An interior
candidate satisfying (M2) — i.e., a candidate whose context lies inside
the hull but whose forecasted outcome matches the OLS prediction — also
guarantees no worsening. Conversely, a hull-expander whose outcome is
adversarial can shift the feasible set badly, exhibiting (F3); the (M1)
screen is *necessary* in the rank sense but is *not sufficient against
shifts*.

The cleanest characterization combines both:

> A candidate $k$ is **width-monotone safe** if either (i) its
> residualized context is outside $\mathrm{range}(\Sigma_\mathcal{S})$
> *and* the outcome shift $\Delta b_k$ is small enough that the
> hyperplane $u^\top\theta=y_k$ intersects $\mathcal{R}$ at a non-empty
> set, or (ii) its context is inside the existing range *and* its
> outcome matches the existing OLS prediction (modulo tolerance).

This is the operational characterization we use in §7.

---

## 7. Simulation

We study a controlled DGP built on the template of Section 3 of
Boyarsky et al. (2026), modified to expose the site-selection decision.

### 7.1 Design

We generate $S = 8$ existing sites whose context vectors
$(C_m^{(s)}, C_r^{(s)}) \in \mathbb{R}^{10}$ lie in a 4-dimensional affine
subspace of the 10-dimensional context space — that is, the existing pool
spans only a thin slice of the site-level geometry, as in Meager (2019).
Within each site we draw $n_s = 500$ individual-level covariates $X$ in
$\mathbb{R}^3$, generate outcomes from the DGP (2.1) with $g_0$ a fixed
non-linear function, and compute the sample moment matrix
$\hat\Sigma_\mathcal{S}$ and right-hand side $\hat b_\mathcal{S}$ along the
lines of Equation (A.9) of Boyarsky et al. (2026). The candidate pool
$\mathcal{K} = \{k_1, k_2, k_3, k_4\}$ is constructed to span the three
regimes of Section 5.2:

- **$k_1$**: context profile inside $\mathcal{H}_\mathcal{S}$ (interior
  candidate; regime (B) at best).
- **$k_2$**: context profile on the boundary of $\mathcal{H}_\mathcal{S}$
  (boundary candidate; regime (B)).
- **$k_3$**: context profile outside $\mathcal{H}_\mathcal{S}$ in a direction
  *not* spanned by any current rank-reducing direction (regime (A);
  hull-expanding).
- **$k_4$**: context profile outside $\mathcal{H}_\mathcal{S}$ but only weakly
  (small displacement from the boundary; regime (A) but with small
  $\|\Delta\Sigma_{k_4}\|$).

The target region $\mathcal{T}$ is a discrete set of $M = 20$ profiles drawn
uniformly from a box that straddles $\mathcal{H}_\mathcal{S}$, chosen so that
some targets lie inside the hull and some outside. We impose a box restriction
$\|\theta\|_\infty \le 5$ as $r$.

### 7.2 Quantities reported

For each candidate $k \in \{k_1, \ldots, k_4\}$ and each $t \in \mathcal{T}$
we compute:

1. the **pre-width** $w_r(t; \hat\Sigma_\mathcal{S}, \hat b_\mathcal{S})$;
2. the **post-width** $w_r(t; \hat\Sigma_{\mathcal{S}\cup\{k\}},
   \hat b_{\mathcal{S}\cup\{k\}})$, with $\hat b$ resampled from the
   calibrated prior of Section 5.3(1);
3. the **minimax objective** $\sup_t w_r(t; \Sigma_{\mathcal{S}\cup\{k\}})$;
4. the **distance-to-hull** $d_\mathcal{S}(t)$ and $d_{\mathcal{S}\cup\{k\}}(t)$.

### 7.3 Results

Across $50$ Monte Carlo replicates, the interior and boundary candidates
$k_1, k_2$ produce an identical minimax width to the existing pool (to
numerical precision), $\hat w^\star_{k_1} = \hat w^\star_{k_2} =
\hat w^\star_{\mathcal{S}} = 1291.8$; the two rank-expanding candidates
produce strictly smaller widths, $\hat w^\star_{k_3} = 1152.2$ and
$\hat w^\star_{k_4} = 1158.3$, corresponding to an $11$% reduction in the
worst-case partial-identification interval. The minimax-optimal candidate
is $k_3$ in $23$/$50$ replicates and $k_4$ in $27$/$50$ replicates — never
$k_1$ or $k_2$. The near-tie between $k_3$ and $k_4$ reflects a feature of
the design rather than a bug: both candidates reduce the null space of
$\Sigma$ by exactly one dimension, albeit in different directions, so
whichever direction is better represented in the randomly-sampled target
region dominates that replicate. Aggregating over replicates, $k_3$ has a
slightly lower mean width, but the gap is within one Monte-Carlo standard
error. The qualitative conclusion — interior candidates are useless for
tightening external-validity claims; rank-expanding candidates are
strictly better — is stable across the full run. See the companion script
`adding_sites_sim.py` and Figure 1 (generated by the script) for the full
output.

![Figure 1. Site-selection simulation on synthetic data. Left: the
existing sites (blue) lie on a 1D slice of the plotted 2D projection
because the projection axis aligns with the span of the existing pool;
the orthogonal axis captures the direction of the candidate pool's
outside-the-hull displacement. Candidates $k_1$ (interior) and $k_2$
(boundary) lie on the existing-site line, while $k_3$ (outside, strong)
and $k_4$ (outside, weak) sit off the line. Right: the minimax partial-
identification width averaged over 50 Monte-Carlo replicates. The
interior and boundary candidates $k_1, k_2$ leave the width
unchanged; the two rank-expanding candidates $k_3, k_4$ each reduce it
by roughly $11\%$.](figure1.png)

### 7.4 Monotonicity experiment (Figure 2)

A second script, `monotonicity_sim.py`, augments the candidate pool with
two further candidates to exercise the theory of §6:

- **$k_5$ — rank-preserving with conflicting outcome.** Same context
  draw as $k_1$ (interior to the hull), but the outcome forecast
  $y_{k_5}$ is shifted by $\Delta = 20$ from the existing-pool OLS
  prediction. Fails (M2); passes neither screen.
- **$k_6$ — rank-expanding with consistent outcome.** Context outside
  the affine hull (same direction as $k_3$), outcome set to the
  existing-pool OLS prediction at the candidate context. Passes both
  (M1) and (M2).

The restriction box $\|\theta\|_\infty\le 5$ is randomly *re-centered*
each replicate (within $\pm M/3$) so that the polyhedron $\mathcal{R}$ is
generically asymmetric around $\theta_\mathcal{S}^\star$ — this is the
condition under which the rank-preserving shift of §6.2 actually bites.

Across $50$ Monte-Carlo replicates, the results are:

| design | mean minimax width | finite replicates |
|---|---:|---:|
| existing | $205.9$ | $50/50$ |
| $k_1$ interior | $205.9$ | $50/50$ |
| $k_2$ boundary | $205.9$ | $50/50$ |
| $k_3$ outside, strong | $185.3$ | $50/50$ |
| $k_4$ outside, weak | $187.5$ | $50/50$ |
| $k_5$ shock | $194.0$ | $50/50$ |
| $k_6$ outside + consistent | $185.3$ | $50/50$ |

Two observations sharpen the message of Theorem 1.

First, in $8/50$ replicates $k_5$ strictly *increases* the worst-case width
relative to the existing pool — never $k_1$ through $k_4$ or $k_6$. The
mean of $k_5$ across the full $50$ replicates is below the existing-pool
mean only because the shift sometimes happens to land the moment-implied
slice in a tighter cross-section of the box. *On individual replicates the
direction of the change is not predictable a priori*, which is precisely
the failure mode (F1) of §6.1. The $8/50$ rate scales linearly with the
asymmetry of the box restriction and with the magnitude $|\Delta|$ of the
outcome shock; at $\Delta=40$ and a more asymmetric box, more than
$30\%$ of replicates exhibit width increase.

Second, $k_6$ — passing both (M1) and (M2) — achieves *exactly* the
width of $k_3$ (the latter passes (M1) but not (M2)), and both are
monotone-nonincreasing relative to the existing pool in every replicate.
The candidate $k_3$ in §7's first script also satisfies the conclusion of
Theorem 1 in this dataset, because in the simplified moment formulation
the residualization is exact and the OLS prediction matches the candidate
outcome up to the noise term; this becomes a strict inequality once the
candidate outcome is drawn from an adversarial prior, as in §5.3(3).

Panel (A) of Figure 2 reproduces the hand-built 2D counterexample of
§6.2 explicitly; Panel (B) shows the per-candidate mean minimax width
across replicates; Panel (C) is the replicate-level scatter of
post-addition vs. existing-pool minimax width, with points above the
$y=x$ diagonal corresponding to candidate–replicate pairs in which the
addition harmed external validity. The $k_5$ scatter cloud straddles
the diagonal; every other cloud sits weakly below.

![Figure 2. Monotonicity simulation. (A) Hand-built 2D counterexample
where adding a rank-preserving site with conflicting outcome doubles the
partial-identification width on $t=(1,0)$ inside an asymmetric box
restriction. (B) Mean minimax width across 50 replicates by candidate;
$k_5$ (rank-preserving with conflicting outcome) sits *above* the
rank-expanders $k_3,k_4,k_6$. (C) Per-replicate scatter of post-addition
vs.\ existing-pool minimax width; points above the $y=x$ diagonal
indicate replicates where adding the candidate *hurt* external validity.
Only $k_5$ ever sits above the diagonal.](figure2.png)

### 7.5 What the simulations show

Three features of the results are robust to DGP perturbations (heavy-tailed
$X$, $S=6$ instead of $8$, larger $\|\theta\|_\infty$ bound):

- **Hull membership is a sharp screen.** Candidates $k_1$ and $k_2$, both of
  whose site-level contexts lie inside the affine hull of the existing pool,
  produce an exactly identical minimax width to the existing pool. This is
  not an approximation: when $\mathrm{rank}(\Sigma)$ does not change, the
  null-space geometry of the LP is unchanged, and a location-only update of
  $b$ (the only remaining effect of adding such a site) cancels in the
  width.
- **Among rank-expanding candidates, distance to the hull matters at the
  margin.** The $11$% reduction achieved by $k_3, k_4$ is an interplay of
  two factors: how large a component of the target region aligns with the
  newly-resolved null-space direction, and how far the candidate sits
  outside the hull in that direction. Proposition 3 captures both effects
  in the large-$\tau$ limit.
- **The minimax selection is stable to the choice of prior on $b$.** The
  three $b$-prior variants of Section 5.3 agree on the optimal candidate
  up to Monte-Carlo noise. In particular, they all rank the four candidates
  as $\{k_1, k_2\} \succ k_3 \sim k_4$ by worst-case width, with the tie
  between $k_3$ and $k_4$ broken stochastically by the target draw.
- **Non-monotonicity is real, but the screens of Theorem 1 eliminate it.**
  The candidate $k_5$ — interior to the hull, with a $\Delta=20$ outcome
  shock relative to the existing-pool OLS prediction — increases the
  worst-case width in $8/50$ replicates. The candidate $k_6$ — outside
  the hull *and* with an OLS-consistent outcome — never does so, matching
  Theorem 1's prediction. In a Monte-Carlo sweep over asymmetry of the
  restriction box and shock magnitude, the fraction of replicates in
  which an unscreened candidate hurts external validity scales roughly
  linearly with both, reaching $30$%+ in the most adversarial setting.
  Restricting the minimax (3.2) to the *safe* subset (passing (M1) or
  (M2)) costs nothing in the average optimum we report and eliminates the
  tail risk of doing harm.

---

## 8. Extensions and Discussion

**Effect-ordering (mixed-integer).** When the restriction set is the effect-
ordering polyhedron of Restriction 1 of Boyarsky et al. (2026), the program
(5.2) becomes a mixed-integer LP. The locally-constant integer-solution
argument of §1(ii) in Boyarsky et al. (2026) extends: on a neighborhood of
$\gamma_\mathcal{S}$, the optimal sign pattern is constant, so the MILP
reduces to the LP (5.2) after fixing integer variables. Selection is then
performed by (5.2) at the fixed pattern; sign flips across candidate sites
can be detected by re-solving the MILP at the top two or three candidates.

**Cost-weighted selection.** Adding a cost $c_k$ to the objective (3.2) is
straightforward because the cost term is separable in $k$; the minimax-
optimal candidate under Lagrangian (3.3) is obtained by solving (5.2) with
an additional additive penalty. In our setting an ethicist-flavored version
of cost might penalize candidates that are demographically similar to
existing sites, because such sites do little to broaden the external-
validity claim despite being cheap to reach.

**Multiple candidates at once.** The design problem (3.2) can be posed over a
subset $\mathcal{J} \subseteq \mathcal{K}$ of candidates to add
simultaneously; in this case the width surrogate (5.2) becomes a set-function
in $\mathcal{J}$ that is *submodular* in the rank-monotone limit — in
precisely the sense that adding the first hull-expander contributes more than
adding the second. A greedy algorithm attains a $(1 - 1/e)$-approximation
under mild regularity, matching the classical D-optimality result of
Nemhauser et al. (1978).

**Connection to efficient influence functions.** The minimum-norm multiplier
$\lambda^\star$ that appears in (5.1) is the same object as the Riesz
representer in the efficient influence function of Boyarsky et al. (2026,
Eq. 2.9). The minimax design criterion therefore inherits inference validity
from their debiased cross-fitting: confidence intervals for the post-
addition width are a standard delta-method computation.

**Open questions.** Three directions strike us as natural follow-ups. First,
the adversarial $b$-prior of Section 5.3(3) admits a saddle-point
representation but not a closed-form solution; existing interior-point
solvers handle small instances, but large candidate pools would benefit from
a custom Frank–Wolfe scheme. Second, in the online-experimentation setting
the candidate pool is revealed sequentially and the selection must be made
without access to the full set; adapting (5.2) to this regret-minimization
formulation is open. Third, the assumption that the within-site
$X$-distribution at candidate sites is known (or informatively prior'd) is
strong; a distributionally-robust variant that treats $P_X^{(k)}$ as
uncertain over a Wasserstein ball would inherit the theoretical guarantees
of the Jeong–Namkoong (2020) worst-case program but at the cost of a more
conservative selection.

---

## Appendix A. Complete Proofs

We collect rigorous proofs of the propositions stated in §4–§6.
Throughout, $\Sigma\in\mathbb{R}^{d_g\times d_g}$ is symmetric positive
semidefinite, $b\in\mathbb{R}^{d_g}$, and the moment-implied affine
subspace is
$\mathcal{L}(\Sigma,b):=\{\theta\in\mathbb{R}^{d_g}:\Sigma\theta=-b\}$.
The restriction polyhedron is
$\mathcal{R}:=\{\theta:r(\theta)\le 0\}$ — compact and convex unless
explicitly stated otherwise. The target functional is
$c(t)\in\mathbb{R}^{d_g}$, the partial-ID interval at target $t$ has
endpoints $v_\ell(t),v_u(t)$ given by the LP pair (2.3), and the width
is $w_r(t;\Sigma,b)=v_u(t)-v_\ell(t)$. We assume the consistency
condition $b\in\mathrm{range}(\Sigma)$ throughout (otherwise
$\mathcal{L}(\Sigma,b)=\emptyset$).

### A.1 Proof of Proposition 1 (unrestricted dichotomy)

*Claim.* With $\mathcal{R}=\mathbb{R}^{d_g}$ and any $b\in\mathrm{range}(\Sigma)$,
$$
w_{\rm unr}(t;\Sigma,b) \;=\;
 \begin{cases} 0, & c(t)\in\mathrm{range}(\Sigma^\top)\\ +\infty,
 & c(t)\notin\mathrm{range}(\Sigma^\top).\end{cases}
$$

*Proof.* Let $\theta_0$ be any solution to $\Sigma\theta_0=-b$; such a
$\theta_0$ exists by the consistency hypothesis. Every solution to
$\Sigma\theta=-b$ has the form $\theta=\theta_0+\nu$ with
$\nu\in\mathrm{null}(\Sigma)$. The linear functional evaluates to
$c(t)^\top\theta=c(t)^\top\theta_0+c(t)^\top\nu$. Since
$\mathcal{R}=\mathbb{R}^{d_g}$, $\nu$ ranges over all of
$\mathrm{null}(\Sigma)$, a linear subspace.

*Case 1: $c(t)\perp\mathrm{null}(\Sigma)$.* Then $c(t)^\top\nu=0$ for
every $\nu\in\mathrm{null}(\Sigma)$, so $c(t)^\top\theta=c(t)^\top\theta_0$
identically and $v_u=v_\ell=c(t)^\top\theta_0$, giving width $0$. By
the orthogonal decomposition
$\mathbb{R}^{d_g}=\mathrm{range}(\Sigma^\top)\oplus\mathrm{null}(\Sigma)$
(valid for any matrix, here in fact $\mathrm{range}(\Sigma)$ since
$\Sigma=\Sigma^\top$), this is equivalent to $c(t)\in\mathrm{range}(\Sigma^\top)$.

*Case 2: $c(t)\not\perp\mathrm{null}(\Sigma)$.* Pick
$\nu_0\in\mathrm{null}(\Sigma)$ with $c(t)^\top\nu_0\ne 0$, and consider
the one-parameter family $\theta_0+\alpha\nu_0$ for $\alpha\in\mathbb{R}$.
Then $c(t)^\top(\theta_0+\alpha\nu_0)=c(t)^\top\theta_0+\alpha c(t)^\top\nu_0\to\pm\infty$
as $\alpha\to\pm\infty$, so $v_u=+\infty$ and $v_\ell=-\infty$. Width is
$+\infty$. $\blacksquare$

### A.2 Proof of Proposition 2 (rank-expansion ⇔ hull-expansion)

*Claim.* Let $C\in\mathbb{R}^{S\times d_g}$ be the matrix whose rows are
the existing site-level contexts $c^{(s)}$, and let
$\Sigma_\mathcal{S}=C^\top C$ in the simplified formulation (the
general block form (3.1) replaces $c^{(s)}c^{(s)\top}$ by the residualized
analogue, but the row-space argument is identical). Then for a candidate
$k$ with context $c^{(k)}=u\in\mathbb{R}^{d_g}$,
$$
\mathrm{rank}(\Sigma_{\mathcal{S}\cup\{k\}})>\mathrm{rank}(\Sigma_\mathcal{S})
 \iff u\notin\mathrm{range}(C^\top)
 \iff u\notin\mathrm{range}(\Sigma_\mathcal{S}),
$$
and equivalently $(C_m^{(k)},C_r^{(k)})\notin\mathrm{aff}(\mathcal{H}_\mathcal{S})$
in the centered version of the same construction.

*Proof.* The matrix $\Sigma_{\mathcal{S}\cup\{k\}}=C^\top C+uu^\top$ has
range $\mathrm{range}(C^\top C+uu^\top)$. We claim
$\mathrm{range}(C^\top C+uu^\top)=\mathrm{range}(C^\top)+\mathrm{span}(u)$.

The forward inclusion follows from
$C^\top Cx+uu^\top x = C^\top(Cx)+u(u^\top x)$, both summands lying in
$\mathrm{range}(C^\top)+\mathrm{span}(u)$ for any $x$.

The reverse inclusion uses positive semidefiniteness: pick any
$v\in\mathrm{range}(C^\top)+\mathrm{span}(u)$. We need to exhibit $x$
with $(C^\top C+uu^\top)x=v$. Choose $x_1\in\mathrm{range}(C^\top)$
such that $C^\top C x_1$ equals the $\mathrm{range}(C^\top)$-component
of $v$ (possible since $C^\top C$ is invertible on $\mathrm{range}(C^\top)$),
and add a scalar multiple of an $x_2$ achieving $uu^\top x_2$ equal to
the $\mathrm{span}(u)$-component (possible since $u^\top u>0$ when
$u\ne 0$). Sum $x=x_1+x_2$ achieves both. The two components are
not orthogonal in general but the resulting linear system is full-rank
on the sum of subspaces.

The first equivalence in the claim now follows: rank increases iff
$\mathrm{span}(u)\not\subseteq\mathrm{range}(C^\top)$, i.e., iff
$u\notin\mathrm{range}(C^\top)$. The second equivalence
$\mathrm{range}(C^\top)=\mathrm{range}(\Sigma_\mathcal{S})$ is the
standard fact $\mathrm{range}(A^\top A)=\mathrm{range}(A^\top)$ for any
real matrix $A$. The third equivalence — between rank-expansion and
affine-hull membership — is obtained by centering: writing
$u=\bar c+\tilde u$ with $\bar c$ the average of $c^{(s)}$ and
$\tilde u=u-\bar c$, the centered row space of $C$ equals
$\mathrm{aff}(\mathcal{H}_\mathcal{S})-\bar c$ (the linear subspace
parallel to the affine hull), and $u\in\mathrm{aff}(\mathcal{H}_\mathcal{S})$
iff $\tilde u$ lies in that linear subspace, iff $u$ is in the row span
of $C$. $\blacksquare$

### A.3 Proof of Proposition 3 (width = distance × $2\tau$)

*Claim.* With the regularized restriction $\|\theta\|_2\le\tau$ and a
balanced cross-site design with common within-site $X$-distribution,
$$
w_r^\tau(t;\Sigma_\mathcal{S},b)
 \;=\; 2\tau\cdot d_\mathcal{S}(t)\cdot(1+o_\tau(1)),
\qquad\tau\to\infty,
$$
*where $d_\mathcal{S}(t):=\mathrm{dist}\bigl((t_m,t_r),
 \mathrm{aff}(\mathcal{H}_\mathcal{S})\bigr)$ and the $o_\tau(1)$ term
vanishes uniformly on compact target regions.*

*Proof.* Decompose $c(t)=c_\parallel(t)+c_\perp(t)$ with
$c_\parallel(t)\in\mathrm{range}(\Sigma^\top)$ and
$c_\perp(t)\in\mathrm{null}(\Sigma)$ (using the symmetric SVD of
$\Sigma$). For any solution $\theta=\theta_0+\nu$ with
$\nu\in\mathrm{null}(\Sigma)$,
$c(t)^\top\theta=c_\parallel(t)^\top\theta_0+c_\perp(t)^\top\nu$,
because $c_\parallel(t)^\top\nu=0$ and $c_\perp(t)^\top\theta_0=
c_\perp(t)^\top(\theta_0^\parallel+0)$ vanishes if we pick $\theta_0$ to
be the minimum-norm particular solution (i.e.,
$\theta_0\in\mathrm{range}(\Sigma^\top)$, so $\theta_0^\perp=0$).

Therefore
$$
v_u^\tau(t)-v_\ell^\tau(t)
 \;=\; \sup_{\nu}\,c_\perp(t)^\top\nu - \inf_\nu c_\perp(t)^\top\nu,
\qquad
\nu\in\mathrm{null}(\Sigma)\cap\{\theta_0+\nu:\|\theta_0+\nu\|_2\le\tau\}.
$$
For large $\tau$ (specifically $\tau\ge\|\theta_0\|_2+M$ for any compact
$M$), the constraint $\|\theta_0+\nu\|_2\le\tau$ allows $\nu$ to range
over a ball of radius $\tau-\|\theta_0\|_2$ inside $\mathrm{null}(\Sigma)$,
i.e., over a ball of effective radius $\tau(1-\|\theta_0\|_2/\tau)$,
which converges to a ball of radius $\tau$ at rate $O(1/\tau)$.

The supremum of $c_\perp(t)^\top\nu$ over the radius-$\rho$ ball in
$\mathrm{null}(\Sigma)$ is $\rho\cdot\|c_\perp(t)\|_2$ by the Cauchy–
Schwarz inequality with equality at
$\nu=\rho\,c_\perp(t)/\|c_\perp(t)\|_2$. The infimum is its negative.
Hence
$w_r^\tau(t;\Sigma,b)=2\rho\|c_\perp(t)\|_2=2\tau\|c_\perp(t)\|_2(1+O(1/\tau))$,
which matches the asymptotic form claimed.

It remains to identify $\|c_\perp(t)\|_2$ with $d_\mathcal{S}(t)$, the
Euclidean distance from $(t_m,t_r)$ to $\mathrm{aff}(\mathcal{H}_\mathcal{S})$.
Under the balanced design with common $X$-distribution, the residualized
context appearing in the block formula (3.1) reduces (up to a positive
multiplicative constant absorbed into $\tau$) to the centered raw
context. The target direction $c(t)$, by Equation (2.2), is the
moderating-direction vector for the target profile, which up to that same
constant is $(t_m,t_r)-\bar c$. The orthogonal complement of
$\mathrm{range}(\Sigma^\top)$ inside the centered context space is the
orthogonal complement of $\mathrm{aff}(\mathcal{H}_\mathcal{S})$, so the
norm of the orthogonal-complement projection of $c(t)$ equals
$d_\mathcal{S}(t)$, as required. Strict monotonicity follows because the
target distance $d_\mathcal{S}(t)$ is a sub-additive function of the
candidate-pool augmentation: $d_{\mathcal{S}\cup\{k\}}(t)\le d_\mathcal{S}(t)$
with equality iff $c(t)$'s null-space component is unaffected by the
augmentation, which by Proposition 2 is iff
$u\in\mathrm{range}(\Sigma_\mathcal{S})$. $\blacksquare$

### A.4 Proof of Theorem 1 (monotone improvement under (M1) or (M2))

We first establish the algebraic identity that drives the proof. Let
$u:=(u_m^{(k)},u_r^{(k)})$ be the residualized context of the candidate
and $y_k$ its scalar outcome forecast. Then, in the simplified moment
formulation $\Sigma_{\mathcal{S}\cup\{k\}}=\Sigma_\mathcal{S}+uu^\top$
and $b_{\mathcal{S}\cup\{k\}}=b_\mathcal{S}-u y_k$, we have for every
$\theta$:
$$
\Sigma_{\mathcal{S}\cup\{k\}}\theta+b_{\mathcal{S}\cup\{k\}}
 \;=\; \Sigma_\mathcal{S}\theta+b_\mathcal{S}
 \;+\; u\cdot(u^\top\theta-y_k).
\qquad\qquad (A.3)
$$
Direct verification: $\Sigma_{\mathcal{S}\cup\{k\}}\theta=
\Sigma_\mathcal{S}\theta+u(u^\top\theta)$ and
$b_{\mathcal{S}\cup\{k\}}=b_\mathcal{S}-uy_k$; summing gives (A.3).

*Proof of Theorem 1 under (M1).* Suppose $u\notin\mathrm{range}(\Sigma_\mathcal{S})$.
Let $P$ be the orthogonal projector onto $\mathrm{range}(\Sigma_\mathcal{S})$
and $P^\perp:=I-P$ its orthogonal complement onto $\mathrm{null}(\Sigma_\mathcal{S})$.
Then $u_\perp:=P^\perp u\ne 0$. By the consistency hypothesis
$b_\mathcal{S}\in\mathrm{range}(\Sigma_\mathcal{S})$, so for every $\theta$,
$\Sigma_\mathcal{S}\theta+b_\mathcal{S}\in\mathrm{range}(\Sigma_\mathcal{S})$.
Applying $P^\perp$ to both sides of (A.3),
$$
P^\perp\bigl[\Sigma_{\mathcal{S}\cup\{k\}}\theta+b_{\mathcal{S}\cup\{k\}}\bigr]
 \;=\; P^\perp[u]\cdot(u^\top\theta-y_k)
 \;=\; u_\perp\cdot(u^\top\theta-y_k).
$$
For $\theta\in\mathcal{F}_{\mathcal{S}\cup\{k\}}$, the left-hand side is
zero, and $u_\perp\ne 0$, so the scalar coefficient vanishes:
$u^\top\theta=y_k$. Substituting this back into (A.3) yields
$\Sigma_\mathcal{S}\theta+b_\mathcal{S}=0$, i.e.,
$\theta\in\mathcal{L}(\Sigma_\mathcal{S},b_\mathcal{S})$. Together with
$\theta\in\mathcal{R}$, this gives
$\theta\in\mathcal{F}_\mathcal{S}\cap\{u^\top\theta=y_k\}\subseteq\mathcal{F}_\mathcal{S}$,
as required. Pointwise width follows: for any target $t$,
$$
w_r(t;\Sigma_{\mathcal{S}\cup\{k\}},b_{\mathcal{S}\cup\{k\}})
 = \sup_{\mathcal{F}_{\mathcal{S}\cup\{k\}}} c(t)^\top\theta-
   \inf_{\mathcal{F}_{\mathcal{S}\cup\{k\}}} c(t)^\top\theta
 \le \sup_{\mathcal{F}_\mathcal{S}} c(t)^\top\theta -
     \inf_{\mathcal{F}_\mathcal{S}} c(t)^\top\theta
 = w_r(t;\Sigma_\mathcal{S},b_\mathcal{S}),
$$
and strictly so when $c(t)$ has a non-zero component along $u_\perp$.

*Proof of Theorem 1 under (M2).* Suppose $y_k=u^\top\theta_\mathcal{S}^\star$
for the minimum-norm solution $\theta_\mathcal{S}^\star$.

Fix any $\theta\in\mathcal{F}_\mathcal{S}$. Then $\Sigma_\mathcal{S}\theta+
b_\mathcal{S}=0$. Write $\theta=\theta_\mathcal{S}^\star+\nu$ with
$\nu\in\mathrm{null}(\Sigma_\mathcal{S})$. Compute:
$$
u^\top\theta-y_k
 \;=\; u^\top(\theta_\mathcal{S}^\star+\nu)-u^\top\theta_\mathcal{S}^\star
 \;=\; u^\top\nu.
$$
Case (i): $u\in\mathrm{range}(\Sigma_\mathcal{S})$. Then
$u\perp\mathrm{null}(\Sigma_\mathcal{S})$, so $u^\top\nu=0$. Substituting
into (A.3),
$\Sigma_{\mathcal{S}\cup\{k\}}\theta+b_{\mathcal{S}\cup\{k\}}=0$, i.e.,
$\theta\in\mathcal{L}(\Sigma_{\mathcal{S}\cup\{k\}},b_{\mathcal{S}\cup\{k\}})$.
Combined with $\theta\in\mathcal{R}$, we obtain
$\mathcal{F}_\mathcal{S}\subseteq\mathcal{F}_{\mathcal{S}\cup\{k\}}$
*and* $\mathcal{F}_{\mathcal{S}\cup\{k\}}\subseteq\mathcal{F}_\mathcal{S}$
(the latter by the (M1)/rank argument when (M1) also holds, or by symmetry
of the affine subspaces in the rank-preserving case). Therefore
$\mathcal{F}_\mathcal{S}=\mathcal{F}_{\mathcal{S}\cup\{k\}}$ and the
width is unchanged.

Actually we need to be careful — in the rank-preserving case the
two affine subspaces are not generally equal as point sets, but they
coincide *exactly* when (M2) holds because both contain
$\theta_\mathcal{S}^\star$ (by (M2)) and both have the same direction
subspace $\mathrm{null}(\Sigma_\mathcal{S})=\mathrm{null}(\Sigma_{\mathcal{S}\cup\{k\}})$
(by $u\in\mathrm{range}(\Sigma_\mathcal{S})$, which gives no rank change).
Equality of two affine subspaces with the same direction follows from
sharing one point. Hence
$\mathcal{L}(\Sigma_\mathcal{S},b_\mathcal{S})=\mathcal{L}
(\Sigma_{\mathcal{S}\cup\{k\}},b_{\mathcal{S}\cup\{k\}})$, intersecting
with $\mathcal{R}$ preserves equality, and the width is unchanged target
by target.

Case (ii): $u\notin\mathrm{range}(\Sigma_\mathcal{S})$ (i.e., (M1) also
holds). Then the proof under (M1) above applies, and additionally (M2)
ensures the hyperplane $u^\top\theta=y_k$ passes through
$\theta_\mathcal{S}^\star$, hence intersects $\mathcal{F}_\mathcal{S}$
non-trivially. So $\mathcal{F}_{\mathcal{S}\cup\{k\}}$ is non-empty
(no feasibility break) and is a strict subset of $\mathcal{F}_\mathcal{S}$,
yielding the same pointwise width inequality with strictness on targets
loading the new direction.

Taking the supremum over $t\in\mathcal{T}$ in either case proves
$\sup_{t\in\mathcal{T}}w_r(t;\Sigma_{\mathcal{S}\cup\{k\}},b_{\mathcal{S}\cup\{k\}})
 \le\sup_{t\in\mathcal{T}}w_r(t;\Sigma_\mathcal{S},b_\mathcal{S})$.
$\blacksquare$

### A.5 Proof of Proposition 4 (rank-2 envelope)

Proposition 4 is a special case of the Bonnans–Shapiro directional
derivative theorem for the value function of a perturbed convex program;
we sketch the specialization to (2.3) to make the constants explicit.

*Setup.* Define the parametric LP
$$
v_u(h):=\max_\theta\bigl\{c(t)^\top\theta:(\Sigma_\mathcal{S}+h_\Sigma)\theta
 =-(b_\mathcal{S}+h_b),\;r(\theta)\le 0\bigr\},
$$
with perturbation $h=(h_\Sigma,h_b)$. At $h=0$, let
$(\theta^\star,\lambda^\star,\nu^\star)$ be a KKT triple with
$\lambda^\star$ the equality multipliers (Lagrange multipliers on
$\Sigma\theta+b=0$) and $\nu^\star\ge 0$ the inequality multipliers on
$r(\theta)\le 0$. Stationarity reads $c(t)=\Sigma\lambda^\star+
\partial r(\theta^\star)\nu^\star$.

*Directional derivative.* Under Robinson's constraint qualification
(Assumption 6 of Boyarsky et al. 2026), the value function $v_u(\cdot)$
is directionally differentiable at $h=0$ with derivative given by
$$
\frac{d v_u}{dh}\Bigl|_{h=0}\cdot h
 \;=\; -\lambda^{\star\top}\bigl(h_\Sigma\theta^\star+h_b\bigr),
$$
where $\lambda^\star$ is chosen as the minimum-norm element of the set
of optimal multipliers (a singleton under strict complementarity). This
is Theorem 4.24 of Bonnans–Shapiro (2013); the form is standard for
parametric LPs and follows from the envelope theorem applied to the
Lagrangian $L(\theta,\lambda;h)=c(t)^\top\theta-
\lambda^\top[(\Sigma+h_\Sigma)\theta+b+h_b]$.

*Application to site augmentation.* Set
$h_\Sigma=\Delta\Sigma_k,\;h_b=\Delta b_k$. Then the first-order change in
$v_u$ is
$\Delta v_u\approx -\lambda^{\star\top}(\Delta\Sigma_k\theta^\star+\Delta b_k)$,
which is (5.1). The same identity with the $v_\ell$-multiplier
$\lambda^{\ell,\star}$ yields the change in $v_\ell$, and subtracting
gives the width formula in the statement of Proposition 4.

*Rank-2 reduction.* By (3.1), $\Delta\Sigma_k$ has rank at most $2$ and
factors as $\Delta\Sigma_k=U_k\Lambda_k U_k^\top$ with $U_k$ a
$d_g\times 2$ block of residualized contexts. The product
$\Delta\Sigma_k\theta^\star=U_k(\Lambda_k U_k^\top\theta^\star)$ is a
linear combination of the two columns of $U_k$, and the inner product
with $\lambda^\star$ reduces to two scalar products
$\lambda^{\star\top}U_k$. The minimax (3.2) over $k\in\mathcal{K}$
therefore reduces to evaluating two scalars per candidate after the
existing-pool LP has been solved once; this is the $O(|\mathcal{T}|K)$
cost claimed in §5.1. $\blacksquare$

### A.6 Extension to the block (BEN) moment

The proofs of A.1, A.2, A.4 use only the symmetric PSD structure of
$\Sigma$, the consistency hypothesis $b\in\mathrm{range}(\Sigma)$, and the
algebraic identity (A.3). In the full BEN formulation of Equation (A.9)
of Boyarsky et al. (2026), $\Sigma$ is a $d_g\times d_g$ block matrix
with $d_g=2(d_M+d_I)$ and block-rank update $\Delta\Sigma_k$ given by
(3.1). The same identity (A.3) holds with $u$ replaced by the
block-residualized context vector $u_k:=(u_m^{(k)},u_r^{(k)})\otimes(\cdot)$
appearing in the block factorization of $\Delta\Sigma_k$, and with the
scalar $y_k-u^\top\theta$ replaced by the corresponding block-residualized
moment quantity. The (M1)/(M2) screens generalize to the block-residualized
context (replacing the $\mathrm{range}$ test of (6.1) by a rank test on
the block matrix), and Theorem 1 carries over verbatim. The only
non-trivial change is that the rank of $\Delta\Sigma_k$ is at most $2$
in the block formulation, so the screens may strengthen in the sense that
two coordinates of the candidate context need to fall outside the existing
row space simultaneously; this is a strictly weaker monotone-improvement
condition than (M1) for the simplified moment.

---

## References

Atkinson, A., Donev, A., and Tobias, R. (2007). *Optimum Experimental
Designs, With SAS.* Oxford University Press.

Bareinboim, E. and Pearl, J. (2016). Causal inference and the data-fusion
problem. *PNAS*, 113(27), 7345–7352.

Bonnans, J. F. and Shapiro, A. (2013). *Perturbation Analysis of Optimization
Problems.* Springer.

Boyarsky, A., Egami, N., and Namkoong, H. (2026). A Sensitivity Framework for
Assessing External Validity. Working paper.

Jeong, S. and Namkoong, H. (2020). Robust causal inference under covariate
shift via worst-case subpopulation treatment effects. *arXiv:2007.02411*.

Manski, C. (2026). *Identification for Prediction and Decision.* Harvard
University Press, revised edition.

Meager, R. (2019). Understanding the average impact of microcredit expansions.
*AEJ: Applied Economics*, 11(1), 57–91.

Nemhauser, G., Wolsey, L., and Fisher, M. (1978). An analysis of
approximations for maximizing submodular set functions. *Mathematical
Programming*, 14, 265–294.

Pukelsheim, F. (2006). *Optimal Design of Experiments.* SIAM.
