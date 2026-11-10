# Algorithm

The mathematics approx-chol implements: how a symmetric GDD matrix $A$ becomes the approximate
inverse $B$, and which matrix $B$ targets. Terms are those of `glossary.md`. Planned, not on
`main`: signed elimination and the Schur tree (`design/cholesky.md`, F1, F2).

## Overview

Approximate Cholesky factors Laplacians. Exact elimination of a vertex adds a clique on its
neighbors, and these cliques make the factors dense. Approximate Cholesky replaces each clique by a
sparse random graph whose expectation is the clique. Each elimination then adds few edges, and the
factorization equals the matrix in expectation.

The input class is wider than Laplacians: symmetric GDD matrices, with entries of either sign, not
necessarily dominant as given, and possibly singular. The algorithm therefore has three stages:

1. **Reduce** each component of $A$ to a WDD matrix: scale it to be diagonally dominant and, if
   it is balanced, switch its off-diagonal entries to be nonpositive.
2. **Factor** it by approximate Cholesky: a balanced one as a Laplacian with its surplus on a
   ground vertex; a frustrated one, as configured, on its Gremban cover or by signed elimination.
3. **Solve** by mapping $b$ into the factored matrix's coordinates, applying the factorization,
   and mapping the result back.

Stage 1 changes $A$ only by the clamp $E$ and the dropped surplus $E'$, both at rounding level for
a GDD input whose scaling settles; it is exact for the target matrix $A_r = A + E + E'$. All randomness sits in stage 2:
with exact elimination, $B$ would be $A_r^+$. That is what "$B$ targets $A_r^+$" means.

## Pseudocode

```
build(A, k, seed):
    for each component C of A:
        d ← dominance_scaling(A_C)                        -- step 2
        E ← clamp of A_C under d                          -- step 3
        Ā ← D(A_C + E)D                                   -- WDD
        if C is balanced:                                 -- step 4
            S ← switching that makes SĀS a Z-matrix
            Ã ← SĀS                                       -- SDDM
            if C is numerically floating:
                set the surpluses of Ã to 0               -- E′; Ã is a Laplacian
                z ← DS·1
        else:
            S ← I
            Ã ← Ā                                         -- WDD, size |C|
        M ← Ã, or for a frustrated C on the cover route its Gremban cover   -- step 5
        if bipartite: eliminate F exactly; M ← its Schur complement         -- step 8
        L ← M with its surplus on a ground vertex g, or on a cover as
            surplus edges, as configured                                     -- step 5, 6
        (Π, 𝓛, 𝓓) ← approximate_cholesky(L, rule, k, seed)                  -- step 7

solve(b), per component:
    b ← Qb                                                -- project onto z^⟂ if singular
    b̃ ← SDb                                               -- scaled and switched
    lift b̃ into L's coordinates
    y ← Πᵀ 𝓛⁻ᵀ 𝓓⁺ 𝓛⁻¹ Π (lifted b̃)
    descend y to Ã's coordinates, giving ỹ
    x ← Q DS ỹ                                            -- back to A, project again
```

## Build

### 1. Components

$A$ is block diagonal over the components of its graph, so each later step works on one component,
and $B$ is block diagonal too. A component without an edge is a single vertex $i$, solved by
$x_i = b_i / a_{ii}$, or $x_i = 0$ when $a_{ii} = 0$. An inadmissible row is rejected, so
$\Delta > 0$ on every other component; the steps below concern those.

### 2. Dominance scaling

GDD means some positive diagonal $D$ makes $DAD$ WDD; finding it is an eigenvalue problem. Row $i$
of $DAD$ is dominant iff its row ratio $\rho_i \le 1$. Writing $d = \Delta^{-1/2} u$ turns the row
ratios into $\rho_i = (Tu)_i / u_i$ with

$$
T = \Delta^{-1/2}\, |A - \Delta|\, \Delta^{-1/2},
$$

the absolute value taken entrywise: a nonnegative matrix, irreducible on a component. By
Perron–Frobenius it has a Perron root $\lambda$ and a Perron vector $u > 0$ with $Tu = \lambda u$.
For every $u > 0$ the Collatz–Wielandt bound

$$
\min_i \rho_i \;\le\; \lambda \;\le\; \max_i \rho_i
$$

holds, with equality for the Perron vector. Hence $A$ is GDD iff $\lambda \le 1$, and then
$d = \Delta^{-1/2} u$ for the Perron vector is a dominance scaling.

The eigenvector need not be exact: the row ratios of any positive iterate certify dominance
($\max_i \rho_i \le 1$) or refute it ($\min_i \rho_i > 1$), up to the slack
$\tau = \max(t, \beta)$, $t$ the caller's tolerance, never below the rounding band $\beta$. $u$ is
computed by an inexact Noda iteration; its budget counts the inner solves' steps.

```
dominance_scaling(A_C):
    T ← Δ^{-1/2} |A_C − Δ| Δ^{-1/2}
    u ← 1
    loop, until a step budget runs out:
        ρ ← (Tu)ᵢ / uᵢ for every i                 -- row ratios of d = Δ^{-1/2} u
        if maxᵢ ρᵢ ≤ 1 + τ and settled: return Δ^{-1/2} u   -- dominance certified
        if minᵢ ρᵢ > 1 + τ: return Δ^{-1/2} u       -- refuted; the clamp covers the rest
        α ← maxᵢ ρᵢ                                -- Collatz–Wielandt: α ≥ λ
        u ← approximate solution of (αI − T) u′ = u, taken when positive and better:
            certified, a smaller eigen-residual; else a smaller maxᵢ ρᵢ   -- Noda step
    return Δ^{-1/2} u                             -- unsettled; the clamp covers the rest
```

For $\alpha > \lambda$, $\alpha I - T$ is a nonsingular M-matrix, so its inverse is nonnegative and
maps a positive $u$ to a positive one; with exact solves, $\alpha$ decreases toward $\lambda$. Jia,
Lin and Liu keep both properties for inexact solves whose residual $f$ satisfies $|f| \le \gamma u$
entrywise with $\gamma < 1$, and prove convergence to $\lambda$ under a slightly stronger bound.
approx-chol's solves bound no residual; an iterate is taken only once it is positive and ranks
better: before certification by its largest row ratio, after it by its eigen-residual, and it is
settled once $\max_i \rho_i < 1 - \beta$ or its eigen-residual is within $\beta$. Every returned $u$ is positive, so its row ratios bound $\lambda$ soundly, but
convergence is not guaranteed: if the budget runs out unsettled, the clamp can exceed rounding
level even on a GDD input.

**Sources**

- GDD and factor width: Boman, Chen, Parekh, Toledo. "On factor width and symmetric H-matrices."
  *Linear Algebra Appl.* 405, 2005, pp. 239–248.
- Perron–Frobenius (Thm. 2.1.4), Collatz–Wielandt bound (Thm. 2.1.11), nonsingular M-matrices and
  their nonnegative inverse (Ch. 6, §1; Thm. 6.2.3, N38): Berman, Plemmons. *Nonnegative Matrices
  in the Mathematical Sciences.* SIAM, 1994 (Classics in Applied Mathematics 9).
- Dominance scaling of a positive definite symmetric M-matrix given as $VV^\top$ with two nonzeros
  per column of $V$, by a randomized method: Daitch, Spielman. "Faster approximate lossy
  generalized flow via interior point algorithms." *STOC*, 2008, pp. 451–460. arXiv:0803.0988, §4.
- Inexact Noda iteration, positivity and monotonicity (§3.1, Lemma 3), convergence (§4, Thm. 1):
  Jia, Lin, Liu. "A positivity preserving inexact Noda iteration for computing the
  smallest eigenpair of a large irreducible M-matrix." *Numer. Math.* 130, 2015, pp. 645–679.
  arXiv:1309.3926.

### 3. Clamp

In floating point a computed $d$ can leave some $\rho_i$ slightly above $1$, most often on a
component that is singular or nearly so, where $\lambda$ is $1$ or close to it. The clamp $E$ is the
least nonnegative diagonal that makes $D(A + E)D$ WDD under this $d$. It is zero when no row is
violated. On an input that is not GDD, where the scaling is refuted, $E$ can be large and $A_r$ far
from $A$. Every later step works on $\bar A = D(A + E)D$, which is WDD.

### 4. Switching and classification

A symmetric WDD matrix is SDDM iff its off-diagonal entries are nonpositive.

- **Balanced:** if every cycle has an even number of positive entries, a signature $S$ switches
  $\bar A$ to a Z-matrix: $\tilde A = S\bar A S$. Switching is a similarity by $\pm 1$, so it keeps
  the spectrum; a solution maps back by $S$.
- **Frustrated:** otherwise no switching works; $\tilde A = \bar A$, with $S = I$, keeps its
  positive entries. Step 5 factors it.

The classification below is numerical, on the computed surpluses.

- **Balanced, numerically floating:** the surpluses of $\tilde A$ are at rounding level, and the
  component is treated as floating: they are set to $0$, the dropped surplus $E'$. A surplus inside
  the band cannot be told apart from rounding, and keeping it would give $B$ an eigenvalue of order
  $1/\text{surplus}$ that rounding alone decided. $\tilde A$ is then a Laplacian,
  $\tilde A \mathbf{1} = 0$, and $z = DS\mathbf{1}$ spans the kernel of $A_r$.
- **Frustrated:** $A + E$ is nonsingular whatever its surplus, its smallest eigenvalue set by the
  frustration; a rounding-level surplus is kept, as no $1/\text{surplus}$ eigenvalue reaches $B$.
- **Otherwise grounded:** $\tilde A$ has a positive surplus.

**Sources**

- Balance: Harary. "On the notion of balance of a signed graph." *Michigan Math. J.* 2(2), 1953,
  pp. 143–146. His signed graph has edge signs $-\operatorname{sgn} a_{ij}$, so his even number of
  negative edges is our even number of positive entries.
- Switching: Zaslavsky. "Signed graphs." *Discrete Appl. Math.* 4, 1982, pp. 47–74, §3,
  Prop. 3.2; erratum 5, 1983, p. 248.

### 5. Factoring a frustrated component

Approximate Cholesky samples cliques of nonnegative weights. A frustrated $\tilde A$ has entries of
both signs, so it is factored on its Gremban cover, a Z-matrix of twice the size, or by signed
elimination, on $\tilde A$ itself. The configuration chooses; both target the same $A_r$.

**Gremban cover.** Split $\bar A = \bar\Delta + \bar N + \bar P$ into its diagonal, its negative
off-diagonal entries and its positive ones, and set

$$
\hat A = \begin{pmatrix} \bar\Delta + \bar N & -\bar P \\ -\bar P & \bar\Delta + \bar N \end{pmatrix}.
$$

As a graph, every vertex $i$ gets two copies $i^+$ and $i^-$. A negative entry $\bar a_{ij}$ becomes
the edges $\{i^+, j^+\}$ and $\{i^-, j^-\}$, a positive one the edges $\{i^+, j^-\}$ and
$\{i^-, j^+\}$, each of weight $|\bar a_{ij}|$. Every off-diagonal entry of $\hat A$ is then
nonpositive, and each row keeps the absolute off-diagonal sum and the diagonal of $\bar A$, so
$\hat A$ has $\bar A$'s surpluses: it is SDDM because $\bar A$ is WDD.

A solution is read off the cover. If $\hat A \begin{pmatrix} p \\ q \end{pmatrix} =
\begin{pmatrix} b \\ -b \end{pmatrix}$, the two block rows are

$$
(\bar\Delta + \bar N)p - \bar P q = b, \qquad -\bar P p + (\bar\Delta + \bar N)q = -b;
$$

subtracting them gives $\bar A(p - q) = 2b$, so $x = (p - q)/2$ solves $\bar A x = b$. A positive
entry ties $i^+$ to $j^-$, so following a cycle with an odd number of positive entries leads from
$i^+$ to $i^-$: the cover of a frustrated component is connected. On a balanced component it would
fall apart into two copies of $S\bar A S$, which is why switching, at half the size, is used there.

**Where a cover puts the surplus.** $\hat A$ commutes with exchanging $i^+$ and $i^-$, so it splits
into two invariant parts: on $\begin{pmatrix} x \\ -x \end{pmatrix}$ it acts as $\bar A$, on
$\begin{pmatrix} z \\ z \end{pmatrix}$ as $\bar\Delta + \bar N - \bar P$, the unsigned matrix with
$\bar A$'s surpluses. A right-hand side $\begin{pmatrix} b \\ -b \end{pmatrix}$ lies in the first.

- **Ground vertex** (step 6): $\hat A$'s surpluses on edges to $g$, a Laplacian of size $2n + 1$.
  The unsigned part has smallest eigenvalue at most the mean surplus, an eigenvalue of the cover
  that $\bar A$ does not have; when the surplus is near rounding, its errors can swamp the solve.
- **Surplus edges:** each $s_i$ on an edge $\{i^+, i^-\}$ of weight $s_i/2$, a Laplacian of size
  $2n$. On $\begin{pmatrix} x \\ -x \end{pmatrix}$ the edge acts as the surplus $s_i$; on
  $\begin{pmatrix} z \\ z \end{pmatrix}$ it vanishes, so the unsigned part is a Laplacian whose
  only null vector, $\mathbf{1}$, no lifted right-hand side reaches. The edges join two real
  vertices, so they add fill where a ground vertex, pinned last, adds little.
- **Auto:** surplus edges when the total surplus is within the rounding band of the trace, a ground
  vertex otherwise. A ground vertex within the band can leave PCG without progress where the
  surplus sits on a light region (#352).

**Signed elimination.** $\tilde A$ is factored as it is, of size $n$, with signed entries
(`glossary.md`, Elimination): each elimination keeps it WDD, the cancellation of entries of
opposite sign turning into surplus, so it has a ground vertex $g$ even without an input surplus,
which is pinned. $\tilde A$ is nonsingular, so every pivot but $g$'s is positive.

**Sources**

- Gremban cover, his expanded matrix, and $\hat A(u, -u) = (\bar A u, -\bar A u)$: Gremban.
  *Combinatorial preconditioners for sparse, symmetric, diagonally dominant linear systems.* PhD
  thesis, CMU-CS-96-123, 1996, §7.1, Def. 7.2 and Lemma 7.3.
- The cover's graph, connected iff the graph is unbalanced: Zaslavsky (1982), as in step 4, §6.
- Surplus edges, a one-step reduction of size $2n$ preserving approximate solutions: Kelner,
  Orecchia, Sidford, Zhu. "A simple, combinatorial algorithm for solving SDD systems in
  nearly-linear time." *STOC*, 2013. arXiv:1301.6628, App. A.
- The two invariant parts: Diaz-Diaz, Devriendt, Lambiotte. "Gremban expansion for signed
  networks." arXiv:2509.14193, 2025.
- Elimination directly on signed SDD matrices: Boman, Chen, Hendrickson, Toledo.
  "Maximum-weight-basis preconditioners." *Numer. Linear Algebra Appl.* 11, 2004, pp. 695–721;
  for connection Laplacians, of which a signed graph is the case $\pm 1$: Kyng, Lee, Peng,
  Sachdeva, Spielman. "Sparsified Cholesky and multigrid solvers for connection Laplacians."
  *STOC*, 2016. arXiv:1512.01892.

### 6. Augmentation

An SDDM matrix is a Laplacian with one vertex deleted: the ground vertex $g$, with an edge
$\{i, g\}$ of weight $s_i$ for every positive surplus. A grounded $\tilde A$, balanced or a cover
with a ground vertex, gets $g$ back, so the matrix factored is a connected Laplacian $L$; signed
elimination keeps $g$ beside a WDD matrix instead, pinned.

**Source:** Gremban (1996), as in step 5, §4.2, Def. 4.1 and Lemma 4.2.

### 7. Approximate Cholesky

Exact elimination of a pivot $v$ with star weights $w_{vi}$ and $\omega_v = \sum_i w_{vi}$ removes
its star and adds the clique on its neighbors, with weights

$$
\frac{w_{vi}\, w_{vj}}{\omega_v}.
$$

Approximate Cholesky adds a clique sample instead.

```
approximate_cholesky(L, rule, k, seed):
    while more than one vertex remains:
        v ← a vertex with the fewest edge copies, g last under signed elimination
        𝓓_vv ← a_vv;  column v of 𝓛 ← eᵥ + (a_{vi} / a_vv)ᵢ
        remove the star of v
        add a clique sample on the neighbors of v, by rule with fill-in level k
    the last vertex gets 𝓓 = 0
```

- **Order:** the next pivot is a vertex with the fewest incident edge copies in the current graph,
  so cliques stay small. The order is dynamic, recomputed after each elimination; $g$ is never a
  pivot, it is pinned as the last slot.
- **Rule:** a clique tree or a Schur tree (`glossary.md`). Both keep the clique's expectation;
  the clique tree is the default, and the Schur tree is opt-in.
- **Clique tree** (Gao, Kyng, Spielman, §4.3): every edge is split into $k$ parallel copies,
  $k$ the fill-in level. When $v$ is eliminated, the copies of each edge $\{v, i\}$ are regrouped
  into $c_i = \min(\text{count}, k)$ copies of weight $w_{vi}/c_i$, $w_{vi}$ their total weight (GKS's
  merge cap $l$, set to $k$). The neighbors are put in ascending order of the per-copy weight $w_{vi}/c_i$; GKS order
  by $w_{vi}$, and any order keeps the expectation. Each of the $c_i$ copies, for every neighbor
  $i$ but the last, joins $i$ to one later neighbor $j$, chosen with probability
  $w_{vj} / \sum_{l>i} w_{vl}$, and carries weight

  $$
  \frac{w_{vi} \sum_{l>i} w_{vl}}{c_i\, \omega_v}.
  $$

  The expected weight of $\{i, j\}$ is then the clique's $w_{vi} w_{vj} / \omega_v$, and the sample
  is connected, since every neighbor but the last joins a later one. With $k = 1$ it is a random
  spanning tree of the clique; larger $k$ adds edges and lowers the variance.
- **Schur tree:** one spanning tree of the clique, drawn by a weighted Prüfer code, each neighbor
  chosen with probability $w_{vi}/\omega_v$; edge $\{i, j\}$ is present with probability
  $(w_{vi} + w_{vj})/\omega_v$ and carries $w_{vi} w_{vj}/(w_{vi} + w_{vj})$, so its expected weight
  is the clique's. With a split $k$, each neighbor becomes $k$ copies of weight $w_{vi}/k$ for this
  draw, and the tree is contracted back; no copy persists.
- **Signed entries:** under signed elimination both rules draw on the magnitudes $w_{vi} = |a_{vi}|$,
  and each sampled edge takes the sign of the exact entry it stands for. A star is a tree, hence
  balanced, so a switching local to the star makes it a Z-matrix's star; the sample keeps the
  clique's expectation, positive semidefiniteness and null vector. Merging with an entry of the
  other sign cancels into surplus at both ends.
- **Expectation:** given the eliminations so far and the chosen pivot, the sample's expected
  Laplacian is the clique's. The errors therefore have conditional mean zero, a martingale, and in
  original coordinates
  $\mathbb{E}[\Pi^\top \mathcal{L}\mathcal{D}\mathcal{L}^\top \Pi] = L$, also for the dynamic order,
  since each pivot is chosen from the past (Kyng, Sachdeva prove it for a random order).
- **Zero pivot:** $L$ is a connected Laplacian, hence singular with kernel $\operatorname{span}
  \mathbf{1}$. Connected samples keep every intermediate graph connected, so exactly one pivot value
  is zero, the last one, $g$ where there is one, and $\mathcal{D}^+$ replaces $\mathcal{D}^{-1}$.
  Under signed elimination the matrix with $g$ is no Laplacian, so $g$'s pivot value need not be
  zero; it is not computed but set to $0$, which pins $g$.

**Sources**

- Approximate Cholesky: Kyng, Sachdeva. "Approximate Gaussian elimination for Laplacians – fast,
  sparse, and simple." *FOCS*, 2016, pp. 573–582. arXiv:1605.02353, §4.1 (random order).
- Clique tree, split and merge, expectation of the factorization: Gao, Kyng, Spielman. "Robust
  and practical solution of Laplacian equations by approximate elimination." arXiv:2303.00709,
  2023, §4.1 (Claim 4.1), §4.3 (Alg. 6), App. B.
- Schur tree: Baumann, Kyng, Zöcklein. "VAC: a volume-sampling-based elimination rule for
  approximate Cholesky factorization." arXiv:2609.20241, 2026; with a split $k$: Chaitanya, Musco,
  Gionis. "CAST: canonical approximate Schur tree for approximate Cholesky on graphs."
  arXiv:2609.09255, 2026, Alg. 1.
- Minimum degree, with the degree as the number of distinct neighbors: George, Liu. "The evolution
  of the minimum degree ordering algorithm." *SIAM Review* 31(1), 1989, pp. 1–19. Counting edge copies
  is approx-chol's variant.

### 8. Bipartite factor-width-2 form

A bipartite factor-width-2 form has no edges inside $F$. Steps 2–5 apply to the whole $A$; after
a cover, $F$ and $K$ comprise both copies of their vertices, and $F$ keeps no internal edges in
$\tilde A$. Under signed elimination $F$ is eliminated from $\tilde A$ itself, with signed entries. Its pivots do not touch each other, so $F$ is eliminated exactly as
one block: the diagonal $\tilde A_{FF}$ enters $\mathcal{D}$, and $\tilde A_{KF} \tilde
A_{FF}^{-1}$ enters $\mathcal{L}$. An isolated vertex of $F$ has a zero row and drops out; every
other vertex of $F$ has a positive diagonal.

What remains is the Schur complement $\tilde H$ of $\tilde A$ onto $K$,

$$
\tilde H = \tilde A_{KK} - \tilde A_{KF}\, \tilde A_{FF}^{-1}\, \tilde A_{FK},
$$

which is SDDM on a cover and WDD under signed elimination, since each elimination keeps either.
Eliminating a vertex $f \in F$ adds the clique
on its neighbors with weights $w_{fi} w_{fj} / a_{ff}$ and raises their surpluses (glossary,
Elimination). These cliques can make $\tilde H$ dense, so $\tilde H$ is formed exactly, or sampled: each
star of $F$ is replaced by a clique sample on its neighbors in $K$, unbiased like the clique; its
ground share counts in $a_{ff}$ but is never drawn, and each neighbor's share $w_{fk} s_f / a_{ff}$
raises its surplus exactly, as in the exact formation. A sample then joins a star's neighbors
as its clique does, and a neighbor's surplus does not depend on the draw (#437). The stars are independent,
since $F$ has no internal edges: on a cover they are formed with a ground vertex, and whether
$\tilde H$'s surplus then goes to a ground vertex or to surplus edges is decided on $\tilde H$
(step 5), after formation. Steps 6–7 then factor $\tilde H$.

## Solve

For one component let

$$
G = \Pi^\top \mathcal{L}^{-\top} \mathcal{D}^{+} \mathcal{L}^{-1} \Pi,
$$

and $Q$ the orthogonal projection onto $z^\perp$ (the identity on a nonsingular component).

1. **Project:** $b \mapsto Qb$, the nearest right-hand side meeting the range condition.
2. **Reduce:** $b \mapsto SDb$.
3. **Lift:** for a cover, form $\begin{pmatrix} SDb \\ -SDb \end{pmatrix}$; this is the
   right-hand side of the cover. If grounded, append $-\mathbf{1}^\top \tilde b$ at $g$, so the
   lifted vector sums to zero like every vector in the range of $L$; under signed elimination
   append $0$, as $g$ is pinned.
4. **Factor solve:** apply $G$.
5. **Descend:** if grounded, subtract $y_g$ from every entry and drop $g$, which pins $g$ at zero;
   for a cover, take $(p - q)/2$.
6. **Return:** $x = DS\tilde y$, then project with $Q$, which gives the orthogonal gauge.

The lift and the descent are transposes up to a constant: steps 3 and 5 are $R$ and $cR^\top$ for
one matrix $R$, the product of the cover's and the ground's lifts, with $c = \tfrac12$ for a cover
and $c = 1$ otherwise. So

$$
B = c\, Q\, DS\, R^\top G\, R\, SD\, Q,
$$

symmetric and PSD. For a bipartite factor-width-2 form, steps 3–5 act on $\tilde H$, with exact
block substitution around them: before, $\tilde b_K \mapsto \tilde b_K - \tilde A_{KF} \tilde A_{FF}^{-1} \tilde b_F$;
after, $\tilde y_F = \tilde A_{FF}^{-1} (\tilde b_F - \tilde A_{FK} \tilde y_K)$.

A Schur build's $B$ takes only $b_K$: steps 1–6 run on $\tilde H$, with the lift restricted to
$K$ and $Q$ the projection onto the complement of $\tilde H$'s null vector in $K$'s coordinates;
no substitution through $F$ runs.

## Target

With exact elimination of a Laplacian, $G$ is a {1,2}-inverse of $L$ ($LGL = L$, $GLG = G$), since
$\mathcal{L}$ is invertible and $\mathcal{D}^+$ is one of $\mathcal{D}$. Then $B = A_r^+$:

- **grounded, balanced:** $R^\top G R = \tilde A^{-1}$, since $\tilde A$ is $L$ with $g$ deleted,
  so $B = A_r^{-1}$;
- **frustrated, signed elimination:** the matrix with $g$ is no Laplacian and $g$'s pivot is set
  to $0$, not computed, so $G$ is no generalized inverse of it. But $g$ is pinned: the
  substitutions run on the principal block without $g$, which is $\tilde A$ factored exactly, so
  $R^\top G R = \tilde A^{-1}$ and $B = A_r^{-1}$;
- **frustrated, cover:** the ground's lift and descent turn $G$ into $\hat A^{-1}$ if grounded and
  into a {1}-inverse $Y$ of $\hat A$ if floating, as with surplus edges. With $J = \begin{pmatrix} I \\ -I \end{pmatrix}$,
  $\hat A J = J \bar A$, so $Jb = \hat A J \bar A^{-1} b$ lies in the range of $\hat A$, and
  $Y J b = J \bar A^{-1} b + n$ with $n \in \ker \hat A = \operatorname{span}
  \begin{pmatrix} \mathbf{1} \\ \mathbf{1} \end{pmatrix}$. Since $J^\top n = 0$,
  $\tfrac12 J^\top Y J = \bar A^{-1}$, so $B = A_r^{-1}$;
- **floating, balanced:** $X = DS\, G\, SD$ is a {1,2}-inverse of $A_r$, but its range is not
  $z^\perp$ in general. $QXQ$ is again a {1,2}-inverse, now with range $z^\perp$ and kernel
  $\operatorname{span} z$; the {1,2}-inverse with that range and kernel is unique, and it is
  $A_r^+$. The projection on both sides is what makes $B$ the pseudoinverse;
- **bipartite:** exact block elimination of $F$ and an exact factorization of $\tilde H$ together
  factor $\tilde A$ exactly, so one of the cases above applies; a Schur build's $B$ is $H^+$ by the
  same cases applied to $\tilde H$.

Approximate Cholesky replaces $G$ by a random operator whose factorization has $L$ as its
expectation. The inverse is not unbiased, $\mathbb{E}[B] \ne A_r^+$ in general: "targets" refers to
exact elimination, and the fill-in level $k$ trades the variance of the factorization against
cost.

**Source**

- Uniqueness of the {1,2}-inverse with prescribed range and kernel: Ben-Israel, Greville.
  *Generalized Inverses: Theory and Applications*, 2nd ed. Springer, 2003, Ch. 2, §6, Thm. 12(c).
