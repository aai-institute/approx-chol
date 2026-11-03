# Glossary

Terms the code and docs use. Where the literature differs, the entry says which meaning we
take. "(not: …)" lists synonyms we don't use.

## Symbols

| Symbol | Meaning |
|---|---|
| $A$, $a_{ij}$, $n$ | input matrix, its entries, its size |
| $C$, $A_C$ | a component, the principal submatrix on it |
| $e_i$, $\mathbf{1}$ | standard basis vector, all-ones vector |
| $\varepsilon$ | machine epsilon |
| $\mathbb{E}$ | expectation |
| $M(A)$ | comparison matrix |
| $s$, $s_i$ | surplus |
| $L$ | Laplacian |
| $w_e$, $w_{ij}$, $\chi_e$ | edge weight, incidence vector $e_i - e_j$ |
| $\deg(v)$ | degree |
| $S = \operatorname{diag}(\sigma)$ | signature matrix |
| $\Delta$, $N$, $P$ | diagonal, negative and positive off-diagonal parts of $A$ |
| $\hat A$, $p$, $q$ | Gremban cover, the two halves of its solution |
| $d$, $D = \operatorname{diag}(d)$ | scaling vector |
| $\tilde A$, $\tilde b$ | reduced matrix, reduced right-hand side |
| $z$ | null vector |
| $\rho_i$, $\nu_i$, $\beta$ | row ratio, violation, rounding band |
| $\eta$, $E = \operatorname{diag}(\eta)$ | clamp |
| $E'$, $A_r$ | dropped surplus, target matrix |
| $F$, $K$, $\Delta_F$, $\Delta_K$, $A_{FK}$ | eliminated side, kept side, their diagonal blocks, coupling block (bipartite factor-width-2 form) |
| $H$ | Schur complement |
| $v$, $w_{vi}$, $\omega_v$, $a_{vv}$ | pivot, its star weights, their sum, its pivot value |
| $\Pi$ | elimination order |
| $\mathcal{L}$, $\mathcal{D}$ | unit lower-triangular and diagonal factors |
| $B$ | approximate inverse |
| $b$, $x$ | right-hand side, solution |

## Basics

- $A$ is a real symmetric $n \times n$ matrix throughout.
- **PSD / PD:** $x^\top A x \ge 0$ for every $x$ / $x^\top A x > 0$ for every $x \ne 0$.
- **Machine epsilon** $\varepsilon$: the spacing from $1$ to the next larger value of the scalar
  type, $2^{-52}$ for `f64` and $2^{-23}$ for `f32` (Rust's `EPSILON`), twice the unit roundoff.
- **CSR:** compressed sparse row storage of $A$: row pointers, column indices, values. **Full**
  stores every entry; **upper triangle** stores $a_{ij}$ for $j \ge i$ only.
- **Symmetric up to rounding:** $|a_{ij} - a_{ji}|$ is within a tolerance relative to the entries.

## Graph structure

- **Graph of $A$:** vertices $1, \dots, n$, an edge $\{i, j\}$ for every $a_{ij} \ne 0$, $i \ne j$.
  The edge's **entry** is $a_{ij}$; the edge is **positive** or **negative** by the sign of its
  entry.
- **Neighbors** of $v$: the vertices sharing an edge with $v$. Its **degree** $\deg(v)$ is their
  number, in the current graph during elimination. **Isolated:** degree $0$.
- **Path:** a sequence of distinct vertices $v_1, \dots, v_k$ in which consecutive vertices share an
  edge.
- **Cycle:** a path $v_1, \dots, v_k$ with $k \ge 3$ whose ends $v_k$ and $v_1$ also share an edge.
  Its entries are $a_{v_1 v_2}, \dots, a_{v_{k-1} v_k}, a_{v_k v_1}$.
- **Connected:** every two vertices are joined by a path.
- **Component** $C$: a maximal connected set of vertices; $A_C$ is the principal submatrix on it.
  Grouping indices by component makes $A$ block diagonal.
- **Tree:** a connected graph without cycles; on $k$ vertices it has $k - 1$ edges.
- **Spanning tree** of a graph $G$: a subgraph of $G$ that is a tree containing every vertex of $G$.
- **Star** of $v$: $v$ together with its edges to its neighbors.
- **Clique** on a vertex set $U$: an edge between every two vertices of $U$, $|U|(|U| - 1)/2$ in
  all.

## Dominance and matrix classes

- **Surplus** (not: excess, slack) of row $i$: $s_i = a_{ii} - \sum_{j \ne i} |a_{ij}|$. It uses
  $a_{ii}$, not $|a_{ii}|$, so dominance here includes a nonnegative diagonal.
- **Weakly diagonally dominant (WDD):** every surplus is $\ge 0$. **Strictly diagonally dominant:**
  every surplus is $> 0$. Unqualified "diagonally dominant" means weak. ("SDD" is not used: it
  also reads as "symmetric diagonally dominant".)
- **Z-matrix:** every off-diagonal entry is $\le 0$.
- **Comparison matrix** $M(A)$: $m_{ii} = |a_{ii}|$; $m_{ij} = -|a_{ij}|$ for $i \ne j$. It is a
  Z-matrix.
- **Laplacian** of a graph with edge weights $w_e \ge 0$: $L = \sum_e w_e \chi_e \chi_e^\top$.
  Equivalently, a symmetric Z-matrix with every row sum $0$. $L\mathbf{1} = 0$, so it is singular.
- **SDDM:** $L + \operatorname{diag}(s)$ with $L$ a Laplacian and $s \ge 0$; equivalently, a
  symmetric WDD Z-matrix. It is nonsingular iff every component has some $s_i > 0$. The name
  ("symmetric diagonally dominant M-matrix") is historical; literature often requires
  nonsingularity, we don't.
- **Factor width** of a PSD $A$: the least $k$ such that $A = VV^\top$ with at most $k$ nonzeros per
  column of $V$.
- **Factor-width-2 form:** $A = VWV^\top$ given by $V$ with at most two nonzeros per column and $W$
  nonnegative diagonal.
- **Generalized diagonally dominant (GDD):** some positive diagonal $D$ makes $DAD$ WDD. Weak, so
  $A$ may be singular. For symmetric $A$ these are equivalent (Boman, Chen, Parekh, Toledo 2005,
  "On factor width and symmetric H-matrices"):
  - $A$ is GDD;
  - $A$ has factor width at most 2;
  - $a_{ii} \ge 0$ for every $i$, and $M(A)$ is PSD.
- **Bipartite factor-width-2 form:** $A$ with a given split of its vertices into an
  **eliminated side** $F$ and a **kept side** $K$, with no off-diagonal entry inside a side:

  $$
  A = \begin{pmatrix} \Delta_F & A_{FK} \\ A_{KF} & \Delta_K \end{pmatrix},
  $$

  $\Delta_F$, $\Delta_K$ diagonal. The split is part of the input; a bipartite graph may admit
  several.

## Signs and reduction

- **Signature matrix** $S = \operatorname{diag}(\sigma)$, $\sigma_i \in \{-1, +1\}$.
- **Switching** (Zaslavsky; not: sign flip, fold) by $S$: $A \mapsto SAS$, i.e.
  $a_{ij} \mapsto \sigma_i \sigma_j a_{ij}$. It keeps the spectrum.
- **Balanced** component: some switching makes $SAS$ a Z-matrix there. Equivalently, every cycle
  has an even number of positive entries. **Frustrated:** not balanced.
- **Gremban cover** (not: double cover, Gremban expansion, lift), for a frustrated component:
  split $A = \Delta + N + P$ into its diagonal, its negative off-diagonal entries and its positive
  ones.

  $$
  \hat A = \begin{pmatrix} \Delta + N & -P \\ -P & \Delta + N \end{pmatrix},
  $$

  a $2n \times 2n$ Z-matrix with $A$'s surpluses, so WDD iff $A$ is. If
  $\hat A \begin{pmatrix} p \\ q \end{pmatrix} = \begin{pmatrix} b \\ -b \end{pmatrix}$, then
  $x = (p - q)/2$ is a solution of $Ax = b$, even when $\hat A$ is singular.
- **Scaling vector** $d > 0$, $D = \operatorname{diag}(d)$. It is a **dominance scaling** (not:
  balancing, equilibration) of $A$ if $DAD$ is WDD.
- **Reduction** of a GDD component, given a dominance scaling $D$. It is exact.
  - Balanced, with $S$ switching $DAD$ to a Z-matrix: $\tilde A = SDADS$, of size $n$;
    $\tilde b = SDb$, and $x = DS\tilde y$ where $\tilde A \tilde y = \tilde b$.
  - Frustrated: $\tilde A$ is the Gremban cover of $DAD$, of size $2n$;
    $\tilde b = \begin{pmatrix} Db \\ -Db \end{pmatrix}$, and $x = D(p - q)/2$ where
    $\tilde A \begin{pmatrix} p \\ q \end{pmatrix} = \tilde b$.

  The **reduced matrix** $\tilde A$ is SDDM. Reduction need not preserve the orthogonal gauge; it
  is imposed as $z^\top x = 0$ in $A$'s coordinates.
- **Edge weight:** the magnitude of an off-diagonal entry of $\tilde A$, $w_{ij} = |a_{ij}|\, d_i d_j$.

## Singularity and gauges

Entries here concern one GDD component $A_C$, with $n = |C|$; apply them per component.

- **Grounded:** $M(A)$ is nonsingular. Then $DAD$ has a positive surplus, and $A$ is nonsingular.
- **Floating:** $M(A)$ is singular. Then every surplus of $DAD$ is $0$, and $d$ is unique up to a
  positive factor.
  - Balanced: $\tilde A$ is a Laplacian, so $A$ is singular, with null space spanned by the
    **null vector** $z = DS\mathbf{1}$, i.e. $z_i = \pm d_i$.
  - Frustrated: $A$ is nonsingular. The cover of $DAD$ is singular, and the recovery
    $x = D(p - q)/2$ cancels its null direction.

  So a component is singular iff it is balanced and floating. An isolated vertex with $a_{ii} = 0$
  is one, with $z = e_i$.
- **Ground vertex** (Gremban augmentation): an extra vertex $g$ with an edge $\{i, g\}$ of weight
  $s_i$ for each $s_i > 0$. An SDDM matrix is the Laplacian of the augmented graph with row and
  column $g$ deleted.
- **Range condition:** on a singular component, $Ax = b$ has a solution iff $z^\top b = 0$; the
  solutions are then the line $x_0 + t z$, $t \in \mathbb{R}$.
- **Projection** of $b$: $b - \frac{z^\top b}{z^\top z} z$, the nearest right-hand side meeting the
  range condition.
- **Gauge:** a rule that picks exactly one point on that line.
  - **Orthogonal:** $z^\top x = 0$.
  - **Pinned** (not: anchored, grounded) at vertex $r$: $x_r = 0$.
- **Regauge:** moving a solution along $z$ to another gauge; to the orthogonal one,
  $x - \frac{z^\top x}{z^\top z} z$.
- **Pseudoinverse** $A^+$: the Moore–Penrose inverse. $A^+ b$ is the unique $x \perp \ker A$ with
  $Ax$ equal to the orthogonal projection of $b$ onto $\operatorname{range} A$.

## Rounding

Terms for dominance of a rounded $A$ under a computed scaling $d$.

- **Row ratio** under $d$, for $a_{ii} > 0$: $\rho_i = \sum_{j \ne i} |a_{ij}|\, d_j / (a_{ii} d_i)$.
  Row $i$ of $DAD$ has surplus $a_{ii} d_i^2 (1 - \rho_i)$.
- **Inadmissible row:** $a_{ii} < 0$, or $a_{ii} = 0$ with an off-diagonal entry. No scaling makes
  it dominant, and $A$ is then not GDD (for $a_{ii} < 0$ not even PSD).
- **Violation** of row $i$: $\nu_i = \max(0, \rho_i - 1)$, and $0$ for an empty row. A component's
  violation is $\max_i \nu_i$.
- **Clamp:** $E = \operatorname{diag}(\eta)$ with
  $\eta_i = \max\bigl(0, \sum_{j \ne i} |a_{ij}|\, d_j / d_i - a_{ii}\bigr)$, which is
  $a_{ii} \nu_i$ on admissible rows. For this $d$, it is the componentwise least nonnegative
  diagonal that makes $D(A + E)D$ WDD.
- **Clamped matrix** $A + E$ ($E = 0$ when no row is violated); $E$ is the **perturbation**.
  Reduction refers to the clamped matrix.
- **Rounding band** $\beta$: the tolerance on $|1 - \rho_i|$ below which a deviation is attributed
  to rounding, used both for violations and for numerically floating. A violation inside the band
  does not prove that $A$ is GDD.
- **Numerically floating:** $|1 - \rho_i| \le \beta$ for every row with $a_{ii} > 0$ under the
  computed $d$. It does not prove singularity; a balanced component that is numerically floating is
  treated as floating.
- **Dropped surplus** $E' \le 0$: on a numerically floating balanced component, the surpluses of
  the reduced matrix, inside the band, are set to $0$; $E'$ is that change in $A$'s coordinates,
  and $0$ elsewhere.
- **Target matrix** $A_r = A + E + E'$: the matrix $B$ targets.

## Elimination and factorization

- **Schur complement** of $A$ onto a vertex set $K$, eliminating $F$:
  $H = A_{KK} - A_{KF} A_{FF}^+ A_{FK}$, numbered by $K$. For a bipartite factor-width-2 form:
  $H = \Delta_K - A_{KF} \Delta_F^+ A_{FK}$.
- **Pivot:** the vertex $v$ being eliminated. Its **pivot value** is $a_{vv}$ at that moment, the
  entry of $\mathcal{D}$ it contributes.
- **Elimination** of $v$ from an SDDM matrix: the Schur complement onto the other vertices. With
  star weights $w_{vi} = -a_{vi}$, $\omega_v = \sum_i w_{vi}$ and $a_{vv} = \omega_v + s_v$:
  - if $a_{vv} > 0$, it removes the star, adds the clique on $v$'s neighbors with weights
    $w_{vi} w_{vj} / a_{vv}$, and raises each neighbor's surplus by $w_{vi} s_v / a_{vv}$;
  - if $a_{vv} = 0$, $v$ is isolated and nothing changes.
- **Elimination order:** the permutation in which pivots are eliminated, and its permutation
  matrix $\Pi$.
- **Clique sample:** a random weighted graph on the pivot's neighbors whose Laplacian has
  expectation equal to the clique's. It is **connected** if it is connected on those neighbors.
- **Approximate Cholesky:** elimination with each clique replaced by a clique sample, e.g. a
  spanning tree of the clique: $\deg(v) - 1$ edges instead of $\deg(v)(\deg(v) - 1)/2$. Applied to
  a Laplacian $L$, it produces $\mathcal{L}$ and $\mathcal{D}$ with
  $\mathbb{E}[\Pi^\top \mathcal{L}\mathcal{D}\mathcal{L}^\top \Pi] = L$. With connected samples, a
  connected Laplacian has exactly one zero pivot value (in exact arithmetic).
- **Seed:** the value that initializes the random choices of a factorization.
- **Factorization** (not: decomposition): writing a matrix as a product of factors, and the
  result. The **approximate Cholesky factorization** of a Laplacian $L$ is
  $(\Pi, \mathcal{L}, \mathcal{D})$; that of $A$ is the reduction maps with the approximate
  Cholesky factorizations of the reduced matrices, and $B$ is the operator it gives.
- **Factor:** one matrix of a factorization: $\mathcal{L}$ or $\mathcal{D}$.
- **Approximate inverse** $B$ (not: $Z$, $M^{-1}$): the linear operator $b \mapsto x$ given by the
  reduction maps and the approximate Cholesky factorizations of the reduced matrices. It targets
  $A_r^{-1}$, and $A_r^+$ on singular components.
- **Preconditioner:** an operator used to speed up an iterative solver; $B$ is used as one.
- **Solve** (not: apply): applying $B$ to a right-hand side or a block. Exact solutions are called
  "the solution of …", never "solve".
- **Right-hand side** (column): one vector $b$ that $B$ is applied to. A **block** is several
  right-hand sides applied at once.
