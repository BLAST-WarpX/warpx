# PR #6994 Evidence: Conformal Embedded Boundary + Hybrid Ohm's-Law Field Solve

## 1. Summary

This PR gives the hybrid Ohm's-law (kinetic-ion / fluid-electron) field solver a working **conformal embedded-boundary (EB)** capability: a curved metal wall (e.g. a cylinder) is represented as a masked/mirrored PEC boundary inside the algebraic `E = eta*J - (J - J_i)xB/(n q) - grad(Pe)/(n q) + ...` solve, driven by the ECT (enlarged-cell / conformal-FDTD) `curl` machinery reused from the EM side. It is large because making a masked EB well-posed inside an *algebraic* (not time-stepped) Ohm solve touched many layers at once — covered-cell current handling, PEC mirror fills for `J`/`E`/`B`, resistivity/pressure sign-safety on mirrored density, an EB-aware nodal Hall interpolation, isotropic finite-difference operators to kill a grid `m=4` mode, and a particle **dielectric standoff** that is the actual Yee-stability lever — *plus* a long research arc of experimental stabilizers (Marder div-clean, ECT-flux and LSQ wall currents with a cut-metric div-clean, a self-declared-deprecated covered-B curl-fill) and their PICMI/pybind surface, most of which are default-off and separable. The genuine minimum-viable core is small; the bulk of the LOC is accuracy/hardening/experimental work that can be split into follow-ups.

## 2. Master table

| Feature (key) | Files | Why required | Breaks if removed | Separable | MVP keeper | Risk | ~LOC |
|---|---|---|---|---|---|---|---|
| `hybrid-eb-solve` | HybridPICSolveE.cpp; HybridPICModel.{H,cpp}; CMakeLists.txt; Make.package | Well-posedness of a masked EB in the algebraic Ohm solve: covered-J zeroing, J/E/B mirror fills + band pinning, `abs(rho)` for eta, Pe EB handling, `eb_resistive_only_partial`, EB-aware Hall mask | CORE revert → ~10x wall div(J), covered-J pollution of nodal JxB, non-finite blow-up (E-band>J-band), eta sign inversion, RKF45 crawl | partial | yes | high | ~1600 |
| `eb-j-boundary` | EBJBoundary.{H,cpp} | Layer 1 (~1000 loc) = the E/J/Ji PEC edge BC + deposit fold + nodal-scalar BC + cyl correction the standoff run depends on | Full revert crashes/diverges (deep-zero/tangential-E asserts, 2nd→1st order at wall, ~10x div(J), lost wall charge) | partial | yes | high | 2593 |
| `em-side-ect` | WarpXFaceExtensions.cpp; EmbeddedBoundaryInit.{H,cpp}; WarpXInitEB.cpp; WarpXFaceInfoBox.H; ComputeCurlA.cpp; EvolveB.cpp; EvolveECTRho.cpp | MVP subset (~90 loc): dispatch hooks (A), collocated nodal mask (B), partial=2 E mask (C) make conformal collocated+Yee run | Revert subset → collocated init fails, Yee degrades to bare-staircase with wrong wall B | partial | yes | high | 560 |
| `dielectric-standoff` | ParticleScraper.H; MultiParticleContainer.{H,cpp}; WarpXEvolve.cpp; WarpX.{H,cpp} | Threads an `offset` into `scrapeParticlesAtEB`; >=3-cell fluid standoff is the actual Yee-stability lever | Runtime instability: near-wall plasma hits O(1) covered-B/Hall artifacts, m=4 seeds, Yee liftoff blows up | partial | yes | low | 24 |
| `external-vector-potential` | ExternalVectorPotential.{H,cpp}; WarpXPushFieldsHybridPIC.cpp | Delta 1: fill B_ext=curl(A_ext) through wall so a uniform external bias stays uniform under conformal EB; Delta 2: warn-only initial-B guard | Delta 1 revert → 0.025 T B_ext step at wall, spurious Hall gradient, wrong wall physics (byte-identical for staircase) | yes | yes | low | 80 |
| `python-picmi-fields` | Source/Python/WarpX.cpp; picmi.py; Fields.H; MacroscopicProperties.cpp | Thin MVP kwarg slice + `edge_cent_offset` registration expose conformal wall + standoff from PICMI | Drop whole group → no PICMI access to conformal EB/standoff + `edge_cent_offset` compile break | partial | yes | medium | 697 |
| `tests-examples` | ohm_solver_plasma_cylinder_liftoff/{CMakeLists.txt,analysis.py,analysis_default_regression.py,inputs_..._picmi.py} | Only end-to-end CI exercise of the collocated conformal wall (PR policy: features need a test) | Revert → no CI coverage of collocated conformal wall | partial | yes | high | 697 |
| `isotropic-operators` | IsotropicOperators.H; FiniteDifferenceSolver.H | Cancels cos(4θ) truncation error (Mehrstellen 9pt / Patra-Karttunen 27pt + corner-curl), suppresses grid m=4; defaults TRUE | No crash; quality regression: axis-vs-diagonal anisotropy re-seeds m=4 | yes | no | medium | 304 |
| `accurate-wall-current` (`conformal_ect_lsq`) | ConformalEctLSQ.cpp | Payoff (not stability): LSQ wall current 20-100x more accurate than masked (rel-L2 ~0.0073-0.0087 vs ~0.30-0.43); default-off | Flag off → byte-identical; flag on + file gone → loss of accurate current (no test asserts it) | yes | no | high | 776 |
| `marder-divclean` | MarderCorrection.cpp | Default-off opt-in stabilizer; measured to **destabilize** the standoff run; standoff (not Marder) gives Yee stability | Nothing in MVP (all entry points early-return at default alpha<=0) | yes | no | high | 758 |

## 3. Minimum-viable core

These groups MUST stay together for the conformal-wall + dielectric-standoff MVP to function and be reviewable, in dependency order:

1. **`em-side-ect` (MVP subset only, ~90 loc)** — the EvolveB/EvolveECTRho dispatch hooks (A), `MarkUpdateCellsNodalLevelSet` collocated mask (B), and the partial=2 E mask (C). Without these the conformal wall does not initialize (collocated) or silently degrades to bare-staircase Yee. *This is the wiring everything else stands on.*
2. **`eb-j-boundary` (Layer 1 only, ~1000 loc)** — the E/J/Ji PEC edge-field BC + deposit fold + nodal-scalar (rho Dirichlet / Pe Neumann) + cylindrical correction. This *is* the field boundary condition the feature is named after; reverting it crashes/diverges the run.
3. **`hybrid-eb-solve` (CORE only)** — covered-J zeroing (not skipping), J/E/B mirror-fill wiring with the E band pinned to the J band, `rho_val_eta = abs(rho)`, Pe EB handling, `eb_resistive_only_partial`, and the (currently **uncommitted**) `eb_hall_mask`/`InterpMasked` EB-aware nodal Hall interpolation. This is the minimal well-posedness layer for a masked EB in the algebraic Ohm solve.
4. **`dielectric-standoff` (~24 loc)** — the `eb_particle_scrape_offset` plumbing. Per the brief, a >=3-cell standoff is the actual Yee-stability lever; without it the Yee liftoff blows up. Default 0.0 keeps all other runs bit-identical.
5. **`external-vector-potential` Delta 1 (~40 loc)** — curl-through-wall for the external vacuum A, gated on `UseConformalEBSolve()`. Required only when a conformal-EB deck also runs a nonzero external bias (the liftoff reversal coil), but that *is* the MVP deck, so it ships with the core.
6. **`python-picmi-fields` (thin MVP slice)** — `use_conformal_eb`, `eb_bc_*`, `eb_deposit_fold`, `eb_rho_dirichlet`, plus the `edge_cent_offset` `FieldType` registration (a compile dependency). The deck cannot run without these.
7. **`tests-examples` (trimmed)** — the `ohm_solver_plasma_cylinder_liftoff` collocated smoke test, pruned to the flags the MVP actually uses. PR policy requires a test to ship with the feature.

**Strongly recommended companion (technically separable):** `isotropic-operators` defaults TRUE and is *not* byte-identical when reverted (it re-exposes the cos(4θ) m=4 mode), and it conditions the plasma-current fill-band widen (`HybridPICModel.cpp:290-294`). Keep it in the MVP flagged as a self-contained numerics change, or the MVP must pin the fill band to 1 and accept the m=4 mode.

> **Blocking pre-req:** `eb_hall_mask`/`InterpMasked` is **uncommitted (working-tree only)**. A `git diff development..HEAD` review MISSES it entirely. It must be committed before the PR is measured.

## 4. Split into separate PRs

### PR-A — External-A initial-B diagnostic (`external-vector-potential` Delta 2)
- **Contents:** `CheckInitialB()` warn-only guard on the pre-existing split-field subtract.
- **Rationale:** Fully standalone; does not read `UseConformalEBSolve()`; targets base split-field machinery already on `development`. Trivial, ship anytime.
- **Dependencies:** none.

### PR-B — Isotropic finite-difference operators (`isotropic-operators`)
- **Contents:** `IsotropicOperators.H` (Mehrstellen/Patra-Karttunen Laplacian + corner-curl), the two default-true flags, and the fill-band-widen conditional (`HybridPICModel.cpp:290-294`).
- **Rationale:** Self-contained numerical-quality feature; does not depend on the conformal wall or standoff. Move the fill-band conditional with it (or, if deferred, the MVP must pin the band to 1 to stay byte-consistent).
- **Dependencies:** none, but if it *lands after* the MVP, coordinate the fill-band default so the MVP stays byte-consistent.
- **Note:** the `FiniteDifferenceSolver.H` hunks in the same diff are ECT-Ampere/curvature, not isotropic — do not attribute them here.

### PR-C — Marder / div-clean stabilizers (`marder-divclean`)
- **Contents:** `MarderCorrection.cpp` (E-field Marder cleaner + geometry-blind div(B)/div(J) clean), its 3 FieldPush call sites, marder/div-clean flag parsing, ~16 header members/enums, 4 Marder CTests, and the Marder/div-clean PICMI kwargs + WarpX.cpp test bindings.
- **Rationale:** Default-off and **measured to destabilize the standoff run** — affirmatively NOT wanted in the MVP. Could split further into (A) E-Marder (CI smoke coverage) vs (B) div(B)/div(J) clean (no CI).
- **Dependencies:** consumes infra the MVP brings in (`ComputeDivE`, `ApplyPECBoundaryToField`, `distance_to_eb`); nothing in the MVP depends on it → land **after** the MVP.

### PR-D — Accurate wall current + cut-metric div-clean (`accurate-wall-current`)
- **Contents:** `ConformalEctLSQ.cpp` (Phase-1 LSQ centroid overwrite + Phase-2b `BuildConformalDualAreas`/`ApplyConformalDivClean`), the 2 call sites, 5 `query()`s, header decls, CMake/Make.package lines, the one Marder gate (`MarderCorrection.cpp:748`), and the `conformal_ect_lsq`/`conformal_ect_j`/`divclean` PICMI kwargs + bindings.
- **Rationale:** This is the *payoff* accuracy feature, not stability. Fully opt-in (`m_conformal_ect_lsq` defaults false → byte-identical). Phase-2b div-clean is **not required for stability with the standoff** and adds thousands of CG solves per RKF45 substage.
- **Dependencies:** needs the MVP's conformal-wall infra (`distance_to_eb`, `edge_cent_offset`, covered-J mirror fill) but that infra does not need it → land **after** the MVP. **Caveat:** the covered-J mirror-fill restructure (`HybridPICModel.cpp:759-791`) serves the masked path too and must remain — only the `ect_lsq` sub-branch leaves. Add a CTest (none currently wired).

### PR-E — Covered-B curl-fill layer (`eb-j-boundary` Layer 2, ~750 loc)
- **Contents:** the flag-gated `quadratic_gather` MLS covered-B fill (`solve_spd`, tap-weight cache, `normal_weight_floor`/`cut_blend`/`cut_clamp`/`corner_skip`, S_CORNER detector + chamfer, `WARPX_BCURL_DIAG_*`), plus the `conformal_b_curl_fill{,_freeze,_blend,_clamp,_corner_skip}` PICMI kwargs.
- **Rationale:** Guarded behind `m_conformal_b_curl_fill`/`m_conformal_ect_lsq`, default byte-identical, orthogonal to the standoff MVP. **Self-declared DEPRECATED** (runtime warning `HybridPICModel.cpp:31`) — a candidate to *drop* rather than split.
- **Dependencies:** the friction is that Layer 2 is interleaved *inside* the shared `ApplyPECBoundaryToField`/`FoldEBDepositToField` (signature grew to 18 params). Splitting = surgically editing shared code, not deleting files: keep the core signature + linear mirror + fold + nodal scalar, move the quadratic machinery out.

### PR-F — EM ECT hardening & cleanup (`em-side-ect` remainder, ~470 loc)
Split into three independent PRs:
- **F1 — `neigh`→`neighbor` rename** (~200 diff lines, zero behavior change): standalone no-op cleanup, land anytime.
- **F2 — GPU atomic borrow-race fix** (`Gpu::Atomic::If`, refs #2257/#2298): standalone bug fix on the *pre-existing* EM ECT path; testable with existing `ohms_law_embedded_boundary`. **Land early** — it fixes a real GPU corruption in any ECT run.
- **F3 — Multi-box seam-sync (E) + ECT curvature correction (D):** both default-dormant. Seam-sync is gated by `m_ect_needs_seam_sync=(nbox>1||periodic)` and has **no MVP deck coverage** (liftoff/eb_diffusion are single-box) — the single largest reviewable-risk hunk; give it its own PR with a multi-box test. Curvature is opt-in (`conformal_ect_curvature`, byte-identical off) and needs its own convergence test. The `distance_to_eb` nodal-sync hunk in `WarpXInitEB.cpp` is technically seam-sync but cheap/safe — keep it with the MVP.

### PR-G — Unrelated bundled fixes (from `python-picmi-fields`)
- **`MacroscopicProperties.cpp` 1D fix:** zeroes index-type comps 1,2 (`WARPX_DIM_1D_Z`) and corrects the parser z-coord `j*dx`→`i*dx`. Standalone correctness fix, tiny PR.
- **`InverseBremsstrahlungCollisions` (picmi.py):** backs upstream collision PR #5839 not yet on `development` — obviously its own PR.
- **`Fields.H` `area_mod`/ECTRhofield/Venl doc-comments:** cosmetic, trivially separable.
- **WarpX.cpp test-only pybind methods (~317 lines):** the Marder/ECT-LSQ/div-clean bindings should follow their C++ features (PR-C/PR-D/PR-E); the Layer-1 bindings (`hybrid_apply_eb_boundary_to_edge_field`, fold, nodal scalar, `hybrid_solve_e`, `hybrid_calculate_plasma_current`) may stay with the MVP as unit-test hooks.

**Recommended ordering:** F2 (bug fix) → MVP (core groups §3) → PR-B (isotropic) → PR-D (accurate current) / PR-C (Marder) / PR-E (covered-B) in any order after the MVP. F1, PR-A, PR-G land anytime.

## 5. Review-risk hotspots

- **`em-side-ect` multi-box seam-sync** (`WarpXFaceExtensions.cpp`, `EvolveB.cpp`): the largest single reviewable risk, OwnerMask-gated borrowing + reductions + deferred phase-2 B update + SumBoundary, with **no MVP deck coverage** (all decks single-box). Top split candidate; verify bit-identity when `m_ect_needs_seam_sync` is off.
- **`hybrid-eb-solve` E-band-must-not-exceed-J-band** (`HybridPICModel.H:448-457`, commit `3c254866f`): filling E where J=0 is a documented non-finite blow-up through the curl(E)→B→curl(B)→J→E loop. Confirm the E fill band is pinned to the J band (`m_eb_fill_band_cells`).
- **`hybrid-eb-solve` uncommitted `eb_hall_mask`/`InterpMasked`:** working-tree-only — invisible to `git diff development..HEAD`. Must be committed or the reviewed diff is wrong.
- **`eb-j-boundary` shared 18-parameter `ApplyPECBoundaryToField`:** Layer 1 (MVP) and Layer 2 (deprecated covered-B fill) are interleaved in one function; a reviewer must separate the load-bearing mirror/fold/parity logic from the flag-gated quadratic machinery.
- **`em-side-ect` GPU atomic borrow-race fix** (`Gpu::Atomic::If`, refs #2257/#2298): correctness-critical on the pre-existing EM path; a GPU-only double-lend race that corrupts `S_mod`.
- **`accurate-wall-current` hand-rolled Cholesky / auto-dropping normal equations** (`ConformalEctLSQ.cpp`, `lsq_centroid_weights`, sliver gate → `normal_linear_value`): numerically delicate, no CTest asserts on the accurate current.
- **`marder-divclean` cadence/stencil branching** (`MarderCorrection.cpp`): three cadence sites × nodal/Yee × RZ axis special-cases, and it is measured to destabilize the standoff — ensure it is off by default and split out.
- **`tests-examples` 694-line research CLI:** most flags (b-curl-fill family, div-clean, ect-lsq, marder, equilibrium-b) gate abandoned/experimental code the CI never touches — prune to `use_conformal_eb`/`grid_type`/`standoff`/`wall_supported`.

## 6. Known caveats

> **Marder destabilizes the standoff.** The substep-E Marder cleaner (`marder-divclean`, part A) is *measured to destabilize* the standoff run and is affirmatively NOT wanted in the MVP. It is default-off (entry points early-return at `alpha<=0`); keep it that way and split it out (PR-C). The standoff — not Marder — provides Yee stability.

> **Phase-2b cut-metric div-clean is NOT required with the standoff.** `ApplyConformalDivClean` (`accurate-wall-current`) zeroes the cut-metric divergence (band-RMS 0.99→5e-7) but only *halves* the Yee divergence the RKF45 stiffness actually responds to (`ConformalEctLSQ.cpp` header comment H:502-505), so it does not clear stiffness. On the liftoff growth it was one of 7 NULL levers (cut-metric and Cartesian modes both left J-scale and spurious-jz unchanged), and the growth reproduced identically on the masked path → not LSQ-specific. It is gated off by default (`iters=0`) and, if enabled, destabilizes nothing but also fixes nothing while adding thousands of CG solves per RKF45 substage. Task #15 is settled: Phase-2b is a separate PR, not an MVP stability requirement.

> **The standoff is the stability lever.** Per the brief, a >=3-cell dielectric standoff (`dielectric-standoff`, `eb_particle_scrape_offset`) is what makes the Yee conformal-EB run stable; several `hybrid-eb-solve` CORE band-aids are belt-and-suspenders once the plasma is held off the wall.

> **Convergence-order limitations.** The `accurate-wall-current` LSQ overwrite converges the *wall current* (cylinder wall rel-L2 ~0.0073-0.0087, order ~+0.55-0.6) — a 20-100x accuracy gain over the masked/matched-area path (~0-order, 0.30-0.43) — but the wall order is ~1st-order-plus, **not 2nd order** in these measurements. Reverting `eb-j-boundary` Layer 1 drops conformal-wall convergence from 2nd to 1st order at the wall. `isotropic-operators` is a numerical-*quality* feature (m=4 suppression), not an order-improving one; removing it causes a quality regression, not an order or stability loss.

> **CI coverage gaps.** The shipped `tests-examples` CI test covers **collocated only** (not Yee) and does **not** cover the standoff (`standoff_cells=0.0`, `grid_type=collocated`, marder off) — so as written it is neither a Yee-stability nor a standoff-stability test; its MVP value is collocated-wall coverage. `accurate-wall-current` has **no CTest**; `marder-divclean` E-path has finite-field smoke CI only and the div(B)/div(J) clean has none; the multi-box seam-sync has no deck coverage. A follow-up should add a Yee + standoff stability variant and a `conformal_ect_lsq` convergence test.
