# Plan (for review): EM curved-cavity ECT test + ECT-aware boundary stencils for the hybrid E-solve

Branch context: `eb_cylindrical_correction` (off `ohms_law_conformal_EB`).
Status: **plan only — review before implementing.**

## Why (established this session, primary-source + measurement grounded)

- The conformal **B-advance (ECT) is correct and ~2nd-order on a curved wall** — measured
  directly: a pure EM-Maxwell circular-cavity, `method='ECT'`, resonant-frequency
  convergence = **~2nd order (freq err 2.1e-4 → 1.5e-5 across N=64/128/256, ~100x below
  Yee's clean 1st-order)**. So the cut-cell B-update reproduces Dey-Mittra/Xiao-Liu.
- The **hybrid** is 1st-order on the *identical* geometry + `J0(J11 r/R)` Neumann mode +
  identical ECT machinery (decay-rate eigenvalue order ~1.0, field L2 ~1.0). So the gap is
  the **hybrid-specific near-wall E coupling, not the B-scheme** (Dey-Mittra/USC are off the
  table — they'd replace an already-2nd-order B-advance).
- Mechanism (the user's framing): the EM solver **integrates** E forward with the
  masked-but-conformal-B system (covered E edges stay masked at their value; the conformal
  B-update + `edge_lengths` weighting carry the wall to 2nd order). The hybrid instead
  computes E **algebraically** from Ohm's law `E = eta*J - (J x B - grad pe)/(e n)`,
  `J = curl(B)/mu0`, and imposes the wall via the **level-set mirror** (a pointwise,
  ~1st-order fill). Because E is not integrated by an EB-aware operator, it does not inherit
  the ECT's conformality — its near-wall stencils must be made ECT-aware explicitly.
- Code facts: `EvolveECartesian` (EM E-update) and `CalculateCurrentAmpereCartesian` (hybrid
  Ampere curl, HybridPICSolveE.cpp:33-180) both use ONLY `eb_update_E` masking — no
  `face_areas`/`edge_lengths`/area-scaling. The hybrid additionally runs
  `ApplyPECBoundaryToField` (EBJBoundary.cpp) on E/J (and B on the collocated path) = the
  level-set mirror.

## Part A — EM curved-cavity ECT validation test (low risk; do first)

Formalize `em_circle_cavity.py` (already drafted, validated) into a registered test that
guards "ECT is 2nd-order, Yee is 1st-order, on a curved PEC wall."

- **Deck**: 2D circular PEC cavity (`sqrt(x*x+z*z)-R`), `ElectromagneticSolver(method=ECT|Yee)`,
  TM^y_01 standing mode `By = B1 J0(J11 r/R)` via an afterinit field-wrapper (parser has no
  Bessel), E=0; oscillates at `omega = c J11 / R`.
- **Metric**: resonant-frequency (eigenvalue) error vs `omega`, from a cosine fit of the
  J0-modal amplitude over a few periods. (Field-L2-at-fixed-time is unreliable — it sits near
  a phase node; use the frequency.) **Gotcha**: drop the t=0 diagnostic dump (predates the
  afterinit write -> amplitude 0, degenerates the fit).
- **CI-speed form**: a small single/2-resolution regression assert (e.g. n=48,96, ~2-3
  periods) asserting `ECT freq err` is small and `<< Yee freq err` and the ECT order `> ~1.7`.
  Keep the full N=64/128/256 convergence as the unregistered diagnostic (like
  `cylinder_edge_order.py`). Register via `add_warpx_test` + a checksum.
- **Value**: a permanent regression guard that the EM ECT stays 2nd-order on curves, and the
  reference oracle for Part C (the E-solve fix must bring the hybrid toward this).

## Part B — DONE (2026-06-22): localized to the covered B that curl(B)->J reads

Probe `ampere_curl_order.py` feeds the EXACT analytic mode `By = B1 J0(J11 r/R)`
(written at the true per-component staggered coords) into the hybrid near-wall path
with NO time stepping, via two new bindings (`hybrid_calculate_plasma_current`;
`fill_covered_centers`/`cyl_correction` args added to
`hybrid_apply_eb_boundary_to_face_field`), and measures the radial-shell L2 order of
`J = curl(B)/mu0` (`hybrid_current_fp_plasma`) and `E = eta J` (`Efield_fp`).

**Findings (decisive, grid-independent, asymptotic over N=48/96/192):**
1. SANITY: By matches analytic to roundoff (~5e-19), Jy==0 -> coords/forms correct.
2. The Ampere curl `curl(B)->J` is INTRINSICALLY 2nd order: deep [0,.45] and mid
   [.45,.54] shells converge at order 2.00 on BOTH staggered (ECT) and collocated.
3. The near-wall band [.54,.60) DIVERGES (order ~ -0.4..-0.7, rel-L2 ~18x analytic).
   This is the staircase signature of the NEUMANN mode: `dBy/dr=0` but `By(R)=B1 J0(J11)
   != 0` at the wall, while the covered cells hold `By=0`; the plain-masked curl
   straddling the wall sees a spurious `~By(R)/h` jump -> O(1/h) current. The J/E mirror
   fill only rewrites COVERED cells, so the spurious current on the last FLUID edges
   survives.
4. CEILING A/B: writing the smooth analytic Bessel CONTINUATION into the covered cells
   (covered B exact) -> ALL shells incl the wall recover order 2.00. So the curl is fine;
   **the covered-B value the curl reads is the sole near-wall 1st-order entry.**
5. The level-set mirror as a covered-B fill (fill_covered_centers=True) cuts the near-wall
   error ~20x but does NOT restore order (still diverging) -- the mirror VALUE is only ~3%
   accurate and NON-converging on the curved wall (~1st order); curl differentiates it.
6. The surface-of-revolution RADIAL (cyl) weighting does NOT help B: it is a no-op for the
   AXIAL component (By/Bz along the axis) by construction (mirror_combine scales only the
   radial+azimuthal parts). The radial Jacobian is for flux-type J/rho, not axial B.

=> The fix target is a genuinely **2nd-order-accurate covered-B EXTENSION feeding the
Ampere curl** (NOT a new area-scaled curl: the curl is already 2nd order; NOT the J/E
mirror or the Ohm assembly: E order == J order in vacuum; NOT the radial Jacobian).

## Part B (original gating plan, superseded by the result above)

We know it is the coupling, not the scheme, but the *exact* offending stencil is not yet
pinned, and one clue cautions against assuming: the `eb_cylindrical_correction` radial-mirror
change was **byte-inert on the staggered ECT run**, which suggests the level-set mirror's
cut-edge values may not actually feed the ECT circulation (covered edges carry
`edge_length=0`). So the 1st-order source is plausibly the **plain-masked Ampere curl
`curl(B)->J` at the cut cells** and/or the **algebraic-E assembly**, not (only) the mirror.
Pin it before building, via cheap A/Bs (no new solver):

1. **Mirror vs not**: does disabling/var ying the level-set mirror near the wall change the
   hybrid cylinder order? (The cyl-correction A/B already hints "no" on staggered — confirm.)
2. **Ampere-curl conformality**: instrument `J = curl(B)/mu0` at cut edges vs the analytic
   `curl(J0 mode)`; is J already 1st-order at the wall even though B is 2nd? (Prime suspect.)
3. **Algebraic vs integrated**: compare the hybrid resistive E at the wall to the EM
   integrated E on the same field — isolate whether the *assembly* (not the curl) loses order.
4. **Near-wall error map** of E/J vs the analytic mode (where is the 1st-order concentrated:
   the cut edges, the masked band, the deep interior?).

Output: the specific stencil(s) to make ECT-aware in Part C. This prevents building the wrong
fix (the lesson from "validate before Dey-Mittra").

## Part C — PROTOTYPE VALIDATED (2026-06-22): 2nd-order covered-B extension for the curl

`ampere_curl_order.py --extrap-b` fills the covered B band by a 2nd-order, fluid-ONLY
extension (even radial reflection to the interior image + h-scaled local quadratic
least-squares with rcond truncation) before the Ampere curl. The near-wall current
recovers a clean >=2nd order on BOTH grids, robust across resolution pairs:
staggered wall-band 2.04 (48/96) & 2.79 (96/192); collocated 2.44 & 2.39 (was ~ -0.5).
So the fix is confirmed: **a 2nd-order-accurate covered-B extension feeding curl(B)->J
is sufficient.** Lessons for the C++ port (each found the hard way; see memory
[[cect-convergence-followup]]): quadratic (not linear); reflect to the interior image
and interpolate there (not one-sided extrapolation -> overshoots); non-dimensionalize
the fit by h (else ill-conditioned and worse as h->0); generous stencil + rcond so
degenerate near-wall stencils degrade gracefully; uniform (not Gaussian) weight.

**C++ port plan:** upgrade the level-set mirror gather in `ApplyPECBoundaryToField`
(EBJBoundary.cpp) -- it already has the right STRUCTURE (even reflection to the image
via the true normal) but a ~1st-order (~3%) gather -- to an h-scaled quadratic MLS at
the image with singular-value truncation; fill covered CENTERS in a >=2-cell band; on
the staggered path apply this B fill to `Bfield_fp` BEFORE `CalculateCurrentAmpere`
(production currently does not fill B for the curl there). Gate behind a flag
(byte-identical off). The radial cylindrical Jacobian is NOT the lever (no-op for axial
B). Validate via ampere_curl_order.py and the evolved cylinder_edge_order.py.

## Part C (original) — ECT-aware boundary stencils for the E-solve

Make the hybrid's near-wall E-path conformal/area-scaled, consistent with the ECT B-update,
so the algebraic E fed into the Faraday circulation is 2nd-order at the curved wall. Likely
components (final set fixed by Part B):

- **Conformal Ampere curl `curl(B) -> J`**: introduce an area-scaled cut-cell curl for the
  Ampere current, the dual of `EvolveRhoCartesianECT`'s `curl(E) -> B` (which uses
  `Rho = sum(E*edge_length)/face_area`). For J on a cut edge, weight the B-face differences by
  the conformal `edge_lengths`/`face_areas` metrics already built on the staggered ECT path
  (`WarpXInitData.cpp:1616-1652`). This is **new machinery** — WarpX has the conformal
  curl(E)->B (ECT) but not a conformal curl(B)->J; it is the technical core of the change.
- **Cut-edge E treatment**: where Ohm's law is spurious (covered centers), supply the E that
  the circulation needs from the conformal stencil rather than the 1st-order pointwise mirror
  — i.e. mask + conformal weighting (the EM approach) rather than mirror-fill, OR a 2nd-order
  conformal fill. Keep the resistive/Hall/`grad pe` terms consistent (the same metrics).
- **Gate** behind a flag (default off -> byte-identical) for A/B and a safe PR, like
  `use_conformal_eb`.

## Part D — Validation

- Re-run the hybrid cylinder `cylinder_edge_order.py` (decay-rate eigenvalue + field L2):
  success = the order lifts from ~1.0 toward ~2.0, approaching the EM ECT reference (Part A).
- div(B)=0 probe (the conformal Ampere curl must not inject divergence into the
  continuity/`SyncCurrent`); stability over the substepping; flat-wall regression
  (flag-off byte-identical; flag-on flat = no-op / 2.79 preserved on the square).
- GPU portability and the 3D/XZ guards (RZ is already cylindrical — separate).

## Risks / open questions

- **Mechanism not fully pinned** (Part B gates Part C). If Part B shows the mirror (not the
  Ampere curl) is the source, Part C shifts to a conformal/2nd-order E/J fill instead of a new
  conformal curl.
- A conformal Ampere curl is the **dual-mesh** operation; getting the discrete duality right
  (so div/continuity and the C̃=C^T FIT identity hold) is the main correctness risk — this is
  exactly where Zagorodnov's FIT material-matrix view (refs in hand) is the guide.
- The hybrid substeps (RKF45) and the Hall/`grad pe` terms add cut-cell stencils beyond the
  Ampere curl; the order is capped by the *worst* of them, so Part B must check all near-wall
  contributions, not just `curl(B)`.
- Effort: Part A small; Part B cheap (diagnostics); **Part C is the substantial C++** (new
  conformal operator + plumbing + tests).
