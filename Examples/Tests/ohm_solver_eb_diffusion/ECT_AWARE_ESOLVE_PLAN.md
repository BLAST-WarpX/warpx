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

## Part B — Localize the hybrid 1st-order E-path source (gating; cheap; MUST precede C)

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

## Part C — ECT-aware boundary stencils for the E-solve (the change, contingent on B)

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
