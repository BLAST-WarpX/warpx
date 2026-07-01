# PR #6994 — Conformal Embedded Boundary + Dielectric Standoff for the Hybrid Ohm's-Law Solver

**Draft writeup / current-state record.** Companion to `PR_EVIDENCE.md` (per-change justification).

## 1. What this delivers

A working **conformal embedded-boundary (EB)** capability for the hybrid Ohm's-law
(kinetic-ion / fluid-electron) field solver — a curved metal wall represented as a
masked/mirrored PEC boundary inside the algebraic Ohm solve — plus a **dielectric
particle standoff** that holds the plasma off the wall. Supported on both grid modes:
**collocated (nodal, level-set EB)** and **Yee (staggered, enlarged-cell/ECT EB)**.

The headline validation: on the `ohm_solver_plasma_cylinder_liftoff` theta-pinch
reversal deck (n=128, nppc=500), **both grid modes run stably through the full field
ramp + compression** (12000 steps ≈ 2.3× the ramp) with a 3-cell standoff.

## 2. Stability matrix (corrected)

Deck: `--equilibrium-b --holmstrom --resistive-only-partial`, rtol 1e-3, 12000 steps,
TAU_RAMP = 8e-6 s ≈ step 5185 (peak reversed field 0.85 T).

| config | standoff | result |
|---|---|---|
| collocated + conformal | **3 cells** | **STEP 12000, clean** (t=1.85e-5 s) |
| Yee (masked) + conformal | **3 cells** | **STEP 12000, clean** |
| Yee + LSQ wall current | **3 cells** | **STEP 12000, clean** |
| any mode | **0** | unstable (RKF45 substep runaway at the wall) |

**The dielectric standoff (≥3 cells) is the stability lever.** With the plasma held
off the wall, the O(1) near-wall current sheet is removed and the run is stable through
the whole liftoff. Mechanism confirmed independently by the divergence measurement (§4).

> **Methodology note (reproducibility).** Run PICMI decks with **plain `python deck.py --flag ...` (NO leading `--`)**. The deck uses `argparse.parse_known_args()`; a leading `--` (an ipython idiom) makes plain-python argparse treat all following tokens as positional and silently drop every flag to defaults. Verify the actual config in `warpx_used_inputs` (`max_step`, `warpx.grid_type`, `eb_particle_scrape_offset`). An earlier batch of runs was invalidated by this and gave a spurious "both modes choke" result.

## 3. Performance — cost of conformal walls (local A6000, steady-state, n=128 nppc=500)

All configs at 2 RKF45 substeps/step (early regime), so these compare per-substage compute:

| config | s/step | vs staircase baseline |
|---|---|---|
| baseline (no-conformal staircase EB) | 0.080 | 1.00× |
| conformal collocated | 0.060 | 0.75× |
| conformal Yee (masked) | 0.054 | 0.68× |
| conformal Yee + LSQ | 0.392 | 4.9× |

- **Conformal walls are not a per-step cost** vs the staircase baseline (measured *faster*
  here at equal substep count).
- **The LSQ accurate wall current is ~7.3× the conformal masked-Yee** — pure per-substage
  cost (hand-rolled Cholesky per wall edge; identical substep counts confirm it is compute,
  not stiffness). Heavy, opt-in, and separable.

## 4. Accuracy / divergence (validate_lsq, deck-matched, n=128)

Wall-band `div(J)` (RMS, `H·divJ/J`), 200 steps:

| config | J-scale (rms Jx) | **wall div(J)** | interior div |
|---|---|---|---|
| Yee+LSQ, standoff=0 (plasma at wall) | 1.15e6 | **1.02** | 6.6e-18 |
| Yee+LSQ, standoff=3 | 5.1e4 | **0.053** | 1.4e-16 |
| Yee+LSQ, standoff=3, +8 div-clean loops | 5.1e4 | **0.053** (unchanged) | 1.4e-16 |

- **Standoff cuts the wall divergence ~20×** (removes the plasma-at-wall current sheet,
  J-scale 1.15e6 → 5.1e4).
- **The Phase-2b cut-metric div-clean is a no-op on the solver's plain divergence**
  (0 vs 8 loops identical) — it zeroes the cut-metric divergence, an orthogonal quantity.
- **Substep-E Marder div-clean actively destabilizes** the standoff run.
- → **Zero correction loops needed for the MVP.**

## 5. MVP scope + PR split (see PR_EVIDENCE.md for per-change detail)

**Minimum-viable core (keep together):** em-side-ect wiring subset → eb-j-boundary
Layer 1 (PEC edge BC) → hybrid-eb-solve CORE (covered-J zeroing, mirror fills, `abs(rho)`
eta, EB-aware Hall mask) → **dielectric standoff** → external-A Δ1 → thin PICMI slice →
trimmed collocated test. Ships **both grid modes** (collocated + Yee masked).

**Split into follow-up PRs:**
- **PR-D — accurate LSQ wall current + Phase-2b div-clean.** The accuracy payoff, but
  ~7× per-step cost and *not* a stability requirement (§3, §4). Opt-in, byte-identical off.
- **PR-C — Marder div-clean.** Default-off; measured to *destabilize* the standoff → split.
- **PR-B — isotropic operators** (m=4 suppression, default-true numerics).
- **PR-E — covered-B curl-fill.** Self-declared deprecated → **drop candidate.**
- **PR-F — EM-ECT hardening:** F2 (GPU atomic borrow-race fix) land early; F1 rename;
  F3 multi-box seam-sync (biggest review risk, no deck coverage).
- **PR-A** external-A diagnostic; **PR-G** unrelated bundled fixes.

**Two must-fix items before the PR is measured:**
1. **`eb_hall_mask`/`InterpMasked` is uncommitted** (working-tree only) → invisible to the
   PR diff. Commit it (MVP core) or the reviewed diff is wrong.
2. **CI gap:** the shipped test is **collocated-only, no Yee, no standoff.** Add a
   Yee + standoff stability variant.

## 6. Current state for the implicit-solver transition

- The **explicit RKF45 hybrid EB solve is validated stable through the full liftoff** with
  the standoff (both grid modes) — a solid base to hand to the implicit development.
- The standoff is a physical model of a quartz liner (plasma reaching the liner is
  absorbed). Production value: **3 cells** (grid-scale rule; 1–2 cells insufficient because
  the near-wall Hall/curl coupling reaches ~2 cells).
- Deferred to implicit / follow-up: accurate wall current (LSQ) at acceptable cost;
  div-consistent + accurate wall curl in one explicit operator (shown impossible explicitly —
  needs implicit curl-curl); multi-box seam-sync coverage; convergence-order push (wall
  currently ~1st-order-plus, not 2nd).

## 7. Artifacts

- Side-by-side liftoff video: `liftoff_out/liftoff_mvp_coll_vs_yee.mp4` (collocated vs
  Yee-masked, both standoff=3, full ramp + compression + bounces).
- Perf harness: `Docs/eb_fill_review/perf_sweep.sh`; div harness: `validate_lsq.py`;
  video: `tools_liftoff_compare_movie.py`.
- Per-change evidence: `Docs/eb_fill_review/PR_EVIDENCE.md`.
