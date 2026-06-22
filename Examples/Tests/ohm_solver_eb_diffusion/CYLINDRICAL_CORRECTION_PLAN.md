# `eb_cylindrical_correction` — radial mirror correction (piece ③)

Branch: `eb_cylindrical_correction` (off `ohms_law_conformal_EB`).
Scope of THIS implementation: **piece ③ only** — the radial-metric correction to
the level-set mirror fill of J, ρ, and B. (Pieces ① arc-edge and ② face-area are
deferred; see the planning verdict below.) Opt-in flag, default off → flag-off is
byte-identical.

## Why only piece ③ (planning verdict)

A planning pass established that, of the user's three ideas:
- **① arc-length on the circulation edges is ~null (O(h³))**: the edges the ECT
  circulates are straight Cartesian grid edges, already exact; the wall-following
  contour contributes nothing (tangential-E = 0 on the PEC). Skip.
- **② cut-face chord→sector area is a real O(h/R)** correction but feeds the
  enlarged-cell borrowing (itself chord-based) and risks injecting div(B); modest
  expected payoff. Deferred behind ③.
- **③ radial mirror correction is the real lever**: the level-set normal is
  already radial/accurate on a smooth cylinder; the error is that the mirror does
  a *planar* reflection and assigns the mirrored J/ρ/B strength with flat-plane
  weights, missing the cylindrical-metric Jacobian. It is O(h/R), self-contained,
  and ports to BOTH the staggered and collocated grids (the collocated B wall
  treatment IS this mirror), so it may also stabilize the collocated mirror's
  high-resolution blow-up.

## The correction (derivation)

The mirror fills a band/covered point by reflecting the field from an image point
in the fluid. For component `c` the current value is
```
field_c = w_n·(n·F_im)·n_c  +  w_t·(F_im,c − (n·F_im)·n_c)
```
where `n = nv` is the unit boundary normal (radial on a cylinder), `(n·F_im)·n_c`
is the **normal (radial)** part projected onto axis `c`, the remainder is the
**tangential** part, and the parity weights are `w_n,w_t ∈ {1, s/d_im}` (odd =
`s/d_im`, the linear-to-zero-at-wall profile). This is exact for a FLAT wall.

For a wall that is a **surface of revolution about a symmetry axis** (default Z),
decompose the tangential part further into **azimuthal** (in the transverse
plane) and **axial** (along the symmetry axis). The divergence/curl operators in
cylindrical coordinates carry the metric factor `r`: the natural odd/even
variables near the wall are the **r-weighted** flux quantities `r·F_radial` and
`r·F_azimuthal` (e.g. div B = (1/r)∂(r B_r)/∂r + …; Faraday couples r·E_θ). The
purely **axial** component carries no radial metric. Reflecting the r-weighted
variable along the radial ray from the image radius `r_im` to the fill radius
`r_e` therefore scales the radial and azimuthal parts by the **radial Jacobian**
```
λ = r_im / r_e            (r = transverse radius about the symmetry axis)
```
while the axial part keeps factor 1:
```
field_c = w_n·λ·(radial)_c  +  w_t·( λ·(azimuthal)_c + 1·(axial)_c )
```
With `λ = 1` (flat wall, or flag off) this is **algebraically identical** to the
planar formula — verified term by term, so the implementation reduces to the
existing expression bit-for-bit when the flag is off.

For the **scalar** ρ (`ApplyEBBoundaryToNodalScalar`), the same r-weighting gives
`ρ(fill) = λ·(profile)·ρ(image)`, i.e. multiply the mirrored scalar by λ.

This is a **hypothesis to validate**, not a proven 2nd-order scheme: it corrects
the leading O(h/R) metric error in the reflection magnitude. It does NOT correct
the image-point *position* (planar vs inverse-point differ at O(h²/R) — higher
order, skipped). If the cylinder edge-order does not lift, the form (the power of
λ per component, the ρ factor) is the first thing to revisit.

## Integration

- Flag: `hybrid_pic_model.eb_cylindrical_correction` (bool, default false) +
  `eb_cyl_axis` (`x|y|z`, default `z`). Requires `use_conformal_eb` + 3D + EB.
  Members `m_eb_cylindrical_correction`, `m_eb_cyl_axis` on HybridPICModel.
- `EBJBoundary.cpp`: a device helper `mirror_combine(...)` replaces the inline
  parity expression in all three vector spots (direct pass, cascade, Jacobi) and
  applies λ when cyl is on; λ computed per-point from the image/fill transverse
  radii (3D only). The scalar fill multiplies by λ.
- `ApplyPECBoundaryToField` / `ApplyEBBoundaryToNodalScalar` gain trailing
  `bool eb_cyl=false, int eb_cyl_axis=2` params (defaulted → existing callers
  unaffected). The hybrid passes `m_eb_cylindrical_correction`/`m_eb_cyl_axis` at
  the J, E, B, and ρ mirror call sites.
- Assumes the symmetry axis passes through the transverse origin (the cylinder is
  centered at x=y=0 for axis=z); documented; generalizing to an offset axis is a
  later sub-param.

## Validation

- `cylinder_edge_order.py`: add a "conformal ECT + cyl-correction" (staggered) and
  a "mirror + cyl-correction" (collocated) config; run N=96/192/384; success = the
  order rises above the ~1.0 baseline (toward 2.0 is the stretch goal), and the
  collocated mirror stays finite at N=384.
- Flat-wall regression: flag-off byte-identical (gated); flag-on on the rotated
  square must be a near-no-op (λ→1 as R→∞) and must not regress the 2.79 order.
- div(B) probe on the collocated B fill (the metric scaling must not inject div B).
