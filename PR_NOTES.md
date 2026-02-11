# EB Particle Boundary Condition — Design Notes for PR

## Problem (Issue #6566)

Embedded boundary particle scraping unconditionally absorbs all particles via a
hard-coded `ParticleBoundaryProcess::Absorb()` functor. Users setting reflecting
particle BCs on the domain walls observe particle loss when an EB is present,
because the EB scraper overrides the domain BC and destroys every particle that
crosses the EB surface.

## Root Cause

Three files form the critical path:

1. `ParticleBoundaryProcess.H` — only `Absorb` and `NoOp` functors existed;
   no `Reflect` functor.
2. `MultiParticleContainer.cpp:1130` — `ScrapeParticlesAtEB()` hard-codes
   `ParticleBoundaryProcess::Absorb()` for every species.
3. `WarpXEvolve.cpp:690-696` — domain BCs are applied first
   (`ApplyBoundaryConditions`), then EB scraping unconditionally absorbs.

The scraper template in `ParticleScraper.H` was *already* designed for pluggable
functors — it just never had a Reflect implementation.

## Changes in This Branch

### 1. Specular Reflect functor (`ParticleBoundaryProcess.H`)

Added `ParticleBoundaryProcess::Reflect` that performs:
- **Position reflection**: `x_new = x − 2·φ·n̂` (mirrors particle across EB surface)
- **Velocity reflection**: `v_new = v − 2(v·n̂)n̂` (reverses normal component)

Handles all dimension variants: 3D, XZ, RZ (with cylindrical ↔ Cartesian velocity
projection), and 1D_Z.

### 2. Scraper interface extended (`ParticleScraper.H`)

The functor call was changed from:
```cpp
f(ptd, ip, pos, normal, engine);
```
to:
```cpp
f(ptd, ip, pos, normal, phi_value, engine);
```

`phi_value` (the signed distance, negative inside the EB) is needed so the
functor can compute the correct reflected position. `NoOp` and `Absorb` accept
and ignore it.

### 3. User-facing configuration

| Interface | Parameter | Values |
|-----------|-----------|--------|
| PICMI Python | `picmi.EmbeddedBoundary(..., particle_boundary_condition='Reflecting')` | `'Absorbing'` (default), `'Reflecting'` |
| Classic input | `warpx.eb_particle_boundary_condition = Reflecting` | `Absorbing` (default), `Reflecting` |

Parsed in `MakeWarpX()` via `query_enum_sloppy`, stored as
`WarpX::eb_particle_boundary` (`ParticleBoundaryType` enum).

A low-priority warning is emitted when the parameter is unspecified, so existing
users are informed of the default absorbing behavior.

### 4. Functor dispatch (`MultiParticleContainer::ScrapeParticlesAtEB`)

Branches on `WarpX::eb_particle_boundary`:
- `Reflecting` → `ParticleBoundaryProcess::Reflect()`
- anything else → `ParticleBoundaryProcess::Absorb()`

Injection/reposition call sites (`AddParticles.cpp`, `WarpXParticleContainer.cpp`)
remain `Absorb` — particles created inside the EB should be removed.

## Functor Interface — Extensibility Assessment

The current scraper-to-functor interface is:

```cpp
void operator()(PData& ptd, int i,
                const RealVect& pos, const RealVect& normal,
                amrex::Real phi_value,
                RandomEngine const& engine)
```

### Sufficient for these future BCs

| BC Type | What it does | Why the interface is sufficient |
|---------|-------------|-------------------------------|
| **Diffuse reflect** | Re-emit with cosine-weighted random direction from surface | `normal` + `engine` generate the random direction; `phi_value` repositions the particle |
| **Thermal** | Rethermalize velocity at wall temperature | `uth` stored as functor struct member; `normal` orients half-Maxwellian; `engine` samples it; `phi_value` repositions |
| **Stochastic absorb/reflect** | Velocity-dependent reflection probability | Reflection model parser stored as functor member; velocity read from `ptd`; `engine` for random draw |
| **Accommodation coefficient** | Partial thermalization (mix of specular + diffuse) | `alpha`, `uth` as struct members; everything else available |

Per-BC configuration data belongs on the **functor struct** (e.g., `Thermal{uth=...}`),
not in the scraper call. The scraper arguments are geometry data; the functor carries
physics parameters. This separation is clean.

### Not sufficient for

| BC Type | Why |
|---------|-----|
| **Secondary electron emission** | Functor modifies one particle in-place; cannot create new particles. Would need a two-phase approach: (1) flag & record collision data, (2) create new particles in a separate pass. |
| **Sputtering / surface chemistry** | Same architectural limitation — requires particle creation. |

These would need a different design (likely a post-scrape particle injection step),
regardless of what data the scraper passes.

### One optional future improvement

`phi_value` approximates the penetration distance at the end of the timestep.
For more precise intersection timing (needed if a BC depends on exact collision
energy), the scraper could additionally pass a `dt_fraction` from bisection — the
`ParticleBoundaryBuffer` already computes this. Not needed for the BCs above, but
noted for completeness.

## Files Changed

| File | Lines | Summary |
|------|-------|---------|
| `Source/EmbeddedBoundary/ParticleBoundaryProcess.H` | +90 | `Reflect` functor; updated `NoOp`/`Absorb` signatures for `phi_value` |
| `Source/EmbeddedBoundary/ParticleScraper.H` | +1 −1 | Pass `phi_value` to functor |
| `Source/Particles/MultiParticleContainer.cpp` | +10 −2 | Dispatch on `WarpX::eb_particle_boundary` |
| `Source/WarpX.H` | +6 | New `eb_particle_boundary` static member |
| `Source/WarpX.cpp` | +22 | Parse `warpx.eb_particle_boundary_condition` with validation and warning |
| `Python/pywarpx/picmi.py` | +17 | `particle_boundary_condition` param on `EmbeddedBoundary` |
