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
   no reflection capability.
2. `MultiParticleContainer.cpp:1130` — `ScrapeParticlesAtEB()` hard-codes
   `ParticleBoundaryProcess::Absorb()` for every species.
3. `WarpXEvolve.cpp:690-696` — domain BCs are applied first
   (`ApplyBoundaryConditions`), then EB scraping unconditionally absorbs.

The scraper template in `ParticleScraper.H` was *already* designed for pluggable
functors — it just never had a reflection implementation.

## Changes in This Branch

### 1. Bisection-based specular reflection (`ParticleScraper.H`)

Added `reflectParticlesAtEB()`, a new function template that handles curved EB
surfaces correctly using the same bisection algorithm as
`ParticleBoundaryBuffer::FindEmbeddedBoundaryIntersection`. For each particle
that crosses into the EB (`phi < 0`):

1. **Bisect** along the particle trajectory (using `amrex::bisect` + `UpdatePosition`)
   to find the exact contact point where `phi = 0`.
2. **Compute normal** at the contact point (not at the penetrated position).
3. **Reflect velocity** specularly: `v_new = v − 2(v·n̂)n̂`.
4. **Advance** from the contact point with reflected velocity for the remaining
   fraction of the timestep.

This replaces the earlier `ParticleBoundaryProcess::Reflect` functor which used
the simpler approximation `x_new = x − 2·φ·n̂`. That formula is only exact for
planar surfaces; for curved EBs (the primary use case) it accumulates error
proportional to the surface curvature.

Handles all dimension variants: 3D, XZ, RZ (with cylindrical → Cartesian normal
conversion), and 1D_Z.

### 2. Scraper interface extended (`ParticleScraper.H`)

The functor call in `scrapeParticlesAtEB` was changed from:
```cpp
f(ptd, ip, pos, normal, engine);
```
to:
```cpp
f(ptd, ip, pos, normal, phi_value, engine);
```

`phi_value` (the signed distance, negative inside the EB) is passed to the
functor for use by future BCs. `NoOp` and `Absorb` accept and ignore it.

### 3. User-facing configuration

| Interface | Parameter | Values |
|-----------|-----------|--------|
| PICMI Python | `picmi.EmbeddedBoundary(..., particle_boundary_condition='Reflecting')` | `'Absorbing'` (default), `'Reflecting'` |
| Classic input | `warpx.eb_particle_boundary_condition = Reflecting` | `Absorbing` (default), `Reflecting` |

Parsed in `MakeWarpX()` via `query_enum_sloppy`, stored as
`WarpX::eb_particle_boundary` (`ParticleBoundaryType` enum).

A low-priority warning is emitted when the parameter is unspecified, so existing
users are informed of the default absorbing behavior.

### 4. Dispatch (`MultiParticleContainer::ScrapeParticlesAtEB`)

Branches on `WarpX::eb_particle_boundary`:
- `Reflecting` → `reflectParticlesAtEB()` (per-level, with dt and mass)
- anything else → `scrapeParticlesAtEB()` with `ParticleBoundaryProcess::Absorb()`

Injection/reposition call sites (`AddParticles.cpp`, `WarpXParticleContainer.cpp`)
remain `Absorb` — particles created inside the EB should be removed.

### Why reflection is not a functor

Unlike `Absorb` which only needs the particle ID, specular reflection on curved
surfaces requires trajectory data (`dt`, `mass`) and the signed distance array
(`phi`) for bisection — data not available in the functor interface. The
reflection logic therefore lives directly in `reflectParticlesAtEB()` rather than
as a functor passed to `scrapeParticlesAtEB()`.

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
| **Diffuse reflect** | Re-emit with cosine-weighted random direction from surface | `normal` + `engine` generate the random direction; bisection handles position |
| **Thermal** | Rethermalize velocity at wall temperature | `uth` stored as functor struct member; `normal` orients half-Maxwellian; `engine` samples it |
| **Stochastic absorb/reflect** | Velocity-dependent reflection probability | Reflection model parser stored as functor member; velocity read from `ptd`; `engine` for random draw |
| **Accommodation coefficient** | Partial thermalization (mix of specular + diffuse) | `alpha`, `uth` as struct members; everything else available |

Note: BCs involving position correction (diffuse, thermal) would also benefit
from the bisection approach to find the true contact point, similar to what
`reflectParticlesAtEB` does.

### Not sufficient for

| BC Type | Why |
|---------|-----|
| **Secondary electron emission** | Functor modifies one particle in-place; cannot create new particles. Would need a two-phase approach: (1) flag & record collision data, (2) create new particles in a separate pass. |
| **Sputtering / surface chemistry** | Same architectural limitation — requires particle creation. |

These would need a different design (likely a post-scrape particle injection step),
regardless of what data the scraper passes.

## Files Changed

| File | Summary |
|------|---------|
| `Source/EmbeddedBoundary/ParticleBoundaryProcess.H` | Updated `NoOp`/`Absorb` signatures for `phi_value`; removed old `Reflect` functor |
| `Source/EmbeddedBoundary/ParticleScraper.H` | Pass `phi_value` to functor; new `reflectParticlesAtEB()` with bisection |
| `Source/Particles/MultiParticleContainer.cpp` | Dispatch on `WarpX::eb_particle_boundary` |
| `Source/WarpX.H` | New `eb_particle_boundary` static member |
| `Source/WarpX.cpp` | Parse `warpx.eb_particle_boundary_condition` with validation and warning |
| `Python/pywarpx/picmi.py` | `particle_boundary_condition` param on `EmbeddedBoundary` |
