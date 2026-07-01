# Scoping Document — `DielectricEmbeddedBoundary` (second EB for a dielectric liner)

**Author:** S. Eric Clark · **Date:** 2026-07-01 · **Status:** go/no-go review
**Branch context:** `eb_cylindrical_correction` · **Target:** WarpX hybrid-PIC EB machinery

---

## 1. Decision summary

**Feasible: YES-WITH-CAVEATS.** Holding a *second, independent* AMReX embedded boundary (its own `EB2::IndexSpace` + `EBFArrayBoxFactory`) alongside the primary metal/PEC EB is architecturally supported with **no changes to AMReX** and only **additive** changes to WarpX. AMReX's `EB2::IndexSpace` is a static *stack* (not a singleton), each `EB2::Build` pushes a self-contained per-geometry instance, and `makeEBFabFactory(const EB2::IndexSpace*, geom, ba, dm, ngrow, support)` (`AMReX_EBFabFactory.H:167-171`) binds a factory to a *caller-supplied* index space without ever reading `top()`. All the cut-cell interface data Eric wants (area fractions, boundary centroid/normal/area, volume fraction, cell flags) are first-class factory getters (`AMReX_EBFabFactory.H:73-108`) that only require `EBSupport::full`. The precedent already exists in-tree: PML holds its own second `EBFArrayBoxFactory` (`PML.cpp:829`, `PML.H:225-226`).

**Single biggest risk — the global `EB2::IndexSpace::top()` stack.** Every `EB2::Build` rotates a new geometry to `top()`, and WarpX has two live consumers that read `top()` implicitly: `WarpX::ComputeDistanceToEB` (`WarpXInitEB.cpp:124`) and the primary factory rebuild on **regrid** (`WarpXRegrid.cpp:231`, `WarpX.cpp:2345`) plus PML (`PML.cpp:829`). If the dielectric EB is built after `InitEB` and then a regrid fires, the *primary metal* field factory would be silently rebuilt against the *dielectric* geometry — a real correctness bug for any regridding run. Mitigation is well-defined (thread explicit `IndexSpace*` pointers, never rely on `top()` in new code; optionally convert `ComputeDistanceToEB` to derive the index space from the passed factory), but it is the thing that must be gotten right and is the reason this is "with-caveats" rather than a clean "yes."

**Second-order caveat (physics, not plumbing):** exposing dielectric geometry is the *easy* half. There is **no dielectric/ε/surface-charge field BC anywhere in WarpX today** — the EB E-BC is pure PEC (tangential-E → 0) via `mirror_combine` (`EBJBoundary.cpp:710`), which has no ε-jump / D-continuity parity. A new field-BC apply variant is net-new work and is where the real modeling uncertainty lives.

---

## 2. Proposed design — `DielectricEmbeddedBoundary` header sketch

The class *owns* its geometry, factory, distance field, and masks; it never touches the primary EB's storage. It mirrors `WarpX::InitEB`/`ComputeDistanceToEB` but **captures the `IndexSpace*` explicitly** and builds its factory from that pointer.

```cpp
// Source/EmbeddedBoundary/DielectricEmbeddedBoundary.H
#ifndef WARPX_DIELECTRIC_EMBEDDED_BOUNDARY_H_
#define WARPX_DIELECTRIC_EMBEDDED_BOUNDARY_H_

#include <AMReX_EB2.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiCutFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>

#include <memory>
#include <string>

/** A second, query-only embedded boundary describing a dielectric liner.
 *
 *  Distinct from the primary metal/PEC EB (WarpX::fieldEBFactory). Holds its
 *  own EB2::IndexSpace (captured, NOT top()) and EBFArrayBoxFactory so the
 *  dielectric surface has its own resolved cut-cell geometry: area fractions,
 *  boundary centroid/normal/area, volume fraction, cell flags. Also owns a
 *  signed-distance field and (optionally) eb_update-style masks derived from
 *  ITS OWN factory, for use by dielectric-wall field BCs / pressure BC /
 *  particle scraping.
 */
class DielectricEmbeddedBoundary
{
public:
    /** Build the dielectric EB from an implicit-function string, exactly like
     *  warpx.eb_implicit_function -> ParserIF -> EB2::makeShop -> EB2::Build,
     *  but capture the pushed IndexSpace* immediately (see .cpp).
     *
     *  @param implicit_function  parser string in (x,y,z), same XZ/RZ (x,0,z)
     *                            remap convention as ParserIF (WarpXInitEB.cpp:58-62)
     *  @param nlevs              number of AMR levels
     *  @param geoms/ba/dm        MUST match the field grids so the two EBs are
     *                            index-aligned box-for-box
     *  @param ngrow              guard cells for the cut-cell metadata
     *  @param max_coarsening     small for query-only; large only if it feeds an MG solver
     */
    DielectricEmbeddedBoundary (std::string const& implicit_function,
                                int nlevs,
                                amrex::Vector<amrex::Geometry> const& geoms,
                                amrex::Vector<amrex::BoxArray> const& ba,
                                amrex::Vector<amrex::DistributionMapping> const& dm,
                                amrex::IntVect const& ngrow,
                                int max_coarsening_level);

    // Rebuild factory/distance/masks after a regrid (must be hooked into the
    // WarpX regrid path; see integration §4). The IndexSpace* persists across
    // regrids — only the factory (bound to ba/dm) needs rebuilding.
    void remake (int nlevs,
                 amrex::Vector<amrex::BoxArray> const& ba,
                 amrex::Vector<amrex::DistributionMapping> const& dm);

    // ---- Cut-cell interface accessors (forward the AMReX factory getters) ----
    [[nodiscard]] amrex::EBFArrayBoxFactory const& factory (int lev) const
        { return *m_factory[lev]; }

    // Whole-level containers (index by MFIter/box index; MultiCutFab is SPARSE
    // -> guard every access with ok(mfi)).
    [[nodiscard]] amrex::MultiFab      const& volFrac    (int lev) const { return m_factory[lev]->getVolFrac(); }
    [[nodiscard]] amrex::MultiCutFab   const& bndryCent  (int lev) const { return m_factory[lev]->getBndryCent(); }   // surface point, cell-relative dx units
    [[nodiscard]] amrex::MultiCutFab   const& bndryNormal(int lev) const { return m_factory[lev]->getBndryNormal(); } // outward unit normal
    [[nodiscard]] amrex::MultiCutFab   const& bndryArea  (int lev) const { return m_factory[lev]->getBndryArea(); }   // facet area fraction
    [[nodiscard]] amrex::Array<amrex::MultiCutFab const*,AMREX_SPACEDIM>
                                       areaFrac   (int lev) const { return m_factory[lev]->getAreaFrac(); }   // apx/apy/apz in [0,1]
    [[nodiscard]] amrex::Array<amrex::MultiCutFab const*,AMREX_SPACEDIM>
                                       faceCent   (int lev) const { return m_factory[lev]->getFaceCent(); }
    [[nodiscard]] amrex::FabArray<amrex::EBCellFlagFab> const&
                                       cellFlags  (int lev) const { return m_factory[lev]->getMultiEBCellFlagFab(); }

    // Device-side aggregated views for ParallelFor kernels (capture by value).
    [[nodiscard]] amrex::EBDataArrays  ebDataArrays (int lev) const { return m_factory[lev]->getEBDataArrays(); }
    [[nodiscard]] amrex::EBData        ebData (int lev, amrex::MFIter const& mfi) const
        { return m_factory[lev]->getEBData(mfi); }

    // Signed distance to the dielectric surface (nodal), owned by this class,
    // NOT FieldType::distance_to_eb. Feed directly to scrapeParticlesAtEB and
    // to a dielectric-parity field BC.
    [[nodiscard]] amrex::MultiFab const& distance (int lev) const { return *m_distance[lev]; }

    // Optional: eb_update-style masks derived from THIS factory (for a
    // dielectric-wall E/J/B BC that mirrors the PEC mask machinery).
    [[nodiscard]] amrex::Array<amrex::iMultiFab const*,3> updateE (int lev) const;
    [[nodiscard]] amrex::Array<amrex::iMultiFab const*,3> updateB (int lev) const;

    // Convenience per-cell query (host); on device use ebData(...).get<...>().
    [[nodiscard]] bool isCut (int lev, amrex::MFIter const& mfi,
                              amrex::IntVect const& iv) const;

    [[nodiscard]] amrex::EB2::IndexSpace const* indexSpace () const { return m_index_space; }

private:
    // Captured at Build time; owned by the global static EB2 stack, valid until
    // EB2::Finalize (amrex shutdown). Store the POINTER, never a copy.
    amrex::EB2::IndexSpace const*                              m_index_space = nullptr;

    // Query-only factory built via makeEBFabFactory(m_index_space, ...), one per level.
    amrex::Vector<std::unique_ptr<amrex::EBFArrayBoxFactory>>  m_factory;

    // Owned distance + (optional) masks, parallel to WarpX's single-EB storage.
    amrex::Vector<std::unique_ptr<amrex::MultiFab>>            m_distance;
    amrex::Vector<amrex::Array<std::unique_ptr<amrex::iMultiFab>,3>> m_eb_update_E;
    amrex::Vector<amrex::Array<std::unique_ptr<amrex::iMultiFab>,3>> m_eb_update_B;

    std::string m_implicit_function;
    amrex::Vector<amrex::Geometry> m_geoms;
    amrex::IntVect m_ngrow;
    int m_max_coarsening_level = 0;
};
#endif
```

**Constructor body (essential sequence — the `.cpp`):**

```cpp
// 1. Parse implicit function exactly like WarpXInitEB.cpp:93-95
auto parser = utils::parser::makeParser(implicit_function, {"x","y","z"});
ParserIF const pif(parser.compile<3>());
auto gshop = amrex::EB2::makeShop(pif, parser);

// 2. Build -> PUSHES a second IndexSpace onto the global stack (WarpXInitEB.cpp:102 analogue)
amrex::EB2::Build(gshop, geoms[maxlev], maxlev, m_max_coarsening_level);

// 3. CAPTURE the pushed pointer IMMEDIATELY (do not rely on top() later)
m_index_space = amrex::EB2::TopIndexSpace();          // AMReX_EB2.H:65-66

// 4. Factory from the CAPTURED index space (order-independent; never reads top())
for (int lev=0; lev<nlevs; ++lev) {
    m_factory[lev] = amrex::makeEBFabFactory(
        m_index_space, geoms[lev], ba[lev], dm[lev],
        {ngrow[0],ngrow[1],ngrow[2]}, amrex::EBSupport::full);   // AMReX_EBFabFactory.cpp:237-246
}

// 5. Signed distance off THIS level (explicit-Level overload; NOT the top()-based one)
for (int lev=0; lev<nlevs; ++lev) {
    auto const& eb_level = m_index_space->getLevel(geoms[lev]);
    amrex::FillSignedDistance(*m_distance[lev], eb_level, *m_factory[lev], 1); // AMReX_EB_utils.H:59
    // nodal-sync FillBoundary, as WarpXInitEB.cpp:142-144
}
```

Interface data is exposed by simply **forwarding the AMReX getters** (`getAreaFrac`/`getBndryCent`/`getBndryNormal`/`getBndryArea`/`getVolFrac`/`getMultiEBCellFlagFab`, `AMReX_EBFabFactory.H:73-108`) plus the device-side `getEBDataArrays()`/`getEBData(mfi)` (`:135-138`). No new AMReX plumbing.

---

## 3. AMReX feasibility

**Can two EB IndexSpaces/factories coexist? YES (verified from AMReX source in `build_eb/_deps/fetchedamrex-src/`).**

- `EB2::IndexSpace` is a **static stack**: `static Vector<std::unique_ptr<IndexSpace>> m_instance` (`AMReX_EB2.H:62`, def `AMReX_EB2.cpp:24`), with `push`/`erase`/`pop`/`clear`/`top`/`size`. `top()` returns `m_instance.back()` (`AMReX_EB2.H:51`). It can hold N>1 instances concurrently.
- Each `EB2::Build(gshop, geom, ...)` constructs a self-contained `IndexSpaceImp<G>` (own `m_gslevel`/`m_geom`/`m_domain`, `AMReX_EB2.H:119-122`) and pushes it (`AMReX_EB2.H:160-197`). Two EBs built from different implicit functions on the **same** `Geometry` each key their cut-cell data by the same domain box with **no shared mutable state** (`AMReX_EB2_IndexSpaceI.H:99-119`).
- `makeEBFabFactory(const EB2::IndexSpace*, geom, ba, dm, ngrow, support)` (`AMReX_EBFabFactory.cpp:237-246`) does `index_space->getLevel(a_geom)` and **never reads `top()`** — order-independent and safe against the primary EB.
- The factory owns its cut-cell metadata via `std::shared_ptr<EBDataCollection> m_ebdc` + a raw `EB2::Level const* m_parent` (`AMReX_EBFabFactory.H:144-145`). Query-only use never calls `create()` (no field FABs allocated), so cost = one `EBSupport::full` metadata set over `ba/dm`.
- In-tree precedent: PML already holds a second `EBFArrayBoxFactory` in one run (`PML.cpp:829`, `PML.H:225-226`).

**Exact build call sequence:** `makeShop(pif, parser)` → `EB2::Build(gshop, geom, req_crse, max_crse)` → `m_index_space = EB2::TopIndexSpace()` (capture) → `makeEBFabFactory(m_index_space, geom, ba, dm, ngrow, EBSupport::full)` per level → `FillSignedDistance(mf, index_space->getLevel(geom), factory, 1)`.

**Global-state / lifetime hazards & management:**

| Hazard | Where | Management |
|---|---|---|
| `top()` rotates to last-built geometry | `AMReX_EB2.H:51` | Capture `IndexSpace*` at build; **never** rely on `top()` in new code. |
| WarpX regrid rebuilds primary factory via `top()`-reading overload | `WarpXRegrid.cpp:231`, `WarpX.cpp:2345` | Build dielectric EB, but ensure regrid path pins the *primary* to its own captured `IndexSpace*` (see §4 change 1). Highest-priority correctness item. |
| `ComputeDistanceToEB` reads `top()` | `WarpXInitEB.cpp:124` | For the dielectric, use the explicit-Level `FillSignedDistance` (`AMReX_EB_utils.H:59`). Optionally refactor primary path to derive from `fieldEBFactory(lev).getEBIndexSpace()`. |
| Lifetime: IndexSpaces freed only at `EB2::Finalize` (amrex shutdown, `AMReX_EB2.cpp:45-51`) | — | Captured `const EB2::IndexSpace*` is stable for the whole run. Store pointer, never a copy. Do not `erase()` it (dangles factory `Level*`). |
| Memory: doubles EB build + one full cut-cell metadata set; coarse hierarchy duplicated | `EB2::Build` `max_coarsening` | Pass a **small** `max_coarsening_level` for query-only (the primary uses `maxLevel()+20` only for MG). |
| GPU: each factory builds its own `Gpu::DeviceVector<Array4>` | `AMReX_EBFabFactory.cpp:24-92` | Independent; no shared device state, no new race. |
| Two surfaces sharing ONE cell | AMReX per-cell model | AMReX single-EB "multi-valued" means multiple VoFs of the **same** surface, not two different surfaces. The design is clean only where the liner **stands off** from the metal; coincident/crossing surfaces are outside AMReX's model. |

---

## 4. WarpX integration points (ordered)

WarpX currently assumes **exactly one** EB everywhere: one build (`WarpXInitEB.cpp:76`), one per-level factory (`m_field_factory`, `WarpX.H:1468`; `fieldEBFactory` hard-casts it, `WarpX.H:928-930`), one `FieldType::distance_to_eb`, one pair of `eb_update_E`/`eb_update_B` masks, one input param `warpx.eb_implicit_function`. The dielectric EB does **not** reuse any of these; it holds parallel storage and only *feeds geometry* into apply paths.

1. **EB build site — `WarpX::InitEB` / ctor.** *Change:* add a `warpx.dielectric_eb_implicit_function` input; after `InitEB` (`WarpX.cpp:333`), construct `DielectricEmbeddedBoundary` (a new WarpX member). *Single-EB assumption:* `InitEB` builds one geometry and runs in the ctor before grids exist — dielectric factory construction must follow `AllocLevelData` (`WarpX.cpp:2301`) like the primary. **Hook the regrid path** (`WarpXRegrid.cpp:231`) to call `DielectricEmbeddedBoundary::remake(...)` or the dielectric goes stale after regrid — *and* ensure the primary factory rebuild there is pinned to the metal `IndexSpace*` (this is the biggest-risk change).

2. **Fields.** *No change to how field MultiFabs are allocated.* Field MFs are built with the single primary factory (`WarpX.cpp:2345`); the dielectric EB **cannot** retroactively make existing field MFs cut-aware — it only supplies geometry/metrics/masks. This is a hard boundary of the design and is acceptable for a field-BC/material model (not a new cut-cell solver).

3. **Hybrid E / J / B wall BC — `ApplyPECBoundaryToField`** (`EBJBoundary.H:222`, call sites `HybridPICModel.cpp:926` (E), `:774` (J), `:604`/`:1737` (B)). *Change:* substitute the dielectric distance + dielectric masks, **and add a new non-PEC parity mode.** `mirror_combine` (`EBJBoundary.cpp:710`) only encodes PEC parities (tangential-E→0) via `normal_odd`; there is **no ε-jump / D-continuity / surface-charge path**. This new apply variant is the main net-new physics work and the modeling unknown.

4. **Electron-pressure BC — `ApplyEBBoundaryToNodalScalar`** (`EBJBoundary.H:266`, call `HybridPICModel.cpp:968` with `odd=false`). *Change:* pass the **dielectric** distance field. `odd=false` = even/Neumann (zero normal gradient) is already the natural dielectric-wall condition — **reusable as-is**, cleanest field hook.

5. **Particle scraper — `scrapeParticlesAtEB`** (`ParticleScraper.H:156`, call sites `MultiParticleContainer.cpp:1243`, `WarpXParticleContainer.cpp:347`). *Change:* pass the dielectric distance MF + a dielectric callable. The API already reads phi from the **passed** distance field and computes the normal on-the-fly (`ParticleScraper.H:176,201`); the standoff `offset` param is documented precedent (`:192-195`). **Zero core change** — the cleanest drop-in for optional particle reflection at the liner.

6. **(Only if needed) ECT curl area fractions** (`HybridPICSolveE.cpp:602-615`). A dielectric *liner* is a field-BC/material problem, not a new cut-metric problem; do **not** touch the ECT curl unless the dielectric must also alter curl(B)=J. Out of scope for the first cut.

---

## 5. Effort & risk

| Piece | Files touched | Rough LOC | Size | Notes / prototype-first? |
|---|---|---|---|---|
| `DielectricEmbeddedBoundary` class (build + factory + distance + accessors) | +2 new (`.H`/`.cpp`), `Make.package`, `CMakeLists.txt` | ~250–400 | **M** | **Prototype first.** Pure additive; no behavior change until wired. |
| Input param + ctor/regrid wiring; pin primary factory to captured `IndexSpace*` | `WarpX.cpp`, `WarpXRegrid.cpp`, `WarpXInitEB.cpp` | ~60–120 | **M** | **Highest risk** (regrid/`top()` correctness). Do the `top()`-decoupling carefully; add a regridding test. |
| Particle scraper hook (dielectric distance + reflection callable) | `MultiParticleContainer.cpp` / call site | ~30–80 | **S** | **Cleanest.** No core change; good first *end-to-end* demonstrator. |
| Electron-pressure Neumann BC on dielectric distance | `HybridPICModel.cpp` | ~20–50 | **S** | Reuses `odd=false`. |
| Dielectric E/J/B field BC — new non-PEC parity (ε / D-continuity / surface charge) | `EBJBoundary.H/.cpp`, `HybridPICModel.cpp` | ~200–500+ | **L** | **The real work + modeling uncertainty.** Prototype the *math* (D-continuity/surface-charge discretization) before committing. |
| Dielectric masks (`MarkUpdate*` against 2nd factory) | reuse `EmbeddedBoundaryInit.H` builders | ~40–100 | **S–M** | Builders already take factory as a param (`EmbeddedBoundaryInit.H:40-177`) — hand them the 2nd factory unchanged. |

**Prototype-first order:** (a) class + query getters, (b) scraper hook (fastest end-to-end proof a second EB drives real behavior), (c) pressure Neumann, (d) *then* the field-BC physics.

---

## 6. Open questions for the morning

1. **Store fields or geometry-only?** Recommendation is geometry-only (query factory + owned distance/masks). Fields stay on the single primary factory. Confirm no dielectric quantity needs its own cut-aware *field* MultiFab (would break the "fields use one factory" contract).
2. **Regrid support required?** If production runs regrid, the primary-factory-vs-`top()` fix (§4.1) is mandatory and is the top risk. If runs are single-grid/static, we can defer the regrid hardening and ship faster.
3. **RZ / dims.** EB is 2D/3D only (`WarpXInitEB.cpp:82-84`); dielectric parser must follow the XZ/RZ `(x,0,z)` remap (`WarpXInitEB.cpp:58-62`); RZ maps r=√(x²+y²). Is RZ in scope for v1, or 3D/XZ first?
4. **Composition with the primary metal EB.** Must the liner **stand off** from the metal (AMReX cannot represent two surfaces in one cell)? Define the minimum standoff (≥1 cell) and whether the two masks are allowed to overlap.
5. **Field-BC physics target.** Full D-continuity + surface charge + electron-pressure Neumann + reflection, or a reduced first model (e.g. Neumann pressure + reflection only, no ε-jump)? This sets whether the **L** field-BC item is in v1.
6. **MR / multi-box.** Multi-level EB interp defaults to `TopIndexSpaceIfPresent()` (`AMReX_FillPatchUtil_I.H:805,831`) — only a hazard for AMR EB runs. Multi-box distance seams need the same nodal-sync as the primary (`WarpXInitEB.cpp:142-144`). Confirm target is single-level (as most hybrid runs) so MR FillPatch is not in scope.
7. **`max_coarsening_level` for the dielectric** — small (query-only) vs large (only if it ever feeds an MG solver). Recommend small to halve the memory hit.

---

## 7. Recommendation

**PROTOTYPE-FIRST — proceed to a bounded PoC, not a full implementation.** The AMReX feasibility is proven and the design is clean and additive; the two unknowns (regrid/`top()` correctness and the net-new dielectric field-BC physics) are exactly what a prototype should retire before committing to the **L** field-BC work.

**Minimal first prototype step (1–2 days):** Build the `DielectricEmbeddedBoundary` class that (1) does the capture-`IndexSpace*` `EB2::Build`, (2) constructs a query factory via `makeEBFabFactory(m_index_space, ...)`, (3) fills its own signed-distance field, and (4) wire it into **`scrapeParticlesAtEB` only** (the zero-core-change hook, `ParticleScraper.H:156`) on the existing `ohm_solver_eb_diffusion`/cylinder testbed. Success criterion: particles scrape/reflect on a dielectric surface that is *geometrically distinct* from the metal EB, with the primary metal EB machinery (distance, masks, PEC BCs) **provably unchanged** (diff a step-0 dump against a baseline run). That single end-to-end result validates the coexistence, the `top()` decoupling, and index-alignment of the two factories — after which the field-BC physics can be scoped as its own follow-on.
