/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_COLLIDED_DIFFUSION_ADMISSION_H_
#define WARPX_RADIATION_COLLIDED_DIFFUSION_ADMISSION_H_

#include "Particles/WarpXParticleContainer.H"
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <cmath>
#include <limits>
#include <string>

namespace warpx::radiation
{
struct CollidedDiffusionAdmissionComponents
{
    amrex::GpuArray<int, 3> initial_u{-1,-1,-1};
};

struct CollidedDiffusionAdmissionControls
{
    amrex::Real max_flux_fraction = amrex::Real(1)/3;
    amrex::Real max_pressure_anisotropy = amrex::Real(0.1);
};

struct CollidedDiffusionAdmissionReport
{
    // In RZ, per-cell inventory is [E,cPr,cLz/rbar,cPz]. Its component sums
    // are not a global Cartesian four-vector; reconstruct Lz with each rbar.
    // The admission flux screen in these work-conjugate coordinates is a
    // conservative additional restriction, not a physical Lorentz cone.
    bool valid = false;
    std::string reason;
    amrex::Long eligible = 0; //!< Changed native rays in desired bulk cells.
    amrex::Long admitted = 0;
    amrex::Long retained = 0; //!< All remaining valid native rays.
    amrex::GpuArray<amrex::Real, 4> actual_admitted_four_inventory{};
    amrex::MultiFab admitted_four_inventory; //!< Cell E/q in the declared coordinate basis.
};

/** Explicit registration before InitData; the only schema-changing operation. */
[[nodiscard]] CollidedDiffusionAdmissionComponents
RegisterCollidedDiffusionAdmissionState (WarpXParticleContainer& species);

/** Snapshot PRIVATE candidate before NativePhotonPathReplay::Prepare.
 * No other momentum-changing operation may intervene before admission. The
 * physical path preserves u exactly when no collision occurred; unequal u
 * proves a collision, while equal-u rays are conservatively retained.
 * Cartesian and RZ replay reset markers to final u for wall-only histories, so reflection
 * alone cannot supply this proof. Physical-collision histories retain markers.
 */
[[nodiscard]] CollidedDiffusionAdmissionReport SnapshotCollidedDiffusionAdmission (
    WarpXParticleContainer::Base& photons, CollidedDiffusionAdmissionComponents const& components);

/** Collective census conversion; actual four-inventory transfers with no kick.
 * Supports double-precision, level-zero periodic Cartesian 1D Z, 2D XZ, and
 * 3D XYZ meshes, or native RZ with reflecting radial/periodic axial boundaries.
 * RZ converts actual cLz to qtheta=cLz/rbar and tests pressure in the physical
 * local radial/theta/z basis. Both native AMReX coordinate tags 0/1 are allowed.
 * Photons must be redistributed and belong to the supplied level-zero layout.
 * New leakage births must follow this operation. All validation precedes commit;
 * numerical failure leaves private photons and fields unchanged. Allocation or
 * process failure during commit requires discarding the complete caller trial.
 * Successful field writes affect valid cells only; caller owns ghost filling.
 */
[[nodiscard]] CollidedDiffusionAdmissionReport TryAdmitCollidedPhotons (
    WarpXParticleContainer::Base& photons, CollidedDiffusionAdmissionComponents const& components,
    amrex::MultiFab& private_energy, amrex::MultiFab& private_momentum,
    amrex::iMultiFab const& desired_bulk_mask, amrex::Geometry const& geometry,
    CollidedDiffusionAdmissionControls const& controls = {});

struct CollidedDiffusionCohortMoments
{
    amrex::Real background_energy = 0;
    amrex::GpuArray<amrex::Real, 3> background_momentum{};
    amrex::Real cohort_energy = 0;
    amrex::GpuArray<amrex::Real, 3> cohort_momentum{};
    amrex::GpuArray<amrex::Real, 6> cohort_pressure{}; //!< xx,yy,zz,xy,xz,yz in J.
};

[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
ShouldAdmitCollidedDiffusionCohort (CollidedDiffusionCohortMoments const& m,
    CollidedDiffusionAdmissionControls const& controls) noexcept
{
    using amrex::Math::isfinite;
    constexpr amrex::Real eps = 32*std::numeric_limits<amrex::Real>::epsilon();
    if (!isfinite(controls.max_flux_fraction) || controls.max_flux_fraction < 0 ||
        controls.max_flux_fraction > amrex::Real(1)/3 ||
        !isfinite(controls.max_pressure_anisotropy) || controls.max_pressure_anisotropy <= 0 ||
        controls.max_pressure_anisotropy > amrex::Real(0.1) ||
        !isfinite(m.background_energy) || m.background_energy < 0 ||
        !isfinite(m.cohort_energy) || m.cohort_energy <= 0) { return false; }
    amrex::Real const total = m.background_energy + m.cohort_energy;
    if (!isfinite(total) || total <= 0) { return false; }
    amrex::Real flux2 = 0, background2 = 0, cohort2 = 0, anisotropy2 = 0, trace = 0;
    for (int d = 0; d < 3; ++d) {
        if (!isfinite(m.background_momentum[d]) || !isfinite(m.cohort_momentum[d]) ||
            (m.background_energy == 0 && m.background_momentum[d] != 0)) { return false; }
        amrex::Real const b = m.background_energy > 0 ?
            m.background_momentum[d]/m.background_energy : 0;
        amrex::Real const a = m.cohort_momentum[d]/m.cohort_energy;
        amrex::Real const f = m.background_momentum[d]/total + m.cohort_momentum[d]/total;
        background2 += b*b; cohort2 += a*a; flux2 += f*f;
        if (!isfinite(m.cohort_pressure[d]) || m.cohort_pressure[d] < 0) { return false; }
        trace += m.cohort_pressure[d]/m.cohort_energy;
        amrex::Real const deviation = (m.background_energy/total)/3 +
            m.cohort_pressure[d]/total - amrex::Real(1)/3;
        anisotropy2 += deviation*deviation;
    }
    for (int d = 3; d < 6; ++d) {
        if (!isfinite(m.cohort_pressure[d])) { return false; }
        amrex::Real const ratio = m.cohort_pressure[d]/total;
        anisotropy2 += 2*ratio*ratio;
    }
    return isfinite(background2) && isfinite(cohort2) && isfinite(flux2) &&
        isfinite(anisotropy2) && std::sqrt(background2) <= 1+eps &&
        std::sqrt(cohort2) <= 1+eps && std::abs(trace-1) <= eps &&
        std::sqrt(flux2) <= controls.max_flux_fraction+eps &&
        std::sqrt(anisotropy2) <= controls.max_pressure_anisotropy+eps;
}
}
#endif
