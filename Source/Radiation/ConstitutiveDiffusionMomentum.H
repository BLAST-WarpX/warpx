/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_CONSTITUTIVE_DIFFUSION_MOMENTUM_H
#define WARPX_RADIATION_CONSTITUTIVE_DIFFUSION_MOMENTUM_H

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <string>

namespace warpx::radiation
{

struct ConstitutiveDiffusionMomentumReport
{
    bool valid = false;
    bool failure = false;
    std::string reason;
};

//! \brief Compute the material impulse of the approved conservative FLD reduction
//!        on Cartesian 1D, 2D XZ, 3D, or native RZ meshes.
//!
//! Inputs use cell-integrated units: E and q=c*p are in joules, while
//! outward_face_stress_resultant is the face/time integrated pressure resultant
//! S=sum_faces integral_dt pressure.n area, in kg m/s. For each D cell,
//! output_impulse is I=(q_old-q_new)/c-S in kg m/s. K cells return zero.
//! RZ uses work-conjugate [cPr,cLz/rbar,cPz]; S must already subtract the
//! isotropic radial hoop contribution. The same algebra then preserves axial
//! and angular inventories. Its Euclidean cone is an additional conservative
//! screen in conjugate coordinates, not a physical Cartesian Lorentz cone.
//! Old inventory obeys |q_old|<=E_old; the new isotropic pressure closure
//! additionally requires |q_new|<=E_new/sqrt(3). These necessary moment checks
//! do not establish a physical diffusion ordering or transport invariance.
//! All inputs are read-only; output storage must not overlap any input.
//! Collective rejection leaves output including ghosts unchanged; acceptance
//! writes valid cells only. The caller owns accepted q persistence/checkpoint
//! metadata and the coupled native RequestedWork iteration.
//! Admitted old inventory may be off the new constitutive manifold: its projection
//! onto q_new produces a finite O(q_old-Q) impulse even as dt approaches zero. This
//! is declared instantaneous/stiff-limit constitutive relaxation, not finite-time
//! scattering or proof of causal transient-layer closure. Native collar/kinetic
//! fallback remains necessary for states not justified by this reduction.
bool TryComputeConstitutiveDiffusionMomentum (
    amrex::MultiFab const& old_energy,
    amrex::MultiFab const& old_q,
    amrex::MultiFab const& new_energy,
    amrex::MultiFab const& new_q,
    amrex::MultiFab const& outward_face_stress_resultant,
    amrex::iMultiFab const& diffusion_mask,
    amrex::Geometry const& geometry,
    amrex::MultiFab& output_impulse,
    ConstitutiveDiffusionMomentumReport& report);

} // namespace warpx::radiation

#endif // WARPX_RADIATION_CONSTITUTIVE_DIFFUSION_MOMENTUM_H
