/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_RADIATIONMOMENTUM_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_RADIATIONMOMENTUM_H_

#include "ReducedDiags.H"

#include <AMReX_Array.H>

/** Cell-integrated material, pending, and escaped-boundary radiation impulse
 * accounting.
 *
 * Components are Cartesian in Cartesian geometry and (r,theta,z) in RCYL/RZ.
 * The radial entry is an azimuthally integrated scalar impulse ledger, not the
 * globally conserved Cartesian momentum of a complete cylinder. Pending
 * streaming/diffusion entries are cell- or particle-owned impulses not yet
 * representable in the configured particle momentum precision. Coupled moment
 * transport can additionally report the instantaneous radiation-field momentum;
 * this is distinct from cumulative impulse transferred to material.
 */
class RadiationMomentum : public ReducedDiags
{
public:
    explicit RadiationMomentum (std::string const& rd_name);

    void ComputeDiags (int step) final;
    void WriteCheckpointData (std::string const& dir) final;
    void ReadCheckpointData (std::string const& dir) final;

private:
    bool m_include_moment_inventory = false;
    amrex::GpuArray<amrex::Real, 3> m_cumulative_material_impulse{
        0.0, 0.0, 0.0};
    amrex::GpuArray<amrex::Real, 3> m_cumulative_boundary_impulse{
        0.0, 0.0, 0.0};
    amrex::GpuArray<amrex::Real, 3> m_cumulative_streaming_boundary_impulse{
        0.0, 0.0, 0.0};
    int m_last_accumulated_step = -1;
};

#endif
