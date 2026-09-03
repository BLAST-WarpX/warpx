/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_RADIATIONENERGY_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_RADIATIONENERGY_H_

#include "ReducedDiags.H"

#include <string>

/** Global energy accounting for hybrid multigroup radiation transport. */
class RadiationEnergy : public ReducedDiags
{
public:
    explicit RadiationEnergy (std::string const& rd_name);

    /** Sum streaming, diffusion and current-step material-exchange energy. */
    void ComputeDiags (int step) final;

    /** Preserve the cumulative material exchange across checkpoint/restart. */
    void WriteCheckpointData (std::string const& dir) final;
    void ReadCheckpointData (std::string const& dir) final;

private:
    std::string m_photon_species;
    int m_num_groups = 1;
    amrex::Real m_cumulative_material_exchange = 0.0;
    amrex::Real m_cumulative_boundary_energy_loss = 0.0;
    amrex::Real m_cumulative_streaming_boundary_energy_loss = 0.0;
    amrex::Real m_cumulative_diffusion_boundary_energy_loss = 0.0;
    amrex::Real m_cumulative_numerical_energy_residual = 0.0;
    int m_last_accumulated_step = -1;
};

#endif
