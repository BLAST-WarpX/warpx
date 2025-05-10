/* Copyright 2023 Arianna Formenti
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef TWO_PRODUCT_INITIALIZE_MOMENTUM_H
#define TWO_PRODUCT_INITIALIZE_MOMENTUM_H

#include "TwoProductUtil.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXConst.H"

#include <AMReX_DenseBins.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

namespace {
    // Define shortcuts for frequently-used type names
    using SoaData_type = WarpXParticleContainer::ParticleTileType::ParticleTileDataType;
    using ParticleType = WarpXParticleContainer::ParticleType;
    using ParticleTileType = WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
    using index_type = ParticleBins::index_type;

    /**
     * \brief This function initializes the momentum of the product particles,
     * in a Compton scattering event.
     *
     * @param[in] soa1_in struct of array data of the first colliding species (photon)
     * @param[in] soa2_in struct of array data of the second colliding species (lepton)
     * @param[out] soa1_out struct of array data of the first product species (photon)
     * @param[out] soa2_out struct of array data of the second product species (lepton)
     * @param[in] idx1_in index of first colliding macroparticle (photon)
     * @param[in] idx2_in index of second colliding macroparticle (lepton)
     * @param[in] idx1_out_start index of first product macroparticle (photon)
     * @param[in] idx2_out_start index of second product macroparticle (lepton)
     * @param[in] engine the random engine
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void TwoProductInitializeMomentum (
                            const SoaData_type& soa1_in, const SoaData_type& soa2_in,
                            SoaData_type& soa1_out, SoaData_type& soa2_out,
                            const index_type& idx1_in, const index_type& idx2_in,
                            const index_type& idx1_out_start, const index_type& idx2_out_start,
                            const amrex::RandomEngine& engine)
    {
        using namespace amrex::literals;

        amrex::ParticleReal ux1_out = 0.0_prt, uy1_out = 0.0_prt, uz1_out = 0.0_prt;
        amrex::ParticleReal ux2_out = 0.0_prt, uy2_out = 0.0_prt, uz2_out = 0.0_prt;

        // TODO: Implement angle-dependent Compton scattering (Klein-Nishina cross-section)

        // Fill momentum of product species (note that we actually
        // create 4 products, 2 at the position of each incident particle)
        // - Photon
        soa1_out.m_rdata[PIdx::ux][idx1_out_start] = ux1_out;
        soa1_out.m_rdata[PIdx::uy][idx1_out_start] = uy1_out;
        soa1_out.m_rdata[PIdx::uz][idx1_out_start] = uz1_out;
        soa1_out.m_rdata[PIdx::ux][idx1_out_start + 1] = ux1_out;
        soa1_out.m_rdata[PIdx::uy][idx1_out_start + 1] = uy1_out;
        soa1_out.m_rdata[PIdx::uz][idx1_out_start + 1] = uz1_out;
        // - Electron
        soa2_out.m_rdata[PIdx::ux][idx2_out_start] = ux2_out;
        soa2_out.m_rdata[PIdx::uy][idx2_out_start] = uy2_out;
        soa2_out.m_rdata[PIdx::uz][idx2_out_start] = uz2_out;
        soa2_out.m_rdata[PIdx::ux][idx2_out_start + 1] = ux2_out;
        soa2_out.m_rdata[PIdx::uy][idx2_out_start + 1] = uy2_out;
        soa2_out.m_rdata[PIdx::uz][idx2_out_start + 1] = uz2_out;
    }
}

#endif // TWO_PRODUCT_INITIALIZE_MOMENTUM_H
