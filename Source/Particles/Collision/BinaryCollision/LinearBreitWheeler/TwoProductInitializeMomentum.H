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
     * in a pair-production event. Here only two products are produced.
     * (In this case, conservation of energy and momentum determines
     * the amplitude of the momentum of the particles exactly.)
     * We assume that the emission of the product is isotropic in the center-of-momentum frame.
     *
     * @param[in] soa1_in struct of array data of the first colliding photon species
     * @param[in] soa2_in struct of array data of the second colliding photon species
     * @param[out] soa1_out struct of array data of the first product species
     * @param[out] soa2_out struct of array data of the first product species
     * @param[in] idx1_in index of first colliding macrophoton
     * @param[in] idx2_in index of second colliding macrophoton
     * @param[in] idx1_out_start index of first macrophoton
     * @param[in] idx2_out_start index of second macrophoton
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

        TwoProductComputeProductMomenta(
            soa1_in.m_rdata[PIdx::ux][idx1_in],
            soa1_in.m_rdata[PIdx::uy][idx1_in],
            soa1_in.m_rdata[PIdx::uz][idx1_in],
            soa2_in.m_rdata[PIdx::ux][idx2_in],
            soa2_in.m_rdata[PIdx::uy][idx2_in],
            soa2_in.m_rdata[PIdx::uz][idx2_in],
            ux1_out, uy1_out, uz1_out,
            ux2_out, uy2_out, uz2_out,
            engine);

        // Fill momentum of product species (note that we actually
        // create 4 products, 2 at the position of each incident particle)
        soa1_out.m_rdata[PIdx::ux][idx1_out_start] = ux1_out;
        soa1_out.m_rdata[PIdx::uy][idx1_out_start] = uy1_out;
        soa1_out.m_rdata[PIdx::uz][idx1_out_start] = uz1_out;
        soa1_out.m_rdata[PIdx::ux][idx1_out_start + 1] = ux1_out;
        soa1_out.m_rdata[PIdx::uy][idx1_out_start + 1] = uy1_out;
        soa1_out.m_rdata[PIdx::uz][idx1_out_start + 1] = uz1_out;
        soa2_out.m_rdata[PIdx::ux][idx2_out_start] = ux2_out;
        soa2_out.m_rdata[PIdx::uy][idx2_out_start] = uy2_out;
        soa2_out.m_rdata[PIdx::uz][idx2_out_start] = uz2_out;
        soa2_out.m_rdata[PIdx::ux][idx2_out_start + 1] = ux2_out;
        soa2_out.m_rdata[PIdx::uy][idx2_out_start + 1] = uy2_out;
        soa2_out.m_rdata[PIdx::uz][idx2_out_start + 1] = uz2_out;
    }
}

#endif // TWO_PRODUCT_INITIALIZE_MOMENTUM_H
