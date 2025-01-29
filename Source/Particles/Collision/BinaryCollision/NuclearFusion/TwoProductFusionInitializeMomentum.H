/* Copyright 2022 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_TWO_PRODUCT_FUSION_INITIALIZE_MOMENTUM_H
#define WARPX_TWO_PRODUCT_FUSION_INITIALIZE_MOMENTUM_H

#include "TwoProductFusionUtil.H"
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
     * in a fusion event where only two products are produced.
     * (In this case, conservation of energy and momentum determines
     * the amplitude of the momentum of the particles exactly.)
     * We assume that the emission of the product is isotropic in the center-of-mass frame
     *
     * @param[in] soa1_in struct of array data of the first colliding species
     * @param[in] soa2_in struct of array data of the second colliding species
     * @param[out] soa1_out struct of array data of the first product species
     * @param[out] soa2_out struct of array data of the first product species
     * @param[in] idx1_in index of first colliding macroparticle
     * @param[in] idx2_in index of second colliding macroparticle
     * @param[in] idx1_out_start index of first product macroparticle
     * @param[in] idx2_out_start index of second product macroparticle
     * @param[in] m1_in mass of first colliding species
     * @param[in] m2_in mass of second colliding species
     * @param[in] m1_out mass of first product species
     * @param[in] m2_out mass of second product species
     * @param[in] E_fusion energy released in the fusion reaction
     * @param[in] engine the random engine
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void TwoProductFusionInitializeMomentum (
                            const SoaData_type& soa1_in, const SoaData_type& soa2_in,
                            SoaData_type& soa1_out, SoaData_type& soa2_out,
                            const index_type& idx1_in, const index_type& idx2_in,
                            const index_type& idx1_out_start, const index_type& idx2_out_start,
                            const amrex::ParticleReal& m1_in, const amrex::ParticleReal& m2_in,
                            const amrex::ParticleReal& m1_out, const amrex::ParticleReal& m2_out,
                            const amrex::ParticleReal& E_fusion,
                            const amrex::RandomEngine& engine)
    {
        using namespace amrex::literals;

        amrex::ParticleReal ux1_out = 0.0_prt, uy1_out = 0.0_prt, uz1_out = 0.0_prt;
        amrex::ParticleReal ux2_out = 0.0_prt, uy2_out = 0.0_prt, uz2_out = 0.0_prt;

        TwoProductFusionComputeProductMomenta(
            soa1_in.m_rdata[PIdx::ux][idx1_in],
            soa1_in.m_rdata[PIdx::uy][idx1_in],
            soa1_in.m_rdata[PIdx::uz][idx1_in], m1_in,
            soa2_in.m_rdata[PIdx::ux][idx2_in],
            soa2_in.m_rdata[PIdx::uy][idx2_in],
            soa2_in.m_rdata[PIdx::uz][idx2_in], m2_in,
            ux1_out, uy1_out, uz1_out, m1_out,
            ux2_out, uy2_out, uz2_out, m2_out,
            E_fusion,
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

#endif // WARPX_TWO_PRODUCT_FUSION_INITIALIZE_MOMENTUM_H
