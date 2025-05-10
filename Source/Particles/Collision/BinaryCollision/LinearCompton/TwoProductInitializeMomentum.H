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

        constexpr amrex::ParticleReal inv_c = 1._prt / PhysConst::c;
        constexpr amrex::ParticleReal inv_csq = 1._prt / ( PhysConst::c * PhysConst::c );

        // Extract the momenta of the incident particles
        amrex::ParticleReal const photon_ux = soa1_in.m_rdata[PIdx::ux][idx1_in];
        amrex::ParticleReal const photon_uy = soa1_in.m_rdata[PIdx::uy][idx1_in];
        amrex::ParticleReal const photon_uz = soa1_in.m_rdata[PIdx::uz][idx1_in];
        amrex::ParticleReal const lepton_ux = soa2_in.m_rdata[PIdx::ux][idx2_in];
        amrex::ParticleReal const lepton_uy = soa2_in.m_rdata[PIdx::uy][idx2_in];
        amrex::ParticleReal const lepton_uz = soa2_in.m_rdata[PIdx::uz][idx2_in];

        // Transform the momentum of the incoming photon to the rest frame of the lepton
        // - Prepare parameters of the Lorentz transformation
        amrex::ParticleReal const lepton_u2 = (lepton_ux*lepton_ux + lepton_uy*lepton_uy + lepton_uz*lepton_uz);
        amrex::ParticleReal const lepton_u = std::sqrt(lepton_u2);
        amrex::ParticleReal const lepton_gamma = std::sqrt(1.0_prt + lepton_u2*inv_csq);
        amrex::ParticleReal const lepton_beta = lepton_u / lepton_gamma * inv_c;
        amrex::ParticleReal const lepton_nx = lepton_ux / lepton_u;
        amrex::ParticleReal const lepton_ny = lepton_uy / lepton_u;
        amrex::ParticleReal const lepton_nz = lepton_uz / lepton_u;
        // - Perform the Lorentz transformation
        amrex::ParticleReal const photon_u = std::sqrt(photon_ux*photon_ux + photon_uy*photon_uy + photon_uz*photon_uz);
        amrex::ParticleReal const photon_u_parallel = (photon_ux*lepton_nx + photon_uy*lepton_ny + photon_uz*lepton_nz);
        amrex::ParticleReal const photon_u_rest = lepton_gamma * (photon_u - lepton_beta * photon_u_parallel);
        amrex::ParticleReal const photon_u_parallel_rest = lepton_gamma * ( photon_u_parallel - lepton_beta * photon_u );
        amrex::ParticleReal const photon_ux_rest = photon_ux + lepton_nx * (photon_u_parallel_rest - photon_u_parallel);
        amrex::ParticleReal const photon_uy_rest = photon_uy + lepton_ny * (photon_u_parallel_rest - photon_u_parallel);
        amrex::ParticleReal const photon_uz_rest = photon_uz + lepton_nz * (photon_u_parallel_rest - photon_u_parallel);

        // Find cos and sin of the spherical angle that represent
        // the direction of the incoming photon in the rest frame
        // TODO: Check if we could reuse code from the Coulomb scattering code.
        amrex::ParticleReal const cos_theta = photon_uz_rest / photon_u_rest;
        amrex::ParticleReal cos_phi = 0;
        amrex::ParticleReal sin_phi = 0;
        amrex::ParticleReal sin_theta = 0;
        if (cos_theta*cos_theta < 1.0_prt) {
            sin_theta = std::sqrt( 1.0_prt - cos_theta*cos_theta );
            amrex::ParticleReal const inv_photon_rest_uxy = 1._prt / ( sin_theta * photon_u_rest );
            cos_phi = photon_ux_rest * inv_photon_rest_uxy;
            sin_phi = photon_uy_rest * inv_photon_rest_uxy;
        } else {
            sin_theta = 0._prt;
            // Avoid division by 0; provide arbitrary direction
            // for the phi angle (since theta is 0 or pi anyway)
            cos_phi = 1.0_prt;
            sin_phi = 0.0_prt;
        }

        // Draw scattering angle in the rest frame, from the
        // Klein-Nishina cross-section (See Ozmutl, E. N.
        // "Sampling of Angular Distribution in Compton Scattering"
        // Appl. Radiat. Isot. 43, 6, pp. 713-715 (1992))
        amrex::ParticleReal const k = photon_u_rest * inv_c;
        // By convention, in WarpX, the photon momentum u is normalized by the electron mass m_e, so k corresponds to p/(m_e*c)
        amrex::ParticleReal const c0 = 2._prt * (2._prt*k*k + 2._prt*k + 1._prt) / amrex::Math::powi<3>(2._prt*k + 1._prt);
        amrex::ParticleReal const b = (2._prt + c0) / (2._prt - c0);
        amrex::ParticleReal const a = 2._prt*b - 1._prt;
        // Use rejection method to draw x
        bool reject = true;
        while (reject) {
            // - Draw x with an approximate probability distribution
            amrex::ParticleReal const r1 = amrex::Random(engine);
            amrex::ParticleReal const x = b - (b + 1._prt)*std::pow(0.5_prt*c0, r1);
            // - Calculate approximate probability distribution h
            amrex::ParticleReal const h = a/(b-x);
            // - Calculate expected (exact) probability distribution f
            amrex::ParticleReal const factor = 1._prt + k*(1._prt-x);
            amrex::ParticleReal const f = ( (1._prt+x*x)*factor + k*k*(1._prt-x)*(1._prt-x) )/factor*factor*factor;
            // - Keep x according to rejection rule
            amrex::ParticleReal const r2 = amrex::Random(engine);
            if (r2 < f/h) {
                reject = false;
            }
        }

        // Get scattered momentum in the rest frame
        amrex::ParticleReal const new_photon_u_rest = photon_u_rest/( 1._prt + k*(1._prt-x) );
        // - First in a system of axes aligned with the incoming photon
        amrex::ParticleReal const cos_theta_s = x;
        amrex::ParticleReal const sin_theta_s = std::sqrt( 1._prt - x*x );
        amrex::ParticleReal const phi_s = 2._prt*amrex::Math::pi*amrex::Random(engine);
        amrex::ParticleReal const cos_phi_s = std::cos( phi_s );
        amrex::ParticleReal const sin_phi_s = std::sin( phi_s );
        amrex::ParticleReal const new_photon_uX_rest = new_photon_u_rest * sin_theta_s*cos_phi_s;
        amrex::ParticleReal const new_photon_uY_rest = new_photon_u_rest * sin_theta_s*sin_phi_s;
        amrex::ParticleReal const new_photon_uZ_rest = new_photon_u_rest * cos_theta_s;
        // - Then rotate it to the original system of axes
        amrex::ParticleReal const new_photon_ux_rest = sin_theta * cos_phi * new_photon_uZ_rest
                                                     + cos_theta * cos_phi * new_photon_uX_rest
                                                                 - sin_phi * new_photon_uY_rest;
        amrex::ParticleReal const new_photon_uy_rest = sin_theta * sin_phi * new_photon_uZ_rest
                                                     + cos_theta * sin_phi * new_photon_uX_rest
                                                                   + cos_phi * new_photon_uY_rest;
        amrex::ParticleReal const new_photon_uz_rest = cos_theta * new_photon_uZ_rest
                                                     - sin_theta * new_photon_uX_rest;

        // Transform the momentum of the scattered photon back to the lab frame
        // TODO: abstract the Lorentz transformation, or look for an existing WarpX function that does it
        amrex::ParticleReal const new_photon_u_parallel_rest = - (new_photon_ux_rest*lepton_nx + new_photon_uy_rest*lepton_ny + new_photon_uz_rest*lepton_nz);
        amrex::ParticleReal const new_photon_u = lepton_gamma * ( new_photon_u_rest - lepton_beta * new_photon_u_parallel_rest );
        amrex::ParticleReal const new_photon_u_parallel = lepton_gamma * ( new_photon_u_parallel_rest - lepton_beta * new_photon_u_rest );
        amrex::ParticleReal const new_photon_ux = new_photon_ux_rest - lepton_nx * (new_photon_u_parallel - new_photon_u_parallel_rest);
        amrex::ParticleReal const new_photon_uy = new_photon_uy_rest - lepton_ny * (new_photon_u_parallel - new_photon_u_parallel_rest);
        amrex::ParticleReal const new_photon_uz = new_photon_uz_rest - lepton_nz * (new_photon_u_parallel - new_photon_u_parallel_rest);

        // Fill momentum of product species (note that we actually
        // create 4 products, 2 at the position of each incident particle)
        // - Scattered photon
        soa1_out.m_rdata[PIdx::ux][idx1_out_start] = new_photon_ux;
        soa1_out.m_rdata[PIdx::uy][idx1_out_start] = new_photon_uy;
        soa1_out.m_rdata[PIdx::uz][idx1_out_start] = new_photon_uz;
        soa1_out.m_rdata[PIdx::ux][idx1_out_start + 1] = new_photon_ux;
        soa1_out.m_rdata[PIdx::uy][idx1_out_start + 1] = new_photon_uy;
        soa1_out.m_rdata[PIdx::uz][idx1_out_start + 1] = new_photon_uz;
        // - Scattered electron (obtained by conserving momentum)
        soa2_out.m_rdata[PIdx::ux][idx2_out_start] = lepton_ux - (new_photon_ux - photon_ux);
        soa2_out.m_rdata[PIdx::uy][idx2_out_start] = lepton_uy - (new_photon_uy - photon_uy);
        soa2_out.m_rdata[PIdx::uz][idx2_out_start] = lepton_uz - (new_photon_uz - photon_uz);
        soa2_out.m_rdata[PIdx::ux][idx2_out_start + 1] = lepton_ux - (new_photon_ux - photon_ux);
        soa2_out.m_rdata[PIdx::uy][idx2_out_start + 1] = lepton_uy - (new_photon_uy - photon_uy);
        soa2_out.m_rdata[PIdx::uz][idx2_out_start + 1] = lepton_uz - (new_photon_uz - photon_uz);
    }
}

#endif // TWO_PRODUCT_INITIALIZE_MOMENTUM_H
