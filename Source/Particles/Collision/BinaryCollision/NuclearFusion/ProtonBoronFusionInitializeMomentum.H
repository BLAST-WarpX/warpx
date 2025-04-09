/* Copyright 2021 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_PROTON_BORON_FUSION_INITIALIZE_MOMENTUM_H
#define WARPX_PROTON_BORON_FUSION_INITIALIZE_MOMENTUM_H

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
    using SoaData_type = typename WarpXParticleContainer::ParticleTileType::ParticleTileDataType;
    using ParticleType = typename WarpXParticleContainer::ParticleType;
    using ParticleTileType = typename WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = typename ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
    using index_type = typename ParticleBins::index_type;

    /**
     * \brief This function initializes the momentum of the alpha particles produced from
     * proton-boron fusion. The momentum is initialized by assuming that the fusion of a proton
     * with a boron nucleus into 3 alphas takes place in two steps. In the first step, the proton
     * and the boron fuse into an excited beryllium nucleus and an alpha particle. In the second step, the
     * excited beryllium decays into two alpha particles. The first step produces ~5.56 MeV of kinetic
     * energy while the second step produces ~3.12 MeV of kinetic energy. This two-step process
     * is considered to be the dominant process of proton+boron fusion into alphas (see
     * D.R. Tilley et al. / Nuclear Physics A 745 (2004) 155–362).
     * For each step, we assume in this function that the particles are emitted isotropically in
     * the corresponding center of mass frame (center of mass frame of proton + boron for the
     * creation of first alpha+beryllium and rest frame of beryllium for the creation of second and
     * third alphas). This isotropic assumption is exact for the second step but is only an
     * approximation for the first step.
     *
     * @param[in] soa_1 struct of array data of the first colliding species (can be either proton
     * or boron)
     * @param[in] soa_2 struct of array data of the second colliding species (can be either proton
     * or boron)
     * @param[out] soa_alpha struct of array data of the alpha species
     * @param[in] idx_1 index of first colliding macroparticle
     * @param[in] idx_2 index of second colliding macroparticle
     * @param[in] idx_alpha_start index of first produced alpha macroparticle
     * @param[in] m1 mass of first colliding species
     * @param[in] m2 mass of second colliding species
     * @param[in] engine the random engine
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void ProtonBoronFusionInitializeMomentum (
                            const SoaData_type& soa_1, const SoaData_type& soa_2,
                            SoaData_type& soa_alpha,
                            const index_type& idx_1, const index_type& idx_2,
                            const index_type& idx_alpha_start,
                            const amrex::ParticleReal& m1, const amrex::ParticleReal& m2,
                            const amrex::RandomEngine& engine)
    {
        // General notations in this function:
        //     x_sq denotes the square of x
        //     x_Bestar denotes the value of x in the Beryllium rest frame

        using namespace amrex::literals;

        constexpr amrex::ParticleReal mev_to_joule = PhysConst::q_e*1.e6_prt;
        // Energy produced in the fusion reaction proton + boron11 -> Beryllium8 + alpha
        // cf. https://doi.org/10.1016/j.radphyschem.2022.110727
        // Dominant reaction channel is p + B11 -> alpha_1 + 8Be* -> alpha_1 + alpha_11 + alpha_12 + 8.68 MeV
        // cf. Kelley et al., (2017) http://dx.doi.org/10.1016/j.nuclphysa.2017.07.015
        constexpr amrex::ParticleReal E_fusion = 5.55610759_prt*mev_to_joule;
        // Energy produced when Beryllium8 decays into two alphas
        // cf. Kelley et al., (2017) http://dx.doi.org/10.1016/j.nuclphysa.2017.07.015
        constexpr amrex::ParticleReal E_decay = 3.12600414_prt*mev_to_joule;

        // The constexprs ma_sq and mBe_sq underflow in single precision because we use SI units,
        // which can cause compilation to fail or generate a warning, so we're explicitly setting
        // them as double. Note that nuclear fusion module does not currently work with single
        // precision anyways.
        constexpr double m_alpha = PhysConst::m_u * 4.00260325413_prt;
        // mass of the Be8 excited state (3.03 MeV above ground state)
        constexpr double m_beryllium = PhysConst::m_u*(8.0053095729_prt+0.00325283863_prt);
        constexpr double mBe_sq = m_beryllium*m_beryllium;
        constexpr amrex::ParticleReal c_sq = PhysConst::c * PhysConst::c;

        // Compute the resulting momenta of the alpha and beryllium particles
        // produced in the fusion reaction
        amrex::ParticleReal ux_alpha1 = 0.0, uy_alpha1 = 0.0, uz_alpha1 = 0.0;
        amrex::ParticleReal ux_Be = 0.0, uy_Be = 0.0, uz_Be = 0.0;
        TwoProductFusionComputeProductMomenta (
            soa_1.m_rdata[PIdx::ux][idx_1],
            soa_1.m_rdata[PIdx::uy][idx_1],
            soa_1.m_rdata[PIdx::uz][idx_1], m1,
            soa_2.m_rdata[PIdx::ux][idx_2],
            soa_2.m_rdata[PIdx::uy][idx_2],
            soa_2.m_rdata[PIdx::uz][idx_2], m2,
            ux_alpha1, uy_alpha1, uz_alpha1, static_cast<amrex::ParticleReal>(m_alpha),
            ux_Be, uy_Be, uz_Be, static_cast<amrex::ParticleReal>(m_beryllium),
            E_fusion, engine);

        // Compute momentum of beryllium in lab frame
        const amrex::ParticleReal px_Be = static_cast<amrex::ParticleReal>(m_beryllium) * ux_Be;
        const amrex::ParticleReal py_Be = static_cast<amrex::ParticleReal>(m_beryllium) * uy_Be;
        const amrex::ParticleReal pz_Be = static_cast<amrex::ParticleReal>(m_beryllium) * uz_Be;

        // Compute momentum norm of second and third alphas in Beryllium rest frame
        // Factor 0.5 is here because each alpha only gets half of the decay energy
        constexpr amrex::ParticleReal gamma_Bestar = (1._prt + 0.5_prt*E_decay/(m_alpha*c_sq));
        constexpr amrex::ParticleReal gamma_Bestar_sq_minus_one = gamma_Bestar*gamma_Bestar - 1._prt;
        const amrex::ParticleReal p_Bestar = static_cast<amrex::ParticleReal>(m_alpha)*PhysConst::c*std::sqrt(gamma_Bestar_sq_minus_one);

        // Compute momentum of second alpha in Beryllium rest frame, assuming isotropic distribution
        amrex::ParticleReal px_Bestar, py_Bestar, pz_Bestar;
        ParticleUtils::RandomizeVelocity(px_Bestar, py_Bestar, pz_Bestar, p_Bestar, engine);

        // Next step is to convert momentum of second alpha to lab frame
        amrex::ParticleReal px_alpha2, py_alpha2, pz_alpha2;
        // Preliminary calculation: compute Beryllium velocity v_Be
        const amrex::ParticleReal p_Be_sq   = px_Be*px_Be + py_Be*py_Be + pz_Be*pz_Be;
        const amrex::ParticleReal g_Be      = std::sqrt(1._prt + p_Be_sq / (static_cast<amrex::ParticleReal>(mBe_sq)*c_sq));
        const amrex::ParticleReal mg_Be     = static_cast<amrex::ParticleReal>(m_beryllium)*g_Be;
        const amrex::ParticleReal v_Bex     = px_Be / mg_Be;
        const amrex::ParticleReal v_Bey     = py_Be / mg_Be;
        const amrex::ParticleReal v_Bez     = pz_Be / mg_Be;
        const amrex::ParticleReal v_Be_sq   = v_Bex*v_Bex + v_Bey*v_Bey + v_Bez*v_Bez;

        // Convert momentum of second alpha to lab frame, using equation (13) of F. Perez et al.,
        // Phys.Plasmas.19.083104 (2012)
        if ( v_Be_sq > std::numeric_limits<amrex::ParticleReal>::min() )
        {
            const amrex::ParticleReal vcDps = v_Bex*px_Bestar + v_Bey*py_Bestar + v_Bez*pz_Bestar;
            const amrex::ParticleReal factor0 = (g_Be-1._prt)/v_Be_sq;
            const amrex::ParticleReal factor = factor0*vcDps + static_cast<amrex::ParticleReal>(m_alpha)*gamma_Bestar*g_Be;
            px_alpha2 = px_Bestar + v_Bex * factor;
            py_alpha2 = py_Bestar + v_Bey * factor;
            pz_alpha2 = pz_Bestar + v_Bez * factor;
        }
        else // If Beryllium velocity is zero, we are already in the lab frame
        {
            px_alpha2 = px_Bestar;
            py_alpha2 = py_Bestar;
            pz_alpha2 = pz_Bestar;
        }

        // Compute momentum of third alpha in lab frame, using total momentum conservation
        const amrex::ParticleReal px_alpha3 = px_Be - px_alpha2;
        const amrex::ParticleReal py_alpha3 = py_Be - py_alpha2;
        const amrex::ParticleReal pz_alpha3 = pz_Be - pz_alpha2;

        // Fill alpha species momentum data with the computed momentum (note that we actually
        // create 6 alphas, 3 at the position of the proton and 3 at the position of the boron, so
        // each computed momentum is used twice)
        soa_alpha.m_rdata[PIdx::ux][idx_alpha_start] = ux_alpha1;
        soa_alpha.m_rdata[PIdx::uy][idx_alpha_start] = uy_alpha1;
        soa_alpha.m_rdata[PIdx::uz][idx_alpha_start] = uz_alpha1;
        soa_alpha.m_rdata[PIdx::ux][idx_alpha_start + 1] = ux_alpha1;
        soa_alpha.m_rdata[PIdx::uy][idx_alpha_start + 1] = uy_alpha1;
        soa_alpha.m_rdata[PIdx::uz][idx_alpha_start + 1] = uz_alpha1;
        soa_alpha.m_rdata[PIdx::ux][idx_alpha_start + 2] = px_alpha2/static_cast<amrex::ParticleReal>(m_alpha);
        soa_alpha.m_rdata[PIdx::uy][idx_alpha_start + 2] = py_alpha2/static_cast<amrex::ParticleReal>(m_alpha);
        soa_alpha.m_rdata[PIdx::uz][idx_alpha_start + 2] = pz_alpha2/static_cast<amrex::ParticleReal>(m_alpha);
        soa_alpha.m_rdata[PIdx::ux][idx_alpha_start + 3] = px_alpha2/static_cast<amrex::ParticleReal>(m_alpha);
        soa_alpha.m_rdata[PIdx::uy][idx_alpha_start + 3] = py_alpha2/static_cast<amrex::ParticleReal>(m_alpha);
        soa_alpha.m_rdata[PIdx::uz][idx_alpha_start + 3] = pz_alpha2/static_cast<amrex::ParticleReal>(m_alpha);
        soa_alpha.m_rdata[PIdx::ux][idx_alpha_start + 4] = px_alpha3/static_cast<amrex::ParticleReal>(m_alpha);
        soa_alpha.m_rdata[PIdx::uy][idx_alpha_start + 4] = py_alpha3/static_cast<amrex::ParticleReal>(m_alpha);
        soa_alpha.m_rdata[PIdx::uz][idx_alpha_start + 4] = pz_alpha3/static_cast<amrex::ParticleReal>(m_alpha);
        soa_alpha.m_rdata[PIdx::ux][idx_alpha_start + 5] = px_alpha3/static_cast<amrex::ParticleReal>(m_alpha);
        soa_alpha.m_rdata[PIdx::uy][idx_alpha_start + 5] = py_alpha3/static_cast<amrex::ParticleReal>(m_alpha);
        soa_alpha.m_rdata[PIdx::uz][idx_alpha_start + 5] = pz_alpha3/static_cast<amrex::ParticleReal>(m_alpha);
    }

}

#endif // WARPX_PROTON_BORON_FUSION_INITIALIZE_MOMENTUM_H
