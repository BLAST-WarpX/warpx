/* Copyright 2021 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_BINARY_COLLISION_UTILS_H_
#define WARPX_BINARY_COLLISION_UTILS_H_

#include <string>

#include "Particles/MultiParticleContainer.H"

#include <AMReX_Math.H>

enum struct CollisionType { DeuteriumTritiumToNeutronHeliumFusion,
                            DeuteriumDeuteriumToProtonTritiumFusion,
                            DeuteriumDeuteriumToNeutronHeliumFusion,
                            DeuteriumHeliumToProtonHeliumFusion,
                            ProtonBoronToAlphasFusion,
                            DSMC,
                            PairwiseCoulomb,
                            Undefined };

enum struct NuclearFusionType {
                                DeuteriumTritiumToNeutronHelium,
                                DeuteriumDeuteriumToProtonTritium,
                                DeuteriumDeuteriumToNeutronHelium,
                                DeuteriumHeliumToProtonHelium,
                                ProtonBoronToAlphas,
                                Undefined };

namespace BinaryCollisionUtils{

    NuclearFusionType get_nuclear_fusion_type (const std::string& collision_name,
                                               MultiParticleContainer const * mypc);

    CollisionType get_collision_type (const std::string& collision_name,
                                      MultiParticleContainer const * mypc);

    CollisionType nuclear_fusion_type_to_collision_type (NuclearFusionType fusion_type);

    /**
     * \brief Return (relativistic) collision energy, collision speed and
     * Lorentz factor for transforming between the lab and center-of-momentum
     * frames.
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void get_collision_parameters (
        const amrex::ParticleReal& u1x, const amrex::ParticleReal& u1y,
        const amrex::ParticleReal& u1z, const amrex::ParticleReal& u2x,
        const amrex::ParticleReal& u2y, const amrex::ParticleReal& u2z,
        const amrex::ParticleReal& m1, const amrex::ParticleReal& m2,
        amrex::ParticleReal& E_kin_COM, amrex::ParticleReal& v_rel_COM,
        amrex::ParticleReal& lab_to_COM_lorentz_factor )
    {
        // General notations in this function:
        //     x_sq denotes the square of x
        //     x_star denotes the value of x in the center of mass frame

        using namespace amrex::literals;
        using namespace amrex::Math;

        constexpr double c_sq = PhysConst::c * PhysConst::c;
        constexpr double inv_csq = 1.0 / c_sq;

        // Cast input parameters to double before computing collision properties
        // This is needed to avoid errors when using single-precision particles
        const auto m1_dbl = static_cast<double>(m1);
        const auto m2_dbl = static_cast<double>(m2);
        const auto u1x_dbl = static_cast<double>(u1x);
        const auto u1y_dbl = static_cast<double>(u1y);
        const auto u1z_dbl = static_cast<double>(u1z);
        const auto u2x_dbl = static_cast<double>(u2x);
        const auto u2y_dbl = static_cast<double>(u2y);
        const auto u2z_dbl = static_cast<double>(u2z);

        const double m1_sq = m1_dbl*m1_dbl;
        const double m2_sq = m2_dbl*m2_dbl;

        // Compute Lorentz factor gamma in the lab frame
        const double g1 = std::sqrt( 1.0 + (u1x_dbl*u1x_dbl + u1y_dbl*u1y_dbl + u1z_dbl*u1z_dbl)*inv_csq );
        const double g2 = std::sqrt( 1.0 + (u2x_dbl*u2x_dbl + u2y_dbl*u2y_dbl + u2z_dbl*u2z_dbl)*inv_csq );

        // Compute momenta
        const double p1x = u1x_dbl * m1_dbl;
        const double p1y = u1y_dbl * m1_dbl;
        const double p1z = u1z_dbl * m1_dbl;
        const double p2x = u2x_dbl * m2_dbl;
        const double p2y = u2y_dbl * m2_dbl;
        const double p2z = u2z_dbl * m2_dbl;

        // Square norm of the total (sum between the two particles) momenta in the lab frame
        const double p_total_sq = powi<2>(p1x + p2x) + powi<2>(p1y + p2y) + powi<2>(p1z + p2z);

        // Total energy in the lab frame
        // Note the use of `double` for energy since this calculation is
        // prone to error with single precision.
        const double E_lab = (m1_dbl*g1 + m2_dbl*g2) * c_sq;
        // Total energy squared in the center of mass frame, calculated using the Lorentz invariance
        // of the four-momentum norm
        const double E_star_sq = E_lab*E_lab - c_sq*p_total_sq;

        // Kinetic energy in the center of mass frame
        const double E_star = std::sqrt(E_star_sq);

        // Cast back to chosen precision for output
        E_kin_COM = static_cast<amrex::ParticleReal>(E_star - (m1_dbl + m2_dbl)*c_sq);

        // Square of the norm of the momentum of one of the particles in the center of mass frame
        // Formula obtained by inverting E^2 = p^2*c^2 + m^2*c^4 in the COM frame for each particle
        // The expression below is specifically written in a form that avoids returning
        // small negative numbers due to machine precision errors, for low-energy particles
        const double E_ratio = E_star/((m1_dbl + m2_dbl)*c_sq);
        const double p_star_sq = m1_dbl*m2_dbl*c_sq * ( powi<2>(E_ratio) - 1.0 )
            + powi<2>(m1_dbl - m2_dbl)*c_sq/4.0 * powi<2>( E_ratio - 1.0/E_ratio);

        // Lorentz factors in the center of mass frame
        const double g1_star = std::sqrt(1.0 + p_star_sq / (m1_sq*c_sq));
        const double g2_star = std::sqrt(1.0 + p_star_sq / (m2_sq*c_sq));

        // relative velocity in the center of mass frame, cast back to chosen precision
        v_rel_COM = static_cast<amrex::ParticleReal>(std::sqrt(p_star_sq) * (1.0/(m1_dbl*g1_star) + 1.0/(m2_dbl*g2_star)));

        // Cross sections and relative velocity are computed in the center of mass frame.
        // On the other hand, the particle densities (weight over volume) in the lab frame are used.
        // To take this disrepancy into account, it is needed to multiply the
        // collision probability by the ratio between the Lorentz factors in the
        // COM frame and the Lorentz factors in the lab frame (see
        // Perez et al., Phys.Plasmas.19.083104 (2012)). The correction factor
        // is calculated here.
        lab_to_COM_lorentz_factor = static_cast<amrex::ParticleReal>(g1_star*g2_star/(g1*g2));
    }

    /**
     * \brief Subtract given weight from particle and set its ID to invalid
     * if the weight reaches zero.
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void remove_weight_from_colliding_particle (
        amrex::ParticleReal& weight, uint64_t& idcpu,
        const amrex::ParticleReal reaction_weight )
    {
        // Remove weight from given particle
        amrex::Gpu::Atomic::AddNoRet(&weight, -reaction_weight);

        // If the colliding particle weight decreases to zero, remove particle by
        // setting its id to invalid
        if (weight <= std::numeric_limits<amrex::ParticleReal>::min())
        {
#if defined(AMREX_USE_OMP)
#pragma omp atomic write
            idcpu = amrex::ParticleIdCpus::Invalid;
#else
            amrex::Gpu::Atomic::Exch(
                (unsigned long long *)&idcpu,
                (unsigned long long)amrex::ParticleIdCpus::Invalid
            );
#endif
        }
    }
}

#endif // WARPX_BINARY_COLLISION_UTILS_H_
