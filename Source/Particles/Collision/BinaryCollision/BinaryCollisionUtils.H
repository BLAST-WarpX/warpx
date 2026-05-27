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
                            Bremsstrahlung,
                            DSMC,
                            PairwiseCoulomb,
                            LinearBreitWheeler,
                            LinearCompton,
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

    void CheckFusionEnergy (NuclearFusionType fusion_type,
                            amrex::ParticleReal mass_before,
                            amrex::ParticleReal mass_after);

    CollisionType get_collision_type (const std::string& collision_name,
                                      MultiParticleContainer const * mypc);

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    CollisionType nuclear_fusion_type_to_collision_type (NuclearFusionType fusion_type)
    {
        CollisionType collision_type = CollisionType::Undefined;
        if (fusion_type == NuclearFusionType::DeuteriumTritiumToNeutronHelium) {
            collision_type = CollisionType::DeuteriumTritiumToNeutronHeliumFusion;
        }
        else if (fusion_type == NuclearFusionType::DeuteriumDeuteriumToProtonTritium) {
            collision_type = CollisionType::DeuteriumDeuteriumToProtonTritiumFusion;
        }
        else if (fusion_type == NuclearFusionType::DeuteriumDeuteriumToNeutronHelium) {
            collision_type = CollisionType::DeuteriumDeuteriumToNeutronHeliumFusion;
        }
        else if (fusion_type == NuclearFusionType::DeuteriumHeliumToProtonHelium) {
            collision_type = CollisionType::DeuteriumHeliumToProtonHeliumFusion;
        }
        else if (fusion_type == NuclearFusionType::ProtonBoronToAlphas) {
            collision_type = CollisionType::ProtonBoronToAlphasFusion;
        }
        else {
            AMREX_IF_ON_HOST((
                WARPX_ABORT_WITH_MESSAGE("Invalid nuclear fusion type");
            ))
        }
        return collision_type;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    bool is_two_product_fusion_type (const CollisionType collision_type)
    {
        return ((collision_type == CollisionType::DeuteriumTritiumToNeutronHeliumFusion) ||
            (collision_type == CollisionType::DeuteriumDeuteriumToProtonTritiumFusion) ||
            (collision_type == CollisionType::DeuteriumDeuteriumToNeutronHeliumFusion) ||
            (collision_type == CollisionType::DeuteriumHeliumToProtonHeliumFusion));
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    bool is_two_product_fusion_type (const NuclearFusionType fusion_type)
    {
        const CollisionType collision_type = nuclear_fusion_type_to_collision_type(fusion_type);
        return is_two_product_fusion_type(collision_type);
    }

    /**
     * \brief Return (relativistic) collision energy, collision speed and
     * Lorentz factor for transforming between the lab and center-of-momentum
     * frames.
     */
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void get_collision_parameters (
        const amrex::ParticleReal& p1x, const amrex::ParticleReal& p1y,
        const amrex::ParticleReal& p1z, const amrex::ParticleReal& p2x,
        const amrex::ParticleReal& p2y, const amrex::ParticleReal& p2z,
        const amrex::ParticleReal& m1, const amrex::ParticleReal& m2,
        amrex::ParticleReal& E_kin_COM, amrex::ParticleReal& v_rel_COM,
        amrex::ParticleReal& lab_to_COM_lorentz_factor )
    {
        // General notations in this function:
        //     x_sq denotes the square of x
        //     x_star denotes the value of x in the center of mass frame

        using namespace amrex::literals;
        using namespace amrex::Math;

        constexpr double c = PhysConst::c;
        constexpr auto c2 = PhysConst::c2_v<double>;
        constexpr double inv_c = 1./PhysConst::c;

        // Cast input parameters to double before computing collision properties
        // This is needed to avoid errors when using single-precision particles
        const auto m1_dbl = static_cast<double>(m1);
        const auto m2_dbl = static_cast<double>(m2);
        const auto p1x_dbl = static_cast<double>(p1x);
        const auto p1y_dbl = static_cast<double>(p1y);
        const auto p1z_dbl = static_cast<double>(p1z);
        const auto p2x_dbl = static_cast<double>(p2x);
        const auto p2y_dbl = static_cast<double>(p2y);
        const auto p2z_dbl = static_cast<double>(p2z);

        const double m1_sq = m1_dbl*m1_dbl;
        const double m2_sq = m2_dbl*m2_dbl;

        const double p1_sq = p1x_dbl*p1x_dbl + p1y_dbl*p1y_dbl + p1z_dbl*p1z_dbl;
        const double p2_sq = p2x_dbl*p2x_dbl + p2y_dbl*p2y_dbl + p2z_dbl*p2z_dbl;
        const double p1_dot_p2 = p1x_dbl*p2x_dbl + p1y_dbl*p2y_dbl + p1z_dbl*p2z_dbl;

        // Total energy of each particle in the lab frame.
        // Note the use of `double` for energy since this calculation is prone to error with single precision.
        const double E1_lab = std::sqrt( m1_sq * c2 + p1_sq ) * c;
        const double E2_lab = std::sqrt( m2_sq * c2 + p2_sq ) * c;

        // Compute the kinetic energy in the COM frame, and the squared momentum of each
        // particle in the COM frame, in a way that avoids catastrophic cancellation
        // for low-energy collisions.
        //
        // Starting from the Lorentz-invariant
        //     E_star^2 = E_lab^2 - c^2 p_total^2
        //              = (E1 + E2)^2 - c^2 (p1 + p2)^2
        // and expanding using E_i^2 = m_i^2 c^4 + c^2 p_i^2, one gets
        //     E_star^2 = m1^2 c^4 + m2^2 c^4 + 2 ( E1 E2 - c^2 p1.p2 )
        // and therefore
        //     E_star^2 - (m1 + m2)^2 c^4 = 2 ( E1 E2 - m1 m2 c^4 - c^2 p1.p2 ).            (*)
        //
        // The reverse Cauchy-Schwarz inequality for future-pointing timelike four-vectors
        // guarantees E1 E2 - c^2 p1.p2 >= m1 m2 c^4 (with equality iff the two particles
        // have the same four-velocity), so the RHS of (*) is non-negative.
        //
        // To preserve this non-negativity numerically, we compute (E1 E2 - m1 m2 c^4)
        // as a sum of inherently non-negative products by using the identity
        //     E1 E2 - m1 m2 c^4 = m2 c^2 KE1 + m1 c^2 KE2 + KE1 KE2
        // where KE_i = E_i - m_i c^2 is computed without cancellation as
        //     KE_i = p_i^2 c^2 / (E_i + m_i c^2).
        const double KE1_lab = p1_sq * c2 / (E1_lab + m1_dbl * c2);
        const double KE2_lab = p2_sq * c2 / (E2_lab + m2_dbl * c2);
        const double E1E2_minus_m1m2c4 = m2_dbl * c2 * KE1_lab
                                       + m1_dbl * c2 * KE2_lab
                                       + KE1_lab * KE2_lab;

        // s_minus_threshold = E_star^2 - (m1 + m2)^2 c^4 >= 0 (in J^2 units)
        const double s_minus_threshold = 2.0 * (E1E2_minus_m1m2c4 - c2 * p1_dot_p2);
        const double m_sum_c2 = (m1_dbl + m2_dbl) * c2;
        const double E_star_sq = m_sum_c2 * m_sum_c2 + s_minus_threshold;
        const double E_star = std::sqrt(E_star_sq);

        // Kinetic energy in the COM frame, computed as
        //     E_star - (m1 + m2) c^2 = (E_star^2 - (m1 + m2)^2 c^4) / (E_star + (m1 + m2) c^2)
        // to avoid catastrophic cancellation between E_star and (m1 + m2) c^2.
        // For two massless particles, E_kin_COM = E_star exactly, and the denominator
        // above can be zero (degenerate collinear case), so just use E_star directly.
        if (m_sum_c2 == 0.0) {
            E_kin_COM = static_cast<amrex::ParticleReal>(E_star);
        } else {
            E_kin_COM = static_cast<amrex::ParticleReal>(
                s_minus_threshold / (E_star + m_sum_c2)
            );
        }

        // Find the norm of the momentum of each particle in the COM frame, using the
        // factored Kallen form of the relativistic two-body momentum:
        //     p_star^2 = (s - (m1 + m2)^2 c^2) (s - (m1 - m2)^2 c^2) / (4 s)
        // with s = E_star^2 / c^2. The two factors in the numerator differ by 4 m1 m2 c^2,
        // so we obtain both from `s_minus_threshold` without further cancellation.
        double p_star_sq;
        if (m1_dbl + m2_dbl == 0) {
            p_star_sq = powi<2>( 0.5 * E_star * inv_c );
        } else {
            p_star_sq = s_minus_threshold
                      * (s_minus_threshold + 4.0 * m1_dbl * m2_dbl * c2 * c2)
                      / (4.0 * E_star_sq * c2);
        }

        // Energy of each particle in the center of mass frame
        const double E1_star = std::sqrt(m1_sq*c2 + p_star_sq) * c;
        const double E2_star = std::sqrt(m2_sq*c2 + p_star_sq) * c;

        // relative velocity in the center of mass frame, cast back to chosen precision
        v_rel_COM = PhysConst::c * static_cast<amrex::ParticleReal>(std::sqrt(p_star_sq) * c * (1.0/E1_star + 1.0/E2_star));

        // Cross sections and relative velocity are computed in the center of mass frame.
        // On the other hand, the particle densities (weight over volume) in the lab frame are used.
        // To take this discrepancy into account, it is needed to multiply the
        // collision probability by the ratio between the Lorentz factors in the
        // COM frame and the Lorentz factors in the lab frame (see Perez et al.,
        // Phys.Plasmas.19.083104 (2012)). The correction factor is calculated here.
        lab_to_COM_lorentz_factor = static_cast<amrex::ParticleReal>(E1_star*E2_star/(E1_lab*E2_lab));
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
