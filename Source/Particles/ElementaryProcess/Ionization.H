/* Copyright 2019-2020 Andrew Myers, Axel Huebl,
 * Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_IONIZATION_H_
#define WARPX_IONIZATION_H_

#include "Particles/Gather/FieldGather.H"
#include "Particles/Gather/GetExternalFields.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_REAL.H>
#include <AMReX_Random.H>

#include <AMReX_BaseFwd.H>

#include <cmath>

struct IonizationFilterFunc
{
    const amrex::Real* AMREX_RESTRICT m_ionization_energies;
    const amrex::Real* AMREX_RESTRICT m_adk_prefactor;
    const amrex::Real* AMREX_RESTRICT m_adk_exp_prefactor;
    const amrex::Real* AMREX_RESTRICT m_adk_power;
    const amrex::Real* AMREX_RESTRICT m_adk_correction_factors;

    int comp;
    int m_atomic_number;
    int m_do_adk_correction = 0;

    GetParticlePosition<PIdx> m_get_position;
    GetExternalEBField m_get_externalEB;
    amrex::ParticleReal m_Ex_external_particle;
    amrex::ParticleReal m_Ey_external_particle;
    amrex::ParticleReal m_Ez_external_particle;
    amrex::ParticleReal m_Bx_external_particle;
    amrex::ParticleReal m_By_external_particle;
    amrex::ParticleReal m_Bz_external_particle;

    amrex::Array4<const amrex::Real> m_ex_arr;
    amrex::Array4<const amrex::Real> m_ey_arr;
    amrex::Array4<const amrex::Real> m_ez_arr;
    amrex::Array4<const amrex::Real> m_bx_arr;
    amrex::Array4<const amrex::Real> m_by_arr;
    amrex::Array4<const amrex::Real> m_bz_arr;

    amrex::IndexType m_ex_type;
    amrex::IndexType m_ey_type;
    amrex::IndexType m_ez_type;
    amrex::IndexType m_bx_type;
    amrex::IndexType m_by_type;
    amrex::IndexType m_bz_type;

    amrex::XDim3 m_dinv;
    amrex::XDim3 m_xyzmin;

    bool m_galerkin_interpolation;
    int m_nox;
    int m_n_rz_azimuthal_modes;

    amrex::Dim3 m_lo;

    IonizationFilterFunc (const WarpXParIter& a_pti, int lev, amrex::IntVect ngEB,
                          amrex::FArrayBox const& exfab,
                          amrex::FArrayBox const& eyfab,
                          amrex::FArrayBox const& ezfab,
                          amrex::FArrayBox const& bxfab,
                          amrex::FArrayBox const& byfab,
                          amrex::FArrayBox const& bzfab,
                          amrex::Vector<amrex::ParticleReal>& E_external_particle,
                          amrex::Vector<amrex::ParticleReal>& B_external_particle,
                          const amrex::Real* AMREX_RESTRICT a_ionization_energies,
                          const amrex::Real* AMREX_RESTRICT a_adk_prefactor,
                          const amrex::Real* AMREX_RESTRICT a_adk_exp_prefactor,
                          const amrex::Real* AMREX_RESTRICT a_adk_power,
                          const amrex::Real* AMREX_RESTRICT a_adk_correction_factors,
                          int a_comp,
                          int a_atomic_number,
                          int a_do_adk_correction,
                          int a_offset = 0) noexcept;

    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator() (const PData& ptd, int i, amrex::RandomEngine const& engine) const noexcept
    {
        using namespace amrex::literals;

        const int ion_lev = ptd.m_runtime_idata[comp][i];
        if (ion_lev < m_atomic_number)
        {
            constexpr amrex::Real c = PhysConst::c;
            constexpr amrex::Real c2_inv = amrex::Real(1.)/c/c;

            // gather E and B
            amrex::ParticleReal xp, yp, zp;
            m_get_position(i, xp, yp, zp);

            amrex::ParticleReal ex = m_Ex_external_particle;
            amrex::ParticleReal ey = m_Ey_external_particle;
            amrex::ParticleReal ez = m_Ez_external_particle;
            amrex::ParticleReal bx = m_Bx_external_particle;
            amrex::ParticleReal by = m_By_external_particle;
            amrex::ParticleReal bz = m_Bz_external_particle;
            m_get_externalEB(i, ex, ey, ez, bx, by, bz);

            doGatherShapeN(xp, yp, zp, ex, ey, ez, bx, by, bz,
                           m_ex_arr, m_ey_arr, m_ez_arr, m_bx_arr, m_by_arr, m_bz_arr,
                           m_ex_type, m_ey_type, m_ez_type, m_bx_type, m_by_type, m_bz_type,
                           m_dinv, m_xyzmin, m_lo, m_n_rz_azimuthal_modes,
                           m_nox, m_galerkin_interpolation);
            m_get_externalEB(i, ex, ey, ez, bx, by, bz);

            // Compute electric field amplitude in the particle's frame of
            // reference (particularly important when in boosted frame).
            const amrex::ParticleReal ux = ptd.m_rdata[PIdx::ux][i];
            const amrex::ParticleReal uy = ptd.m_rdata[PIdx::uy][i];
            const amrex::ParticleReal uz = ptd.m_rdata[PIdx::uz][i];

            const auto ga = static_cast<amrex::Real>(
                std::sqrt(1. + (ux*ux + uy*uy + uz*uz) * c2_inv));
            const amrex::Real E = std::sqrt(
                               - ( ux*ex + uy*ey + uz*ez ) * ( ux*ex + uy*ey + uz*ez ) * c2_inv
                               + ( ga   *ex + uy*bz - uz*by ) * ( ga   *ex + uy*bz - uz*by )
                               + ( ga   *ey + uz*bx - ux*bz ) * ( ga   *ey + uz*bx - ux*bz )
                               + ( ga   *ez + ux*by - uy*bx ) * ( ga   *ez + ux*by - uy*bx )
                               );

            // Compute probability of ionization p
            amrex::Real w_dtau = (E <= 0._rt) ? 0._rt : 1._rt/ ga * m_adk_prefactor[ion_lev] *
                std::pow(E, m_adk_power[ion_lev]) *
                std::exp( m_adk_exp_prefactor[ion_lev]/E );
            // if requested, do Zhang's correction of ADK
            if (m_do_adk_correction) {
                const amrex::Real r = E / m_adk_correction_factors[3];
                w_dtau *= std::exp(m_adk_correction_factors[0]*r*r+m_adk_correction_factors[1]*r+
                                   m_adk_correction_factors[2]);
            }

            const amrex::Real p = 1._rt - std::exp( - w_dtau );

            const amrex::Real random_draw = amrex::Random(engine);
            if (random_draw < p)
            {
                return true;
            }
        }
        return false;
    }
};

struct IonizationTransformFunc
{
    template <typename DstData, typename SrcData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (DstData& /*dst*/, SrcData& src,
        int i_src, int /*i_dst*/,
        amrex::RandomEngine const& /*engine*/) const noexcept
    {
        src.m_runtime_idata[0][i_src] += 1;
    }
};

#endif //WARPX_IONIZATION_H_
