/* Copyright 2019 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_QED_PAIR_GENERATION_H_
#define WARPX_QED_PAIR_GENERATION_H_

#include "Particles/Gather/FieldGather.H"
#include "Particles/Gather/GetExternalFields.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "QEDInternals/BreitWheelerEngineWrapper.H"
#include "QEDInternals/QedChiFunctions.H"
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

#include <picsar_qed/math/cmath_overloads.hpp>
#include <picsar_qed/physics/breit_wheeler/breit_wheeler_engine_core.hpp>
#include <picsar_qed/physics/gamma_functions.hpp>
#include <picsar_qed/physics/unit_conversion.hpp>

#include <cmath>

/** @file
 *
 * This file contains the implementation of the elementary process
 * functors needed for Breit-Wheeler pair generation (one photon generates
 * an electron-positron pair).
 */

/**
 * \brief Filter functor for the Breit-Wheeler process.
 *
 * The functor gathers E and B at the photon position, computes the photon
 * quantum parameter chi, looks up the instantaneous pair-production rate
 * dN/dt, and decides — by sampling a single random number — whether the
 * photon undergoes pair production during the current timestep. The
 * probability of an event is `1 - exp(-dN/dt * dt)`.
 */
class PairGenerationFilterFunc
{
public:

    PairGenerationFilterFunc () = default;

    /**
    * \brief Constructor of the PairGenerationFilterFunc functor.
    *
    * @param[in] dndt_table_view view of the Breit-Wheeler dN/dt lookup table
    * @param[in] bw_minimum_chi_phot minimum chi parameter below which the
    *            Breit-Wheeler process is ignored
    * @param[in] dt simulation timestep (SI units)
    * @param[in] a_pti particle iterator to iterate over photons
    * @param[in] lev   the mesh-refinement level
    * @param[in] ngEB  number of guard cells allocated for the E and B MultiFabs
    * @param[in] exfab,eyfab,ezfab,bxfab,byfab,bzfab field FArrayBoxes
    * @param[in] E_external_particle reference to external E field
    * @param[in] B_external_particle reference to external B field
    * @param[in] a_offset offset to apply to the particle indices
    */
    PairGenerationFilterFunc (BW_dndt_table_view dndt_table_view,
                              amrex::ParticleReal bw_minimum_chi_phot,
                              amrex::Real dt,
                              const WarpXParIter& a_pti, int lev, amrex::IntVect ngEB,
                              amrex::FArrayBox const& exfab,
                              amrex::FArrayBox const& eyfab,
                              amrex::FArrayBox const& ezfab,
                              amrex::FArrayBox const& bxfab,
                              amrex::FArrayBox const& byfab,
                              amrex::FArrayBox const& bzfab,
                              amrex::Vector<amrex::ParticleReal>& E_external_particle,
                              amrex::Vector<amrex::ParticleReal>& B_external_particle,
                              int a_offset = 0);

    /**
    * \brief Functor call. Determines if a given photon undergoes pair
    * production during the current timestep.
    *
    * @param[in] ptd particle tile data
    * @param[in] i particle index
    * @param[in] engine random number generator engine
    * @return true if a pair has to be generated, false otherwise
    */
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator() (const PData& ptd, int const i,
                     amrex::RandomEngine const& engine) const noexcept
    {
        using namespace amrex;
        namespace pxr_m = picsar::multi_physics::math;
        namespace pxr_p = picsar::multi_physics::phys;
        namespace pxr_bw = picsar::multi_physics::phys::breit_wheeler;

        // gather E and B at the photon position
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

        const auto ux = ptd.m_rdata[PIdx::ux][i];
        const auto uy = ptd.m_rdata[PIdx::uy][i];
        const auto uz = ptd.m_rdata[PIdx::uz][i];

        constexpr amrex::ParticleReal m_e = PhysConst::m_e;
        const auto px = m_e*ux;
        const auto py = m_e*uy;
        const auto pz = m_e*uz;

        const auto chi_phot = QedUtils::chi_photon(
            px, py, pz, ex, ey, ez, bx, by, bz);

        const auto gamma_photon = pxr_p::compute_gamma_photon<
                amrex::ParticleReal, pxr_p::unit_system::SI>(
                    px, py, pz);

        // Skip photons that cannot pair-produce
        if (gamma_photon < pxr_m::two<amrex::ParticleReal> ||
            chi_phot < m_bw_minimum_chi_phot) {
            return false;
        }

        const auto u_norm = std::sqrt(ux*ux + uy*uy + uz*uz);
        const auto energy = u_norm*m_e*PhysConst::c;

        bool is_out = false;
        const auto dndt = pxr_bw::get_dN_dt<
            amrex::ParticleReal,
            BW_dndt_table_view,
            pxr_p::unit_system::SI>(
                energy, chi_phot, m_dndt_table_view,
                pxr_m::one<amrex::ParticleReal>, &is_out);

        const auto prob = amrex::Real(1) - std::exp(
            -static_cast<amrex::Real>(dndt) * m_dt);

        return amrex::Random(engine) < prob;
    }

private:

    BW_dndt_table_view m_dndt_table_view;
    amrex::ParticleReal m_bw_minimum_chi_phot;
    amrex::Real m_dt;

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
};

/**
 * \brief Transform functor for the Breit-Wheeler process
 */
class PairGenerationTransformFunc
{
public:

    /**
    * \brief Constructor of the PairGenerationTransformFunc functor.
    *
    * A BreitWheelerGeneratePairs functor is passed by value. However, it contains
    * only few integer and real parameters and few pointers to the raw data of the
    * lookup tables. Therefore, it should be rather lightweight to copy.
    *
    * @param[in] generate_functor functor to be called to determine the properties of the generated pairs
    * @param[in] a_pti particle iterator to iterate over photons undergoing QED pair generation process
    * @param[in] lev   the mesh-refinement level
    * @param[in] ngEB  number of guard cells allocated for the E and B MultiFabs
    * @param[in] exfab constant reference to the FArrayBox of the x component of the electric field
    * @param[in] eyfab constant reference to the FArrayBox of the y component of the electric field
    * @param[in] ezfab constant reference to the FArrayBox of the z component of the electric field
    * @param[in] bxfab constant reference to the FArrayBox of the x component of the magnetic field
    * @param[in] byfab constant reference to the FArrayBox of the y component of the magnetic field
    * @param[in] bzfab constant reference to the FArrayBox of the z component of the magnetic field
    * @param[in] E_external_particle reference to external E field
    * @param[in] B_external_particle reference to external B field
    * @param[in] a_offset offset to apply to the particle indices
    */
    PairGenerationTransformFunc(BreitWheelerGeneratePairs generate_functor,
                                const WarpXParIter& a_pti, int lev, amrex::IntVect ngEB,
                                amrex::FArrayBox const& exfab,
                                amrex::FArrayBox const& eyfab,
                                amrex::FArrayBox const& ezfab,
                                amrex::FArrayBox const& bxfab,
                                amrex::FArrayBox const& byfab,
                                amrex::FArrayBox const& bzfab,
                                amrex::Vector<amrex::ParticleReal>& E_external_particle,
                                amrex::Vector<amrex::ParticleReal>& B_external_particle,
                                int a_offset = 0);

    /**
    * \brief Functor call. It determines the properties of the generated pair
    * and it sets to -1 the id of the source photon
    *
    * @param[in,out] dst1 target species 1 (either electrons or positrons)
    * @param[in,out] dst2 target species 2 (either electrons or positrons)
    * @param[in] src source species (photons)
    * @param[in] i_src particle index of the source species
    * @param[in] i_dst1 particle index of target species 1
    * @param[in] i_dst2 particle index of target species 2
    * @param[in] engine random number generator engine
    */
    template <typename DstData, typename SrcData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (DstData& dst1, DstData& dst2, SrcData& src,
        int const i_src, int const i_dst1, int const i_dst2,
        amrex::RandomEngine const& engine) const noexcept
    {
        using namespace amrex;

        // gather E and B
        amrex::ParticleReal xp, yp, zp;
        m_get_position(i_src, xp, yp, zp);

        amrex::ParticleReal ex = m_Ex_external_particle;
        amrex::ParticleReal ey = m_Ey_external_particle;
        amrex::ParticleReal ez = m_Ez_external_particle;
        amrex::ParticleReal bx = m_Bx_external_particle;
        amrex::ParticleReal by = m_By_external_particle;
        amrex::ParticleReal bz = m_Bz_external_particle;

        m_get_externalEB(i_src, ex, ey, ez, bx, by, bz);

        doGatherShapeN(xp, yp, zp, ex, ey, ez, bx, by, bz,
                       m_ex_arr, m_ey_arr, m_ez_arr, m_bx_arr, m_by_arr, m_bz_arr,
                       m_ex_type, m_ey_type, m_ez_type, m_bx_type, m_by_type, m_bz_type,
                       m_dinv, m_xyzmin, m_lo, m_n_rz_azimuthal_modes,
                       m_nox, m_galerkin_interpolation);

        //Despite the names of the variables, positrons and electrons
        //can be exchanged, since the physical process is completely
        //symmetric with respect to this exchange.
        const auto& ux = src.m_rdata[PIdx::ux][i_src];
        const auto& uy = src.m_rdata[PIdx::uy][i_src];
        const auto& uz = src.m_rdata[PIdx::uz][i_src];
        auto& e_ux = dst1.m_rdata[PIdx::ux][i_dst1];
        auto& e_uy = dst1.m_rdata[PIdx::uy][i_dst1];
        auto& e_uz = dst1.m_rdata[PIdx::uz][i_dst1];
        auto& p_ux = dst2.m_rdata[PIdx::ux][i_dst2];
        auto& p_uy = dst2.m_rdata[PIdx::uy][i_dst2];
        auto& p_uz = dst2.m_rdata[PIdx::uz][i_dst2];
        m_generate_functor(
            ux, uy, uz,
            ex, ey, ez,
            bx, by, bz,
            e_ux, e_uy, e_uz,
            p_ux, p_uy, p_uz,
            engine);

        src.m_idcpu[i_src] = amrex::ParticleIdCpus::Invalid;  // destroy photon after pair generation
    }

private:

    BreitWheelerGeneratePairs
    m_generate_functor; /*!< A copy of the functor to generate pairs. It contains only pointers to the lookup tables.*/

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
};

#endif //WARPX_QED_PAIR_GENERATION_H_
