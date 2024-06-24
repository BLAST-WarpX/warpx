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

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_REAL.H>

#include <AMReX_BaseFwd.H>

/** @file
 *
 * This file contains the implementation of the elementary process
 * functors needed for Breit-Wheeler pair generation (one photon generates
 * and electron-positron pair).
 */

/**
 * \brief Filter functor for the Breit Wheeler process
 */
class PairGenerationFilterFunc
{
public:

    /**
    * \brief Constructor of the PairGenerationFilterFunc functor.
    *
    * @param[in] opt_depth_runtime_comp index of the optical depth component
    */
    PairGenerationFilterFunc(int const opt_depth_runtime_comp)
        : m_opt_depth_runtime_comp(opt_depth_runtime_comp)
        {}

    /**
    * \brief Functor call. This method determines if a given (photon) particle
    * should undergo pair generation.
    *
    * @param[in] ptd particle tile data
    * @param[in] i particle index
    * @return true if a pair has to be generated, false otherwise
    */
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator() (const PData& ptd, int const i, amrex::RandomEngine const&) const noexcept
    {
        using namespace amrex;

        const amrex::ParticleReal opt_depth =
            ptd.m_runtime_rdata[m_opt_depth_runtime_comp][i];
        return (opt_depth < 0.0_rt);
    }

private:
    int m_opt_depth_runtime_comp = 0; /*!< Index of the optical depth component of the species.*/
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

    const BreitWheelerGeneratePairs
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
