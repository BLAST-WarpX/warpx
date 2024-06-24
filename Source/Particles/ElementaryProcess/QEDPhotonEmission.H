/* Copyright 2019 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_QED_PHOTON_EMISSION_H_
#define WARPX_QED_PHOTON_EMISSION_H_

#include "Particles/Gather/FieldGather.H"
#include "Particles/Gather/GetExternalFields.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "QEDInternals/QuantumSyncEngineWrapper.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_REAL.H>

#include <AMReX_BaseFwd.H>

#include <algorithm>
#include <limits>

/** @file
 *
 * This file contains the implementation of the elementary process
 * functors needed for QED photon emission (an electron or a positron
 * emits a photon).
 */

/**
 * \brief Filter functor for the QED photon emission process
 */
class PhotonEmissionFilterFunc
{
public:

    /**
    * \brief Constructor of the PhotonEmissionFilterFunc functor.
    *
    * @param[in] opt_depth_runtime_comp Index of the optical depth component
    */
    PhotonEmissionFilterFunc(int const opt_depth_runtime_comp)
        : m_opt_depth_runtime_comp(opt_depth_runtime_comp)
        {}

    /**
    * \brief Functor call. This method determines if a given (electron or positron)
    * particle should undergo QED photon emission.
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
    int m_opt_depth_runtime_comp; /*!< Index of the optical depth component of the source species*/
};

/**
 * \brief Transform functor for the QED photon emission process
 */
class PhotonEmissionTransformFunc
{

public:

    /**
    * \brief Constructor of the PhotonEmissionTransformFunc functor.
    *
    * A QuantumSynchrotronGeneratePhotonAndUpdateMomentum functor is passed by value. However, it contains
    * only few integer and real parameters and few pointers to the raw data of the
    * lookup tables. Therefore, it should be rather lightweight to copy.
    *
    * Also a QuantumSynchrotronGetOpticalDepth has to be passed, since the
    * optical depth has to be re-initialized after each photon emission.
    *
    * @param[in] opt_depth_functor functor to re-initialize the optical depth of the source particles
    * @param[in] opt_depth_runtime_comp index of the optical depth component of the source species
    * @param[in] emission_functor functor to generate photons and update momentum of the source particles
    * @param[in] a_pti particle iterator to iterate over electrons or positrons undergoing QED photon emission process
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
    PhotonEmissionTransformFunc (
        QuantumSynchrotronGetOpticalDepth opt_depth_functor,
        int opt_depth_runtime_comp,
        QuantumSynchrotronPhotonEmission emission_functor,
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
    * \brief Functor call. It determines the properties of the generated photon
    * and updates the momentum of the source particle
    *
    * @param[in,out] dst target species (photons)
    * @param[in, out] src source species (either electrons or positrons)
    * @param[in] i_src particle index of the source species
    * @param[in] i_dst particle index of target species
    * @param[in] engine random number generator engine
    */
    template <typename DstData, typename SrcData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (DstData& dst, SrcData& src, int i_src, int i_dst,
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

        auto& ux = src.m_rdata[PIdx::ux][i_src];
        auto& uy = src.m_rdata[PIdx::uy][i_src];
        auto& uz = src.m_rdata[PIdx::uz][i_src];
        auto& g_ux = dst.m_rdata[PIdx::ux][i_dst];
        auto& g_uy = dst.m_rdata[PIdx::uy][i_dst];
        auto& g_uz = dst.m_rdata[PIdx::uz][i_dst];
        m_emission_functor(
            ux, uy, uz,
            ex, ey, ez,
            bx, by, bz,
            g_ux, g_uy, g_uz,
            engine);

        //Initialize the optical depth component of the source species.
        src.m_runtime_rdata[m_opt_depth_runtime_comp][i_src] =
            m_opt_depth_functor(engine);
    }

private:
    const QuantumSynchrotronGetOpticalDepth
        m_opt_depth_functor;  /*!< A copy of the functor to initialize the optical depth of the source species. */

    const int m_opt_depth_runtime_comp = 0;  /*!< Index of the optical depth component of source species*/

    const QuantumSynchrotronPhotonEmission
        m_emission_functor;  /*!< A copy of the functor to generate photons. It contains only pointers to the lookup tables.*/

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
* \brief Free function to call to remove immediately
* low energy photons by setting their ID to -1.
* Photons with extremely small energy are removed regardless of
* the value of the energy_threshold
*
* @tparam PTile particle tile type
* @param[in,out] ptile a particle tile
* @param[in] old_size the old number of particles
* @param[in] num_added the number of photons added to the tile
* @param[in] energy_threshold the energy threshold
*/
template <typename PTile>
void cleanLowEnergyPhotons(
    PTile& ptile,
    const int old_size, const int num_added,
    const amrex::ParticleReal energy_threshold)
{
    auto& soa = ptile.GetStructOfArrays();
    auto p_idcpu = soa.GetIdCPUData().data() + old_size;
    const auto p_ux = soa.GetRealData(PIdx::ux).data() + old_size;
    const auto p_uy = soa.GetRealData(PIdx::uy).data() + old_size;
    const auto p_uz = soa.GetRealData(PIdx::uz).data() + old_size;

    //The square of the energy threshold
    const auto energy_threshold2 = std::max(
        energy_threshold*energy_threshold,
        std::numeric_limits<amrex::ParticleReal>::min());

    amrex::ParallelFor(num_added, [=] AMREX_GPU_DEVICE (int ip) noexcept
    {
        const auto ux = p_ux[ip];
        const auto uy = p_uy[ip];
        const auto uz = p_uz[ip];

        //The square of the photon energy (in SI units)
        // ( Particle momentum is stored as gamma * velocity.)
        constexpr amrex::ParticleReal me_c = PhysConst::m_e*PhysConst::c;
        const auto phot_energy2 = (ux*ux + uy*uy + uz*uz)*me_c*me_c;

        if (phot_energy2 < energy_threshold2) {
            p_idcpu[ip] = amrex::ParticleIdCpus::Invalid;
        }
    });
}


#endif //WARPX_QED_PHOTON_EMISSION_H_
