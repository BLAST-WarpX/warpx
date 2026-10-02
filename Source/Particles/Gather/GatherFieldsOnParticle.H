/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_GATHER_GATHERFIELDSONPARTICLE_H_
#define WARPX_PARTICLES_GATHER_GATHERFIELDSONPARTICLE_H_

#include "Particles/Gather/FieldGather.H"
#include "Particles/Gather/GetExternalFields.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"

#include <AMReX_Array4.H>
#include <AMReX_Dim3.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

/**
 * \brief Functor gathering the E and B fields (grid and external) at a particle position.
 *
 * It bundles everything a device kernel needs to call doGatherShapeN on one particle of a
 * tile, so that elementary processes running outside the pusher (QED, ionization, ...) do
 * not have to carry the gather set-up themselves.
 */
class GatherFieldsOnParticle
{
public:
    GatherFieldsOnParticle () = default;

    /**
     * @param[in] a_pti particle iterator of the tile
     * @param[in] lev the mesh-refinement level
     * @param[in] ngEB number of guard cells allocated for the E and B MultiFabs
     * @param[in] exfab,eyfab,ezfab FArrayBoxes of the electric field components
     * @param[in] bxfab,byfab,bzfab FArrayBoxes of the magnetic field components
     * @param[in] E_external_particle,B_external_particle uniform external fields
     * @param[in] a_offset offset to apply to the particle indices
     */
    GatherFieldsOnParticle (
        const WarpXParIter& a_pti, int lev, amrex::IntVect ngEB,
        amrex::FArrayBox const& exfab,
        amrex::FArrayBox const& eyfab,
        amrex::FArrayBox const& ezfab,
        amrex::FArrayBox const& bxfab,
        amrex::FArrayBox const& byfab,
        amrex::FArrayBox const& bzfab,
        amrex::Vector<amrex::ParticleReal> const& E_external_particle,
        amrex::Vector<amrex::ParticleReal> const& B_external_particle,
        int a_offset = 0);

    /**
     * \brief Gather the fields at the position of particle `ip`.
     *
     * @param[in] ip index of the particle
     * @param[out] ex,ey,ez,bx,by,bz field components at the particle position
     */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void operator() (long ip,
                     amrex::ParticleReal& ex, amrex::ParticleReal& ey, amrex::ParticleReal& ez,
                     amrex::ParticleReal& bx, amrex::ParticleReal& by, amrex::ParticleReal& bz)
        const noexcept
    {
        amrex::ParticleReal xp, yp, zp;
        m_get_position(ip, xp, yp, zp);

        ex = m_Ex_external_particle;
        ey = m_Ey_external_particle;
        ez = m_Ez_external_particle;
        bx = m_Bx_external_particle;
        by = m_By_external_particle;
        bz = m_Bz_external_particle;
        m_get_externalEB(ip, ex, ey, ez, bx, by, bz);

        doGatherShapeN(xp, yp, zp, ex, ey, ez, bx, by, bz,
                       m_ex_arr, m_ey_arr, m_ez_arr, m_bx_arr, m_by_arr, m_bz_arr,
                       m_ex_type, m_ey_type, m_ez_type, m_bx_type, m_by_type, m_bz_type,
                       m_dinv, m_xyzmin, m_lo, m_n_rz_azimuthal_modes,
                       m_nox, m_galerkin_interpolation);
    }

private:
    GetParticlePosition<PIdx> m_get_position;
    GetExternalEBField m_get_externalEB;
    amrex::ParticleReal m_Ex_external_particle = 0;
    amrex::ParticleReal m_Ey_external_particle = 0;
    amrex::ParticleReal m_Ez_external_particle = 0;
    amrex::ParticleReal m_Bx_external_particle = 0;
    amrex::ParticleReal m_By_external_particle = 0;
    amrex::ParticleReal m_Bz_external_particle = 0;

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
    amrex::Dim3 m_lo;

    bool m_galerkin_interpolation = false;
    int m_nox = 0;
    int m_n_rz_azimuthal_modes = 0;
};

#endif // WARPX_PARTICLES_GATHER_GATHERFIELDSONPARTICLE_H_
