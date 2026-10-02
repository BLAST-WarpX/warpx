/* Copyright 2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Particles/Gather/GatherFieldsOnParticle.H"

#include "WarpX.H"

#include <AMReX_Box.H>

#include <algorithm>
#include <array>

GatherFieldsOnParticle::GatherFieldsOnParticle (
    const WarpXParIter& a_pti, int lev, amrex::IntVect ngEB,
    amrex::FArrayBox const& exfab,
    amrex::FArrayBox const& eyfab,
    amrex::FArrayBox const& ezfab,
    amrex::FArrayBox const& bxfab,
    amrex::FArrayBox const& byfab,
    amrex::FArrayBox const& bzfab,
    amrex::Vector<amrex::ParticleReal> const& E_external_particle,
    amrex::Vector<amrex::ParticleReal> const& B_external_particle,
    int a_offset):
    m_get_position{a_pti, a_offset},
    m_get_externalEB{a_pti, a_offset},
    m_Ex_external_particle{E_external_particle[0]},
    m_Ey_external_particle{E_external_particle[1]},
    m_Ez_external_particle{E_external_particle[2]},
    m_Bx_external_particle{B_external_particle[0]},
    m_By_external_particle{B_external_particle[1]},
    m_Bz_external_particle{B_external_particle[2]},
    m_ex_arr{exfab.array()},
    m_ey_arr{eyfab.array()},
    m_ez_arr{ezfab.array()},
    m_bx_arr{bxfab.array()},
    m_by_arr{byfab.array()},
    m_bz_arr{bzfab.array()},
    m_ex_type{exfab.box().ixType()},
    m_ey_type{eyfab.box().ixType()},
    m_ez_type{ezfab.box().ixType()},
    m_bx_type{bxfab.box().ixType()},
    m_by_type{byfab.box().ixType()},
    m_bz_type{bzfab.box().ixType()},
    m_galerkin_interpolation{WarpX::galerkin_interpolation},
    m_nox{WarpX::nox},
    m_n_rz_azimuthal_modes{WarpX::n_rz_azimuthal_modes}
{
    using namespace amrex::literals;

    amrex::Box box = a_pti.tilebox();
    box.grow(ngEB);

    const std::array<amrex::Real,3>& dx = WarpX::CellSize(std::max(lev, 0));
    m_dinv = amrex::XDim3{1._rt/dx[0], 1._rt/dx[1], 1._rt/dx[2]};

    // Lower corner of tile box physical domain (take into account Galilean shift)
    m_xyzmin = WarpX::LowerCorner(box, lev, 0._rt);

    m_lo = amrex::lbound(box);
}
