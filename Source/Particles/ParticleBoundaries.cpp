/* Copyright 2021 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ParticleBoundaries.H"
#include "WarpX.H"

#include "Utils/Parser/ParserUtils.H"

ParticleBoundaries::ParticleBoundaries () noexcept
{
    SetAll(ParticleBoundaryType::Absorbing);
    data.reflect_all_velocities = false;
}

void
ParticleBoundaries::Set_reflect_all_velocities (bool flag)
{
    data.reflect_all_velocities = flag;
}

void
ParticleBoundaries::SetAll (ParticleBoundaryType bc)
{
    data.xmin_bc = bc;
    data.xmax_bc = bc;
    data.ymin_bc = bc;
    data.ymax_bc = bc;
    data.zmin_bc = bc;
    data.zmax_bc = bc;
}

void
ParticleBoundaries::SetThermalVelocity (amrex::Real u_th)
{
    data.m_uth = u_th;
}

void
ParticleBoundaries::SetBoundsX (ParticleBoundaryType bc_lo, ParticleBoundaryType bc_hi)
{
    data.xmin_bc = bc_lo;
    data.xmax_bc = bc_hi;
}

void
ParticleBoundaries::SetBoundsY (ParticleBoundaryType bc_lo, ParticleBoundaryType bc_hi)
{
    data.ymin_bc = bc_lo;
    data.ymax_bc = bc_hi;
}

void
ParticleBoundaries::SetBoundsZ (ParticleBoundaryType bc_lo, ParticleBoundaryType bc_hi)
{
    data.zmin_bc = bc_lo;
    data.zmax_bc = bc_hi;
}

bool
ParticleBoundaries::CheckAll (ParticleBoundaryType bc) const
{
    return (data.xmin_bc == bc && data.xmax_bc == bc
#ifdef WARPX_DIM_3D
         && data.ymin_bc == bc && data.ymax_bc == bc
#endif
         && data.zmin_bc == bc && data.zmax_bc == bc);
}

void
ParticleBoundaries::BuildReflectionModelParsers ()
{
    reflection_model_xlo_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(reflection_model_xlo_str, {"v"}));
    data.reflection_model_xlo = reflection_model_xlo_parser->compile<1>();
    reflection_model_xhi_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(reflection_model_xhi_str, {"v"}));
    data.reflection_model_xhi = reflection_model_xhi_parser->compile<1>();
#ifdef WARPX_DIM_3D
    reflection_model_ylo_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(reflection_model_ylo_str, {"v"}));
    data.reflection_model_ylo = reflection_model_ylo_parser->compile<1>();
    reflection_model_yhi_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(reflection_model_yhi_str, {"v"}));
    data.reflection_model_yhi = reflection_model_yhi_parser->compile<1>();
#endif
    reflection_model_zlo_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(reflection_model_zlo_str, {"v"}));
    data.reflection_model_zlo = reflection_model_zlo_parser->compile<1>();
    reflection_model_zhi_parser = std::make_unique<amrex::Parser>(
        utils::parser::makeParser(reflection_model_zhi_str, {"v"}));
    data.reflection_model_zhi = reflection_model_zhi_parser->compile<1>();
}
