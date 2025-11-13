/* Copyright 2021 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ParticleBoundaries.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"

namespace warpx::particles
{

    namespace detail
    {
        void set_to_periodic_if_field_boundary_is_periodic (
            amrex::Array<ParticleBoundaryType, AMREX_SPACEDIM>& particle_boundary_lo,
            amrex::Array<ParticleBoundaryType, AMREX_SPACEDIM>& particle_boundary_hi,
            const amrex::Array<bool, AMREX_SPACEDIM>& is_field_boundary_periodic
        )
        {
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                if (is_field_boundary_periodic[idim]){
                    particle_boundary_lo[idim] = ParticleBoundaryType::Periodic;
                    particle_boundary_hi[idim] = ParticleBoundaryType::Periodic;
                }
            }
        }

        void check_consistency (
            const amrex::Array<ParticleBoundaryType, AMREX_SPACEDIM>& particle_boundary_lo,
            const amrex::Array<ParticleBoundaryType, AMREX_SPACEDIM>& particle_boundary_hi,
            const amrex::Array<bool, AMREX_SPACEDIM>& is_field_boundary_periodic
        )
        {
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                if (is_field_boundary_periodic[idim] ||
                    particle_boundary_lo[idim] == ParticleBoundaryType::Periodic ||
                    particle_boundary_hi[idim] == ParticleBoundaryType::Periodic ) {
                    // to ensure both lo and hi are set to periodic consistently for both field and particles.
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        (particle_boundary_lo[idim] == ParticleBoundaryType::Periodic) &&
                        (particle_boundary_hi[idim] == ParticleBoundaryType::Periodic),
                        "field and particle boundary must be periodic in both lo and hi");
                }
            }
        }
    }

    std::pair<
        amrex::Array<ParticleBoundaryType, AMREX_SPACEDIM>,
        amrex::Array<ParticleBoundaryType, AMREX_SPACEDIM>
    >
    parse_particle_boundaries (
        const amrex::Array<bool, AMREX_SPACEDIM>& is_field_boundary_periodic)
    {
        auto particle_boundary_lo =
            amrex::Array<ParticleBoundaryType, AMREX_SPACEDIM>{
                AMREX_D_DECL(
                    ParticleBoundaryType::Default,
                    ParticleBoundaryType::Default,
                    ParticleBoundaryType::Default)};
        auto particle_boundary_hi = particle_boundary_lo;

        const auto pp_boundary = amrex::ParmParse{"boundary"};

        const bool particle_boundary_specified =
            pp_boundary.contains("particle_lo") ||
            pp_boundary.contains("particle_hi");

        if (! particle_boundary_specified){
            detail::set_to_periodic_if_field_boundary_is_periodic(
                particle_boundary_lo, particle_boundary_hi,
                is_field_boundary_periodic);
        }
        else{
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                pp_boundary.query_enum_sloppy("particle_lo", particle_boundary_lo[idim], "-_", idim);
                pp_boundary.query_enum_sloppy("particle_hi", particle_boundary_hi[idim], "-_", idim);
            }

            detail::check_consistency(
                particle_boundary_lo, particle_boundary_hi,
                is_field_boundary_periodic);
        }

        return {particle_boundary_lo, particle_boundary_hi};
    }
}

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
