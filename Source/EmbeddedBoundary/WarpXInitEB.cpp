/* Copyright 2021 Lorenzo Giacomel
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "WarpX.H"

#include "EmbeddedBoundary/Enabled.H"
#ifdef AMREX_USE_EB
#  include "Fields.H"
#  include "Utils/Parser/ParserUtils.H"
#  include "Utils/TextMsg.H"

#   include <AMReX_BLProfiler.H>
#   include <AMReX_BoxArray.H>
#   include <AMReX_Config.H>
#   include <AMReX_EB2.H>
#   include <AMReX_EB2_GeometryShop.H>
#   include <AMReX_EB2_IF_Base.H>
#   include <AMReX_EB2_IndexSpace_STL.H>
#   include <AMReX_EB_utils.H>
#   include <AMReX_GpuQualifiers.H>
#   include <AMReX_ParmParse.H>
#   include <AMReX_REAL.H>
#   include <AMReX_SPACE.H>

#  include <cstdlib>
#  include <string>

using namespace ablastr::fields;
using namespace amrex;
#endif

#ifdef AMREX_USE_EB
namespace {
    class ParserIF
        : public amrex::GPUable
    {
    public:
        explicit
        ParserIF (const amrex::ParserExecutor<3>& a_parser)
            : m_parser(a_parser)
            {}

        ParserIF (const ParserIF& rhs) noexcept = default;
        ParserIF (ParserIF&& rhs) noexcept = default;
        ParserIF& operator= (const ParserIF& rhs) = delete;
        ParserIF& operator= (ParserIF&& rhs) = delete;

        ~ParserIF() = default;

        AMREX_GPU_HOST_DEVICE inline
        amrex::Real operator() (AMREX_D_DECL(amrex::Real x, amrex::Real y,
                                             amrex::Real z)) const noexcept {
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            return m_parser(x,amrex::Real(0.0),y);
#else
            return m_parser(x,y,z);
#endif
        }

        inline amrex::Real operator() (const amrex::RealArray& p) const noexcept {
            return this->operator()(AMREX_D_DECL(p[0],p[1],p[2]));
        }

    private:
        amrex::ParserExecutor<3> m_parser; //! function parser with three arguments (x,y,z)
    };
}
#endif

// TODO: new per_level implementation might be added to particleSinks?
void
WarpX::InitEB ([[maybe_unused]] const std::string& name)
{
    if (!EB::enabled() && name == "eb2") {
        throw std::runtime_error("InitEB only works when EBs are enabled at runtime");
    }
#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE("EBs only implemented in 2D and 3D");
#endif

#ifdef AMREX_USE_EB
    BL_PROFILE("InitEB");

    const amrex::ParmParse pp_warpx("warpx");
    pp_warpx.query("build_eb_data_per_level", m_build_eb_data_per_level);

    amrex::ParmParse pp_name(name);
    std::string impf;
    if (name == "eb2") {
        pp_warpx.query("eb_implicit_function", impf);
    } else { pp_name.query("implicit_function", impf); }

    std::string geom_type;
    pp_name.query("geom_type", geom_type);

    if (! impf.empty()) {
        auto eb_if_parser = utils::parser::makeParser(impf, {"x", "y", "z"});
        ParserIF const pif(eb_if_parser.compile<3>());
        auto gshop = amrex::EB2::makeShop(pif, eb_if_parser);
         // The last argument of amrex::EB2::Build is the maximum coarsening level
         // to which amrex should try to coarsen the EB.  It will stop after coarsening
         // as much as it can, if it cannot coarsen to that level.  Here we use a big
         // number (e.g., maxLevel()+20) for multigrid solvers.  Because the coarse
         // level has only 1/8 of the cells on the fine level, the memory usage should
         // not be an issue.
        if (m_build_eb_data_per_level) {
            // Build the EB data independently at each mesh-refinement level's own resolution.
            for (int ilev = 0; ilev <= maxLevel(); ++ilev) {
                amrex::EB2::Build(gshop, Geom(ilev), 0, 20);
                if (name == "eb2") {
                    m_eb_index_space.push_back(&(amrex::EB2::IndexSpace::top()));
                }
            }
        } else {
            // Build the EB data once at the finest level and coarsen it for coarser levels.
            amrex::EB2::Build(gshop, Geom(maxLevel()), maxLevel(), maxLevel()+20);
            if (name == "eb2") {
                m_eb_index_space.push_back(&(amrex::EB2::IndexSpace::top()));
            }
        }
    } else if (geom_type == "stl") {
        std::string stl_file;
        pp_name.get("stl_file", stl_file);
        amrex::Real stl_scale = 1._rt;
        pp_name.queryAdd("stl_scale", stl_scale);
        std::vector<amrex::Real> stl_center{0.0_rt, 0.0_rt, 0.0_rt};
        pp_name.queryAdd("stl_center", stl_center);
        bool stl_reverse_normal = false;
        pp_name.queryAdd("stl_reverse_normal", stl_reverse_normal);
        bool stl_use_bvh = true;
        pp_name.queryAdd("stl_use_bvh", stl_use_bvh);
        amrex::EB2::IndexSpace::push(std::make_unique<amrex::EB2::IndexSpaceSTL>
                                    (stl_file, stl_scale,
                                    amrex::Array<amrex::Real,3>{stl_center[0], stl_center[1], stl_center[2]},
                                    int(stl_reverse_normal), Geom(maxLevel()), maxLevel(),
                                    maxLevel()+20, 4, true,
                                    amrex::EB2::ExtendDomainFace(), amrex::EB2::NumCoarsenOpt(),
                                    stl_use_bvh, false));
    } else {
        if (geom_type.empty()) {
            pp_name.add("geom_type", std::string("all_regular"));
        }
        // See the comment above on amrex::EB2::Build for the hard-wired number 20.
        if (m_build_eb_data_per_level) {
            // Build the EB data independently at each mesh-refinement level's own resolution.
            for (int ilev = 0; ilev <= maxLevel(); ++ilev) {
                amrex::EB2::Build(Geom(ilev), 0, 20);
                if (name == "eb2") {
                    m_eb_index_space.push_back(&(amrex::EB2::IndexSpace::top()));
                }
            }
        } else {
            // Build the EB data once at the finest level and coarsen it for coarser levels.
            amrex::EB2::Build(Geom(maxLevel()), maxLevel(), maxLevel()+20);
            if (name == "eb2") {
                m_eb_index_space.push_back(&(amrex::EB2::IndexSpace::top()));
            }
        }
    }
#endif
}

void
WarpX::InitEB ()
{
    InitEB("eb2");
}

// TODO: new per_level implementation might be added to particleSinks?
void
WarpX::ComputeDistanceToEB ([[maybe_unused]] const std::string& field_name)
{
    if (!EB::enabled() and !ParticleSink::enabled()) {
        throw std::runtime_error("ComputeDistanceToEB only works when EBs or particle sinks are enabled at runtime");
    }

#ifdef AMREX_USE_EB
    BL_PROFILE("ComputeDistanceToEB");
    const amrex::EB2::IndexSpace* eb_is = &amrex::EB2::IndexSpace::top();

    for (int lev = 0; lev <= maxLevel(); ++lev) {
        const amrex::EB2::Level& eb_level = eb_is->getLevel(Geom(lev));
        auto* distance_mf = m_fields.get(field_name, lev);

        // 6 Arguments: level, geom, boxarray, dmap, ngrow vector, support
        amrex::EBFArrayBoxFactory sink_eb_fact(
            eb_level,
            Geom(lev),
            distance_mf->boxArray(),
            distance_mf->DistributionMap(),
            amrex::Vector<int>{2, 2, 2},
            amrex::EBSupport::full
        );

        amrex::FillSignedDistance(*distance_mf, eb_level, sink_eb_fact, 1);
        distance_mf->FillBoundary(Geom(lev).periodicity());
    }
#endif
}

void
WarpX::ComputeDistanceToEB ()
{
    if (!EB::enabled()) {
        throw std::runtime_error("ComputeDistanceToEB only works when EBs are enabled at runtime");
    }
#ifdef AMREX_USE_EB
    BL_PROFILE("ComputeDistanceToEB");
    using warpx::fields::FieldType;
    for (int lev=0; lev<=maxLevel(); lev++) {
        auto const* eb_index_space = GetEBIndexSpace(lev);
        const amrex::EB2::Level& eb_level = eb_index_space->getLevel(Geom(lev));
        auto const eb_fact = fieldEBFactory(lev);
        amrex::FillSignedDistance(*m_fields.get(FieldType::distance_to_eb, lev), eb_level, eb_fact, 1);
    }
#endif
}
