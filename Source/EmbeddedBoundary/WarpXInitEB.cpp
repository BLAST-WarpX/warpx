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

#  include <ablastr/utils/Communication.H>

#   include <AMReX_BLProfiler.H>
#   include <AMReX_BoxArray.H>
#   include <AMReX_Config.H>
#   include <AMReX_EB2.H>
#   include <AMReX_EB2_GeometryShop.H>
#   include <AMReX_EB2_IF_Base.H>
#   include <AMReX_EB_utils.H>
#   include <AMReX_GpuQualifiers.H>
#   include <AMReX_ParmParse.H>
#   include <AMReX_REAL.H>
#   include <AMReX_SPACE.H>

#  include <cmath>
#  include <cstdlib>
#  include <string>

using namespace ablastr::fields;

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

void
WarpX::InitEB ()
{
    if (!EB::enabled()) {
        throw std::runtime_error("InitEB only works when EBs are enabled at runtime");
    }

#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE("EBs only implemented in 2D and 3D");
#endif

#ifdef AMREX_USE_EB
    BL_PROFILE("InitEB");

    const amrex::ParmParse pp_warpx("warpx");
    std::string impf;
    pp_warpx.query("eb_implicit_function", impf);
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
        amrex::EB2::Build(gshop, Geom(maxLevel()), maxLevel(), maxLevel()+20);
    } else {
        amrex::ParmParse pp_eb2("eb2");
        if (!pp_eb2.contains("geom_type")) {
            std::string const geom_type = "all_regular";
            pp_eb2.add("geom_type", geom_type); // use all_regular by default
        }
        // See the comment above on amrex::EB2::Build for the hard-wired number 20.
        amrex::EB2::Build(Geom(maxLevel()), maxLevel(), maxLevel()+20);
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
    const amrex::EB2::IndexSpace& eb_is = amrex::EB2::IndexSpace::top();
    for (int lev=0; lev<=maxLevel(); lev++) {
        const amrex::EB2::Level& eb_level = eb_is.getLevel(Geom(lev));
        auto const eb_fact = fieldEBFactory(lev);
        auto * const distance_to_eb = m_fields.get(FieldType::distance_to_eb, lev);
        amrex::FillSignedDistance(*distance_to_eb, eb_level, eb_fact, 1);

        // EXPERIMENT (opt-in, diagnostic): overwrite the discrete signed distance
        // with the EXACT analytic distance to a cylinder of radius R centred on
        // the z-axis, phi = R - sqrt(x^2+y^2) (fluid positive), clamped to the SAME
        // band [min,max] = +/- ls_roof that FillSignedDistance produced. This lets
        // us A/B whether the discrete-levelset faceting / normal error (the small
        // m=4/m=8 azimuthal signature) feeds the solution. Enable by setting
        //   hybrid_pic_model.eb_analytic_cylinder_radius = R  (<=0 disables;
        // byte-identical). Read from the hybrid_pic_model namespace so the PICMI
        // HybridPICSolver kwarg (eb_analytic_cylinder_radius) flushes it reliably.
        {
            using namespace amrex::literals;
            amrex::Real R_cyl = -1.0_rt;
            amrex::ParmParse const pp_hybrid("hybrid_pic_model");
            pp_hybrid.query("eb_analytic_cylinder_radius", R_cyl);
            if (R_cyl > 0.0_rt) {
                amrex::Real const lo = distance_to_eb->min(0);  // -ls_roof clamp
                amrex::Real const hi = distance_to_eb->max(0);  // +ls_roof clamp
                auto const problo = Geom(lev).ProbLoArray();
                auto const dx = Geom(lev).CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
                for (amrex::MFIter mfi(*distance_to_eb, amrex::TilingIfNotGPU());
                     mfi.isValid(); ++mfi)
                {
                    amrex::Box const gbx = mfi.growntilebox();
                    amrex::Array4<amrex::Real> const fab = distance_to_eb->array(mfi);
                    amrex::ParallelFor(gbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        // distance_to_eb is NODAL: node (i,j,k) sits at problo + i*dx
                        amrex::Real const x = problo[0] + amrex::Real(i) * dx[0];
                        amrex::Real const y = problo[1] + amrex::Real(j) * dx[1];
                        amrex::Real const rr = std::sqrt(x * x + y * y);
                        amrex::Real phi = R_cyl - rr;  // fluid (r<R) positive
                        phi = amrex::min(hi, amrex::max(lo, phi));
                        fab(i, j, k) = phi;
                    });
                }
            }
        }

        // distance_to_eb is nodal and FillSignedDistance computes each box's
        // values (valid + ghosts) independently from that box's local facet
        // search, so the shared box-seam nodes can disagree slightly between
        // neighbouring boxes. The eb_update masks and the EB-fill classification
        // are derived node-by-node from this distance, so an unreconciled seam
        // node can be classified inconsistently across boxes (fluid on one side,
        // covered on the other) right where a box boundary crosses the wall.
        // Reconcile the shared seam nodes (owner with the smaller global box
        // index wins) and propagate to ghosts, using the ablastr nodal-sync
        // FillBoundary (FillBoundaryAndSync under the hood) so the whole
        // downstream EB machinery is seam-consistent in multi-box runs.
        ablastr::utils::communication::FillBoundary(
            *distance_to_eb, WarpX::do_single_precision_comms,
            Geom(lev).periodicity(), /*nodal_sync=*/true);
    }
#endif
}
