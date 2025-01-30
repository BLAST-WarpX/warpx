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

#  include <AMReX.H>
#  include <AMReX_Array.H>
#  include <AMReX_Array4.H>
#  include <AMReX_BLProfiler.H>
#  include <AMReX_Box.H>
#  include <AMReX_BoxArray.H>
#  include <AMReX_BoxList.H>
#  include <AMReX_Config.H>
#  include <AMReX_EB2.H>
#  include <AMReX_EB_utils.H>
#  include <AMReX_FabArray.H>
#  include <AMReX_FabFactory.H>
#  include <AMReX_GpuControl.H>
#  include <AMReX_GpuDevice.H>
#  include <AMReX_GpuQualifiers.H>
#  include <AMReX_IntVect.H>
#  include <AMReX_Loop.H>
#  include <AMReX_MFIter.H>
#  include <AMReX_MultiFab.H>
#  include <AMReX_iMultiFab.H>
#  include <AMReX_ParmParse.H>
#  include <AMReX_Parser.H>
#  include <AMReX_REAL.H>
#  include <AMReX_SPACE.H>
#  include <AMReX_Vector.H>

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

#ifdef AMREX_USE_EB
void
WarpX::ComputeEdgeLengths (ablastr::fields::VectorField& edge_lengths,
                           const amrex::EBFArrayBoxFactory& eb_fact) {
    BL_PROFILE("ComputeEdgeLengths");

#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE("ComputeEdgeLengths only implemented in 2D and 3D");
#endif

    auto const &flags = eb_fact.getMultiEBCellFlagFab();
    auto const &edge_centroid = eb_fact.getEdgeCent();
    for (int idim = 0; idim < 3; ++idim){
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        if (idim == 1) {
            edge_lengths[1]->setVal(0.);
            continue;
        }
#endif
        for (amrex::MFIter mfi(flags); mfi.isValid(); ++mfi){
            amrex::Box const box = mfi.tilebox(edge_lengths[idim]->ixType().toIntVect(),
                                               edge_lengths[idim]->nGrowVect());
            amrex::FabType const fab_type = flags[mfi].getType(box);
            auto const &edge_lengths_dim = edge_lengths[idim]->array(mfi);

            if (fab_type == amrex::FabType::regular) {
                // every cell in box is all regular
                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    edge_lengths_dim(i, j, k) = 1.;
                });
            } else if (fab_type == amrex::FabType::covered) {
                // every cell in box is all covered
                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    edge_lengths_dim(i, j, k) = 0.;
                });
            } else {
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                int idim_amrex = idim;
                if (idim == 2) { idim_amrex = 1; }
                auto const &edge_cent = edge_centroid[idim_amrex]->const_array(mfi);
#elif defined(WARPX_DIM_3D)
                auto const &edge_cent = edge_centroid[idim]->const_array(mfi);
#endif
                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (edge_cent(i, j, k) == amrex::Real(-1.0)) {
                        // This edge is all covered
                        edge_lengths_dim(i, j, k) = 0.;
                    } else if (edge_cent(i, j, k) == amrex::Real(1.0)) {
                        // This edge is all open
                        edge_lengths_dim(i, j, k) = 1.;
                    } else {
                        // This edge is cut.
                        edge_lengths_dim(i, j, k) = 1 - amrex::Math::abs(amrex::Real(2.0)
                                                                        * edge_cent(i, j, k));
                    }

                });
            }
        }
    }
}


void
WarpX::ComputeFaceAreas (VectorField& face_areas,
                         const amrex::EBFArrayBoxFactory& eb_fact) {
    BL_PROFILE("ComputeFaceAreas");

#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE("ComputeFaceAreas only implemented in 2D and 3D");
#endif

    auto const &flags = eb_fact.getMultiEBCellFlagFab();
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    //In 2D the volume frac is actually the area frac.
    auto const &area_frac = eb_fact.getVolFrac();
#elif defined(WARPX_DIM_3D)
    auto const &area_frac = eb_fact.getAreaFrac();
#endif

    for (int idim = 0; idim < 3; ++idim) {
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        if (idim == 0 || idim == 2) {
            face_areas[idim]->setVal(0.);
            continue;
        }
#endif
        for (amrex::MFIter mfi(flags); mfi.isValid(); ++mfi) {
            amrex::Box const box = mfi.tilebox(face_areas[idim]->ixType().toIntVect(),
                                               face_areas[idim]->nGrowVect());
            amrex::FabType const fab_type = flags[mfi].getType(box);
            auto const &face_areas_dim = face_areas[idim]->array(mfi);
            if (fab_type == amrex::FabType::regular) {
                // every cell in box is all regular
                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    face_areas_dim(i, j, k) = amrex::Real(1.);
                });
            } else if (fab_type == amrex::FabType::covered) {
                // every cell in box is all covered
                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    face_areas_dim(i, j, k) = amrex::Real(0.);
                });
            } else {
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                auto const &face = area_frac.const_array(mfi);
#elif defined(WARPX_DIM_3D)
                auto const &face = area_frac[idim]->const_array(mfi);
#endif
                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    face_areas_dim(i, j, k) = face(i, j, k);
                });
            }
        }
    }
}


void
WarpX::ScaleEdges (ablastr::fields::VectorField& edge_lengths,
                   const std::array<amrex::Real,3>& cell_size) {
    BL_PROFILE("ScaleEdges");

#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE("ScaleEdges only implemented in 2D and 3D");
#endif

    for (int idim = 0; idim < 3; ++idim){
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        if (idim == 1) { continue; }
#endif
        for (amrex::MFIter mfi(*edge_lengths[0]); mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.tilebox(edge_lengths[idim]->ixType().toIntVect(),
                                                edge_lengths[idim]->nGrowVect() );
            auto const &edge_lengths_dim = edge_lengths[idim]->array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                edge_lengths_dim(i, j, k) *= cell_size[idim];
            });
        }
    }
}

void
WarpX::ScaleAreas (ablastr::fields::VectorField& face_areas,
                  const std::array<amrex::Real,3>& cell_size) {
    BL_PROFILE("ScaleAreas");

#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_XZ) && !defined(WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE("ScaleAreas only implemented in 2D and 3D");
#endif

    for (int idim = 0; idim < 3; ++idim) {
#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        if (idim == 0 || idim == 2) { continue; }
#endif
        for (amrex::MFIter mfi(*face_areas[0]); mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.tilebox(face_areas[idim]->ixType().toIntVect(),
                                                face_areas[idim]->nGrowVect() );
            amrex::Real const full_area = cell_size[(idim+1)%3]*cell_size[(idim+2)%3];
            auto const &face_areas_dim = face_areas[idim]->array(mfi);

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                face_areas_dim(i, j, k) *= full_area;
            });

        }
    }
}

void
WarpX::MarkReducedShapeCells (
    std::unique_ptr<amrex::iMultiFab> & eb_reduce_particle_shape,
    amrex::EBFArrayBoxFactory const & eb_fact,
    int const particle_shape_order )
{
    // Pre-fill array with 0, including in the ghost cells outside of the domain.
    // (The guard cells in the domain will be updated by `FillBoundary` at the end of this function.)
    eb_reduce_particle_shape->setVal(0, eb_reduce_particle_shape->nGrow());

    // Extract structures for embedded boundaries
    amrex::FabArray<amrex::EBCellFlagFab> const& eb_flag = eb_fact.getMultiEBCellFlagFab();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*eb_reduce_particle_shape); mfi.isValid(); ++mfi) {

        const amrex::Box& box = mfi.tilebox();
        amrex::Array4<int> const & eb_reduce_particle_shape_arr = eb_reduce_particle_shape->array(mfi);

        // Check if the box (including one layer of guard cells) contains a mix of covered and regular cells
        const amrex::Box& eb_info_box = mfi.tilebox(amrex::IntVect::TheCellVector()).grow(1);
        amrex::FabType const fab_type = eb_flag[mfi].getType( eb_info_box );

        if (fab_type == amrex::FabType::regular) { // All cells in the box are regular

            // Every cell in box is regular: do not reduce particle shape in any cell
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                eb_reduce_particle_shape_arr(i, j, k) = 0;
            });

        } else if (fab_type == amrex::FabType::covered) { // All cells in the box are covered

            // Every cell in box is fully covered: reduce particle shape
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                eb_reduce_particle_shape_arr(i, j, k) = 1;
            });

        } else { // The box contains a mix of covered and regular cells

            auto const & flag = eb_flag[mfi].array();

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                // Check if any of the neighboring cells over which the particle shape might extend
                // are either partially or fully covered. In this case, set eb_reduce_particle_shape_arr
                // to one for this cell, to indicate that the particle should use an order 1 shape
                // (This ensures that the particle never deposits any charge in a partially or
                // fully covered cell, even with higher-order shapes)
                // Note: in the code below `particle_shape_order/2` corresponds to the number of neighboring cells
                // over which the shape factor could extend, in each direction.
                int const i_start = i-particle_shape_order/2;
                int const i_end = i+particle_shape_order/2;
#if AMREX_SPACEDIM > 1
                int const j_start = j-particle_shape_order/2;
                int const j_end = j+particle_shape_order/2;
#else
                int const j_start = j;
                int const j_end = j;
#endif
#if AMREX_SPACEDIM > 2
                int const k_start = k-particle_shape_order/2;
                int const k_end = k+particle_shape_order/2;
#else
                int const k_start = k;
                int const k_end = k;
#endif
                int reduce_shape = 0;
                for (int i_cell = i_start; i_cell <= i_end; ++i_cell) {
                    for (int j_cell = j_start; j_cell <= j_end; ++j_cell) {
                        for (int k_cell = k_start; k_cell <= k_end; ++k_cell) {
                            // `isRegular` returns `false` if the cell is either partially or fully covered.
                            if ( !flag(i_cell, j_cell, k_cell).isRegular() ) {
                                reduce_shape = 1;
                            }
                        }
                    }
                }
                eb_reduce_particle_shape_arr(i, j, k) = reduce_shape;
            });

        }

    }

    // FillBoundary to set the values in the guard cells
    eb_reduce_particle_shape->FillBoundary(Geom(0).periodicity());

}

void
WarpX::MarkUpdateCellsStairCase (
    std::array< std::unique_ptr<amrex::iMultiFab>,3> & eb_update,
    ablastr::fields::VectorField const& field,
    amrex::EBFArrayBoxFactory const & eb_fact )
{

    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // Extract structures for embedded boundaries
    amrex::FabArray<amrex::EBCellFlagFab> const& eb_flag = eb_fact.getMultiEBCellFlagFab();

    for (int idim = 0; idim < 3; ++idim) {

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(*field[idim]); mfi.isValid(); ++mfi) {

            const amrex::Box& box = mfi.tilebox();
            amrex::Array4<int> const & eb_update_arr = eb_update[idim]->array(mfi);

            // Check if the box (including one layer of guard cells) contains a mix of covered and regular cells
            const amrex::Box& eb_info_box = mfi.tilebox(amrex::IntVect::TheCellVector()).grow(1);
            amrex::FabType const fab_type = eb_flag[mfi].getType( eb_info_box );

            if (fab_type == amrex::FabType::regular) { // All cells in the box are regular

                // Every cell in box is regular: update field in every cell
                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    eb_update_arr(i, j, k) = 1;
                });

            } else if (fab_type == amrex::FabType::covered) { // All cells in the box are covered

                // Every cell in box is fully covered: do not update field
                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    eb_update_arr(i, j, k) = 0;
                });

            } else { // The box contains a mix of covered and regular cells

                auto const & flag = eb_flag[mfi].array();
                auto index_type = field[idim]->ixType();

                amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                    // Stair-case approximation: If neighboring cells of this gridpoint
                    // are either partially or fully covered: do not update field

                    // The number of cells that we need to check depend on the index type
                    // of the `eb_update_arr` in each direction.
                    // If `eb_update_arr` is nodal in a given direction, we need to check the cells
                    // to the left and right of this nodal gridpoint.
                    // For instance, if `eb_update_arr` is nodal in the first dimension, we need
                    // to check the cells at index i-1 and at index i, since, with AMReX indexing conventions,
                    // these are the neighboring cells for the nodal gripoint at index i.
                    // If `eb_update_arr` is cell-centerd in a given direction, we only need to check
                    // the cell at the same position (e.g., in the first dimension: the cell at index i).
                    int const i_start = ( index_type.nodeCentered(0) )? i-1 : i;
#if AMREX_SPACEDIM > 1
                    int const j_start = ( index_type.nodeCentered(1) )? j-1 : j;
#else
                    int const j_start = j;
#endif
#if AMREX_SPACEDIM > 2
                    int const k_start = ( index_type.nodeCentered(2) )? k-1 : k;
#else
                    int const k_start = k;
#endif
                    // Loop over neighboring cells
                    int eb_update_flag = 1;
                    for (int i_cell = i_start; i_cell <= i; ++i_cell) {
                        for (int j_cell = j_start; j_cell <= j; ++j_cell) {
                            for (int k_cell = k_start; k_cell <= k; ++k_cell) {
                                // If one of the neighboring is either partially or fully covered
                                // (i.e. if they are not regular cells), do not update field
                                // (`isRegular` returns `false` if the cell is either partially or fully covered.)
                                if ( !flag(i_cell, j_cell, k_cell).isRegular() ) {
                                    eb_update_flag = 0;
                                }
                            }
                        }
                    }
                    eb_update_arr(i, j, k) = eb_update_flag;
                });

            }

        }

    }

}

void
WarpX::MarkUpdateECellsECT (
    std::array< std::unique_ptr<amrex::iMultiFab>,3> & eb_update_E,
    ablastr::fields::VectorField const& edge_lengths )
{

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(*eb_update_E[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const amrex::Box& tbx = mfi.tilebox( eb_update_E[0]->ixType().toIntVect(), eb_update_E[0]->nGrowVect() );
        const amrex::Box& tby = mfi.tilebox( eb_update_E[1]->ixType().toIntVect(), eb_update_E[1]->nGrowVect() );
        const amrex::Box& tbz = mfi.tilebox( eb_update_E[2]->ixType().toIntVect(), eb_update_E[2]->nGrowVect() );

        amrex::Array4<int> const & eb_update_Ex_arr = eb_update_E[0]->array(mfi);
        amrex::Array4<int> const & eb_update_Ey_arr = eb_update_E[1]->array(mfi);
        amrex::Array4<int> const & eb_update_Ez_arr = eb_update_E[2]->array(mfi);

        amrex::Array4<amrex::Real> const & lx_arr = edge_lengths[0]->array(mfi);
        amrex::Array4<amrex::Real> const & lz_arr = edge_lengths[2]->array(mfi);
#if defined(WARPX_DIM_3D)
        amrex::Array4<amrex::Real> const & ly_arr = edge_lengths[1]->array(mfi);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        amrex::Dim3 const lx_lo = amrex::lbound(lx_arr);
        amrex::Dim3 const lx_hi = amrex::ubound(lx_arr);
        amrex::Dim3 const lz_lo = amrex::lbound(lz_arr);
        amrex::Dim3 const lz_hi = amrex::ubound(lz_arr);
#endif

        amrex::ParallelFor (tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Do not update Ex if the edge on which it lives is fully covered
                eb_update_Ex_arr(i, j, k) = (lx_arr(i, j, k) == 0)? 0 : 1;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
#ifdef WARPX_DIM_3D
                // In 3D: Do not update Ey if the edge on which it lives is fully covered
                eb_update_Ey_arr(i, j, k) = (ly_arr(i, j, k) == 0)? 0 : 1;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                // In XZ and RZ: Ey is associated with a mesh node,
                // so we need to check if  the mesh node is covered
                if((lx_arr(std::min(i  , lx_hi.x), std::min(j  , lx_hi.y), k)==0)
                 ||(lx_arr(std::max(i-1, lx_lo.x), std::min(j  , lx_hi.y), k)==0)
                 ||(lz_arr(std::min(i  , lz_hi.x), std::min(j  , lz_hi.y), k)==0)
                 ||(lz_arr(std::min(i  , lz_hi.x), std::max(j-1, lz_lo.y), k)==0)) {
                    eb_update_Ey_arr(i, j, k) = 0;
                } else {
                    eb_update_Ey_arr(i, j, k) = 1;
                }
#endif
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Do not update Ez if the edge on which it lives is fully covered
                eb_update_Ez_arr(i, j, k) = (lz_arr(i, j, k) == 0)? 0 : 1;
            }
        );

    }
}

void
WarpX::MarkUpdateBCellsECT (
    std::array< std::unique_ptr<amrex::iMultiFab>,3> & eb_update_B,
    ablastr::fields::VectorField const& face_areas,
    ablastr::fields::VectorField const& edge_lengths )
{

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for ( amrex::MFIter mfi(*eb_update_B[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const amrex::Box& tbx = mfi.tilebox( eb_update_B[0]->ixType().toIntVect(), eb_update_B[0]->nGrowVect() );
        const amrex::Box& tby = mfi.tilebox( eb_update_B[1]->ixType().toIntVect(), eb_update_B[1]->nGrowVect() );
        const amrex::Box& tbz = mfi.tilebox( eb_update_B[2]->ixType().toIntVect(), eb_update_B[2]->nGrowVect() );

        amrex::Array4<int> const & eb_update_Bx_arr = eb_update_B[0]->array(mfi);
        amrex::Array4<int> const & eb_update_By_arr = eb_update_B[1]->array(mfi);
        amrex::Array4<int> const & eb_update_Bz_arr = eb_update_B[2]->array(mfi);

#ifdef WARPX_DIM_3D
        amrex::Array4<amrex::Real> const & Sx_arr = face_areas[0]->array(mfi);
        amrex::Array4<amrex::Real> const & Sy_arr = face_areas[1]->array(mfi);
        amrex::Array4<amrex::Real> const & Sz_arr = face_areas[2]->array(mfi);
        amrex::ignore_unused(edge_lengths);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        amrex::Array4<amrex::Real> const & Sy_arr = face_areas[1]->array(mfi);
        amrex::Array4<amrex::Real> const & lx_arr = edge_lengths[0]->array(mfi);
        amrex::Array4<amrex::Real> const & lz_arr = edge_lengths[2]->array(mfi);
#endif
        amrex::ParallelFor (tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
#ifdef WARPX_DIM_3D
                // In 3D: do not update Bx if the face on which it lives is fully covered
                eb_update_Bx_arr(i, j, k) = (Sx_arr(i, j, k) == 0)? 0 : 1;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                //In XZ and RZ, Bx lives on a z-edge ; do not update if fully covered
                eb_update_Bx_arr(i, j, k) = (lz_arr(i, j, k) == 0)? 0 : 1;
#endif
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Do not update By if the face on which it lives is fully covered
                eb_update_By_arr(i, j, k) = (Sy_arr(i, j, k) == 0)? 0 : 1;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
#ifdef WARPX_DIM_3D
                // In 3D: do not update Bz if the face on which it lives is fully covered
                eb_update_Bz_arr(i, j, k) = (Sz_arr(i, j, k) == 0)? 0 : 1;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                //In XZ and RZ, Bz lives on a x-edge ; do not update if fully covered
                eb_update_Bz_arr(i, j, k) = (lx_arr(i, j, k) == 0)? 0 : 1;
#endif
            }
        );

    }
}

void
WarpX::MarkExtensionCells ()
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

#ifndef WARPX_DIM_RZ
    auto const &cell_size = CellSize(maxLevel());

#if !defined(WARPX_DIM_3D) && !defined(WARPX_DIM_XZ)
    WARPX_ABORT_WITH_MESSAGE("MarkExtensionCells only implemented in 2D and 3D");
#endif

    for (int idim = 0; idim < 3; ++idim) {
#if defined(WARPX_DIM_XZ)
        if (idim == 0 || idim == 2) {
            m_flag_info_face[maxLevel()][idim]->setVal(0.);
            m_flag_ext_face[maxLevel()][idim]->setVal(0.);
            continue;
        }
#endif
        for (amrex::MFIter mfi(*m_fields.get(FieldType::Bfield_fp, Direction{idim}, maxLevel())); mfi.isValid(); ++mfi) {
            auto* face_areas_idim_max_lev =
                m_fields.get(FieldType::face_areas, Direction{idim}, maxLevel());

            const amrex::Box& box = mfi.tilebox(face_areas_idim_max_lev->ixType().toIntVect(),
                                                face_areas_idim_max_lev->nGrowVect() );

            auto const &S = face_areas_idim_max_lev->array(mfi);
            auto const &flag_info_face = m_flag_info_face[maxLevel()][idim]->array(mfi);
            auto const &flag_ext_face = m_flag_ext_face[maxLevel()][idim]->array(mfi);
            const auto &lx = m_fields.get(FieldType::edge_lengths, Direction{0}, maxLevel())->array(mfi);
            const auto &ly = m_fields.get(FieldType::edge_lengths, Direction{1}, maxLevel())->array(mfi);
            const auto &lz = m_fields.get(FieldType::edge_lengths, Direction{2}, maxLevel())->array(mfi);
            auto const &mod_areas_dim = m_fields.get(FieldType::area_mod, Direction{idim}, maxLevel())->array(mfi);

            const amrex::Real dx = cell_size[0];
            const amrex::Real dy = cell_size[1];
            const amrex::Real dz = cell_size[2];

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Minimal area for this cell to be stable
                mod_areas_dim(i, j, k) = S(i, j, k);
                double S_stab;
                if (idim == 0){
                    S_stab = 0.5 * std::max({ly(i, j, k) * dz, ly(i, j, k + 1) * dz,
                                                    lz(i, j, k) * dy, lz(i, j + 1, k) * dy});
                }else if (idim == 1){
#ifdef WARPX_DIM_XZ
                    S_stab = 0.5 * std::max({lx(i, j, k) * dz, lx(i, j + 1, k) * dz,
                                             lz(i, j, k) * dx, lz(i + 1, j, k) * dx});
#elif defined(WARPX_DIM_3D)
                    S_stab = 0.5 * std::max({lx(i, j, k) * dz, lx(i, j, k + 1) * dz,
                                             lz(i, j, k) * dx, lz(i + 1, j, k) * dx});
#endif
                }else {
                    S_stab = 0.5 * std::max({lx(i, j, k) * dy, lx(i, j + 1, k) * dy,
                                             ly(i, j, k) * dx, ly(i + 1, j, k) * dx});
                }

                // Does this face need to be extended?
                // The difference between flag_info_face and flag_ext_face is that:
                //     - for every face flag_info_face contains a:
                //          * 0 if the face needs to be extended
                //          * 1 if the face is large enough to lend area to other faces
                //          * 2 if the face is actually intruded by other face
                //       Here we only take care of the first two cases. The entries corresponding
                //       to the intruded faces are going to be set in the function ComputeFaceExtensions
                //     - for every face flag_ext_face contains a:
                //          * 1 if the face needs to be extended
                //          * 0 otherwise
                //       In the function ComputeFaceExtensions, after the cells are extended, the
                //       corresponding entries in flag_ext_face are set to zero. This helps to keep
                //       track of which cells could not be extended
                flag_ext_face(i, j, k) = int(S(i, j, k) < S_stab && S(i, j, k) > 0);
                if(flag_ext_face(i, j, k)){
                    flag_info_face(i, j, k) = 0;
                }
                // Is this face available to lend area to other faces?
                // The criterion is that the face has to be interior and not already unstable itself
                if(int(S(i, j, k) > 0 && !flag_ext_face(i, j, k))) {
                    flag_info_face(i, j, k) = 1;
                }
            });
        }
    }
#endif
}
#endif

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
        amrex::FillSignedDistance(*m_fields.get(FieldType::distance_to_eb, lev), eb_level, eb_fact, 1);
    }
#endif
}
