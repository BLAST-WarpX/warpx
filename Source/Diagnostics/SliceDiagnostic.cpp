/* Copyright 2019-2020 Luca Fedeli, Revathi Jambunathan, Weiqun Zhang
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SliceDiagnostic.H"

#include "Fields.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/utils/Communication.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_Dim3.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_SPACE.H>

#include <cmath>
#include <memory>
#include <sstream>

using namespace amrex;
using warpx::fields::FieldType;

/* \brief
 *  The functions creates the slice for diagnostics based on the user-input.
 *  The slice can be 1D, 2D, or 3D and it inherits the index type of the underlying data.
 *  The implementation assumes that the slice is aligned with the coordinate axes.
 *  The input parameters are modified if the user-input does not comply with requirements of coarsenability or if the slice extent is not contained within the simulation domain.
 *  First a slice multifab (smf) with cell size equal to that of the simulation grid is created such that it extends from slice.dim_lo to slice.dim_hi and shares the same index space as the source multifab (mf)
 *  The values are copied from src mf to dst smf using amrex::ParallelCopy
 *  If interpolation is required, then on the smf, using data points stored in the ghost cells, the data in interpolated.
 *  If coarsening is required, then a coarse slice multifab is generated (cs_mf) and the
 *  values of the refined slice (smf) is averaged down to obtain the coarse slice.
 *  \param mf is the source multifab containing the field data
 *  \param dom_geom is the geometry of the domain and used in the function to obtain the
 *  CellSize of the underlying grid.
 *  \param slice_realbox defines the extent of the slice
 *  \param slice_cr_ratio provides the coarsening ratio for diagnostics
 */

std::unique_ptr<MultiFab>
CreateSlice( const MultiFab& mf, const Vector<Geometry> &dom_geom,
             RealBox &slice_realbox, IntVect &slice_cr_ratio )
{
    std::unique_ptr<MultiFab> smf;
    std::unique_ptr<MultiFab> cs_mf;

    Vector<int> slice_ncells(AMREX_SPACEDIM);
    const int nghost = 1;
    auto nlevels = static_cast<int>(dom_geom.size());
    const int ncomp = (mf).nComp();

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( nlevels==1,
       "Slice diagnostics does not work with mesh refinement yet (TO DO).");

    const auto conversionType = (mf).ixType();
    IntVect SliceType(AMREX_D_DECL(0,0,0));
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim )
    {
        SliceType[idim] = conversionType.nodeCentered(idim);
    }

    const RealBox& real_box = dom_geom[0].ProbDomain();
    RealBox slice_cc_nd_box;
    const int default_grid_size = 32;
    int slice_grid_size = default_grid_size;

    bool interpolate = false;
    bool coarsen = false;

    // same index space as domain //
    IntVect slice_lo(AMREX_D_DECL(0,0,0));
    IntVect slice_hi(AMREX_D_DECL(1,1,1));
    IntVect interp_lo(AMREX_D_DECL(0,0,0));

    CheckSliceInput(real_box, slice_cc_nd_box, slice_realbox, slice_cr_ratio,
                    dom_geom, SliceType, slice_lo,
                    slice_hi, interp_lo);
    int configuration_dim = 0;
    // Determine if interpolation is required and number of cells in slice //
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {

        // Flag for interpolation if required //
        if ( interp_lo[idim] == 1) {
            interpolate = true;
        }

        // For the case when a dimension is reduced //
        if ( ( slice_hi[idim] - slice_lo[idim]) == 1) {
            slice_ncells[idim] = 1;
        }
        else {
            slice_ncells[idim] = ( slice_hi[idim] - slice_lo[idim] + 1 )
                                    / slice_cr_ratio[idim];

            const int refined_ncells = slice_hi[idim] - slice_lo[idim] + 1 ;
            if ( slice_cr_ratio[idim] > 1) {
                coarsen = true;

                // modify slice_grid_size if >= refines_cells //
                if ( slice_grid_size >= refined_ncells ) {
                    slice_grid_size = refined_ncells - 1;
                }

            }
            configuration_dim += 1;
        }
    }
    if (configuration_dim==1) {
      ablastr::warn_manager::WMRecordWarning("Diagnostics",
         "The slice configuration is 1D and cannot be visualized using yt.");
    }

    // Slice generation with index type inheritance //
    const Box slice(slice_lo, slice_hi);

    Vector<BoxArray> sba(1);
    sba[0].define(slice);
    sba[0].maxSize(slice_grid_size);

    // Distribution mapping for slice can be different from that of domain //
    Vector<DistributionMapping> sdmap(1);
    sdmap[0] = DistributionMapping{sba[0]};

    smf = std::make_unique<MultiFab>(amrex::convert(sba[0],SliceType), sdmap[0],
                           ncomp, nghost);

    // Copy data from domain to slice that has same cell size as that of //
    // the domain mf. src and dst have the same number of ghost cells    //
    const amrex::IntVect nghost_vect(AMREX_D_DECL(nghost, nghost, nghost));
    ablastr::utils::communication::ParallelCopy(*smf, mf, 0, 0, ncomp, nghost_vect, nghost_vect, WarpX::do_single_precision_comms);

    // interpolate if required on refined slice //
    if (interpolate) {
       InterpolateSliceValues( *smf, interp_lo, slice_cc_nd_box, dom_geom,
                               ncomp, nghost, slice_lo, slice_hi, SliceType, real_box);
    }


    if (!coarsen) {
       return smf;
    }
    else {
        Vector<BoxArray> crse_ba(1);
        crse_ba[0] = sba[0];
        crse_ba[0].coarsen(slice_cr_ratio);

        AMREX_ALWAYS_ASSERT(crse_ba[0].size() == sba[0].size());

        cs_mf = std::make_unique<MultiFab>(amrex::convert(crse_ba[0],SliceType),
                                           sdmap[0], ncomp,nghost);

        const MultiFab& mfSrc = *smf;
        MultiFab& mfDst = *cs_mf;

        auto & warpx = WarpX::GetInstance();

        using ablastr::fields::Direction;

        MFIter mfi_dst(mfDst);
        for (MFIter mfi(mfSrc); mfi.isValid(); ++mfi) {

            Array4<Real const> const& Src_fabox = mfSrc.const_array(mfi);

            const Box& Dst_bx = mfi_dst.validbox();
            Array4<Real> const& Dst_fabox = mfDst.array(mfi_dst);

            const int scomp = 0;
            const int dcomp = 0;

            const IntVect cctype(AMREX_D_DECL(0,0,0));
            if( SliceType==cctype ) {
                amrex::amrex_avgdown(Dst_bx, Dst_fabox, Src_fabox, dcomp, scomp,
                                     ncomp, slice_cr_ratio);
            }
            const IntVect ndtype(AMREX_D_DECL(1,1,1));
            if( SliceType == ndtype ) {
                amrex::amrex_avgdown_nodes(Dst_bx, Dst_fabox, Src_fabox, dcomp,
                                           scomp, ncomp, slice_cr_ratio);
            }
            if( SliceType == warpx.m_fields.get(FieldType::Efield_aux, Direction{0}, 0)->ixType().toIntVect() ) {
                amrex::amrex_avgdown_edges(Dst_bx, Dst_fabox, Src_fabox, dcomp,
                                           scomp, ncomp, slice_cr_ratio, 0);
            }
            if( SliceType == warpx.m_fields.get(FieldType::Efield_aux, Direction{1}, 0)->ixType().toIntVect() ) {
                amrex::amrex_avgdown_edges(Dst_bx, Dst_fabox, Src_fabox, dcomp,
                                           scomp, ncomp, slice_cr_ratio, 1);
            }
            if( SliceType == warpx.m_fields.get(FieldType::Efield_aux, Direction{2}, 0)->ixType().toIntVect() ) {
                amrex::amrex_avgdown_edges(Dst_bx, Dst_fabox, Src_fabox, dcomp,
                                           scomp, ncomp, slice_cr_ratio, 2);
            }
            if( SliceType == warpx.m_fields.get(FieldType::Bfield_aux, Direction{0}, 0)->ixType().toIntVect() ) {
                amrex::amrex_avgdown_faces(Dst_bx, Dst_fabox, Src_fabox, dcomp,
                                           scomp, ncomp, slice_cr_ratio, 0);
            }
            if( SliceType == warpx.m_fields.get(FieldType::Bfield_aux, Direction{1}, 0)->ixType().toIntVect() ) {
                amrex::amrex_avgdown_faces(Dst_bx, Dst_fabox, Src_fabox, dcomp,
                                           scomp, ncomp, slice_cr_ratio, 1);
            }
            if( SliceType == warpx.m_fields.get(FieldType::Bfield_aux, Direction{2}, 0)->ixType().toIntVect() ) {
                amrex::amrex_avgdown_faces(Dst_bx, Dst_fabox, Src_fabox, dcomp,
                                           scomp, ncomp, slice_cr_ratio, 2);
            }

            if ( mfi_dst.isValid() ) {
                ++mfi_dst;
            }

        }
        return cs_mf;
    }
}


/* \brief
 *  This function modifies the slice input parameters under certain conditions.
 *  The coarsening ratio, slice_cr_ratio is modified if the input is not an exponent of 2.
 *  for example, if the coarsening ratio is 3, 5 or 6, which is not an exponent of 2,
 *  then the value of coarsening ratio is modified to the nearest exponent of 2.
 *  The default value for coarsening ratio is 1.
 *  slice_realbox.lo and slice_realbox.hi are set equal to the simulation domain lo and hi
 *  if for the user-input for the slice lo and hi coordinates are outside the domain.
 *  If the slice_realbox.lo and slice_realbox.hi coordinates do not align with the data
 *  points and the number of cells in that dimension is greater than 1, and if the extent of
 *  the slice in that dimension is not coarsenable, then the value lo and hi coordinates are
 *  shifted to the nearest coarsenable point to include some extra data points in the slice.
 *  If slice_realbox.lo==slice_realbox.hi, then that dimension has only one cell and no
 *  modifications are made to the value. If the lo and hi do not align with a data point,
 *  then it is flagged for interpolation.
 *  \param real_box a Real box defined for the underlying domain.
 *  \param slice_realbox a Real box for defining the slice dimension.
 *  \param slice_cc_nd_box a Real box for defining the modified lo and hi of the slice
 *  such that the coordinates align with the underlying data points.
 *  If the dimension is reduced to have only one cell, the slice_realbox is not modified and  *  instead the values are interpolated to the coordinate from the nearest data points.
 *  \param slice_cr_ratio contains values of the coarsening ratio which may be modified
 *  if the input values do not satisfy coarsenability conditions.
 *  \param slice_lo and slice_hi are the index values of the slice
 *  \param interp_lo are set to 0 or 1 if they are flagged for interpolation.
 *  The slice shares the same index space as that of the simulation domain.
 */


void
CheckSliceInput( const RealBox real_box, RealBox &slice_cc_nd_box,
               RealBox &slice_realbox, IntVect &slice_cr_ratio,
               Vector<Geometry> dom_geom, IntVect const SliceType,
               IntVect &slice_lo, IntVect &slice_hi, IntVect &interp_lo)
{

    IntVect slice_lo2(AMREX_D_DECL(0,0,0));
    for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // Modify coarsening ratio if the input value is not an exponent of 2 for AMR //
        if ( slice_cr_ratio[idim] > 0 ) {
            const int log_cr_ratio =
               static_cast<int>(floor ( log2( double(slice_cr_ratio[idim]))));
            slice_cr_ratio[idim] =
               static_cast<int> (exp2( double(log_cr_ratio) ));
        }

        //// Default coarsening ratio is 1 //
        // Modify lo if input is out of bounds //
        if ( slice_realbox.lo(idim) < real_box.lo(idim) ) {
            slice_realbox.setLo( idim, real_box.lo(idim));
            std::stringstream warnMsg;
            warnMsg << " slice lo is out of bounds. " <<
                       " Modified it in dimension " << idim <<
                       " to be aligned with the domain box.";
            ablastr::warn_manager::WMRecordWarning("Diagnostics",
               warnMsg.str(), ablastr::warn_manager::WarnPriority::low);
        }

        // Modify hi if input in out od bounds //
        if ( slice_realbox.hi(idim) > real_box.hi(idim) ) {
            slice_realbox.setHi( idim, real_box.hi(idim));
            std::stringstream warnMsg;
            warnMsg << " slice hi is out of bounds. " <<
                       " Modified it in dimension " << idim <<
                       " to be aligned with the domain box.";
            ablastr::warn_manager::WMRecordWarning("Diagnostics",
               warnMsg.str(), ablastr::warn_manager::WarnPriority::low);
        }

        const auto very_small_number = 1E-10;

        // Factor to ensure index values computation depending on index type //
        const double fac = ( 1.0 - SliceType[idim] )*dom_geom[0].CellSize(idim)*0.5;
        // if dimension is reduced to one cell length //
        if ( slice_realbox.hi(idim) - slice_realbox.lo(idim) <= 0)
        {
            slice_cc_nd_box.setLo( idim, slice_realbox.lo(idim) );
            slice_cc_nd_box.setHi( idim, slice_realbox.hi(idim) );

            if ( slice_cr_ratio[idim] > 1) { slice_cr_ratio[idim] = 1; }

            // check for interpolation -- compute index lo with floor and ceil
            if ( slice_cc_nd_box.lo(idim) - real_box.lo(idim) >= fac ) {
                slice_lo[idim] = static_cast<int>(
                                 floor( ( (slice_cc_nd_box.lo(idim)
                                 - (real_box.lo(idim) + fac ) )
                                 / dom_geom[0].CellSize(idim)) + fac * very_small_number) );
                slice_lo2[idim] = static_cast<int>(
                                 ceil( ( (slice_cc_nd_box.lo(idim)
                                 - (real_box.lo(idim) + fac) )
                                 / dom_geom[0].CellSize(idim)) - fac * very_small_number) );
            }
            else {
                slice_lo[idim] = static_cast<int>(
                                  std::round( (slice_cc_nd_box.lo(idim)
                                  - (real_box.lo(idim) ) )
                                  / dom_geom[0].CellSize(idim)) );
                slice_lo2[idim] = static_cast<int>(
                                  std::ceil((slice_cc_nd_box.lo(idim)
                                  - (real_box.lo(idim) ) )
                                  / dom_geom[0].CellSize(idim) ) );
            }

            // flag for interpolation -- if reduced dimension location  //
            //                           does not align with data point //
            if ( slice_lo[idim] == slice_lo2[idim]) {
               if ( slice_cc_nd_box.lo(idim) - real_box.lo(idim) < fac ) {
                  interp_lo[idim] = 1;
               }
            }
            else {
               interp_lo[idim] = 1;
            }

            // ncells = 1 if dimension is reduced //
            slice_hi[idim] = slice_lo[idim] + 1;
        }
        else
        {
            // moving realbox.lo and realbox.hi to nearest coarsenable grid point //
            auto index_lo = static_cast<int>(floor(((slice_realbox.lo(idim) +  very_small_number
                            - (real_box.lo(idim))) / dom_geom[0].CellSize(idim))) );
            auto index_hi = static_cast<int>(ceil(((slice_realbox.hi(idim)  - very_small_number
                            - (real_box.lo(idim))) / dom_geom[0].CellSize(idim))) );

            bool modify_cr = true;

            while ( modify_cr ) {
                int lo_new = index_lo;
                int hi_new = index_hi;
                const int mod_lo = index_lo % slice_cr_ratio[idim];
                const int mod_hi = index_hi % slice_cr_ratio[idim];
                modify_cr = false;

                // To ensure that the index.lo is coarsenable //
                if ( mod_lo > 0) {
                   lo_new = index_lo - mod_lo;
                }
                // To ensure that the index.hi is coarsenable //
                if ( mod_hi > 0) {
                   hi_new = index_hi + (slice_cr_ratio[idim] - mod_hi);
                }

                // If modified index.hi is > baselinebox.hi, move the point  //
                // to the previous coarsenable point                         //
                const auto small_number = 0.01;
                if ( (hi_new * dom_geom[0].CellSize(idim))
                      > real_box.hi(idim) - real_box.lo(idim) + dom_geom[0].CellSize(idim)*small_number)
                {
                   hi_new = index_hi - mod_hi;
                }

                if ( (hi_new - lo_new) == 0 ){
                   std::stringstream warnMsg;
                   warnMsg << " Coarsening ratio  " << slice_cr_ratio[idim] << " in dim "<< idim <<
                     "is leading to zero cells for slice." << " Thus reducing cr_ratio by half.\n";

                     ablastr::warn_manager::WMRecordWarning("Diagnostics",
                        warnMsg.str());

                    slice_cr_ratio[idim] = slice_cr_ratio[idim]/2;
                    modify_cr = true;
                }

                if ( !modify_cr ) {
                   index_lo = lo_new;
                   index_hi = hi_new;
                }
                slice_lo[idim] = index_lo;
                slice_hi[idim] = index_hi - 1; // since default is cell-centered
            }
            slice_realbox.setLo( idim, index_lo * dom_geom[0].CellSize(idim)
                                 + real_box.lo(idim) );
            slice_realbox.setHi( idim, index_hi * dom_geom[0].CellSize(idim)
                                 + real_box.lo(idim) );
            slice_cc_nd_box.setLo( idim, slice_realbox.lo(idim) + Real(fac) );
            slice_cc_nd_box.setHi( idim, slice_realbox.hi(idim) - Real(fac) );
        }
    }
}


/* \brief
 *  This function is called if the coordinates of the slice do not align with data points
 *  \param interp_lo is an IntVect which is flagged as 1, if interpolation
     is required in the dimension.
 */
void
InterpolateSliceValues(MultiFab& smf, IntVect interp_lo, RealBox slice_realbox,
                       const Vector<Geometry>& geom, int ncomp, int nghost,
                       IntVect slice_lo, IntVect /*slice_hi*/, IntVect SliceType,
                       const RealBox real_box)
{
    for (MFIter mfi(smf); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        FArrayBox& fabox = smf[mfi];

        for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            if ( interp_lo[idim] == 1 ) {
                InterpolateLo( bx, fabox, slice_lo, geom, idim, SliceType,
                               slice_realbox, 0, ncomp, nghost, real_box);
            }
        }
    }

}

void
InterpolateLo(const Box& bx, FArrayBox &fabox, IntVect slice_lo,
             Vector<Geometry> geom, int idir, IntVect IndType,
             RealBox slice_realbox, int srccomp, int ncomp,
             int /*nghost*/, const RealBox real_box )
{
    auto fabarr = fabox.array();
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);
    const double fac = ( 1.0-IndType[idir] )*geom[0].CellSize(idir) * 0.5;
    const int imin = slice_lo[idir];
    const double minpos = imin*geom[0].CellSize(idir) + fac + real_box.lo(idir);
    const double maxpos = (imin+1)*geom[0].CellSize(idir) + fac + real_box.lo(idir);
    const double slice_minpos = slice_realbox.lo(idir) ;

    switch (idir) {
    case 0:
    {
        if ( imin >= lo.x && imin <= lo.x) {
           for (int n = srccomp; n < srccomp + ncomp; ++n) {
              for (int k = lo.z; k <= hi.z; ++k) {
                 for (int j = lo.y; j <= hi.y; ++j) {
                     for (int i = lo.x; i <= hi.x; ++i) {
                           const double minval = fabarr(i,j,k,n);
                           const double maxval = fabarr(i+1,j,k,n);
                           const double ratio  = (maxval - minval) / (maxpos - minpos);
                           const double xdiff  = slice_minpos - minpos;
                           const double newval = minval + xdiff * ratio;
                           fabarr(i,j,k,n) = static_cast<Real>(newval);
                     }
                 }
              }
           }
        }
        break;
    }
    case 1:
    {
        if ( imin >= lo.y && imin <= lo.y) {
           for (int n = srccomp; n < srccomp+ncomp; ++n) {
              for (int k = lo.z; k <= hi.z; ++k) {
                 for (int j = lo.y; j <= hi.y; ++j) {
                    for (int i = lo.x; i <= hi.x; ++i) {
                        const double minval = fabarr(i,j,k,n);
                        const double maxval = fabarr(i,j+1,k,n);
                        const double ratio  = (maxval - minval) / (maxpos - minpos);
                        const double xdiff  = slice_minpos - minpos;
                        const double newval = minval + xdiff * ratio;
                        fabarr(i,j,k,n) = static_cast<Real>(newval);
                    }
                 }
              }
           }
        }
        break;
    }
    case 2:
    {
        if ( imin >= lo.z && imin <= lo.z) {
           for (int n = srccomp; n < srccomp+ncomp; ++n) {
              for (int k = lo.z; k <= hi.z; ++k) {
                 for (int j = lo.y; j <= hi.y; ++j) {
                    for (int i = lo.x; i <= hi.x; ++i) {
                        const double minval = fabarr(i,j,k,n);
                        const double maxval = fabarr(i,j,k+1,n);
                        const double ratio  = (maxval - minval) / (maxpos - minpos);
                        const double xdiff  = slice_minpos - minpos;
                        const double newval = minval + xdiff * ratio;
                        fabarr(i,j,k,n) = static_cast<Real>(newval);
                    }
                 }
              }
           }
        }
        break;
    }

    }

}
