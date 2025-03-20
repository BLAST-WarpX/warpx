/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 * Authors: Andrew Myers
 */
#include "AddPlasmaUtilities.H"

#include <cmath>

bool find_overlap (const amrex::RealBox& tile_realbox, const amrex::RealBox& part_realbox,
                   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
                   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                   amrex::RealBox& overlap_realbox, amrex::Box& overlap_box, amrex::IntVect& shifted)
{
    using namespace amrex::literals;

    bool no_overlap = false;
    for (int dir=0; dir<AMREX_SPACEDIM; dir++) {
        if ( tile_realbox.lo(dir) <= part_realbox.hi(dir) ) {
            const amrex::Real ncells_adjust = std::floor( (tile_realbox.lo(dir) - part_realbox.lo(dir))/dx[dir] );
            overlap_realbox.setLo( dir, part_realbox.lo(dir) + std::max(ncells_adjust, 0._rt) * dx[dir]);
        } else {
            no_overlap = true; break;
        }
        if ( tile_realbox.hi(dir) >= part_realbox.lo(dir) ) {
            const amrex::Real ncells_adjust = std::floor( (part_realbox.hi(dir) - tile_realbox.hi(dir))/dx[dir] );
            overlap_realbox.setHi( dir, part_realbox.hi(dir) - std::max(ncells_adjust, 0._rt) * dx[dir]);
        } else {
            no_overlap = true; break;
        }
        // Count the number of cells in this direction in overlap_realbox
        overlap_box.setSmall( dir, 0 );
        overlap_box.setBig( dir,
                            int( std::round((overlap_realbox.hi(dir)-overlap_realbox.lo(dir))
                                            /dx[dir] )) - 1);
        shifted[dir] =
            static_cast<int>(std::round((overlap_realbox.lo(dir)-prob_lo[dir])/dx[dir]));
        // shifted is exact in non-moving-window direction.  That's all we care.
    }
    return no_overlap;
}

bool find_overlap_flux (const amrex::RealBox& tile_realbox, const amrex::RealBox& part_realbox,
                        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
                        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                        const PlasmaInjector& plasma_injector,
                        amrex::RealBox& overlap_realbox, amrex::Box& overlap_box, amrex::IntVect& shifted)
{
    using namespace amrex::literals;

    bool no_overlap = false;
    for (int dir=0; dir<AMREX_SPACEDIM; dir++) {
#if (defined(WARPX_DIM_3D))
        if (dir == plasma_injector.flux_normal_axis) {
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        if (2*dir == plasma_injector.flux_normal_axis) {
            // The above formula captures the following cases:
            // - flux_normal_axis=0 (emission along x/r) and dir=0
            // - flux_normal_axis=2 (emission along z) and dir=1
#elif defined(WARPX_DIM_1D_Z)
        if ( (dir==0) && (plasma_injector.flux_normal_axis==2) ) {
#endif
            if (plasma_injector.flux_direction > 0) {
                if (plasma_injector.surface_flux_pos <  tile_realbox.lo(dir) ||
                    plasma_injector.surface_flux_pos >= tile_realbox.hi(dir)) {
                    no_overlap = true;
                    break;
                }
            } else {
                if (plasma_injector.surface_flux_pos <= tile_realbox.lo(dir) ||
                    plasma_injector.surface_flux_pos >  tile_realbox.hi(dir)) {
                    no_overlap = true;
                    break;
                }
            }
            overlap_realbox.setLo( dir, plasma_injector.surface_flux_pos );
            overlap_realbox.setHi( dir, plasma_injector.surface_flux_pos );
            overlap_box.setSmall( dir, 0 );
            overlap_box.setBig( dir, 0 );
            shifted[dir] =
                static_cast<int>(std::round((overlap_realbox.lo(dir)-prob_lo[dir])/dx[dir]));
        } else {
            if ( tile_realbox.lo(dir) <= part_realbox.hi(dir) ) {
                const amrex::Real ncells_adjust = std::floor( (tile_realbox.lo(dir) - part_realbox.lo(dir))/dx[dir] );
                overlap_realbox.setLo( dir, part_realbox.lo(dir) + std::max(ncells_adjust, 0._rt) * dx[dir]);
            } else {
                no_overlap = true; break;
            }
            if ( tile_realbox.hi(dir) >= part_realbox.lo(dir) ) {
                const amrex::Real ncells_adjust = std::floor( (part_realbox.hi(dir) - tile_realbox.hi(dir))/dx[dir] );
                overlap_realbox.setHi( dir, part_realbox.hi(dir) - std::max(ncells_adjust, 0._rt) * dx[dir]);
            } else {
                no_overlap = true; break;
            }
            // Count the number of cells in this direction in overlap_realbox
            overlap_box.setSmall( dir, 0 );
            overlap_box.setBig( dir,
                                int( std::round((overlap_realbox.hi(dir)-overlap_realbox.lo(dir))
                                                /dx[dir] )) - 1);
            shifted[dir] =
                static_cast<int>(std::round((overlap_realbox.lo(dir)-prob_lo[dir])/dx[dir]));
            // shifted is exact in non-moving-window direction.  That's all we care.
        }
    }

    return no_overlap;
}

PlasmaParserWrapper::PlasmaParserWrapper (const std::size_t a_num_user_int_attribs,
                                          const std::size_t a_num_user_real_attribs,
                                          const amrex::Vector< std::unique_ptr<amrex::Parser> >& a_user_int_attrib_parser,
                                          const amrex::Vector< std::unique_ptr<amrex::Parser> >& a_user_real_attrib_parser)

{
    m_user_int_attrib_parserexec_pinned.resize(a_num_user_int_attribs);
    m_user_real_attrib_parserexec_pinned.resize(a_num_user_real_attribs);

    for (std::size_t ia = 0; ia < a_num_user_int_attribs; ++ia) {
        m_user_int_attrib_parserexec_pinned[ia] = a_user_int_attrib_parser[ia]->compile<7>();
    }
    for (std::size_t ia = 0; ia < a_num_user_real_attribs; ++ia) {
        m_user_real_attrib_parserexec_pinned[ia] = a_user_real_attrib_parser[ia]->compile<7>();
    }
}

int** PlasmaParserHelper::getUserIntDataPtrs () {
#ifdef AMREX_USE_GPU
    return m_d_pa_user_int.dataPtr();
#else
    return m_pa_user_int_pinned.dataPtr();
#endif
}

amrex::ParticleReal** PlasmaParserHelper::getUserRealDataPtrs () {
#ifdef AMREX_USE_GPU
    return m_d_pa_user_real.dataPtr();
#else
    return m_pa_user_real_pinned.dataPtr();
#endif
}

amrex::ParserExecutor<7> const* PlasmaParserHelper::getUserIntParserExecData () const {
#ifdef AMREX_USE_GPU
    return m_d_user_int_attrib_parserexec.dataPtr();
#else
    return m_wrapper_ptr->m_user_int_attrib_parserexec_pinned.dataPtr();
#endif
}

amrex::ParserExecutor<7> const* PlasmaParserHelper::getUserRealParserExecData () const {
#ifdef AMREX_USE_GPU
    return m_d_user_real_attrib_parserexec.dataPtr();
#else
    return m_wrapper_ptr->m_user_real_attrib_parserexec_pinned.dataPtr();
#endif
}
