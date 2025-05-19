#include "LowerCornerFunctor.H"

#include <AMReX_Array.H>

#include <limits>

using namespace amrex;

LowerCornerFunctor::LowerCornerFunctor(
    const Vector<Real> cur_time,
    const Real time_of_last_gal_shift,
    const Vector<Real> v_galilean,
    const Vector<Geometry> geom):
    m_cur_time{cur_time},
    m_time_of_last_gal_shift{time_of_last_gal_shift},
    m_v_galilean{v_galilean},
    m_geom{geom}
{}

XDim3
LowerCornerFunctor::operator()(
    const Box& bx, const int lev, const Real time_shift_delta) const
{

    const auto gm_lev = m_geom[lev];
    const auto grid_box = RealBox{bx, gm_lev.CellSize(), gm_lev.ProbLo()};
    const Real* grid_min = grid_box.lo();

    const Real cur_time = m_cur_time[lev];
    const Real time_shift =
        (cur_time + time_shift_delta - m_time_of_last_gal_shift);
    Array<Real,3> galilean_shift = { m_v_galilean[0]*time_shift,
                                                   m_v_galilean[1]*time_shift,
                                                   m_v_galilean[2]*time_shift};
#if defined(WARPX_DIM_3D)
    return { grid_min[0] + galilean_shift[0], grid_min[1] + galilean_shift[1], grid_min[2] + galilean_shift[2] };

#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    return { grid_min[0] + galilean_shift[0], std::numeric_limits<Real>::lowest(), grid_min[1] + galilean_shift[2] };

#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    ignore_unused(galilean_shift);
    return { grid_min[0], std::numeric_limits<Real>::lowest(), std::numeric_limits<Real>::lowest() };

#elif defined(WARPX_DIM_1D_Z)
    return { std::numeric_limits<Real>::lowest(), std::numeric_limits<Real>::lowest(), grid_min[0] + galilean_shift[2] };
#endif
}
