#include <AMReX_Box.H>
#include <AMReX_Dim3.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#ifndef WARPX_UTILS_LOWERCORNERFUNCTOR_
#define WARPX_UTILS_LOWERCORNERFUNCTOR_

class LowerCornerFunctor
{
public:
    LowerCornerFunctor (
        amrex::Vector<amrex::Real> cur_time,
        amrex::Real time_of_last_gal_shift,
        amrex::Vector<amrex::Real> v_galilean,
        amrex::Vector<amrex::Geometry> geom);

    amrex::XDim3 operator() (
        const amrex::Box& bx,
        int lev,
        amrex::Real time_shift_delta) const;

private:
    amrex::Vector<amrex::Real> m_cur_time;
    amrex::Real m_time_of_last_gal_shift;
    amrex::Vector<amrex::Real> m_v_galilean;
    amrex::Vector<amrex::Geometry> m_geom;
};

#endif //WARPX_UTILS_LOWERCORNERFUNCTOR_
