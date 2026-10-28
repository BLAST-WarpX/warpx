/* Copyright 2019 Luca Fedeli
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Laser/LaserProfiles.H"

#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <memory>
#include <set>
#include <string>

using namespace amrex;

void
WarpXLaserProfiles::FieldFunctionLaserProfile::init (
    const amrex::ParmParse& ppl,
    CommonLaserParameters /*params*/)
{
    // Parse the properties of the parse_field_function profile
    utils::parser::Store_parserString(
            ppl, "field_function(X,Y,t)", m_params.field_function);
    m_parser = utils::parser::makeParser(m_params.field_function,{"X","Y","t"});
}

void
WarpXLaserProfiles::FieldFunctionLaserProfile::fill_amplitude (
    const int np, Real const * AMREX_RESTRICT const Xp, Real const * AMREX_RESTRICT const Yp,
    Real t, Real * AMREX_RESTRICT const amplitude) const
{
    auto parser = m_parser.compile<3>();
    amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        amplitude[i] = parser(Xp[i], Yp[i], t);
    });
}
