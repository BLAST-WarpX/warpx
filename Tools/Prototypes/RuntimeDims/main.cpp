/* Copyright 2026 Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Simulation.H"

#include "Utils/WarpXConst.H"
#include "Utils/WarpXDim.H"

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <string>

namespace
{
    /** Make my_constants and the physical constants available in input-file
     *  expressions, cf. warpx::initialization::add_constants */
    void
    add_constants ()
    {
        amrex::ParmParse::SetParserPrefix("my_constants");
        amrex::ParmParse pp_constants("my_constants");
        amrex::Real tmp = PhysConst::c;
        pp_constants.queryAdd("clight", tmp);
        tmp = PhysConst::epsilon_0;
        pp_constants.queryAdd("epsilon0", tmp);
        tmp = PhysConst::mu0;
        pp_constants.queryAdd("mu0", tmp);
        tmp = PhysConst::q_e;
        pp_constants.queryAdd("q_e", tmp);
        tmp = PhysConst::m_e;
        pp_constants.queryAdd("m_e", tmp);
        tmp = PhysConst::m_p;
        pp_constants.queryAdd("m_p", tmp);
        tmp = PhysConst::m_u;
        pp_constants.queryAdd("m_u", tmp);
        tmp = PhysConst::kb;
        pp_constants.queryAdd("kb", tmp);
        tmp = PhysConst::hbar;
        pp_constants.queryAdd("hbar", tmp);
        tmp = MathConst::pi;
        pp_constants.queryAdd("pi", tmp);
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        BL_PROFILE("main()");

        add_constants();

        // runtime dimensionality selection: this replaces the compile-time
        // check in warpx::initialization::check_dims()
        std::string dims;
        const amrex::ParmParse pp_geometry("geometry");
        pp_geometry.get("dims", dims);
        const warpx::Dim dim = warpx::dim_from_geometry_dims(dims);

        amrex::Print() << "WarpX runtime-dimensionality prototype"
                       << " (AMReX " << amrex::Version() << ", AMREX_SPACEDIM="
                       << AMREX_SPACEDIM << ")\n"
                       << "Running with geometry.dims = " << dims << "\n";

        Simulation sim(dim);
        sim.InitData();
        sim.Evolve();
    }
    amrex::Finalize();
    return 0;
}
