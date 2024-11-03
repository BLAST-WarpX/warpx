/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_BOUNDARYHANDLER_H_
#define WARPX_BOUNDARYHANDLER_H_

#include "Utils/Parser/ParserUtils.H"
#include "WarpX.H"

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MLMG.H>
#include <AMReX_REAL.H>
#include <AMReX_Parser.H>

#include <array>
#include <string>
#include <utility>


class PoissonBoundaryHandler
{
public:
    PoissonBoundaryHandler (); // constructor

    /** Read runtime parameters. Called in constructor. */
    void ReadParameters ();

    /**
     * \brief Read the input settings and set the boundary conditions used
     * on each domain boundary for the Poisson solver.
     */
    void DefinePhiBCs (const amrex::Geometry& geom);

    /**
     * \brief Initialize amrex::Parser objects to get the boundary potential
     * values at specified times.
     */
    void BuildParsers ();
    void BuildParsersEB ();

    /**
     * \brief Sets the EB potential string and updates the function parser.
     *
     * \param [in] potential The string value of the potential
     */
    void setPotentialEB(const std::string& potential) {
        potential_eb_str = potential;
        BuildParsersEB();
    }

    struct PhiCalculatorEB {

        amrex::Real t;
        amrex::ParserExecutor<4> potential_eb;

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real operator()(const amrex::Real x, const amrex::Real z) const noexcept {
            using namespace amrex::literals;
            return potential_eb(x, 0.0_rt, z, t);
        }

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Real operator()(const amrex::Real x, const amrex::Real y, const amrex::Real z) const noexcept {
            return potential_eb(x, y, z, t);
        }
    };

    [[nodiscard]] PhiCalculatorEB
    getPhiEB(amrex::Real t) const noexcept
    {
        return PhiCalculatorEB{t, potential_eb};
    }

    bool m_boundary_potential_specified = false;

    // set default potentials to zero in order for current tests to pass
    // but forcing the user to specify a potential might be better
    std::string potential_xlo_str = "0";
    std::string potential_xhi_str = "0";
    std::string potential_ylo_str = "0";
    std::string potential_yhi_str = "0";
    std::string potential_zlo_str = "0";
    std::string potential_zhi_str = "0";
    std::string potential_eb_str = "0";

    amrex::ParserExecutor<1> potential_xlo;
    amrex::ParserExecutor<1> potential_xhi;
    amrex::ParserExecutor<1> potential_ylo;
    amrex::ParserExecutor<1> potential_yhi;
    amrex::ParserExecutor<1> potential_zlo;
    amrex::ParserExecutor<1> potential_zhi;
    amrex::ParserExecutor<1> potential_eb_t;
    amrex::ParserExecutor<4> potential_eb;

    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lobc, hibc;
    std::array<bool, AMREX_SPACEDIM * 2> dirichlet_flag;
    bool has_non_periodic = false;
    bool phi_EB_only_t = true;

private:

    amrex::Parser potential_xlo_parser;
    amrex::Parser potential_xhi_parser;
    amrex::Parser potential_ylo_parser;
    amrex::Parser potential_yhi_parser;
    amrex::Parser potential_zlo_parser;
    amrex::Parser potential_zhi_parser;
    amrex::Parser potential_eb_parser;
};

/** use amrex to directly calculate the electric field since with EB's the
 *
 * simple finite difference scheme in WarpX::computeE sometimes fails
 */
class EBCalcEfromPhiPerLevel {
    private:
    amrex::Vector<
        amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM>
    > m_e_field;

    public:
    EBCalcEfromPhiPerLevel(amrex::Vector<amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM> > e_field)
            : m_e_field(std::move(e_field)) {}

    void operator()(amrex::MLMG & mlmg, int const lev) {
        using namespace amrex::literals;

        mlmg.getGradSolution({m_e_field[lev]});
        for (auto &field: m_e_field[lev]) {
            field->mult(-1._rt);
        }
    }
};

#endif // WARPX_BOUNDARYHANDLER_H_
