/* Copyright 2020 Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_MACROSCOPICPROPERTIES_H_
#define WARPX_MACROSCOPICPROPERTIES_H_

#include "MacroscopicProperties_fwd.H"

#include "Utils/WarpXConst.H"

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

#include <memory>
#include <string>


/**
 * \brief This class contains the macroscopic properties of the medium needed to
 * evaluate macroscopic Maxwell equation.
 */
class MacroscopicProperties
{
public:
    MacroscopicProperties (); // constructor
    /** Read user-defined macroscopic properties. Called in constructor. */
    void ReadParameters ();

    /**
     * \brief Allocate multifabs storing macroscopic multifabs
     *
     * @param[in] ba the box array associated to the multifabs E and B
     * @param[in] dmap the distribution mapping
     * @param[in] ng_EB_alloc guard cells allocated for multifabs E and B
     */
    void AllocateLevelMFs (
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        const amrex::IntVect& ng_EB_alloc );

    /**
     * \brief Initialize multifabs storing macroscopic multifabs
     *
     * @param[in] geom the geometry
     * @param[in] Ex_stag staggering of the Ex field
     * @param[in] Ey_stag staggering of the Ey field
     * @param[in] Ez_stag staggering of the Ez field
     */
    void InitData (
        const amrex::Geometry& geom,
        const amrex::IntVect& Ex_stag,
        const amrex::IntVect& Ey_stag,
        const amrex::IntVect& Ez_stag);

    /** return MultiFab, sigma (conductivity) of the medium. */
    amrex::MultiFab& getsigma_mf  () {return (*m_sigma_mf);}
    /** return MultiFab, epsilon (permittivity) of the medium. */
    amrex::MultiFab& getepsilon_mf  () {return (*m_eps_mf);}
    /** return MultiFab, mu (permeability) of the medium. */
    amrex::MultiFab& getmu_mf  () {return (*m_mu_mf);}

    /** Initializes the Multifabs storing macroscopic properties
     *  with user-defined functions(x,y,z).
     */
    void InitializeMacroMultiFabUsingParser (amrex::MultiFab *macro_mf,
                                  amrex::ParserExecutor<3> const& macro_parser,
                                  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx_lev,
                                  const amrex::RealBox& prob_domain_lev);

    /** Gpu Vector with index type of the conductivity multifab */
    amrex::GpuArray<int, 3> sigma_IndexType;
    /** Gpu Vector with index type of the permittivity multifab */
    amrex::GpuArray<int, 3> epsilon_IndexType;
    /** Gpu Vector with index type of the permeability multifab */
    amrex::GpuArray<int, 3> mu_IndexType;
    /** Gpu Vector with index type of the Ex multifab */
    amrex::GpuArray<int, 3> Ex_IndexType;
    /** Gpu Vector with index type of the Ey multifab */
    amrex::GpuArray<int, 3> Ey_IndexType;
    /** Gpu Vector with index type of the Ez multifab */
    amrex::GpuArray<int, 3> Ez_IndexType;
    /** Gpu Vector with index type of coarsening ratio with default value (1,1,1) */
    amrex::GpuArray<int, 3> macro_cr_ratio;

private:

    /** Conductivity, sigma, of the medium */
    amrex::Real m_sigma = 0.0;
    /** Permittivity, epsilon, of the medium */
    amrex::Real m_epsilon = PhysConst::ep0;
    /** Permeability, mu, of the medium */
    amrex::Real m_mu = PhysConst::mu0;
    /** Multifab for m_sigma */
    std::unique_ptr<amrex::MultiFab> m_sigma_mf;
    /** Multifab for m_epsilon */
    std::unique_ptr<amrex::MultiFab> m_eps_mf;
    /** Multifab for m_mu */
    std::unique_ptr<amrex::MultiFab> m_mu_mf;

    /** Stores initialization type for conductivity : constant or parser */
    std::string m_sigma_s = "constant";
    /** Stores initialization type for permittivity : constant or parser */
    std::string m_epsilon_s = "constant";
    /** Stores initialization type for permeability : constant or parser */
    std::string m_mu_s = "constant";

    /** string for storing parser function */
    std::string m_str_sigma_function;
    std::string m_str_epsilon_function;
    std::string m_str_mu_function;
    /** Parser Wrappers */
    std::unique_ptr<amrex::Parser> m_sigma_parser;
    std::unique_ptr<amrex::Parser> m_epsilon_parser;
    std::unique_ptr<amrex::Parser> m_mu_parser;

};

/**
 * \brief
 * This struct contains only static functions to compute the coefficients for the
 * Lax-Wendroff scheme of macroscopic Maxwells equations using
 * macroscopic properties, namely, conductivity (sigma), permittivity (epsilon).
 * Permeability of the material, mu, is used as (beta/mu) for the E-update
 * defined in MacroscopicEvolveECartesian().
 */
struct LaxWendroffAlgo {

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real alpha (amrex::Real const sigma,
                              amrex::Real const epsilon,
                              amrex::Real dt) {
        using namespace amrex;
        const amrex::Real fac1 = 0.5_rt * sigma * dt / epsilon;
        const amrex::Real alpha = (1._rt - fac1)/(1._rt + fac1);
        return alpha;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real beta (amrex::Real const sigma,
                             amrex::Real const epsilon,
                             amrex::Real dt) {
        using namespace amrex;
        const amrex::Real fac1 = 0.5_rt * sigma * dt / epsilon;
        const amrex::Real beta = dt / ( epsilon * (1._rt + fac1) );
        return beta;
    }

};

/**
 * \brief
 * This struct contains only static functions to compute the coefficients for the
 * BackwardEuler scheme of macroscopic Maxwells equations using
 * macroscopic properties, namely, conductivity (sigma) and permittivity (epsilon).
 * Permeability of the material, mu, is used as (beta/mu) for the E-update
 * defined in MacroscopicEvolveECartesian().
 */
struct BackwardEulerAlgo {

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real alpha (amrex::Real const sigma,
                              amrex::Real const epsilon,
                              amrex::Real dt) {
        using namespace amrex;
        const amrex::Real fac1 = sigma * dt / epsilon;
        const amrex::Real alpha = (1._rt)/(1._rt + fac1);
        return alpha;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real beta (amrex::Real const sigma,
                             amrex::Real const epsilon,
                             amrex::Real dt) {
        using namespace amrex;
        const amrex::Real fac1 = sigma * dt / epsilon;
        const amrex::Real beta = dt / ( epsilon * (1._rt + fac1) );
        return beta;
    }

};

#endif // WARPX_MACROSCOPIC_PROPERTIES_H_
