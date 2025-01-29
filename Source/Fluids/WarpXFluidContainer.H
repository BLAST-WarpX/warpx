/* Copyright 2023 Grant Johnson, Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_WarpXFluidContainer_H_
#define WARPX_WarpXFluidContainer_H_

#include "Evolve/WarpXDtType.H"
#include "Initialization/PlasmaInjector.H"
#include "MultiFluidContainer.H"

#include<AMReX_MultiFab.H>
#include<AMReX_Vector.H>

#include <string>


/**
 * WarpXFluidContainer is the base class from which all concrete
 * fluid container classes derive.
 *
 * WarpXFluidContainer contains the main functions for initialization,
 * interaction with the grid (field gather and current deposition), fluid
 * source and push, advective update and updates for non-inertial terms.
 */
class WarpXFluidContainer
{
public:
    friend MultiFluidContainer;

    WarpXFluidContainer (int ispecies, const std::string &name);
    ~WarpXFluidContainer() = default;

    WarpXFluidContainer (WarpXFluidContainer const &)             = delete;
    WarpXFluidContainer& operator= (WarpXFluidContainer const & ) = delete;
    WarpXFluidContainer(WarpXFluidContainer&& )                   = default;
    WarpXFluidContainer& operator=(WarpXFluidContainer&& )        = default;

    void AllocateLevelMFs (ablastr::fields::MultiFabRegister& m_fields, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, int lev) const;

    void InitData (ablastr::fields::MultiFabRegister& m_fields, amrex::Box init_box, amrex::Real cur_time, int lev);

    void ReadParameters ();

    /**
     * Evolve updates a single timestep (dt) of the cold relativistic fluid equations
     */
    void Evolve (ablastr::fields::MultiFabRegister& fields,
                 int lev,
                 const std::string& current_fp_string,
                 amrex::Real cur_time,
                 bool skip_deposition=false);

    /**
     * AdvectivePush_Muscl takes a single timestep (dt) of the cold relativistic fluid equations
     * using a Muscl-Handcock scheme
     *
     * \brief Advective term, cold-rel. fluids
     *
     * \param[in] lev refinement level
     */
    void AdvectivePush_Muscl (ablastr::fields::MultiFabRegister& m_fields, int lev);


    /**
     * Apply (non-periodic) BC on the fluids (needed for spatial derivative),
     * and communicate N, NU at boundaries
     *
     * \brief Apply non-periodic BC to fluids and communicate boundaries
     *
     * \param[in] lev refinement level
     */
    void ApplyBcFluidsAndComms (ablastr::fields::MultiFabRegister& m_fields, int lev);

#if defined(WARPX_DIM_RZ)
    /**
     * centrifugal_source_rz adds contributions due to curvature acceleration for a
     * single timestep using an SSP-RK3 timestep for RZ specifically
     *
     * \brief Centrifugal source term
     *
     * \param[in] lev refinement level
     */
    void centrifugal_source_rz (ablastr::fields::MultiFabRegister& m_fields, int lev);
#endif

    /**
     * GatherAndPush introduces the Lorentz term in the cold relativistic fluid
     * equations for a single timestep (dt) using Higuera and Cary Push
     *
     * \brief Lorentz Momentum Source
     *
     * \param[in] lev refinement level
     * \param[in] Ex Yee electric field (x)
     * \param[in] Ey Yee electric field (y)
     * \param[in] Ez Yee electric field (z)
     * \param[in] Bx Yee magnetic field (x)
     * \param[in] By Yee magnetic field (y)
     * \param[in] Bz Yee magnetic field (z)
     * \param[in] t Current time
     */
    void GatherAndPush (ablastr::fields::MultiFabRegister& m_fields,
        const amrex::MultiFab& Ex, const amrex::MultiFab& Ey, const amrex::MultiFab& Ez,
        const amrex::MultiFab& Bx, const amrex::MultiFab& By, const amrex::MultiFab& Bz,
        amrex::Real t, int lev);

    /**
     * DepositCurrent interpolates the fluid current density comps. onto the Yee grid and
     * sums the contributions to the particle current density
     *
     * \brief Deposit fluid current density.
     *
     * \param[in] lev refinement level
     * \param[in,out] jx current density MultiFab x comp.
     * \param[in,out] jy current density MultiFab y comp.
     * \param[in,out] jz current density MultiFab z comp.
     */
    void DepositCurrent (ablastr::fields::MultiFabRegister& m_fields,
        amrex::MultiFab& jx, amrex::MultiFab& jy, amrex::MultiFab& jz, int lev);

    /**
     * DepositCharge interpolates the fluid charge density onto the Yee grid and
     * sums the contributions to the particle charge density
     *
     * \brief Deposit fluid charge density.
     *
     * \param[in] lev refinement level
     * \param[in,out] rho charge density MultiFab.
     */
    void DepositCharge (ablastr::fields::MultiFabRegister& m_fields, amrex::MultiFab &rho, int lev, int icomp = 0);

    [[nodiscard]] amrex::Real getCharge () const {return charge;}
    [[nodiscard]] amrex::Real getMass () const {return mass;}

protected:
    int species_id;
    std::string species_name;
    amrex::Real charge;
    amrex::Real mass;

    int do_not_push = 0;
    int do_not_gather = 0;
    int do_not_deposit = 0;
    PhysicalSpecies physical_species;

    // Parser for external fields
    std::string m_B_ext_s = "none";
    std::string m_E_ext_s = "none";

    // Parser for B_external on the particle
    std::unique_ptr<amrex::Parser> m_Bx_parser;
    std::unique_ptr<amrex::Parser> m_By_parser;
    std::unique_ptr<amrex::Parser> m_Bz_parser;
    amrex::ParserExecutor<4> m_Bxfield_parser;
    amrex::ParserExecutor<4> m_Byfield_parser;
    amrex::ParserExecutor<4> m_Bzfield_parser;

    // Parser for E_external on the particle
    std::unique_ptr<amrex::Parser> m_Ex_parser;
    std::unique_ptr<amrex::Parser> m_Ey_parser;
    std::unique_ptr<amrex::Parser> m_Ez_parser;
    amrex::ParserExecutor<4> m_Exfield_parser;
    amrex::ParserExecutor<4> m_Eyfield_parser;
    amrex::ParserExecutor<4> m_Ezfield_parser;

    std::unique_ptr<InjectorDensity,InjectorDensityDeleter> h_inj_rho;
    InjectorDensity* d_inj_rho = nullptr;
    std::unique_ptr<amrex::Parser> density_parser;

    std::unique_ptr<InjectorMomentum,InjectorMomentumDeleter> h_inj_mom;
    InjectorMomentum* d_inj_mom = nullptr;
    std::unique_ptr<amrex::Parser> ux_parser;
    std::unique_ptr<amrex::Parser> uy_parser;
    std::unique_ptr<amrex::Parser> uz_parser;
    std::unique_ptr<amrex::Parser> ux_th_parser;
    std::unique_ptr<amrex::Parser> uy_th_parser;
    std::unique_ptr<amrex::Parser> uz_th_parser;

    // Keep a pointer to TemperatureProperties to ensure the lifetime of the
    // contained Parser
    std::unique_ptr<TemperatureProperties> h_mom_temp;
    std::unique_ptr<VelocityProperties> h_mom_vel;

public:

    // Names of Multifabs that will be added to the mfs register
    std::string name_mf_N = "fluid_density_"+species_name;
    std::string name_mf_NU = "fluid_momentum_density_"+species_name;

};

#endif
