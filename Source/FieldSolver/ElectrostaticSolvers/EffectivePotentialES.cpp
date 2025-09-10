/* Copyright 2024-2025 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */

#include "EffectivePotentialES.H"
#include "Fluids/MultiFluidContainer_fwd.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "Particles/MultiParticleContainer_fwd.H"
#include "Utils/Parser/ParserUtils.H"
#include "WarpX.H"

using namespace amrex;

void EffectivePotentialES::InitData() {
    auto & warpx = WarpX::GetInstance();
    m_poisson_boundary_handler->DefinePhiBCs(warpx.Geom(0));

    // Initialize "sigma" MF which stores the dressing of the Poisson equation.
    // It is a cell-centered multifab.
    auto rho = warpx.GetMultiFabRegister().get(warpx::fields::FieldType::rho_fp, 0);
    auto const& ba = convert(rho->boxArray(), IntVect(AMREX_D_DECL(0,0,0)));
    m_sigma = std::make_unique<MultiFab>(ba, rho->DistributionMap(), 1, 0);
    // Set sigma to 1
    m_sigma->setVal(1.0_rt);
}

void EffectivePotentialES::ComputeSpaceChargeField (
    ablastr::fields::MultiFabRegister& fields,
    MultiParticleContainer& mpc,
    [[maybe_unused]] MultiFluidContainer* mfl,
    int max_level)
{
    WARPX_PROFILE("EffectivePotentialES::ComputeSpaceChargeField");

    using ablastr::fields::MultiLevelScalarField;
    using ablastr::fields::MultiLevelVectorField;
    using warpx::fields::FieldType;

    bool const skip_lev0_coarse_patch = true;

    // grab the simulation fields
    const MultiLevelScalarField rho_fp = fields.get_mr_levels(FieldType::rho_fp, max_level);
    const MultiLevelScalarField rho_cp = fields.get_mr_levels(FieldType::rho_cp, max_level, skip_lev0_coarse_patch);
    const MultiLevelScalarField phi_fp = fields.get_mr_levels(FieldType::phi_fp, max_level);
    const MultiLevelVectorField Efield_fp = fields.get_mr_levels_alldirs(FieldType::Efield_fp, max_level);

    mpc.DepositCharge(rho_fp, 0.0_rt);
    if (mfl) {
        const int lev = 0;
        mfl->DepositCharge(fields, *rho_fp[lev], lev);
    }

    // Apply filter, perform MPI exchange, interpolate across levels
    const Vector<std::unique_ptr<MultiFab> > rho_buf(num_levels);
    auto & warpx = WarpX::GetInstance();
    warpx.SyncRho( rho_fp, rho_cp, amrex::GetVecOfPtrs(rho_buf) );

#ifndef WARPX_DIM_RZ
    for (int lev = 0; lev < num_levels; lev++) {
        // Reflect density over PEC boundaries, if needed.
        warpx.ApplyRhofieldBoundary(lev, rho_fp[lev], PatchType::fine);
    }
#endif

    // set the boundary potentials appropriately
    setPhiBC(phi_fp, warpx.gett_new(0));

    // perform phi calculation
    computePhi(rho_fp, phi_fp, Efield_fp);

    // Compute the electric field. Note that if an EB is used the electric
    // field will be calculated in the computePhi call.
    if (!EB::enabled()) {
        const std::array<Real, 3> beta = {0._rt};
        computeE( Efield_fp, phi_fp, beta );
    }
}

void EffectivePotentialES::computePhi (
    ablastr::fields::MultiLevelScalarField const& rho,
    ablastr::fields::MultiLevelScalarField const& phi,
    ablastr::fields::MultiLevelVectorField const& efield ) const
{
    // Calculate the mass enhancement factor - see  Appendix A of
    // Barnes, Journal of Comp. Phys., 424 (2021), 109852.
    ComputeSigma();

    // Use the AMREX MLMG solver
    computePhi(rho, phi, efield, m_sigma, self_fields_required_precision,
                self_fields_absolute_tolerance, self_fields_max_iters,
                self_fields_verbosity);
}

void EffectivePotentialES::ComputeSigma () const
{
    const ParmParse pp_warpx("warpx");
    // Get the user set value for C_SI (defaults to 4)
    amrex::Real C_SI = 4.0;
    utils::parser::queryWithParser(pp_warpx, "effective_potential_factor", C_SI);

    // Get the user set value for the time filtering parameter (defaults to 0.1)
    amrex::Real time_filter_param = 0.1;
    utils::parser::queryWithParser(pp_warpx, "effective_potential_time_filter_param", time_filter_param);

    int const lev = 0;

    auto& warpx = WarpX::GetInstance();
    auto& mypc = warpx.GetPartContainer();

    // The effective potential dielectric function is given by
    // \varepsilon_{SI} = \varepsilon * (1 + \sum_{i in species} C_{SI}*(w_pi * dt)^2/4)
    // Note the use of the plasma frequency in rad/s (not Hz) and the factor of 1/4,
    // these choices make it so that C_SI = 1 is the marginal stability threshold.
    auto mult_factor = (
        C_SI * warpx.getdt(lev) * warpx.getdt(lev) / (4._rt * PhysConst::ep0)
    );

    // if this is the first step, use the full sigma
    if (warpx.getistep(lev) == 0) time_filter_param = 1._rt;

    // scale sigma down from current value for time filtering
    m_sigma->mult(1.0_rt - time_filter_param, 0);

    // Loop over each species to calculate the Poisson equation dressing
    for (auto const& pc : mypc) {
        // get the species number density per cell
        auto rho_cc = pc->GetNumberDensity(lev);

        // get multiplication factor for this species
        auto const mult_factor_pc = mult_factor * pc->getCharge() * pc->getCharge() / pc->getMass();

        // add species term to sigma:
        // sigma += C_SI / 4 * q^2/(m*eps0) * dt^2 * N
        MultiFab::LinComb(
            *m_sigma,
            1._rt, *m_sigma, 0,
            time_filter_param*mult_factor_pc, *rho_cc, 0,
            0, 1, 0
        );
    }
    m_sigma->plus(time_filter_param, 0);
}

void EffectivePotentialES::computePhi (
    ablastr::fields::MultiLevelScalarField const& rho,
    ablastr::fields::MultiLevelScalarField const& phi,
    ablastr::fields::MultiLevelVectorField const& efield,
    std::unique_ptr<amrex::MultiFab> const& sigma,
    amrex::Real required_precision,
    amrex::Real absolute_tolerance,
    int max_iters,
    int verbosity
) const
{
    // create a vector to our fields, sorted by level
    amrex::Vector<amrex::MultiFab *> sorted_rho;
    amrex::Vector<amrex::MultiFab *> sorted_phi;
    for (int lev = 0; lev < num_levels; ++lev) {
        sorted_rho.emplace_back(rho[lev]);
        sorted_phi.emplace_back(phi[lev]);
    }

    auto & warpx = WarpX::GetInstance();

    std::optional<EBCalcEfromPhiPerLevel> post_phi_calculation;
#ifdef AMREX_USE_EB
    // TODO: double check no overhead occurs on "m_eb_enabled == false"
    std::optional<amrex::Vector<amrex::EBFArrayBoxFactory const *> > eb_farray_box_factory;
#else
    std::optional<amrex::Vector<amrex::FArrayBoxFactory const *> > const eb_farray_box_factory;
#endif
    if (EB::enabled())
    {
        // EB: use AMReX to directly calculate the electric field since with EB's the
        // simple finite difference scheme in WarpX::computeE sometimes fails

        // TODO: maybe make this a helper function or pass Efield_fp directly
        amrex::Vector<
            amrex::Array<amrex::MultiFab *, AMREX_SPACEDIM>
        > e_field;
        for (int lev = 0; lev < num_levels; ++lev) {
            e_field.push_back(
#if defined(WARPX_DIM_1D_Z)
                amrex::Array<amrex::MultiFab*, 1>{
                    efield[lev][2]
                }
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                amrex::Array<amrex::MultiFab*, 1>{
                    efield[lev][0]
                }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                amrex::Array<amrex::MultiFab*, 2>{
                    efield[lev][0], efield[lev][2]
                }
#elif defined(WARPX_DIM_3D)
                amrex::Array<amrex::MultiFab *, 3>{
                    efield[lev][0], efield[lev][1], efield[lev][2]
                }
#endif
            );
        }
        post_phi_calculation = EBCalcEfromPhiPerLevel(e_field);

#ifdef AMREX_USE_EB
        amrex::Vector<
            amrex::EBFArrayBoxFactory const *
        > factories;
        for (int lev = 0; lev < num_levels; ++lev) {
            factories.push_back(&warpx.fieldEBFactory(lev));
        }
        eb_farray_box_factory = factories;
#endif
    }

    ablastr::fields::computeEffectivePotentialPhi(
        sorted_rho,
        sorted_phi,
        *sigma,
        required_precision,
        absolute_tolerance,
        max_iters,
        verbosity,
        warpx.Geom(),
        warpx.DistributionMap(),
        warpx.boxArray(),
        WarpX::grid_type,
        false,
        EB::enabled(),
        WarpX::do_single_precision_comms,
        warpx.refRatio(),
        post_phi_calculation,
        *m_poisson_boundary_handler,
        warpx.gett_new(0),
        eb_farray_box_factory
    );
}
