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
    auto& fields = warpx.GetMultiFabRegister();
    auto rho = fields.get(warpx::fields::FieldType::rho_fp, 0);
    fields.alloc_init(
        warpx::fields::FieldType::effective_potential_sigma, /*lev=*/0,
        convert(rho->boxArray(), IntVect(AMREX_D_DECL(0,0,0))),
        rho->DistributionMap(), 1, IntVect(AMREX_D_DECL(0,0,0)), 1.0_rt
    );
    m_overwrite_sigma = true;
}

void EffectivePotentialES::ComputeSpaceChargeField (
    ablastr::fields::MultiFabRegister& fields,
    [[maybe_unused]] MultiParticleContainer& mpc,
    [[maybe_unused]] MultiFluidContainer* mfl,
    int max_level)
{
    ABLASTR_PROFILE("EffectivePotentialES::ComputeSpaceChargeField");

    using ablastr::fields::MultiLevelScalarField;
    using ablastr::fields::MultiLevelVectorField;
    using warpx::fields::FieldType;

    bool const skip_lev0_coarse_patch = true;

    // grab the simulation fields
    const MultiLevelScalarField rho_fp = fields.get_mr_levels(FieldType::rho_fp, max_level);
    const MultiLevelScalarField rho_cp = fields.get_mr_levels(FieldType::rho_cp, max_level, skip_lev0_coarse_patch);
    const MultiLevelScalarField phi_fp = fields.get_mr_levels(FieldType::phi_fp, max_level);
    const MultiLevelVectorField Efield_fp = fields.get_mr_levels_alldirs(FieldType::Efield_fp, max_level);

    auto & warpx = WarpX::GetInstance();

    // set the boundary potentials appropriately
    setPhiBC(phi_fp, warpx.gett_new(0));

    // Calculate the mass enhancement factor - see  Appendix A of
    // Barnes, Journal of Comp. Phys., 424 (2021), 109852.
    // Also accumulate the total charge density.
    ComputeSigma(rho_fp);

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
    ablastr::fields::MultiLevelVectorField const& efield )
{
    // Use the AMREX MLMG solver
    computePhi(rho, phi, efield, self_fields_required_precision,
                self_fields_absolute_tolerance, self_fields_max_iters,
                self_fields_verbosity);
}

void EffectivePotentialES::ComputeSigma (
    ablastr::fields::MultiLevelScalarField const& rho_fp )
{
    // Reset the rho array
    for (const auto& rho_lev : rho_fp)
    {
        rho_lev->setVal(0.0_rt);
    }

    // Get the user set value for C_SI (defaults to 4)
    amrex::Real C_SI = 4.0;
    const ParmParse pp_warpx("warpx");
    utils::parser::queryWithParser(pp_warpx, "effective_potential_factor", C_SI);

    // Get the user set value for the time filtering parameter (defaults to 0.1)
    amrex::Real time_filter_param = 0.1;
    utils::parser::queryWithParser(pp_warpx, "effective_potential_time_filter_param", time_filter_param);

    // Get the user set value for the density floor in m^-3 (defaults to 0.0)
    amrex::Real density_floor = 0.0;
    utils::parser::queryWithParser(pp_warpx, "effective_potential_density_floor", density_floor);

    int const lev = 0;

    // sigma is a cell-centered array
    amrex::GpuArray<int, 3> const cell_centered = {0, 0, 0};
    // The "coarsening is just 1 i.e. no coarsening"
    amrex::GpuArray<int, 3> const coarsen = {1, 1, 1};

    // GetChargeDensity returns a nodal multifab
    // Below we set all the unused dimensions to have cell-centered values for
    // rho since these values will be interpolated onto a cell-centered grid
    // - if this is not done the Interp function returns nonsense values.
#if defined(WARPX_DIM_3D)
    amrex::GpuArray<int, 3> const nodal = {1, 1, 1};
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    amrex::GpuArray<int, 3> const nodal = {1, 1, 0};
#elif defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    amrex::GpuArray<int, 3> const nodal = {1, 0, 0};
#endif

    auto& warpx = WarpX::GetInstance();
    auto& mypc = warpx.GetPartContainer();

    // The effective potential dielectric function is given by
    // \varepsilon_{SI} = \varepsilon * (1 + \sum_{i in species} C_{SI}*(w_pi * dt)^2/4)
    // Note the use of the plasma frequency in rad/s (not Hz) and the factor of 1/4,
    // these choices make it so that C_SI = 1 is the marginal stability threshold.
    auto mult_factor = (
        C_SI * warpx.getdt(lev) * warpx.getdt(lev) / (4._rt * PhysConst::epsilon_0)
    );

    // if this is the first step, use the full sigma
    if (m_overwrite_sigma) {
        time_filter_param = 1._rt;
        m_overwrite_sigma = false;
    }

    // grab sigma from the multifab registry
    auto sigma = warpx.GetMultiFabRegister().get(warpx::fields::FieldType::effective_potential_sigma, lev);

    // scale sigma down from current value for time filtering
    sigma->mult(1.0_rt - time_filter_param, 0);

    // Loop over each species to calculate the Poisson equation dressing
    for (auto const& pc : mypc) {
        if (pc->do_not_deposit) { continue; }

        // grab the charge density for this species
        // Note: local deposition is done since the guard cells values are added
        // to the valid cells after filtering in `ApplyFilterandSumBoundaryRho` below.
        // The rho boundary condition and inverse volume scaling for RZ is done
        // within the `GetChargeDensity` function.
        auto rho = pc->GetChargeDensity(lev, true);

        // Handle the parallel transfer of guard cells and apply filtering
        warpx.ApplyFilterandSumBoundaryRho(lev, lev, *rho, 0, rho->nComp());

        // Add rho for this species to the total charge density MF
        amrex::MultiFab::Add(*rho_fp[lev], *rho, 0, 0, 1, rho_fp[lev]->nGrowVect());

        // get multiplication factor for this species
        auto const q = std::abs(pc->getCharge());
        auto const mult_factor_pc = mult_factor * q / pc->getMass();

        // update sigma
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(*sigma, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
            Array4<Real> const& sigma_arr = sigma->array(mfi);
            Array4<Real const> const& rho_arr = rho->const_array(mfi);

            // Loop over the cells and update the sigma field
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Interpolate rho to cell-centered value, applying a floor
                // on the density
                auto const rho_cc = std::max(
                    density_floor*q,
                    std::abs(ablastr::coarsen::sample::Interp(
                        rho_arr, nodal, cell_centered, coarsen, i, j, k, 0
                    ))
                );
                // add species term to sigma:
                // C_SI * w_p^2 * dt^2 / 4 = C_SI / 4 * q*rho/(m*eps0) * dt^2
                sigma_arr(i, j, k, 0) += time_filter_param * mult_factor_pc * rho_cc;
            });
        }
    }
    sigma->plus(time_filter_param, 0);
}

void EffectivePotentialES::computePhi (
    ablastr::fields::MultiLevelScalarField const& rho,
    ablastr::fields::MultiLevelScalarField const& phi,
    ablastr::fields::MultiLevelVectorField const& efield,
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

    // grab sigma from the multifab registry
    auto sigma = warpx.GetMultiFabRegister().get(warpx::fields::FieldType::effective_potential_sigma, 0);

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
