/* Copyright 2024 The WarpX Community
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

    // grab the simulation fields
    const MultiLevelScalarField rho_fp = fields.get_mr_levels(FieldType::rho_fp, max_level);
    const MultiLevelScalarField rho_cp = fields.get_mr_levels(FieldType::rho_cp, max_level);
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
    computePhi(rho_fp, phi_fp);

    // Compute the electric field. Note that if an EB is used the electric
    // field will be calculated in the computePhi call.
    const std::array<Real, 3> beta = {0._rt};
    if (!EB::enabled()) { computeE( Efield_fp, phi_fp, beta ); }
}

void EffectivePotentialES::computePhi (
    ablastr::fields::MultiLevelScalarField const& rho,
    ablastr::fields::MultiLevelScalarField const& phi) const
{
    // Calculate the mass enhancement factor - see  Appendix A of
    // Barnes, Journal of Comp. Phys., 424 (2021), 109852.
    // The "sigma" multifab stores the dressing of the Poisson equation. It
    // is a cell-centered multifab.
    auto const& ba = convert(rho[0]->boxArray(), IntVect(AMREX_D_DECL(0,0,0)));
    MultiFab sigma(ba, rho[0]->DistributionMap(), 1, rho[0]->nGrowVect());
    ComputeSigma(sigma);

    // Use the AMREX MLMG solver
    computePhi(rho, phi, sigma, self_fields_required_precision,
                self_fields_absolute_tolerance, self_fields_max_iters,
                self_fields_verbosity);
}

void EffectivePotentialES::ComputeSigma (MultiFab& sigma) const
{
    // Reset sigma to 1
    sigma.setVal(1.0_rt);

    // Get the user set value for C_SI (defaults to 4)
    amrex::Real C_SI = 4.0;
    const ParmParse pp_warpx("warpx");
    utils::parser::queryWithParser(pp_warpx, "effective_potential_factor", C_SI);

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
#elif defined(WARPX_DIM_1D_Z)
    amrex::GpuArray<int, 3> const nodal = {1, 0, 0};
#endif

    auto& warpx = WarpX::GetInstance();
    auto& mypc = warpx.GetPartContainer();

    // The effective potential dielectric function is given by
    // \varepsilon_{SI} = \varepsilon * (1 + \sum_{i in species} C_{SI}*(w_pi * dt)^2/4)
    // Note the use of the plasma frequency in rad/s (not Hz) and the factor of 1/4,
    // these choices make it so that C_SI = 1 is the marginal stability threshold.
    auto mult_factor = (
        C_SI * warpx.getdt(lev) * warpx.getdt(lev) / (4._rt * PhysConst::ep0)
    );

    // Loop over each species to calculate the Poisson equation dressing
    for (auto const& pc : mypc) {
        // grab the charge density for this species
        auto rho = pc->GetChargeDensity(lev, false);

        // Handle the parallel transfer of guard cells and apply filtering
        warpx.ApplyFilterandSumBoundaryRho(lev, lev, *rho, 0, rho->nComp());

        // get multiplication factor for this species
        auto const mult_factor_pc = mult_factor * pc->getCharge() / pc->getMass();

        // update sigma
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( MFIter mfi(sigma, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
            Array4<Real> const& sigma_arr = sigma.array(mfi);
            Array4<Real const> const& rho_arr = rho->const_array(mfi);

            // Loop over the cells and update the sigma field
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k){
                // Interpolate rho to cell-centered value
                auto const rho_cc = ablastr::coarsen::sample::Interp(
                    rho_arr, nodal, cell_centered, coarsen, i, j, k, 0
                );
                // add species term to sigma:
                // C_SI * w_p^2 * dt^2 / 4 = C_SI / 4 * q*rho/(m*eps0) * dt^2
                sigma_arr(i, j, k, 0) += mult_factor_pc * rho_cc;
            });

        }
    }
}


void EffectivePotentialES::computePhi (
    ablastr::fields::MultiLevelScalarField const& rho,
    ablastr::fields::MultiLevelScalarField const& phi,
    amrex::MultiFab const& sigma,
    amrex::Real required_precision,
    amrex::Real absolute_tolerance,
    int max_iters,
    int verbosity
) const
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // create a vector to our fields, sorted by level
    amrex::Vector<amrex::MultiFab *> sorted_rho;
    amrex::Vector<amrex::MultiFab *> sorted_phi;
    for (int lev = 0; lev < num_levels; ++lev) {
        sorted_rho.emplace_back(rho[lev]);
        sorted_phi.emplace_back(phi[lev]);
    }

    std::optional<EBCalcEfromPhiPerLevel> post_phi_calculation;
#ifdef AMREX_USE_EB
    // TODO: double check no overhead occurs on "m_eb_enabled == false"
    std::optional<amrex::Vector<amrex::EBFArrayBoxFactory const *> > eb_farray_box_factory;
#else
    std::optional<amrex::Vector<amrex::FArrayBoxFactory const *> > const eb_farray_box_factory;
#endif
    auto & warpx = WarpX::GetInstance();
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
                    warpx.m_fields.get(FieldType::Efield_fp, Direction{2}, lev)
                }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                amrex::Array<amrex::MultiFab*, 2>{
                    warpx.m_fields.get(FieldType::Efield_fp, Direction{0}, lev),
                    warpx.m_fields.get(FieldType::Efield_fp, Direction{2}, lev)
                }
#elif defined(WARPX_DIM_3D)
                amrex::Array<amrex::MultiFab *, 3>{
                    warpx.m_fields.get(FieldType::Efield_fp, Direction{0}, lev),
                    warpx.m_fields.get(FieldType::Efield_fp, Direction{1}, lev),
                    warpx.m_fields.get(FieldType::Efield_fp, Direction{2}, lev)
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
        sigma,
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
