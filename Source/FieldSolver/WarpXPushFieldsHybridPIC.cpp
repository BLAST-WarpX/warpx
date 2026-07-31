/* Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *          S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Fields.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Fluids/MultiFluidContainer.H"
#include "Fluids/WarpXFluidContainer.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/Communication.H>


using namespace amrex;

void WarpX::HybridPICEvolveFields ()
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    ABLASTR_PROFILE("WarpX::HybridPICEvolveFields()");

    // The below deposition is hard coded for a single level simulation
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        finest_level == 0,
        "Ohm's law E-solve only works with a single level.");

    // Get flag to include external fields.
    const bool add_external_fields = m_hybrid_pic_model->m_add_external_fields;

    // Handle field splitting for Hybrid field push
    if (add_external_fields) {
        // Get the external fields
        m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
            gett_old(0),
            0.5_rt*dt[0]);

        // If using split fields, subtract the external field at the old time
        for (int lev = 0; lev <= finest_level; ++lev) {
            for (int idim = 0; idim < 3; ++idim) {
                MultiFab::Subtract(
                    *m_fields.get(FieldType::Bfield_fp, Direction{idim}, lev),
                    *m_fields.get(FieldType::hybrid_B_fp_external, Direction{idim}, lev),
                    0, 0, 1,
                    m_fields.get(FieldType::Bfield_fp, Direction{idim}, lev)->nGrowVect());
            }
        }
    }

    // The particles have now been pushed to their t_{n+1} positions.
    // Perform charge deposition at t_{n+1} and current deposition at t_{n+1/2}.
    HybridPICDepositRhoAndJ();

    // Electron pressure/temperature update at t=n+1, right after the
    // deposition. With solve_electron_energy_equation on, the QDSMC
    // entropy-transport step advances T_e and emits Pe = n_e k_B T_e at the
    // end (it needs rho_fp = rho^{n+1} and hybrid_rho_fp_temp = rho^{n},
    // which the deposit just above established). Otherwise the algebraic
    // closure fills Pe (and mirrors the implied T_e for diagnostics) at
    // this same point.
    if (m_hybrid_pic_model->m_solve_electron_energy_equation) {
        m_hybrid_pic_model->AdvanceElectronEnergyQDSMC(dt[0]);
    } else {
        m_hybrid_pic_model->CalculateElectronPressure();
    }

    // Get the external current
    m_hybrid_pic_model->GetCurrentExternal();

    // Reference hybrid-PIC multifabs
    ablastr::fields::MultiLevelScalarField rho_fp_temp = m_fields.get_mr_levels(FieldType::hybrid_rho_fp_temp, finest_level);
    ablastr::fields::MultiLevelVectorField current_fp_temp = m_fields.get_mr_levels_alldirs(FieldType::hybrid_current_fp_temp, finest_level);

    // During the above deposition the charge and current density were updated
    // so that, at this time, we have rho^{n} in rho_fp_temp, rho{n+1} in the
    // 0'th index of `rho_fp`, J_i^{n-1/2} in `current_fp_temp` and J_i^{n+1/2}
    // in `current_fp`.

    // Note: E^{n} is recalculated with the accurate J_i^{n} since at the end
    // of the last step we had to "guess" it. It also needs to be
    // recalculated to include the resistivity before evolving B.

    // J_i^{n} is calculated as the average of J_i^{n-1/2} and J_i^{n+1/2}.
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        for (int idim = 0; idim < 3; ++idim) {
            // Perform a linear combination of values in the 0'th index (1 comp)
            // of J_i^{n-1/2} and J_i^{n+1/2} (with 0.5 prefactors), writing
            // the result into the 0'th index of `current_fp_temp[lev][idim]`
            MultiFab::LinComb(
                *current_fp_temp[lev][idim],
                0.5_rt, *current_fp_temp[lev][idim], 0,
                0.5_rt, *m_fields.get(FieldType::current_fp, Direction{idim}, lev), 0,
                0, 1, current_fp_temp[lev][idim]->nGrowVect()
            );
        }
    }

    // Push the B field from t=n to t=n+1/2 using the current and density
    // at t=n, while updating the E field along with B using the electron
    // momentum equation
    m_hybrid_pic_model->BfieldEvolve(
        m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level),
        m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level),
        current_fp_temp, rho_fp_temp,
        m_eb_update_E,
        getistep(0),
        0.5_rt*dt[0],
        SubcyclingHalf::FirstHalf, guard_cells.ng_FieldSolver,
        WarpX::sync_nodal_points
    );

    // Average rho^{n} and rho^{n+1} to get rho^{n+1/2} in rho_fp_temp
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // Perform a linear combination of values in the 0'th index (1 comp)
        // of rho^{n} and rho^{n+1} (with 0.5 prefactors), writing
        // the result into the 0'th index of `rho_fp_temp[lev]`
        MultiFab::LinComb(
            *rho_fp_temp[lev], 0.5_rt, *rho_fp_temp[lev], 0,
            0.5_rt, *m_fields.get(FieldType::rho_fp, lev), 0, 0, 1, rho_fp_temp[lev]->nGrowVect()
        );
    }

    if (add_external_fields) {
        // Get the external fields at E^{n+1/2}
        m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
            gett_old(0) + 0.5_rt*dt[0],
            0.5_rt*dt[0]);
    }

    // Now push the B field from t=n+1/2 to t=n+1 using the n+1/2 quantities
    m_hybrid_pic_model->BfieldEvolve(
        m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level),
        m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level),
        m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level),
        rho_fp_temp,
        m_eb_update_E,
        getistep(0),
        0.5_rt*dt[0],
        SubcyclingHalf::SecondHalf, guard_cells.ng_FieldSolver,
        WarpX::sync_nodal_points
    );

    // Extrapolate the ion current density to t=n+1 using
    // J_i^{n+1} = 1/2 * J_i^{n-1/2} + 3/2 * J_i^{n+1/2}, and recalling that
    // now current_fp_temp = J_i^{n} = 1/2 * (J_i^{n-1/2} + J_i^{n+1/2})
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        for (int idim = 0; idim < 3; ++idim) {
            // Perform a linear combination of values in the 0'th index (1 comp)
            // of J_i^{n-1/2} and J_i^{n+1/2} (with -1.0 and 2.0 prefactors),
            // writing the result into the 0'th index of `current_fp_temp[lev][idim]`
            MultiFab::LinComb(
                *current_fp_temp[lev][idim],
                -1._rt, *current_fp_temp[lev][idim], 0,
                2._rt, *m_fields.get(FieldType::current_fp, Direction{idim}, lev), 0,
                0, 1, current_fp_temp[lev][idim]->nGrowVect()
            );
        }
    }

    if (add_external_fields) {
        m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
            gett_new(0),
            0.5_rt*dt[0]);
    }

    // Update the E field to t=n+1 using the extrapolated J_i^n+1 value
    m_hybrid_pic_model->CalculatePlasmaCurrent(
        m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level),
        m_eb_update_E);
    m_hybrid_pic_model->HybridPICSolveE(
        m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level),
        current_fp_temp,
        m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level),
        m_fields.get_mr_levels(FieldType::rho_fp, finest_level),
        m_eb_update_E, false);
    FillBoundaryE(guard_cells.ng_FieldSolver, WarpX::sync_nodal_points);

    // Handle field splitting for Hybrid field push
    if (add_external_fields) {
        // If using split fields, add the external field at the new time
        for (int lev = 0; lev <= finest_level; ++lev) {
            for (int idim = 0; idim < 3; ++idim) {
                MultiFab::Add(
                    *m_fields.get(FieldType::Bfield_fp, Direction{idim}, lev),
                    *m_fields.get(FieldType::hybrid_B_fp_external, Direction{idim}, lev),
                    0, 0, 1,
                    m_fields.get(FieldType::Bfield_fp, Direction{idim}, lev)->nGrowVect());
                MultiFab::Add(
                    *m_fields.get(FieldType::Efield_fp, Direction{idim}, lev),
                    *m_fields.get(FieldType::hybrid_E_fp_external, Direction{idim}, lev),
                    0, 0, 1,
                    m_fields.get(FieldType::Efield_fp, Direction{idim}, lev)->nGrowVect());
            }
        }
    }

    // Copy the rho^{n+1} values to rho_fp_temp and the J_i^{n+1/2} values to
    // current_fp_temp since at the next step those values will be needed as
    // rho^{n} and J_i^{n-1/2}.
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // copy 1 component value starting at index 0 to index 0
        MultiFab::Copy(*rho_fp_temp[lev], *m_fields.get(FieldType::rho_fp, lev),
                        0, 0, 1, rho_fp_temp[lev]->nGrowVect());
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::Copy(*current_fp_temp[lev][idim], *m_fields.get(FieldType::current_fp, Direction{idim}, lev),
                           0, 0, 1, current_fp_temp[lev][idim]->nGrowVect());
        }
    }
}

void WarpX::HybridPICDepositRhoAndJ ()
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    auto current_fp = m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level);
    auto rho_fp = m_fields.get_mr_levels(FieldType::rho_fp, finest_level);
    if (m_hybrid_pic_model->m_need_per_species_fields) {
        // Per-species deposition at t_{n+1} (rho) and t_{n-1/2} (J): each
        // charged species deposits its charge once into its own MultiFab and
        // the raw deposits are accumulated into the total rho_fp (which gets
        // its guard-cell sum, filtering, boundaries and RZ volume scaling
        // later, via SyncCurrentAndRho); the current deposits accumulate
        // directly into the total current_fp. The per-species charge
        // densities are kept on the grid for the electron-energy-equation
        // sources.
        auto rho_species_sum = m_fields.get_mr_levels("hybrid_rho_species_sum_fp", finest_level);
        for (int lev = 0; lev <= finest_level; ++lev) {
            rho_fp[lev]->setVal(0._rt);
            rho_species_sum[lev]->setVal(0._rt);
            for (int idim = 0; idim < 3; ++idim) { current_fp[lev][idim]->setVal(0._rt); }
        }
        for (auto const & spec : mypc->GetSpeciesNames()) {
            auto & pc = mypc->GetParticleContainerFromName(spec);
            if (pc.getCharge() == 0._prt || pc.do_not_deposit) { continue; }
            auto rho_spec = m_fields.get_mr_levels("rho_fp_" + spec, finest_level);
            pc.DepositCurrent(current_fp, dt[0], -0.5_rt * dt[0]);
            pc.DepositCharge(rho_spec, /*local*/true, /*reset*/true,
                             /*apply_boundary_and_scale_volume*/false,
                             /*interpolate_across_levels*/false);
            // Accumulate the RAW (locally deposited, unsummed) per-species
            // charge density into the total: shape-spread contributions near
            // box edges sit in guard cells at this point and are folded into
            // the valid cells of the total later by SyncCurrentAndRho,
            // exactly as in the single-pass deposition path.
            for (int lev = 0; lev <= finest_level; ++lev) {
                MultiFab::Add(*rho_fp[lev], *rho_spec[lev],
                              0, 0, 1, rho_fp[lev]->nGrowVect());
            }
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            // Radial geometries: apply the inverse-volume scaling to the
            // per-species deposit so it carries a physical charge density,
            // with the same scale-then-guard-sum processing as the totals
            // below. The Joule and Q_ei sources compare the species sum
            // against the physical rho_floor (and recover n_s from it), so
            // a raw radial deposit would engage the floor at healthy
            // densities near the axis and corrupt the species fractions.
            for (int lev = 0; lev <= finest_level; ++lev) {
                ApplyInverseVolumeScalingToChargeDensity(rho_spec[lev], lev);
            }
#endif
            // The per-species charge densities themselves are consumed
            // directly (species fractions in the Joule and Q_ei sources) and
            // need their own guard-cell sum here.
            for (int lev = 0; lev <= finest_level; ++lev) {
                ablastr::utils::communication::SumBoundary(
                    *rho_spec[lev], 0, rho_spec[lev]->nComp(),
                    rho_spec[lev]->nGrowVect(), rho_spec[lev]->nGrowVect(),
                    WarpX::do_single_precision_comms, Geom(lev).periodicity());
            }
            // Species-summed physical charge density (same form as the
            // rho_fp_s numerators), shared by the electron-energy-equation
            // consumers. Accumulated AFTER the guard-cell sum so its valid
            // and ghost cells are final.
            for (int lev = 0; lev <= finest_level; ++lev) {
                MultiFab::Add(*rho_species_sum[lev], *rho_spec[lev],
                              0, 0, 1, rho_species_sum[lev]->nGrowVect());
            }
        }
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        for (int lev = 0; lev <= finest_level; ++lev) {
            ApplyInverseVolumeScalingToChargeDensity(rho_fp[lev], lev);
            ApplyInverseVolumeScalingToCurrentDensity(
                current_fp[lev][0], current_fp[lev][1], current_fp[lev][2], lev);
        }
#endif
    } else {
        // Single-pass deposition (rho at t_{n+1}, J at t_{n-1/2}): no active
        // feature consumes the per-species fields, so skip the per-species
        // deposits and guard-cell sums entirely. Zeroing and the RZ inverse
        // volume scaling are handled inside.
        mypc->DepositCharge(rho_fp, 0._rt);
        mypc->DepositCurrent(current_fp, dt[0], -0.5_rt * dt[0]);
    }

    // TODO: Perhaps add flag here for when using temperature accumulation in Hybrid
    // Perform Temperature Deposition at time t_{n}
    mypc->DepositTemperatures(m_fields, 0.0_rt);

    // Deposit cold-relativistic fluid charge and current
    if (do_fluid_species) {
        int const lev = 0;
        myfl->DepositCharge(m_fields, *m_fields.get(FieldType::rho_fp, lev), lev);
        myfl->DepositCurrent(m_fields,
            *m_fields.get(FieldType::current_fp, Direction{0}, lev),
            *m_fields.get(FieldType::current_fp, Direction{1}, lev),
            *m_fields.get(FieldType::current_fp, Direction{2}, lev),
            lev);
    }

    // Synchronize J and rho:
    // filter (if used), exchange guard cells, interpolate across MR levels
    // and apply boundary conditions
    SyncCurrentAndRho();

    // SyncCurrent does not include a call to FillBoundary, but it is needed
    // for the hybrid-PIC solver since current values are interpolated to
    // a nodal grid
    for (int lev = 0; lev <= finest_level; ++lev) {
        ablastr::utils::communication::FillBoundary(
            *m_fields.get(FieldType::rho_fp, lev),
            m_fields.get(FieldType::rho_fp, lev)->nGrowVect(),
            WarpX::do_single_precision_comms,
            Geom(lev).periodicity(),
            true
        );
        for (int idim = 0; idim < 3; ++idim) {
            ablastr::utils::communication::FillBoundary(
                *m_fields.get(FieldType::current_fp, Direction{idim}, lev),
                m_fields.get(FieldType::current_fp, Direction{idim}, lev)->nGrowVect(),
                WarpX::do_single_precision_comms,
                Geom(lev).periodicity(),
                true
            );
        }
    }
}

void WarpX::HybridPICInitializeRhoJandB ()
{
    // The Ohm's law solver requires two timesteps' values for the charge
    // and current densities. This function is called at the start of
    // the PIC loop (before particles have been pushed for the first time,
    // but after their positions and velocities have been de-synchronized).

    using warpx::fields::FieldType;
    using ablastr::fields::Direction;

    // Deposit rho^n and J_i^{n-1/2} from the particles. This must also run on
    // restart: the checkpoint does not contain rho_fp (and contains current_fp
    // only when written synchronized), while the particles are restored at
    // exactly (x^n, v^{n-1/2}) on both paths, so the deposit deterministically
    // reconstructs both fields. Without it the first restarted step runs the
    // adaptive B integration with rho = 0 everywhere: every node falls into
    // the below-n_floor branch of the Ohm's-law E-solve on top of the full
    // mid-run curl(B), which is catastrophically stiff (or, with the vacuum
    // treatment, silently wrong physics for one step).
    HybridPICDepositRhoAndJ();

    // Fill the electron pressure from the algebraic closure using the freshly
    // deposited rho. On a fresh start this seeds Pe^0 for the iteration-0
    // diagnostics and the first step's B-substep E-solves; on restart it
    // restores Pe(rho^n), which is not checkpointed and would otherwise be
    // zero for the whole first restarted step. From the first step onward,
    // HybridPICEvolveFields refreshes Pe right after each deposition (via the
    // closure, or via the QDSMC entropy transport when
    // solve_electron_energy_equation is on).
    m_hybrid_pic_model->CalculateElectronPressure();

    if (restart_chkfile.empty()) {
        // Handle field splitting for Hybrid field push
        if (m_hybrid_pic_model->m_add_external_fields) {
            // Get the external fields
            // Currently t_new is what t_old will be when entering the solver since
            // after initialization the t_old is set to t_new, then t_new is incremented by dt
            m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
                gett_new(0),
                0.5_rt*dt[0]);

            // If using split fields, add the external field at t=0
            for (int lev = 0; lev <= finest_level; ++lev) {
                for (int idim = 0; idim < 3; ++idim) {
                    // Check to make sure field only contains numeric values
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        m_fields.get(FieldType::hybrid_B_fp_external, Direction{idim}, lev)->is_finite(),
                        "Non-finite value detected in external B-field at t=0."
                    );

                    MultiFab::Add(
                        *m_fields.get(FieldType::Bfield_fp, Direction{idim}, lev),
                        *m_fields.get(FieldType::hybrid_B_fp_external, Direction{idim}, lev),
                        0, 0, 1,
                        m_fields.get(FieldType::Bfield_fp, Direction{idim}, lev)->nGrowVect());
                }
            }
        }
    }

    // Copy the rho_fp values to rho_fp_temp and the current_fp values to
    // current_fp_temp, since the "temp" multifabs are meant to store the
    // particle and current densities from the previous step during the field
    // solve routine and are needed when the first field solve is
    // performed after pushing the particles.
    ablastr::fields::MultiLevelScalarField rho_fp_temp = m_fields.get_mr_levels(FieldType::hybrid_rho_fp_temp, finest_level);
    ablastr::fields::MultiLevelVectorField current_fp_temp = m_fields.get_mr_levels_alldirs(FieldType::hybrid_current_fp_temp, finest_level);
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // copy 1 component value starting at index 0 to index 0
        MultiFab::Copy(*rho_fp_temp[lev], *m_fields.get(FieldType::rho_fp, lev),
                        0, 0, 1, rho_fp_temp[lev]->nGrowVect());
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::Copy(*current_fp_temp[lev][idim], *m_fields.get(FieldType::current_fp, Direction{idim}, lev),
                        0, 0, 1, current_fp_temp[lev][idim]->nGrowVect());
        }
    }
}

void
WarpX::CalculateExternalCurlA() {
    ABLASTR_PROFILE("WarpX::CalculateExternalCurlA()");

    auto & warpx = WarpX::GetInstance();

    // Get reference to External Field Object
    auto* ext_vector = warpx.m_hybrid_pic_model->m_external_vector_potential.get();
    ext_vector->CalculateExternalCurlA();

}
