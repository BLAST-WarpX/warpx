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
#ifdef WARPX_DIM_RZ
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/NativeRZMagneticPressure.H"
#endif
#include "Particles/MultiParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "Fluids/MultiFluidContainer.H"
#include "Fluids/WarpXFluidContainer.H"
#include "WarpX.H"

#include <AMReX_Math.H>

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/Communication.H>

#include <array>
#include <limits>
#include <memory>

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

#ifdef WARPX_DIM_RZ
    if (m_hybrid_pic_model->m_pressure_only) {
        warpx::hybrid::NativeRZMagneticPressureDiagnostics diagnostics;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_hybrid_pic_model->tryNativePressureFieldEvolve(dt[0], getistep(0), diagnostics),
            "Native pressure_only field step rejected; the complete field/caloric state "
            "and adaptive controller were restored.");
        return;
    }
#endif

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
    HybridPICDepositRhoAndJ(/*deposit_energy_auxiliary=*/true);

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

    if (mypc->hasHybridIonization()) {
        // Freeze the old-Z electron density and temperature state. The
        // particle operator then increments accepted ion charge states and
        // deposits the corresponding ionization-potential energy density.
        auto& rho_old =
            *m_fields.get("hybrid_ionization_rho_old_fp", 0);
        auto& electron_source =
            *m_fields.get("hybrid_ionization_electron_source_fp", 0);
        auto& binding_energy =
            *m_fields.get("hybrid_ionization_binding_energy_fp", 0);
        auto& rho = *m_fields.get(FieldType::rho_fp, 0);
        auto& Te = *m_fields.get(
            FieldType::hybrid_electron_temperature_fp, 0);
        MultiFab::Copy(
            rho_old, rho, 0, 0, 1, rho_old.nGrowVect());

        // Ion positions, gett_new, and the QDSMC thermodynamic state are all
        // at the endpoint, so time-dependent coefficients use t^(n+1).
        mypc->doHybridIonization(
            0, rho_old, Te, binding_energy,
            gett_new(0), dt[0]);

        // Recompute ion rho/J with the new Z before Ohm's law. This also
        // refreshes every per-species charge field consumed by the hybrid
        // model. Particle number, mass, position and momentum are unchanged.
        HybridPICDepositRhoAndJ(/*deposit_energy_auxiliary=*/true);

        // The charge increase is exactly the new hybrid-fluid electron
        // density required by quasi-neutrality. Store it independently of the
        // binding-energy ledger for diagnostics and conservation tests.
        MultiFab::LinComb(
            electron_source,
            1.0_rt / PhysConst::q_e, rho, 0,
            -1.0_rt / PhysConst::q_e, rho_old, 0,
            0, 1, electron_source.nGrowVect());
        electron_source.FillBoundary(
            electron_source.nGrowVect(), Geom(0).periodicity());

        m_hybrid_pic_model->ApplyHybridIonizationEnergySource(
            0, rho_old, rho, binding_energy);
    }

    // Get the external current
    m_hybrid_pic_model->GetCurrentExternal();

    // Compute the per-species resistive friction from the start-of-step
    // fields, split into the frozen ion-drift remainder and the lagged
    // coefficient that the E-solves below multiply by the live plasma
    // current (see ComputeResistiveOverlay). The slow moments (Vs, rho_s,
    // T_e) are per-step quantities; the plasma-current response stays live
    // through the coefficient.
    if (m_hybrid_pic_model->m_has_per_species_eta) {
        m_hybrid_pic_model->ComputeResistiveOverlay();
    }

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
    if (m_hybrid_pic_model->m_solve_electron_energy_equation
        && !m_hybrid_pic_model->m_fv_transport_internal_energy)
    {
        m_hybrid_pic_model->QDSMCFillElectronPressureFromTe(rho_fp_temp);
    }
    m_hybrid_pic_model->BfieldEvolve(
        m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level),
        m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level),
        current_fp_temp, rho_fp_temp,
        m_eb_update_E_fp,
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

    // Re-center the per-species friction linearization on the half-step
    // fields before the second half-step B-advance: refresh J_plasma from
    // the accepted B^{n+1/2} and recompute the remainder/coefficient pair
    // about it (see ComputeResistiveOverlay).
    if (m_hybrid_pic_model->m_has_per_species_eta) {
        m_hybrid_pic_model->CalculatePlasmaCurrent(
            m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level),
            m_eb_update_E_fp);
        m_hybrid_pic_model->ComputeResistiveOverlay();
    }

    // Now push the B field from t=n+1/2 to t=n+1 using the n+1/2 quantities
    if (m_hybrid_pic_model->m_solve_electron_energy_equation
        && !m_hybrid_pic_model->m_fv_transport_internal_energy)
    {
        m_hybrid_pic_model->QDSMCFillElectronPressureFromTe(rho_fp_temp);
    }
    m_hybrid_pic_model->BfieldEvolve(
        m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level),
        m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level),
        m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level),
        rho_fp_temp,
        m_eb_update_E_fp,
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
    if (m_hybrid_pic_model->m_solve_electron_energy_equation
        && !m_hybrid_pic_model->m_fv_transport_internal_energy)
    {
        m_hybrid_pic_model->QDSMCFillElectronPressureFromTe(
            m_fields.get_mr_levels(FieldType::rho_fp, finest_level));
    }
    m_hybrid_pic_model->CalculatePlasmaCurrent(
        m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level),
        m_eb_update_E_fp);
    m_hybrid_pic_model->HybridPICSolveE(
        m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level),
        current_fp_temp,
        m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level),
        m_fields.get_mr_levels(FieldType::rho_fp, finest_level),
        m_eb_update_E_fp, false);
    FillBoundaryE(guard_cells.ng_FieldSolver, WarpX::sync_nodal_points);

    // Update Ve_fp and per-species Vs_fp at t=n+1 for the next step's
    // particle-level drag operator and per-species resistive overlay.
    if (m_hybrid_pic_model->m_need_fluid_velocities) {
        m_hybrid_pic_model->CalculateElectronFluidVelocity();
        m_hybrid_pic_model->CalculateIonFluidVelocity();
    }

    // The drag operator also gathers J_plasma (the |J| parser argument) at
    // the particle shape order, which can read beyond the single ghost
    // layer CalculateCurrentAmpere computes: refresh the ghosts, and in
    // radial geometries the below-axis guards.
    if (m_hybrid_pic_model->m_has_resistive_drag) {
        for (int lev = 0; lev <= finest_level; ++lev) {
            ablastr::fields::VectorField J_plasma =
                m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
            for (int idim = 0; idim < 3; ++idim) {
                ablastr::utils::communication::FillBoundary(
                    *J_plasma[idim], J_plasma[idim]->nGrowVect(),
                    WarpX::do_single_precision_comms, Geom(lev).periodicity());
            }
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            ApplyFieldBoundaryOnAxis(J_plasma[0], J_plasma[1], J_plasma[2], lev);
#endif
        }
    }

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

void WarpX::HybridPICDepositRhoAndJ (bool const deposit_energy_auxiliary)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    auto current_fp = m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level);
    auto rho_fp = m_fields.get_mr_levels(FieldType::rho_fp, finest_level);
    bool const deposit_energy_charge_flux =
        m_hybrid_pic_model->m_fv_transport_internal_energy
        && deposit_energy_auxiliary;
    if (deposit_energy_charge_flux) {
        // Esirkepov reconstructs the old particle position from the current
        // position and velocity.  Reject trajectories longer than one cell
        // before entering the deposition kernel, whose stencil assumes that
        // bound.  Check material carriers only; photons and non-depositing
        // species do not contribute to this auxiliary continuity ledger.
        amrex::ParticleReal max_material_dt_inv = 0.0_prt;
        for (auto const& species_name : mypc->GetSpeciesNames()) {
            auto& species =
                mypc->GetParticleContainerFromName(species_name);
            if (species.getCharge() == 0.0_prt || species.do_not_deposit) {
                continue;
            }
            max_material_dt_inv = amrex::max(
                max_material_dt_inv, species.maxParticleDtInv());
        }
        amrex::Real const max_cell_displacement =
            dt[0] * static_cast<amrex::Real>(max_material_dt_inv);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            max_cell_displacement <= 1.0_rt + 64.0_rt
                * std::numeric_limits<amrex::Real>::epsilon(),
            "Nonlinear finite-volume electron-energy transport requires "
            "dt*max_i(|v_i|/dx_i) <= 1 before its charge-conserving "
            "auxiliary current deposition. Reduce the timestep.");
    }
    ablastr::fields::MultiLevelVectorField energy_charge_flux;
    ablastr::fields::MultiLevelScalarField energy_rho_mid;
    ablastr::fields::MultiLevelVectorField energy_velocity_current;
    if (deposit_energy_charge_flux) {
        energy_charge_flux = m_fields.get_mr_levels_alldirs(
            FieldType::hybrid_energy_charge_flux_fp, finest_level);
        energy_rho_mid = m_fields.get_mr_levels(
            FieldType::hybrid_energy_rho_mid_fp, finest_level);
        energy_velocity_current = m_fields.get_mr_levels_alldirs(
            FieldType::hybrid_energy_velocity_current_fp, finest_level);
        for (int lev = 0; lev <= finest_level; ++lev) {
            for (int idim = 0; idim < 3; ++idim) {
                energy_charge_flux[lev][idim]->setVal(0.0_rt);
                energy_velocity_current[lev][idim]->setVal(0.0_rt);
            }
        }
    }
    if (m_hybrid_pic_model->m_need_per_species_fields) {
        // Per-species deposition at t_{n+1} (rho) and t_{n-1/2} (J): each
        // charged species deposits once into its own MultiFabs and the raw
        // deposits are accumulated into the totals rho_fp / current_fp. The
        // per-species fields are synchronized and converted to physical
        // density units below for downstream coupling (electron-energy
        // sources, per-species resistivity and resistive drag).
        auto rho_species_sum = m_fields.get_mr_levels("hybrid_rho_species_sum_fp", finest_level);
        for (int lev = 0; lev <= finest_level; ++lev) {
            rho_fp[lev]->setVal(0._rt);
            rho_species_sum[lev]->setVal(0._rt);
            for (int idim = 0; idim < 3; ++idim) { current_fp[lev][idim]->setVal(0._rt); }
        }
        for (auto const & spec : mypc->GetSpeciesNames()) {
            auto & pc = mypc->GetParticleContainerFromName(spec);
            if (pc.getCharge() == 0._prt || pc.do_not_deposit) { continue; }
            auto J_spec = m_fields.get_mr_levels_alldirs("current_fp_" + spec, finest_level);
            auto rho_spec = m_fields.get_mr_levels("rho_fp_" + spec, finest_level);
            bool is_eos_material = false;
            for (int material = 0;
                 material < m_hybrid_pic_model
                     ->electronThermodynamicsNumMaterials(); ++material)
            {
                is_eos_material = is_eos_material
                    || spec == m_hybrid_pic_model
                        ->electronThermodynamicsMaterialSpeciesName(material);
            }
            ablastr::fields::MultiLevelScalarField ion_count_charge;
            if (is_eos_material) {
                ion_count_charge = m_fields.get_mr_levels(
                    "ni_charge_fp_" + spec, finest_level);
                pc.DepositUnitChargeDensity(
                    ion_count_charge, /*local=*/true, /*reset=*/true,
                    /*apply_boundary_and_scale_volume=*/false,
                    /*interpolate_across_levels=*/false);
            }
            for (auto const & J_lev : J_spec) {
                for (int idim = 0; idim < 3; ++idim) { J_lev[idim]->setVal(0._rt); }
            }
            pc.DepositCurrent(J_spec, dt[0], -0.5_rt * dt[0]);
            if (deposit_energy_charge_flux) {
                pc.DepositCurrent(
                    energy_charge_flux, dt[0], -0.5_rt * dt[0],
                    PushType::Explicit, CurrentDepositionAlgo::Esirkepov);
                pc.DepositCurrent(
                    energy_velocity_current, dt[0], -0.5_rt * dt[0],
                    PushType::Explicit, CurrentDepositionAlgo::Direct);
            }
            pc.DepositCharge(rho_spec, /*local*/true, /*reset*/true,
                             /*interpolate_across_levels*/false);
            // Accumulate the RAW (locally deposited, unsummed) per-species
            // fields into the totals: shape-spread contributions near box
            // edges sit in guard cells at this point and are folded into the
            // valid cells of the totals later by SyncCurrentAndRho, exactly
            // as in the single-pass deposition path.
            for (int lev = 0; lev <= finest_level; ++lev) {
                MultiFab::Add(*rho_fp[lev], *rho_spec[lev],
                              0, 0, 1, rho_fp[lev]->nGrowVect());
                for (int idim = 0; idim < 3; ++idim) {
                    MultiFab::Add(*current_fp[lev][idim], *J_spec[lev][idim],
                                  0, 0, 1, current_fp[lev][idim]->nGrowVect());
                }
            }
            // The per-species fields themselves are consumed directly
            // (Vs = Js/rhos, species fractions, per-species resistivity,
            // resistive drag) and need their own guard-cell sum here;
            // dst_ng = nGrowVect() also leaves the ghosts neighbor-
            // consistent for the drag's particle gathers.
            for (int lev = 0; lev <= finest_level; ++lev) {
                ablastr::utils::communication::SumBoundary(
                    *rho_spec[lev], 0, rho_spec[lev]->nComp(),
                    rho_spec[lev]->nGrowVect(), rho_spec[lev]->nGrowVect(),
                    WarpX::do_single_precision_comms, Geom(lev).periodicity());
                if (is_eos_material) {
                    ablastr::utils::communication::SumBoundary(
                        *ion_count_charge[lev], 0,
                        ion_count_charge[lev]->nComp(),
                        ion_count_charge[lev]->nGrowVect(),
                        ion_count_charge[lev]->nGrowVect(),
                        WarpX::do_single_precision_comms,
                        Geom(lev).periodicity());
                    FinalizeRhoOnLevel(lev, ion_count_charge[lev], PatchType::fine);
                }
                for (int idim = 0; idim < 3; ++idim) {
                    ablastr::utils::communication::SumBoundary(
                        *J_spec[lev][idim], 0, J_spec[lev][idim]->nComp(),
                        J_spec[lev][idim]->nGrowVect(), J_spec[lev][idim]->nGrowVect(),
                        WarpX::do_single_precision_comms, Geom(lev).periodicity());
                }
                // Apply boundary conditions to rho_spec, and then apply volume scaling
                FinalizeRhoOnLevel(lev, rho_spec[lev], PatchType::fine);
                // Apply boundary conditions to J_spec, and then apply volume scaling
                FinalizeJOnLevel(lev, J_spec[lev][0], J_spec[lev][1],
                                 J_spec[lev][2], PatchType::fine);
            }
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            // Below-axis guard cells still hold raw deposit remnants after
            // the fold; fill them by parity reflection (as for E and B) for
            // the drag operator's particle gathers near r = 0.
            if (m_hybrid_pic_model->m_has_resistive_drag) {
                for (int lev = 0; lev <= finest_level; ++lev) {
                    ApplyFieldBoundaryOnAxis(
                        J_spec[lev][0], J_spec[lev][1], J_spec[lev][2], lev);
                }
            }
#endif
            // Species-summed physical charge density (same form as the
            // rho_fp_s numerators), shared by the electron-energy-equation
            // and per-species-resistivity consumers. Accumulated AFTER the
            // guard-cell sum so its valid and ghost cells are final.
            for (int lev = 0; lev <= finest_level; ++lev) {
                MultiFab::Add(*rho_species_sum[lev], *rho_spec[lev],
                              0, 0, 1, rho_species_sum[lev]->nGrowVect());
            }
        }
    } else {
        // Single-pass deposition (rho at t_{n+1}, J at t_{n-1/2}): no active
        // feature consumes the per-species fields, so skip the per-species
        // deposits and guard-cell sums entirely. Zeroing is handled inside;
        // boundary conditions and radial volume scaling are applied below,
        // in FinalizeRho() and FinalizeJ().
        mypc->DepositCharge(rho_fp, 0._rt);
        mypc->DepositCurrent(current_fp, dt[0], -0.5_rt * dt[0]);
    }

    if (deposit_energy_charge_flux) {
        // Co-deposit rho and the direct velocity moment at exactly the same
        // particle midpoint.  Their ratio is therefore exactly constant for
        // a rigidly translating material, including CIC interface tails.
        // Boundary conditions and radial inverse-volume scaling are applied
        // below via FinalizeJOnLevel / FinalizeRhoOnLevel after SumBoundary.
        mypc->DepositCharge(energy_rho_mid, -0.5_rt * dt[0]);
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
    SyncCurrentAndRho();

    // Apply boundary conditions to Rho and J, and then apply volume scaling.
    FinalizeRho();
    FinalizeJ();

    if (deposit_energy_charge_flux) {
        // Sum the dedicated Esirkepov deposits without applying the primary
        // current filter: rho and this auxiliary face flux must retain their
        // exact discrete continuity relation.  Hybrid-PIC is single-level.
        for (int lev = 0; lev <= finest_level; ++lev) {
            auto const& period = Geom(lev).periodicity();
            for (int idim = 0; idim < 3; ++idim) {
                for (auto* auxiliary_current : {
                         energy_charge_flux[lev][idim],
                         energy_velocity_current[lev][idim]})
                {
                    ablastr::utils::communication::SumBoundary(
                        *auxiliary_current, 0, auxiliary_current->nComp(),
                        auxiliary_current->nGrowVect(),
                        auxiliary_current->nGrowVect(), false, period);
                }
            }
            FinalizeJOnLevel(
                lev, energy_charge_flux[lev][0],
                energy_charge_flux[lev][1], energy_charge_flux[lev][2],
                PatchType::fine);
            FinalizeJOnLevel(
                lev, energy_velocity_current[lev][0],
                energy_velocity_current[lev][1],
                energy_velocity_current[lev][2], PatchType::fine);
            ablastr::utils::communication::SumBoundary(
                *energy_rho_mid[lev], 0, energy_rho_mid[lev]->nComp(),
                energy_rho_mid[lev]->nGrowVect(),
                energy_rho_mid[lev]->nGrowVect(), false, period);
            FinalizeRhoOnLevel(lev, energy_rho_mid[lev], PatchType::fine);
            ablastr::utils::communication::FillBoundary(
                *energy_rho_mid[lev], energy_rho_mid[lev]->nGrowVect(),
                false, period, true);
            for (int idim = 0; idim < 3; ++idim) {
                ablastr::utils::communication::FillBoundary(
                    *energy_charge_flux[lev][idim],
                    energy_charge_flux[lev][idim]->nGrowVect(), false,
                    period, true);
                ablastr::utils::communication::FillBoundary(
                    *energy_velocity_current[lev][idim],
                    energy_velocity_current[lev][idim]->nGrowVect(), false,
                    period, true);
            }
        }
    }

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

void
WarpX::HybridPICInitializeElectronPressure ()
{
    using warpx::fields::FieldType;
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_1D_Z) \
    || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D)
    if (m_hybrid_pic_model->m_has_initial_elec_pressure && restart_chkfile.empty()
        && !m_hybrid_pic_model->m_initial_elec_pressure_applied)
    {
        auto const pressure = m_hybrid_pic_model->m_initial_elec_pressure;
        auto const eos = m_hybrid_pic_model->electronThermodynamicsExecutor();
        for (int lev = 0; lev <= finest_level; ++lev) {
            auto& temperature = *m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
            auto const& charge = *m_fields.get(FieldType::rho_fp, lev);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(temperature.nComp() == 1
                    && temperature.ixType().nodeCentered()
                    && temperature.ixType() == charge.ixType(),
                "Initial electron pressure requires scalar collocated nodal temperature and charge.");
            auto const lo = Geom(lev).ProbLoArray();
            auto const dx = Geom(lev).CellSizeArray();
            auto const domain = amrex::convert(Geom(lev).Domain(), temperature.ixType());
            auto const lower = amrex::lbound(domain);
            auto const upper = amrex::ubound(domain);
            auto const periodic = Geom(lev).isPeriodicArray();
            for (amrex::MFIter mfi(temperature); mfi.isValid(); ++mfi) {
                auto const t = temperature.array(mfi);
                auto const rho = charge.const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
#if defined(WARPX_DIM_3D)
                    auto const p = pressure(lo[0]+(i-lower.x)*dx[0],
                                            lo[1]+(j-lower.y)*dx[1],
                                            lo[2]+(k-lower.z)*dx[2]);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                    auto const p = pressure(lo[0]+(i-lower.x)*dx[0], 0.0_rt,
                                            lo[1]+(j-lower.y)*dx[1]);
#else
                    auto const p = pressure(0.0_rt, 0.0_rt, lo[0]+(i-lower.x)*dx[0]);
#endif
                    auto const density = rho(i,j,k);
                    auto const value = density > 0.0_rt ? p*PhysConst::q_e/(density*PhysConst::kb)
                        : eos.minimumTemperature();
                    t(i,j,k) = amrex::Math::isfinite(p) && p >= 0.0_rt
                        && amrex::Math::isfinite(density) && density >= 0.0_rt
                        && (density > 0.0_rt || p == 0.0_rt)
                        && amrex::Math::isfinite(value) && value >= eos.minimumTemperature()
                        && value <= eos.maximumTemperature()
                        ? value : std::numeric_limits<amrex::Real>::quiet_NaN();
                });
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(temperature.is_finite(0,1,0),
                "Initial electron pressure is invalid or requests nonzero pressure in vacuum.");
            ablastr::utils::communication::FillBoundary(temperature, temperature.nGrowVect(),
                false, Geom(lev).periodicity(), true);
            // Scalar even extension, including the coordinate axis. Read only
            // interior/previously exchanged values; each ghost destination is unique.
            for (amrex::MFIter mfi(temperature); mfi.isValid(); ++mfi) {
                auto const t = temperature.array(mfi);
                amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    int ii = i, jj = j, kk = k;
                    if (!periodic[0]) {
                        if (i < lower.x) { ii = 2*lower.x-i; }
                        if (i > upper.x) { ii = 2*upper.x-i; }
                    }
#if AMREX_SPACEDIM >= 2
                    if (!periodic[1]) {
                        if (j < lower.y) { jj = 2*lower.y-j; }
                        if (j > upper.y) { jj = 2*upper.y-j; }
                    }
#endif
#if AMREX_SPACEDIM == 3
                    if (!periodic[2]) {
                        if (k < lower.z) { kk = 2*lower.z-k; }
                        if (k > upper.z) { kk = 2*upper.z-k; }
                    }
#endif
                    if (ii != i || jj != j || kk != k) { t(i,j,k) = t(ii,jj,kk); }
                });
            }
        }
        m_hybrid_pic_model->m_initial_elec_pressure_applied = true;
    }
#endif
    bool const preserve_evolved_temperature =
        m_hybrid_pic_model->m_solve_electron_energy_equation &&
        (!restart_chkfile.empty() || m_hybrid_pic_model->m_has_initial_elec_temp ||
         m_hybrid_pic_model->m_has_initial_elec_pressure ||
         !m_hybrid_pic_model->electronThermodynamicsExecutor().isIdealGas());
    if (preserve_evolved_temperature) {
        for (int lev = 0; lev <= finest_level; ++lev) {
            auto& Te = *m_fields.get(FieldType::hybrid_electron_temperature_fp, lev);
            ablastr::utils::communication::FillBoundary(Te, Te.nGrowVect(), false,
                                                        Geom(lev).periodicity(), true);
            m_hybrid_pic_model->QDSMCFillElectronPressureFromTe(
                lev, *m_fields.get(FieldType::rho_fp, lev));
            ApplyElectronPressureBoundary(lev, PatchType::fine);
            ablastr::utils::communication::FillBoundary(
                *m_fields.get(FieldType::hybrid_electron_pressure_fp, lev),
                WarpX::do_single_precision_comms, Geom(lev).periodicity(), true);
        }
    } else {
        m_hybrid_pic_model->CalculateElectronPressure(
            m_hybrid_pic_model->m_solve_electron_energy_equation);
    }
}

void
WarpX::HybridPICPrepareElectronStateForDiagnostics ()
{
    using warpx::fields::FieldType;
#ifdef WARPX_DIM_RZ
    if (m_hybrid_pic_model->m_pressure_only) {
        if (!m_hybrid_pic_model->m_native_pressure_state_initialized) {
            HybridPICDepositRhoAndJ(/*deposit_energy_auxiliary=*/false);
            m_hybrid_pic_model->InitializeNativePressureState();
        }
        m_hybrid_pic_model->RefreshNativePressureState();
        return;
    }
#endif
    // Use the native density/species/boundary preparation, but preserve the
    // time staggering and any supplied/restored current. The normal PIC
    // bootstrap will deposit its own half-time current after desynchronization.
    auto current = m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level);
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> saved(finest_level + 1);
    for (int lev = 0; lev <= finest_level; ++lev) {
        for (int d = 0; d < 3; ++d) {
            auto const& source = *current[lev][d];
            saved[lev][d] = std::make_unique<amrex::MultiFab>(
                source.boxArray(), source.DistributionMap(), source.nComp(), source.nGrowVect());
            amrex::MultiFab::Copy(*saved[lev][d], source, 0, 0, source.nComp(), source.nGrowVect());
        }
    }
    HybridPICDepositRhoAndJ(/*deposit_energy_auxiliary=*/false);
    HybridPICInitializeElectronPressure();
    for (int lev = 0; lev <= finest_level; ++lev) {
        for (int d = 0; d < 3; ++d) {
            auto& destination = *current[lev][d];
            amrex::MultiFab::Copy(destination, *saved[lev][d], 0, 0, destination.nComp(),
                                  destination.nGrowVect());
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

#ifdef WARPX_DIM_RZ
    if (m_hybrid_pic_model->m_pressure_only) {
        HybridPICPrepareElectronStateForDiagnostics();
        m_hybrid_pic_model->m_restored_moment_history_pending = false;
        m_hybrid_pic_model->m_moment_history_valid = true;
        return;
    }
#endif

    // Restore deposited rho^n and J_i^{n-1/2} history when available. Legacy
    // checkpoints omit rho_fp and save current_fp only when synchronized;
    // reconstruct those deposits from particles at (x^n, v^{n-1/2}). GPU
    // scatter summation need not reproduce the original bits, hence the new
    // checkpoint history. Without either restoration or reconstruction the
    // first restarted step runs the
    // adaptive B integration with rho = 0 everywhere: every node falls into
    // the below-n_floor branch of the Ohm's-law E-solve on top of the full
    // mid-run curl(B), which is catastrophically stiff (or, with the vacuum
    // treatment, silently wrong physics for one step).
    // This initialization deposit reconstructs rho^n and J_i^(n-1/2), but
    // there is no n -> n+1 material trajectory yet.  The first evolved
    // deposit will initialize the auxiliary continuity flux consistently.
    if (!m_hybrid_pic_model->m_restored_moment_history_pending) {
        HybridPICDepositRhoAndJ(/*deposit_energy_auxiliary=*/false);
    }
    m_hybrid_pic_model->m_restored_moment_history_pending = false;
    m_hybrid_pic_model->m_moment_history_valid = true;

    // Fill the electron pressure using deposited or restored rho. On a fresh
    // ideal/polytropic start this seeds Pe^0 and the corresponding T_e for the
    // first step's B-substep E-solves. Initial energy-equation diagnostics
    // prepare the same state independently before desynchronization. Any
    // nonlinear caloric EOS must instead preserve the input T_e seeded by
    // InitData and evaluate its own P(rho,T); running the legacy closure here
    // would silently replace both quantities with ideal-polytropic values. An
    // explicitly initialized temperature profile must likewise survive this
    // bootstrap. On restart every evolved temperature is restored from its
    // checkpoint and likewise must not be replaced by the algebraic closure. Pe
    // is derived, so rebuild it from the preserved T_e and reconstructed rho.
    // This also preserves radiation, Joule and collisional changes to the
    // hybrid electron internal energy across a restart.
    HybridPICInitializeElectronPressure();

    // Handle field splitting for the hybrid field push: stage the external
    // fields and add their contribution to the total field once, at the
    // true start of the run. This entry point executes at the first step
    // of EVERY Evolve() call, and beyond the first entry Bfield_fp already
    // carries the external contribution (segmented stepping through
    // repeated sim.step() calls re-enters here mid-run) -- adding it again
    // double-counts the external field on every re-entry. On restart the
    // checkpointed fields likewise already contain it.
    if (restart_chkfile.empty() && istep[0] == 0) {
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

    if (m_hybrid_pic_model->m_conservative_pressure_work
        && restart_chkfile.empty())
    {
        // The ordinary hybrid startup historically leaves E at its input
        // value until the end of step one.  The exact pressure-work ledger
        // cannot debit -P div(V_work) unless the matching -grad(P)/rho force
        // was actually present in that first particle push.  Build the
        // complete Ohm-law E^0 now from the deposited initial moments and
        // seed both the checkpointed pressure component and its auxiliary
        // gather representation before any particle moves.
        m_hybrid_pic_model->CalculatePlasmaCurrent(
            m_fields.get_mr_levels_alldirs(
                FieldType::Bfield_fp, finest_level),
            m_eb_update_E_fp);
        m_hybrid_pic_model->HybridPICSolveE(
            m_fields.get_mr_levels_alldirs(
                FieldType::Efield_fp, finest_level),
            current_fp_temp,
            m_fields.get_mr_levels_alldirs(
                FieldType::Bfield_fp, finest_level),
            m_fields.get_mr_levels(FieldType::rho_fp, finest_level),
            m_eb_update_E_fp, false);
        FillBoundaryE(
            guard_cells.ng_FieldSolver, WarpX::sync_nodal_points);
        UpdateAuxiliaryData();
    }

    // Seed Ve_fp / Vs_fp (and the J_plasma they derive from) for the first
    // step: collisions run before the first HybridPICEvolveFields, so the
    // resistive drag would otherwise gather the alloc-init zeros. This
    // matters especially on restart, where the checkpointed E already
    // contains the eta*J term while Ve/Vs are not checkpointed.
    if (m_hybrid_pic_model->m_need_fluid_velocities) {
        m_hybrid_pic_model->GetCurrentExternal();
        m_hybrid_pic_model->CalculatePlasmaCurrent(
            m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level),
            m_eb_update_E_fp);
        m_hybrid_pic_model->CalculateElectronFluidVelocity();
        m_hybrid_pic_model->CalculateIonFluidVelocity();
    }
    if (m_hybrid_pic_model->m_has_resistive_drag) {
        for (int lev = 0; lev <= finest_level; ++lev) {
            ablastr::fields::VectorField J_plasma =
                m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
            for (int idim = 0; idim < 3; ++idim) {
                ablastr::utils::communication::FillBoundary(
                    *J_plasma[idim], J_plasma[idim]->nGrowVect(),
                    WarpX::do_single_precision_comms, Geom(lev).periodicity());
            }
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            ApplyFieldBoundaryOnAxis(J_plasma[0], J_plasma[1], J_plasma[2], lev);
#endif
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
