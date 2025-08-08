/* Copyright 2023-2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Roelof Groenewald (TAE Technologies)
 *          S. Eric Clark (Helion Energy)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Evolve/WarpXDtType.H"
#include "Fields.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Fluids/MultiFluidContainer.H"
#include "Fluids/WarpXFluidContainer.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/utils/Communication.H>


using namespace amrex;

void WarpX::HybridPICEvolveFields ()
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    WARPX_PROFILE("WarpX::HybridPICEvolveFields()");

    // The below deposition is hard coded for a single level simulation
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        finest_level == 0,
        "Ohm's law E-solve only works with a single level.");

    // Get requested number of substeps to use
    const int sub_steps = m_hybrid_pic_model->m_substeps;

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
    for (int sub_step = 0; sub_step < sub_steps; sub_step++)
    {
        m_hybrid_pic_model->BfieldEvolveRK(
            m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level),
            m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level),
            current_fp_temp, rho_fp_temp,
            m_eb_update_E,
            0.5_rt/sub_steps*dt[0],
            DtType::FirstHalf, guard_cells.ng_FieldSolver,
            WarpX::sync_nodal_points
        );
    }

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
    for (int sub_step = 0; sub_step < sub_steps; sub_step++)
    {
        m_hybrid_pic_model->BfieldEvolveRK(
            m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level),
            m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level),
            m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level),
            rho_fp_temp,
            m_eb_update_E,
            0.5_rt/sub_steps*dt[0],
            DtType::SecondHalf, guard_cells.ng_FieldSolver,
            WarpX::sync_nodal_points
        );
    }

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

    // Calculate the electron pressure at t=n+1
    m_hybrid_pic_model->CalculateElectronPressure();

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

    // Check that the E-field does not have nan or inf values, otherwise print a clear message
    ablastr::fields::MultiLevelVectorField Efield_fp = m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level);
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        for (int idim = 0; idim < 3; ++idim) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                Efield_fp[lev][idim]->is_finite(),
                "Non-finite value detected in E-field; this indicates more substeps should be used in the field solver."
            );
        }
    }
}

void WarpX::HybridPICDepositRhoAndJ ()
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // Perform charge deposition in component 0 of rho_fp at current time.
    mypc->DepositCharge(m_fields.get_mr_levels(FieldType::rho_fp, finest_level), 0._rt);
    // Perform current deposition at t_{n-1/2}.
    mypc->DepositCurrent(m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level), dt[0], -0.5_rt * dt[0]);

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

    if (restart_chkfile.empty()) {
        // This is not a restart, so the rho_fp and current_fp multifabs are
        // still empty.
        HybridPICDepositRhoAndJ();

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
    WARPX_PROFILE("WarpX::CalculateExternalCurlA()");

    auto & warpx = WarpX::GetInstance();

    // Get reference to External Field Object
    auto* ext_vector = warpx.m_hybrid_pic_model->m_external_vector_potential.get();
    ext_vector->CalculateExternalCurlA();

}
