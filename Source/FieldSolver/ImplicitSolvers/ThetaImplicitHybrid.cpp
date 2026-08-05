/* Copyright 2026 Prabhat Kumar
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Fields.H"
#include "ThetaImplicitHybrid.H"
#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ExternalVectorPotential.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"
#include <ablastr/utils/Communication.H>
#include <ablastr/coarsen/sample.H>

using warpx::fields::FieldType;
using namespace amrex::literals;

void ThetaImplicitHybrid::Define (WarpX* const a_WarpX, bool /*from_restart*/)
{
    BL_PROFILE("ThetaImplicitHybrid::Define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_is_defined,
        "ThetaImplicitHybrid object is already defined!");

    m_WarpX = a_WarpX;
    m_num_amr_levels = 1;
    // E0 for the mass-matrix Jacobian is saved inside ComputeRHS (see the comment
    // there); the default SaveE in PreLinearSolve would save the wrong field.
    m_scheme_saves_E0 = true;

    m_hybrid_pic_model = m_WarpX->get_pointer_HybridPICModel();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_pic_model != nullptr,
        "ThetaImplicitHybrid solver requires hybrid PIC model to be defined");
    // With the electron energy equation on, the implicit scheme handles the
    // transport, compression and Joule heating through the in-loop pe advance
    // (discretely energy-paired); only the symmetric Q_ei ion-electron exchange
    // is applied once per step (it must kick the ion particles). The
    // include_joule_heating flag is therefore inert implicitly, and the
    // Joule-redirect-to-ions option is not supported.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !(m_hybrid_pic_model->m_solve_electron_energy_equation &&
          m_hybrid_pic_model->m_joule_redirect_to_ions),
        "The Joule redirect-to-ions option is not supported with the "
        "theta-implicit hybrid solver (Joule heating enters through the "
        "in-loop energy pairing).");

    /// Set flag for external fields from vector potentials
    m_add_external_fields = m_hybrid_pic_model->m_add_external_fields;

    m_E.Define( m_WarpX, "Efield_fp" );
    m_Eold.Define( m_E );

    // Define B_old MultiFabs
    using ablastr::fields::Direction;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const auto& Bfp_x = m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, lev);
        const auto& dm = Bfp_x->DistributionMap();
        const amrex::IntVect ngb = Bfp_x->nGrowVect();
        
        for (int dir = 0; dir < 3; ++dir) {
            const auto& ba = m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{dir}, lev)->boxArray();
            m_WarpX->m_fields.alloc_init(FieldType::B_old, Direction{dir}, lev, ba, dm, 1, ngb, 0.0_rt);
        }
    }

    const amrex::ParmParse pp("implicit_evolve");
    pp.query("theta", m_theta);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_theta >= 0.5 && m_theta <= 1.0,
        "theta parameter must be between 0.5 and 1.0");
    pp.query("use_darwin_split", m_use_darwin_split);
    pp.query("use_rho_response_divj", m_use_rho_response_divj);

    // Allocate persistent nodal alpha MultiFab for curl-curl preconditioner
    m_alpha_mf.resize(m_num_amr_levels);
    m_alpha_mfarrvec.resize(m_num_amr_levels);
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const auto* rho_mf = m_WarpX->m_fields.get(FieldType::rho_fp, lev);
        auto const& ba = amrex::convert(rho_mf->boxArray(),
                                        amrex::IntVect::TheNodeVector());
        m_alpha_mf[lev].define(ba, rho_mf->DistributionMap(),
                               1, amrex::IntVect::TheZeroVector());
        m_alpha_mfarrvec[lev] = &m_alpha_mf[lev];
    }

    parseNonlinearSolverParams( pp );
    m_nlsolver->Define(m_E, this);

    if (m_use_mass_matrices) { InitializeMassMatrices(); }

    m_is_defined = true;
}

void ThetaImplicitHybrid::PrintParameters () const
{
    BL_PROFILE("ThetaImplicitHybrid::PrintParameters()");

    if (!m_WarpX->Verbose()) { return; }
    amrex::Print() << "\n";
    amrex::Print() << "-----------------------------------------------------------\n";
    amrex::Print() << "-------- THETA IMPLICIT HYBRID PIC SOLVER PARAMETERS ------\n";
    amrex::Print() << "-----------------------------------------------------------\n";
    amrex::Print() << "Time-bias parameter theta:           " << m_theta << "\n";
    PrintBaseImplicitSolverParameters();
    m_nlsolver->PrintParams();
    amrex::Print() << "-----------------------------------------------------------\n\n";
}

int ThetaImplicitHybrid::OneStep ( const amrex::Real  start_time,
                                    const amrex::Real  a_dt,
                                    const int          a_step )
{
    BL_PROFILE("ThetaImplicitHybrid::OneStep()");

    m_dt = a_dt;

    // Handle external field splitting: work with internal fields during the solve
    if (m_add_external_fields) {
        m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
            start_time, 0.5_rt * m_dt);
        SubtractExternalEfield();
        SubtractExternalBfield();
    }

    // Save particle state at t^n
    m_WarpX->SaveParticlesAtImplicitStepStart();

    // Save E^n (or E_T^n if Darwin split)
    m_Eold.Copy(FieldType::Efield_fp);

    // Darwin split: decompose E^n = E_T^n + E_L^n; JFNK iterates on E_T only
    if (m_use_darwin_split) {
        m_hybrid_pic_model->ComputeLongitudinalE(0, GetLinOpBCLo(), GetLinOpBCHi());
        SubtractLongitudinalE(m_Eold);
    }

    // Save B^n
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const ablastr::fields::VectorField Bfp = m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, lev);
        ablastr::fields::VectorField B_old = m_WarpX->m_fields.get_alldirs(FieldType::B_old, lev);
        for (int n = 0; n < 3; ++n) {
            amrex::MultiFab::Copy(*B_old[n], *Bfp[n], 0, 0, 
                                  B_old[n]->nComp(), B_old[n]->nGrowVect());
        }
    }

    // Initial guess: E^{n+θ} = E^n  (or E_T^{n+θ} = E_T^n if Darwin split)
    m_E.Copy(m_Eold);
    
    // Solve nonlinear system for E^{n+θ} (and eventually Pe^{n+θ})
    m_nlsolver->Solve( m_E, m_Eold, start_time, m_dt, a_step );

    const int exit_status = m_nlsolver->GetExitStatus();
    if (exit_status < 0) { return exit_status; }

    // Update WarpX fields to t^{n+θ}
    UpdateWarpXFields( m_E, start_time );
    m_WarpX->reduced_diags->ComputeDiagsMidStep(a_step);

    // Advance particles from t^{n+1/2} to t^{n+1}
    m_WarpX->FinishImplicitParticleUpdate( start_time + m_dt );

    // Advance fields (including the electron pressure state) from t^{n+θ} to t^{n+1}
    FinishFieldUpdate( start_time + m_dt );

    // Electron energy equation: the transport/compression/Joule part of the
    // update already happened inside the Newton solve (in-loop pe advance,
    // energy-paired). What remains is the symmetric Q_ei ion-electron
    // collisional exchange, which must kick the ion particles and therefore
    // runs once per step here, on the t^{n+1} state.
    if (m_hybrid_pic_model->m_solve_electron_energy_equation &&
        m_hybrid_pic_model->m_include_temperature_relaxation) {
        // Without the Q_ei relaxation there is nothing left to do: transport,
        // compression and Joule heating are already handled by the in-loop pe
        // advance, so the energy equation reduces exactly to the gamma-law
        // closure path (no end-of-step redistribute/deposits needed).
        //
        // FinishImplicitParticleUpdate moved particles to t^{n+1} but did not
        // Redistribute them into their valid cells. QDSMCApplyIonHeating does a
        // per-ion NGP lookup into a zero-guard coefficient MultiFab, so ions left
        // in guard cells would read out of bounds. Redistribute first, matching
        // the explicit path's precondition (it redistributes before the QDSMC step).
        m_WarpX->GetPartContainer().Redistribute();
        m_WarpX->GetPartContainer().DepositCharge(
            m_WarpX->m_fields.get_mr_levels(FieldType::rho_fp, m_num_amr_levels - 1),
            0._rt);
        // Sync rho^{n+1} (DepositCharge does not): fold guard-cell deposits into
        // the valid (incl. periodic) nodes so n_e is unbiased at the boundaries.
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            amrex::MultiFab* rf = m_WarpX->m_fields.get(FieldType::rho_fp, lev);
            ablastr::utils::communication::SumBoundary(
                *rf, 0, rf->nComp(), rf->nGrowVect(), rf->nGrowVect(),
                WarpX::do_single_precision_comms, m_WarpX->Geom(lev).periodicity());
            ablastr::utils::communication::FillBoundary(
                *rf, rf->nGrowVect(), WarpX::do_single_precision_comms,
                m_WarpX->Geom(lev).periodicity(), true);
        }
        // Per-species charge densities rho_fp_<spec> at t^{n+1}: the QDSMC Joule
        // and Q_ei sources read them for the species fractions
        // f_s = rho_s / Sigma_t rho_t. The explicit path deposits them every step
        // in HybridPICDepositRhoAndJ; the implicit path does not call that, so
        // without this deposit they stay frozen at their initialization values
        // (stale f_s, and f_s = 0 in cells the plasma has since moved into).
        // Deposited unscaled in RZ (apply_boundary_and_scale_volume = false) to
        // match the explicit convention -- the 2*pi*r factors cancel in f_s.
        {
            auto & mypc = m_WarpX->GetPartContainer();
            for (auto const & spec : mypc.GetSpeciesNames()) {
                auto & pc = mypc.GetParticleContainerFromName(spec);
                if (pc.getCharge() == 0._prt) { continue; }
                pc.DepositCharge(
                    m_WarpX->m_fields.get_mr_levels("rho_fp_" + spec, m_num_amr_levels - 1),
                    /*local*/false, /*reset*/true,
                    /*apply_boundary_and_scale_volume*/false,
                    /*interpolate_across_levels*/false);
            }
        }
        // Deposit the per-species ion temperature T_<nm> for the Q_ei relaxation.
        // The explicit path fills it in HybridPICDepositRhoAndJ; the implicit path
        // does not call that, so it must deposit here (on the redistributed t^{n+1}
        // particles) or the Q_ei exchange reads a stale T_i.
        m_WarpX->GetPartContainer().DepositTemperatures(m_WarpX->m_fields, 0._rt);
        // In-loop integration: transport, compression and Joule heating were
        // advanced inside the Newton solve (energy-paired); only the symmetric
        // Q_ei ion-electron exchange remains, applied on T_e^{n+1} synced from
        // the in-loop pe^{n+1} and the freshly deposited rho^{n+1}.
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            m_hybrid_pic_model->FillTeFromPe(lev);
            m_hybrid_pic_model->ApplyIonElectronEnergyExchange(lev, m_dt);
            m_hybrid_pic_model->FillPeFromTe(lev);
        }
        // Roll the in-loop pressure state so the next step starts from the
        // relaxed pe^{n+1}.
        if (m_pe_old) {
            amrex::MultiFab* pe =
                m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, 0);
            amrex::MultiFab::Copy(*m_pe_old, *pe, 0, 0, pe->nComp(), pe->nGrowVect());
            m_pe_old->FillBoundary(m_WarpX->Geom(0).periodicity());
        }
    } else {
        // No Q_ei step: still mirror T_e = P_e/(n_e k_B) from the in-loop
        // pe^{n+1} so the "Te" diagnostic tracks the evolving pressure --
        // it is otherwise only filled at initialization and would dump as a
        // stale uniform value. Diagnostic-only (rho_fp here is the last
        // solver-state deposit, an O(theta dt) old density).
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            m_hybrid_pic_model->FillTeFromPe(lev);
        }
    }

    return exit_status;
}

void ThetaImplicitHybrid::ComputeRHS ( WarpXSolverVec&        a_RHS,
                                       const WarpXSolverVec&  a_E,
                                       amrex::Real            start_time,
                                       int                    a_nl_iter,
                                       bool                   a_from_jacobian )
{
    BL_PROFILE("ThetaImplicitHybrid::ComputeRHS()");

    UpdateWarpXFields( a_E, start_time );

    const amrex::Real theta_time = start_time + m_theta * m_dt;

    ablastr::fields::MultiLevelVectorField Efield_fp = 
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, m_num_amr_levels - 1);
    ablastr::fields::MultiLevelVectorField Bfield_fp = 
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1);
    ablastr::fields::MultiLevelVectorField current_fp = 
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, m_num_amr_levels - 1);
    ablastr::fields::MultiLevelScalarField rho_fp = 
        m_WarpX->m_fields.get_mr_levels(FieldType::rho_fp, m_num_amr_levels - 1);

    m_hybrid_pic_model->CalculatePlasmaCurrent(Bfield_fp, m_WarpX->GetEBUpdateEFlag());

    if (m_use_mass_matrices_jacobian && a_from_jacobian && m_Ji_save[0]) {
        // Use the ion current frozen at the last nonlinear evaluation, so the push
        // field is a pure function of the Newton variable (see m_Ji_save).
        for (int n = 0; n < 3; ++n) {
            amrex::MultiFab::Copy(*current_fp[0][n], *m_Ji_save[n], 0, 0,
                                  m_Ji_save[n]->nComp(), m_Ji_save[n]->nGrowVect());
        }
    }

    // Particles are pushed with the Newton iterate itself, minus the dissipative
    // part of Ohm's law: E* = a_E - D, D = eta*J_p - eta_h*nabla^2(J_p)
    // (Stanier et al. JCP 2019, Eq. (1); E* = a_E for eta = eta_h = 0). Pushing with the iterate gives the residual a true
    // Jacobian through the particle response -- in particular the electrostatic
    // limit (B = 0) is degenerate with any recomputed push field, which would
    // not depend on the solver variable at all.
    SubtractDissipativeEFromPushField();

    m_WarpX->ApplyFillBoundaryE();

    if (m_add_external_fields) {
        m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
            theta_time, 0.5_rt * m_dt);
        AddExternalBfield();
        AddExternalEfield();
    }

    if (!a_from_jacobian && m_use_mass_matrices_jacobian) {
        // Save the push field E0 for the mass-matrix linear model J = J0 + MM*(E - E0).
        // This must be the same field the linear stage sees at this point of the
        // evaluation (the resistivity-free Ohm's-law E, incl. external fields), not the
        // full Ohm's-law E that Efield_fp holds after ComputeRHS: saving the latter
        // (the default SaveE in PreLinearSolve) puts an O(||R||) offset into MM*(E-E0)
        // and stalls Newton once the fluctuation amplitude grows.
        SaveE();
        // Also capture the ion current that produced this push field (see m_Ji_save).
        for (int n = 0; n < 3; ++n) {
            const amrex::MultiFab& J = *current_fp[0][n];
            if (!m_Ji_save[n]) {
                m_Ji_save[n] = std::make_unique<amrex::MultiFab>(
                    J.boxArray(), J.DistributionMap(), J.nComp(), J.nGrowVect());
            }
            amrex::MultiFab::Copy(*m_Ji_save[n], J, 0, 0, J.nComp(), J.nGrowVect());
        }
    }

    PreRHSOp( theta_time, a_nl_iter, a_from_jacobian );

    {
        // Make the Ohm's-law rho a pure function of the Newton iterate: component 0
        // (deposited at entry positions) carries the previous evaluation's particle
        // state, which pollutes the finite-difference Jacobian matvec at O(hysteresis/eps)
        // in problems where E is deposition-noise dominated (e.g. the electrostatic
        // limit). The post-push component is deposited fresh from x^n each evaluation
        // (Stanier et al. use the half-time moments in Ohm's law for the same reason).
        // At the nonlinear fixed point the two components coincide.
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            amrex::MultiFab* rho = m_WarpX->m_fields.get(FieldType::rho_fp, lev);
            const int nc = rho->nComp()/2;
            const int c_new = rho->nComp() - nc;
            // NOTE: not MultiFab::Copy(*rho, *rho, ...) -- self-copy is UB in AMReX
            for (amrex::MFIter mfi(*rho); mfi.isValid(); ++mfi) {
                const amrex::Box bx = mfi.growntilebox();
                auto const& a = rho->array(mfi);
                amrex::ParallelFor(bx, nc, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                    a(i,j,k,n) = a(i,j,k,c_new + n);
                });
            }
            // The deposit+SumBoundary leaves rho ghosts stale; the electron-pressure
            // advance reads rho through ghosts at box faces, so refresh them here.
            rho->FillBoundary(m_WarpX->Geom(lev).periodicity());
        }
    }

    if (m_use_mass_matrices_jacobian && m_use_rho_response_divj) {
        if (!a_from_jacobian) {
            // Nonlinear evaluation: J and rho are now scaled and synced; capture the
            // linearization base for the rho response (see m_J_base in the header).
            CaptureJRhoBase();
        } else if (m_J_base[0]) {
            // Linear stage: rho = rho_base - (dt/2) div(J - J_base). This replaces both
            // the frozen-rho and retained-rho semantics of the MM linear stage.
            ApplyRhoResponseFromDivJ();
        }
    }

    if (m_add_external_fields) {
        SubtractExternalBfield();
        SubtractExternalEfield();
    }

    // --- Compute full Ohm's law E for Faraday update ---
    // The gamma-law electron pressure is advanced to pe^{n+theta} inside every
    // residual evaluation (in-loop), with its work terms discretely paired to
    // the ion push and Faraday ledgers -- this IS the implicit scheme's
    // implementation of the gamma-law closure (energy-conserving by
    // construction, and the only pe model currently supported implicitly).
    AdvanceElectronPressure( a_from_jacobian );

    m_hybrid_pic_model->HybridPICSolveE(
        Efield_fp, current_fp, Bfield_fp, rho_fp,
        m_WarpX->GetEBUpdateEFlag(),
        true, true   // with resistivity and ∇Pe included (∇Pe is curl-free so doesn't affect Faraday, but needed for self-consistent Newton residual)
    );

    m_WarpX->ApplyFillBoundaryE();

    // Darwin split: always compute E_L from the current E_Ohm so that the JFNK
    // Jacobian (J·v = (F(x+εv) - F(x))/ε) includes ∂E_L/∂E_T and is exactly
    // consistent with the residual R = E_T_Ohm - E_T^n.  Freezing E_L only in
    // GMRES matvecs (a_from_jacobian=true) misses ∂E_L/∂E_T and causes Newton
    // to enter a 2-cycle at late simulation times when E_L is large.
    //
    // To keep the MLMG warm-start seed (hybrid_darwin_phi_fp) from being
    // corrupted by the small ε-perturbations inside each matvec, we save and
    // restore phi around every a_from_jacobian call.  The outer (!a_from_jacobian)
    // call advances phi normally and provides the warm-start seed.
    if (m_use_darwin_split) {
        if (a_from_jacobian) {
            // Save the outer phi so each matvec starts from the same warm seed.
            amrex::MultiFab& phi_mf =
                *m_WarpX->m_fields.get(FieldType::hybrid_darwin_phi_fp, 0);
            amrex::MultiFab phi_save(phi_mf.boxArray(), phi_mf.DistributionMap(),
                                     1, phi_mf.nGrowVect());
            amrex::MultiFab::Copy(phi_save, phi_mf, 0, 0, 1, phi_mf.nGrowVect());
            m_hybrid_pic_model->ComputeLongitudinalE(0, GetLinOpBCLo(), GetLinOpBCHi());
            amrex::MultiFab::Copy(phi_mf, phi_save, 0, 0, 1, phi_mf.nGrowVect());
        } else {
            m_hybrid_pic_model->ComputeLongitudinalE(0, GetLinOpBCLo(), GetLinOpBCHi());
        }
        SubtractLongitudinalE();  // Efield_fp = E_T = E_Ohm - E_L
    }

    // RHS = E_T_ohm - E_T_old  (or E_ohm - E_old when Darwin split is off)
    a_RHS.Copy(FieldType::Efield_fp);
    a_RHS.linComb(1.0, a_RHS, -1.0, m_Eold);
}

void ThetaImplicitHybrid::CaptureJRhoBase ()
{
    using warpx::fields::FieldType;
    using ablastr::fields::Direction;

    const int lev = 0;
    const ablastr::fields::VectorField J = m_WarpX->m_fields.get_alldirs(FieldType::current_fp, lev);
    const amrex::MultiFab* rho = m_WarpX->m_fields.get(FieldType::rho_fp, lev);

    for (int n = 0; n < 3; ++n) {
        if (!m_J_base[n]) {
            m_J_base[n] = std::make_unique<amrex::MultiFab>(
                J[n]->boxArray(), J[n]->DistributionMap(), J[n]->nComp(), J[n]->nGrowVect());
        }
        amrex::MultiFab::Copy(*m_J_base[n], *J[n], 0, 0, J[n]->nComp(), J[n]->nGrowVect());
    }
    if (!m_rho_base) {
        m_rho_base = std::make_unique<amrex::MultiFab>(
            rho->boxArray(), rho->DistributionMap(), rho->nComp(), rho->nGrowVect());
    }
    amrex::MultiFab::Copy(*m_rho_base, *rho, 0, 0, rho->nComp(), rho->nGrowVect());
}

void ThetaImplicitHybrid::ApplyRhoResponseFromDivJ ()
{
    // rho = rho_base - (dt/2) * div(J - J_base), applied to the component of rho that
    // the Ohm's-law solve reads (component 0). J components are interpolated to nodes
    // (rho is nodal in the hybrid model) and differenced centrally; in RZ the m=0
    // cylindrical divergence is used, with the axis limit (1/r)d(r Jr)/dr -> 2 dJr/dr.
    // Higher azimuthal-mode components are left at their base values.
    using namespace amrex::literals;
    using warpx::fields::FieldType;
    using ablastr::fields::Direction;
    using namespace ablastr::coarsen::sample;

    const int lev = 0;
    const ablastr::fields::VectorField J = m_WarpX->m_fields.get_alldirs(FieldType::current_fp, lev);
    amrex::MultiFab* rho = m_WarpX->m_fields.get(FieldType::rho_fp, lev);

    // Start from the base rho (all components)
    amrex::MultiFab::Copy(*rho, *m_rho_base, 0, 0, rho->nComp(), rho->nGrowVect());

    const amrex::Geometry& geom = m_WarpX->Geom(lev);
    const auto dxi = geom.InvCellSizeArray();
    [[maybe_unused]] const amrex::Real rmin = geom.ProbLo(0);
    [[maybe_unused]] const amrex::Real dr = geom.CellSize(0);

    const amrex::GpuArray<int, 3> Jx_stag = m_hybrid_pic_model->Jx_IndexType;
    const amrex::GpuArray<int, 3> Jy_stag = m_hybrid_pic_model->Jy_IndexType;
    const amrex::GpuArray<int, 3> Jz_stag = m_hybrid_pic_model->Jz_IndexType;
    const amrex::GpuArray<int, 3> nodal   = {1, 1, 1};
    const amrex::GpuArray<int, 3> coarsen = {1, 1, 1};
    amrex::ignore_unused(Jy_stag);

    const amrex::Real half_dt = 0.5_rt * m_dt;

    // The response is physical only where plasma exists; in near-floor (vacuum) cells
    // the update would be noise that the 1/n factors of Ohm's law amplify. Restrict the
    // update to cells safely above the density floor and clamp the result at the floor.
    const amrex::Real rho_floor = m_hybrid_pic_model->m_n_floor * PhysConst::q_e;

    // Nodal domain box; the central div stencil (with nodal interpolation of J) reads
    // one cell beyond each node, which exceeds the J guard cells at domain-edge nodes.
    // Interior nodes use central differences; the RZ axis uses a one-sided radial
    // difference; remaining edge nodes keep the base rho (zero response there).
    const amrex::Box dom_nodal = amrex::convert(geom.Domain(), amrex::IntVect::TheNodeVector());
    const amrex::Dim3 dlo = amrex::lbound(dom_nodal);
    const amrex::Dim3 dhi = amrex::ubound(dom_nodal);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*rho, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        amrex::Array4<amrex::Real>       const& rho_arr = rho->array(mfi);
        amrex::Array4<amrex::Real const> const& Jx  = J[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jz  = J[2]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jx0 = m_J_base[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jz0 = m_J_base[2]->const_array(mfi);
#if defined(WARPX_DIM_3D)
        amrex::Array4<amrex::Real const> const& Jy  = J[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jy0 = m_J_base[1]->const_array(mfi);
#endif

        const amrex::Box tb = mfi.tilebox(amrex::IntVect::TheNodeVector());

        amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k) {

            // nodal current response at neighboring nodes
            auto dJx_n = [&] (int ii, int jj, int kk) {
                return Interp(Jx, Jx_stag, nodal, coarsen, ii, jj, kk, 0)
                     - Interp(Jx0, Jx_stag, nodal, coarsen, ii, jj, kk, 0);
            };
            auto dJz_n = [&] (int ii, int jj, int kk) {
                return Interp(Jz, Jz_stag, nodal, coarsen, ii, jj, kk, 0)
                     - Interp(Jz0, Jz_stag, nodal, coarsen, ii, jj, kk, 0);
            };

            amrex::Real div = 0._rt;
#if defined(WARPX_DIM_RZ)
            if (j <= dlo.y || j >= dhi.y || i >= dhi.x) { return; }
            if (i > dlo.x) {
                const amrex::Real r = rmin + i*dr;
                div += (dJx_n(i+1,j,k) - dJx_n(i-1,j,k)) * 0.5_rt * dxi[0]
                     + dJx_n(i,j,k)/r;
            } else {
                // axis: (1/r) d(r Jr)/dr -> 2 dJr/dr, one-sided
                div += 2._rt * (dJx_n(i+1,j,k) - dJx_n(i,j,k)) * dxi[0];
            }
            div += (dJz_n(i,j+1,k) - dJz_n(i,j-1,k)) * 0.5_rt * dxi[1];
#elif defined(WARPX_DIM_XZ)
            if (i <= dlo.x || i >= dhi.x || j <= dlo.y || j >= dhi.y) { return; }
            div += (dJx_n(i+1,j,k) - dJx_n(i-1,j,k)) * 0.5_rt * dxi[0];
            div += (dJz_n(i,j+1,k) - dJz_n(i,j-1,k)) * 0.5_rt * dxi[1];
#elif defined(WARPX_DIM_1D_Z)
            if (i <= dlo.x || i >= dhi.x) { return; }
            div += (dJz_n(i+1,j,k) - dJz_n(i-1,j,k)) * 0.5_rt * dxi[0];
#elif defined(WARPX_DIM_3D)
            auto dJy_n = [&] (int ii, int jj, int kk) {
                return Interp(Jy, Jy_stag, nodal, coarsen, ii, jj, kk, 0)
                     - Interp(Jy0, Jy_stag, nodal, coarsen, ii, jj, kk, 0);
            };
            if (i <= dlo.x || i >= dhi.x || j <= dlo.y || j >= dhi.y ||
                k <= dlo.z || k >= dhi.z) { return; }
            div += (dJx_n(i+1,j,k) - dJx_n(i-1,j,k)) * 0.5_rt * dxi[0];
            div += (dJy_n(i,j+1,k) - dJy_n(i,j-1,k)) * 0.5_rt * dxi[1];
            div += (dJz_n(i,j,k+1) - dJz_n(i,j,k-1)) * 0.5_rt * dxi[2];
#else
            amrex::ignore_unused(i, j, k, dJx_n, dJz_n, dxi, half_dt, dlo, dhi);
#endif
            const amrex::Real rho0v = rho_arr(i,j,k,0);
            if (rho0v > 10._rt * rho_floor) {
                rho_arr(i,j,k,0) = amrex::max(rho0v - half_dt * div, rho_floor);
            }
        });
    }

    // refresh rho guards for downstream interpolations
    rho->FillBoundary(geom.periodicity());
}

void ThetaImplicitHybrid::SubtractDissipativeEFromPushField ()
{
    // D = E_Ohm(with dissipation) - E_Ohm(without) evaluated from the same
    // (B^{n+theta}, rho, pe) state: the ideal, Hall and grad-pe parts cancel
    // exactly, leaving eta*J_p - eta_h*nabla^2(J_p) with the identical stencils,
    // interpolations, floors and axis handling as the residual's Ohm solve --
    // by construction, for any resistivity model. The FD-solver-level entry is
    // used so no boundary condition is applied to Efield_fp as a side effect.
    using namespace amrex::literals;
    using warpx::fields::FieldType;

    if (m_hybrid_pic_model->m_eta_expression == "0.0" &&
        !m_hybrid_pic_model->m_include_hyper_resistivity_term) { return; }

    const int lev = 0;
    const ablastr::fields::VectorField E =
        m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, lev);
    const ablastr::fields::VectorField Ji =
        m_WarpX->m_fields.get_alldirs(FieldType::current_fp, lev);
    const amrex::MultiFab* rho = m_WarpX->m_fields.get(FieldType::rho_fp, lev);
    const amrex::MultiFab* pe =
        m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    const ablastr::fields::VectorField B =
        m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, lev);

    for (int n = 0; n < 3; ++n) {
        if (!m_D[n]) {
            m_D[n] = std::make_unique<amrex::MultiFab>(
                E[n]->boxArray(), E[n]->DistributionMap(), E[n]->nComp(), E[n]->nGrowVect());
            m_E_work[n] = std::make_unique<amrex::MultiFab>(
                E[n]->boxArray(), E[n]->DistributionMap(), E[n]->nComp(), E[n]->nGrowVect());
            // The masked Ohm solves below skip EB-covered (flag = 0) locations,
            // which therefore retain their allocation-time content forever:
            // without this initialization, D at covered nodes is uninitialized
            // arena memory that pollutes E* = a_E - D and the Newton residual.
            m_D[n]->setVal(0.0_rt);
            m_E_work[n]->setVal(0.0_rt);
        }
    }
    const ablastr::fields::VectorField D    = {m_D[0].get(), m_D[1].get(), m_D[2].get()};
    const ablastr::fields::VectorField Ework = {m_E_work[0].get(), m_E_work[1].get(), m_E_work[2].get()};

    ablastr::fields::VectorField Jp =
        m_WarpX->m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    auto& eb_update_E = m_WarpX->GetEBUpdateEFlag()[lev];
    auto* fdtd = m_WarpX->get_pointer_fdtd_solver_fp(lev);
    // with dissipation (solve_for_Faraday = true), into D
    fdtd->HybridPICSolveE(D, Jp, Ji, B, *rho, *pe, eb_update_E, lev,
                          m_hybrid_pic_model, true, true);
    // without dissipation, into Ework
    fdtd->HybridPICSolveE(Ework, Jp, Ji, B, *rho, *pe, eb_update_E, lev,
                          m_hybrid_pic_model, false, true);

    for (int n = 0; n < 3; ++n) {
        amrex::MultiFab::Subtract(*m_D[n], *m_E_work[n], 0, 0, m_D[n]->nComp(), 0);
        m_D[n]->FillBoundary(m_WarpX->Geom(lev).periodicity());
        // E* = a_E - D (all components; the push and the pairing read this)
        amrex::MultiFab::Subtract(*E[n], *m_D[n], 0, 0, E[n]->nComp(), 0);
    }
}

void ThetaImplicitHybrid::AdvanceElectronPressure ( const bool a_from_jacobian )
{
    // pe^{n+theta} = pe^n - theta*dt * [ div(ue pe^n) + (gamma-1) pe^n div(ue) ],
    // with ue = (J_i - J_net)/(e n) evaluated at nodes from the freshly deposited ion
    // current and the plasma (net) current. Forward evaluation in pe is O(dt^2) for
    // the half step and keeps the update an explicit pure function of the iterate.
    using namespace amrex::literals;
    using warpx::fields::FieldType;
    using ablastr::fields::Direction;
    using namespace ablastr::coarsen::sample;

    const int lev = 0;
    amrex::MultiFab* pe = m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    const amrex::MultiFab* rho = m_WarpX->m_fields.get(FieldType::rho_fp, lev);
    const ablastr::fields::VectorField J =
        m_WarpX->m_fields.get_alldirs(FieldType::current_fp, lev);
    const ablastr::fields::VectorField Jp =
        m_WarpX->m_fields.get_alldirs(FieldType::hybrid_current_fp_plasma, lev);
    // Efield_fp holds the push field E* = a_E - D (external-field contributions
    // have already been subtracted again): the exact field the ions were pushed
    // with. theta-Faraday consumes a_E = E* + D, whose dissipative part D enters
    // the pairing as the -D.J_p heating term below.
    const ablastr::fields::VectorField Efld =
        m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, lev);
    for (int n = 0; n < 3; ++n) {
        if (!m_D[n]) {  // no dissipation configured: pair against D = 0
            m_D[n] = std::make_unique<amrex::MultiFab>(
                Efld[n]->boxArray(), Efld[n]->DistributionMap(),
                Efld[n]->nComp(), Efld[n]->nGrowVect());
            m_D[n]->setVal(0.0_rt);
        }
    }
    // Honor include_joule_heating (default off), matching the explicit QDSMC
    // path: with it off, the dissipative work D.J_p (Ohmic + hyper-resistive)
    // is NOT deposited into pe -- it leaves the system as the vacuum/hyper
    // dissipation it is (Faraday still drains it from W_B, so the B-field
    // damping physics is unchanged). Without this, eta_H's grid-scale
    // dissipation at the sharp FRC edge current sheets heats the few
    // electrons there to keV within ~100 steps. The reversible E*.J_e
    // transport/compression pairing is unaffected. Pair against D = 0 by
    // pointing the kernel at zeroed fields.
    const bool jheat = m_hybrid_pic_model->m_include_joule_heating;
    if (!jheat && !m_D_zero[0]) {
        for (int n = 0; n < 3; ++n) {
            m_D_zero[n] = std::make_unique<amrex::MultiFab>(
                Efld[n]->boxArray(), Efld[n]->DistributionMap(),
                Efld[n]->nComp(), Efld[n]->nGrowVect());
            m_D_zero[n]->setVal(0.0_rt);
        }
    }

    const amrex::Geometry& geom = m_WarpX->Geom(lev);

    // The nodal update below interpolates the cell-centered J and J_plasma through
    // guard cells (a node on a box face needs the J value owned by the neighbor box),
    // so their ghosts must be current before the kernel runs. Without this, the two
    // boxes sharing a nodal point compute different pe there and the pressure field
    // becomes multivalued at box seams (breaking the discrete energy pairing).
    for (int d = 0; d < 3; ++d) {
        J[d]->FillBoundary(geom.periodicity());
        Jp[d]->FillBoundary(geom.periodicity());
    }

    if (!m_pe_old) {
#if defined(WARPX_DIM_RZ)
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::n_rz_azimuthal_modes == 1,
            "implicit hybrid in-loop pe advance: the RZ work pairing is implemented for m = 0 only");
#endif
        // First use: initialize pe^n from the algebraic closure on the current density
        m_hybrid_pic_model->CalculateElectronPressure();
        m_pe_old = std::make_unique<amrex::MultiFab>(
            pe->boxArray(), pe->DistributionMap(), pe->nComp(), pe->nGrowVect());
        amrex::MultiFab::Copy(*m_pe_old, *pe, 0, 0, pe->nComp(), pe->nGrowVect());
        m_pe_old->FillBoundary(geom.periodicity());
    }
    if (!m_pe_theta) {
        m_pe_theta = std::make_unique<amrex::MultiFab>(
            pe->boxArray(), pe->DistributionMap(), pe->nComp(), pe->nGrowVect());
        m_pe_scratch = std::make_unique<amrex::MultiFab>(
            pe->boxArray(), pe->DistributionMap(), pe->nComp(), pe->nGrowVect());
    }

    const auto dxi = geom.InvCellSizeArray();
    [[maybe_unused]] const amrex::Real rmin = geom.ProbLo(0);
    [[maybe_unused]] const amrex::Real dr = geom.CellSize(0);
    const amrex::Real theta_dt = m_theta * m_dt;
    const amrex::Real gamma = m_hybrid_pic_model->m_gamma;
    const amrex::Real rho_floor = m_hybrid_pic_model->m_n_floor * PhysConst::q_e;
    const amrex::Real q_e = PhysConst::q_e;
    // Marker-CFL cap on u_e in the pe advance: the theta-centered fixed
    // point contracts only where |u_e| k_max theta dt < 1. The floored-edge
    // u_e = J/(e n_floor) (~1e7 m/s -- fictitious, there is no electron
    // fluid there) violates this, and the local divergence grows a hot-cell
    // tail (~x2 per 150 steps on the 3D FRC). 0.25 dx_min/(theta dt) keeps
    // the contraction <= 0.25*pi everywhere; the physical interior u_e sits
    // 1-2 orders below the cap.
    amrex::Real dx_min = geom.CellSize(0);
    for (int d = 1; d < AMREX_SPACEDIM; ++d) {
        dx_min = std::min(dx_min, geom.CellSize(d));
    }
    const amrex::Real ue_cap = 0.25_rt * dx_min / theta_dt;

    const amrex::GpuArray<int, 3> Jx_stag = m_hybrid_pic_model->Jx_IndexType;
    const amrex::GpuArray<int, 3> Jy_stag = m_hybrid_pic_model->Jy_IndexType;
    const amrex::GpuArray<int, 3> Jz_stag = m_hybrid_pic_model->Jz_IndexType;
    const amrex::GpuArray<int, 3> nodal   = {1, 1, 1};
    const amrex::GpuArray<int, 3> coarsen = {1, 1, 1};
    amrex::ignore_unused(Jy_stag);

    const amrex::Box dom_nodal = amrex::convert(geom.Domain(), amrex::IntVect::TheNodeVector());
    const amrex::Dim3 dlo = amrex::lbound(dom_nodal);
    const amrex::Dim3 dhi = amrex::ubound(dom_nodal);
    amrex::GpuArray<bool, 3> is_per = {true, true, true};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { is_per[d] = geom.isPeriodic(d); }

    // Collocated grid: J, E and rho are all nodal and HybridPICSolveE builds the
    // pressure field with CartesianNodalAlgorithm (centered difference at the node,
    // nodal rho) instead of the Yee edge construction. The work pairing below must
    // use the identical stencil or the electron side subtracts a different discrete
    // work than the ions receive (J-correlated leak).
    const bool J_nodal = J[2]->ixType().nodeCentered();

    // The update must be theta-centered in pe as well: an explicit (forward) pe in the
    // RHS integrates the pressure side of the ion-acoustic oscillation with forward
    // Euler and is numerically unstable (growth ~ exp(omega^2 dt t / 2)). The update is
    // linear in pe, so a short fixed-point iteration (contraction ~ theta*omega*dt)
    // converges the theta-centered value: pe_rhs = (1-theta)*pe^n + theta*pe_iter.
    // Contraction per cycle is ~theta*k_max*Cs*dt; 4 cycles verified sufficient
    // (8 cycles bit-reproduces the energy history on the cold-beam FGI test).
    const int n_pe_iters = 4;
    amrex::MultiFab::Copy(*m_pe_scratch, *m_pe_old, 0, 0, pe->nComp(), pe->nGrowVect());
    for (int pe_it = 0; pe_it < n_pe_iters; ++pe_it) {

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*pe, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        amrex::Array4<amrex::Real>       const& pe_arr  = pe->array(mfi);
        amrex::Array4<amrex::Real const> const& pe_it_arr = m_pe_scratch->const_array(mfi);
        amrex::Array4<amrex::Real const> const& pe0     = m_pe_old->const_array(mfi);
        amrex::Array4<amrex::Real const> const& rho_arr = rho->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jx  = J[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jz  = J[2]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jpx = Jp[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jpz = Jp[2]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jy  = J[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jpy = Jp[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Ex_arr = Efld[0]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Ey_arr = Efld[1]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Ez_arr = Efld[2]->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Dx =
            (jheat ? m_D[0] : m_D_zero[0])->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Dy =
            (jheat ? m_D[1] : m_D_zero[1])->const_array(mfi);
        amrex::Array4<amrex::Real const> const& Dz =
            (jheat ? m_D[2] : m_D_zero[2])->const_array(mfi);

        const amrex::Box tb = mfi.tilebox(amrex::IntVect::TheNodeVector());

        amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k) {

            // Marker-CFL clamp on u_e (see ue_cap above): keeps the
            // theta-centered fixed point contractive in the floored-edge
            // cells, whose fictitious u_e = J/(e n_floor) otherwise makes
            // it locally divergent (hot-cell tail doubling each ~150 steps).
            auto uclamp = [&] (amrex::Real u) {
                return amrex::min(amrex::max(u, -ue_cap), ue_cap);
            };
            // electron velocity at a node: ue = (J_ion - J_net)/(e n)
            auto ue_x = [&] (int ii, int jj, int kk) {
                const amrex::Real n_ = amrex::max(rho_arr(ii,jj,kk,0), rho_floor);
                return uclamp((Interp(Jx, Jx_stag, nodal, coarsen, ii, jj, kk, 0)
                      - Interp(Jpx, Jx_stag, nodal, coarsen, ii, jj, kk, 0)) / n_);
            };
            auto ue_z = [&] (int ii, int jj, int kk) {
                const amrex::Real n_ = amrex::max(rho_arr(ii,jj,kk,0), rho_floor);
                return uclamp((Interp(Jz, Jz_stag, nodal, coarsen, ii, jj, kk, 0)
                      - Interp(Jpz, Jz_stag, nodal, coarsen, ii, jj, kk, 0)) / n_);
            };
            // The fixed point iterates pe^{n+theta} directly:
            //   pe^{n+theta} = pe^n - theta*dt*RHS(pe^{n+theta}),
            // so the RHS reads the current iterate itself.
            auto pec = [&] (int ii, int jj, int kk) { return pe_it_arr(ii,jj,kk); };
            // flux F = ue * pe_centered and velocity divergence, central differences
            auto Fx = [&] (int ii, int jj, int kk) { return ue_x(ii,jj,kk) * pec(ii,jj,kk); };
            auto Fz = [&] (int ii, int jj, int kk) { return ue_z(ii,jj,kk) * pec(ii,jj,kk); };

            // Cartesian: enthalpy-flux transport plus the work pairing term,
            //   d_t pe = -gamma div(ue pe) + (gamma-1) ue.grad(pe) + (gamma-1) Q.
            //
            // Collocated grid (J_nodal): full-Ohm pairing. The electrons absorb the
            // exact complement of the discrete ion work (+sum E.J_i) and the
            // magnetic-energy change (-sum E.J_amp for theta = 1/2 Faraday with
            // mutually adjoint curls), so W carries -E.J_e = E.(J_i - J_amp) with E
            // the solver iterate that pushed the particles and drives Faraday. This
            // is the continuum (gamma-1)[ue.grad(pe) + Q_Joule] with the conservative
            // Joule form Q = eta J_tot.J_e, and it holds for ANY Ohm's-law contents:
            // the Hall term does no work pointwise at a node, and total energy
            // K_i + U_e + W_B is conserved identically, independent of what E is.
            //
            // Yee grid: legacy pressure-channel-only pairing. E_pe,edge =
            // -UpwardD(pe)/max(rho_edge, floor) is the SAME discrete field
            // HybridPICSolveE builds; each edge's work is split half to each
            // adjacent node so the periodic sum cancels the ion pressure-channel
            // work exactly. The remaining J_net.grad(pe)/(en) term uses centered
            // differences (B-channel pairing not exact on the staggered mesh).
            amrex::Real divF = 0._rt, W = 0._rt;
#if defined(WARPX_DIM_1D_Z)
            if ((i <= dlo.x || i >= dhi.x) && !is_per[0]) { return; }
            if (J_nodal) {
                divF += (Fz(i+1,j,k) - Fz(i-1,j,k)) * 0.5_rt * dxi[0];
            } else {
                // Yee: conservative edge flux (matches the RZ construction). The
                // central node form would interpolate J through the SECOND ghost
                // ring at box-face nodes, which the deposited current does not
                // have -- the reads land out of bounds and the seam nodes get
                // garbage flux (observed as a box-seam energy leak).
                auto ue_zedge = [&] (int ie) {
                    const amrex::Real rho_e = amrex::max(
                        0.5_rt*(rho_arr(ie,j,k,0) + rho_arr(ie+1,j,k,0)), rho_floor);
                    return uclamp((Jz(ie,j,k) - Jpz(ie,j,k)) / rho_e);
                };
                divF += (ue_zedge(i)   * 0.5_rt*(pe_it_arr(i,j,k)   + pe_it_arr(i+1,j,k))
                       - ue_zedge(i-1) * 0.5_rt*(pe_it_arr(i-1,j,k) + pe_it_arr(i,j,k)))
                      * dxi[0];
            }
            if (J_nodal) {
                // -E*.J_e over all three components (perp components do work at
                // B != 0), minus the dissipative work D.J_p that Faraday drains
                // from W_B (Joule + hyper-resistive heating -> electrons)
                W += Ex_arr(i,j,k) * (Jx(i,j,k) - Jpx(i,j,k))
                   + Ey_arr(i,j,k) * (Jy(i,j,k) - Jpy(i,j,k))
                   + Ez_arr(i,j,k) * (Jz(i,j,k) - Jpz(i,j,k))
                   - (Dx(i,j,k) * Jpx(i,j,k)
                    + Dy(i,j,k) * Jpy(i,j,k)
                    + Dz(i,j,k) * Jpz(i,j,k));
            } else {
                // Yee: per-component full-Ohm pairing q_d = E*_d (J_d - Jp_d) - D_d Jp_d
                // at each component's own location; the z-staggered component is
                // half-split to the node (Ex/Ey are nodal in 1D)
                auto qz = [&] (int ie) {
                    return Ez_arr(ie,j,k) * (Jz(ie,j,k) - Jpz(ie,j,k))
                         - Dz(ie,j,k) * Jpz(ie,j,k);
                };
                W += Ex_arr(i,j,k) * (Jx(i,j,k) - Jpx(i,j,k)) - Dx(i,j,k) * Jpx(i,j,k)
                   + Ey_arr(i,j,k) * (Jy(i,j,k) - Jpy(i,j,k)) - Dy(i,j,k) * Jpy(i,j,k)
                   + 0.5_rt * (qz(i-1) + qz(i));
            }
#elif defined(WARPX_DIM_XZ)
            if (((i <= dlo.x || i >= dhi.x) && !is_per[0]) ||
                ((j <= dlo.y || j >= dhi.y) && !is_per[1])) { return; }
            if (J_nodal) {
                divF += (Fx(i+1,j,k) - Fx(i-1,j,k)) * 0.5_rt * dxi[0]
                      + (Fz(i,j+1,k) - Fz(i,j-1,k)) * 0.5_rt * dxi[1];
            } else {
                // Yee: conservative edge flux; see the 1D branch for why the
                // central node form cannot be used with the deposited current.
                auto ue_xedge = [&] (int ie) {
                    const amrex::Real rho_e = amrex::max(
                        0.5_rt*(rho_arr(ie,j,k,0) + rho_arr(ie+1,j,k,0)), rho_floor);
                    return uclamp((Jx(ie,j,k) - Jpx(ie,j,k)) / rho_e);
                };
                auto ue_zedge = [&] (int je) {
                    const amrex::Real rho_e = amrex::max(
                        0.5_rt*(rho_arr(i,je,k,0) + rho_arr(i,je+1,k,0)), rho_floor);
                    return uclamp((Jz(i,je,k) - Jpz(i,je,k)) / rho_e);
                };
                divF += (ue_xedge(i)   * 0.5_rt*(pe_it_arr(i,j,k)   + pe_it_arr(i+1,j,k))
                       - ue_xedge(i-1) * 0.5_rt*(pe_it_arr(i-1,j,k) + pe_it_arr(i,j,k)))
                      * dxi[0]
                      + (ue_zedge(j)   * 0.5_rt*(pe_it_arr(i,j,k)   + pe_it_arr(i,j+1,k))
                       - ue_zedge(j-1) * 0.5_rt*(pe_it_arr(i,j-1,k) + pe_it_arr(i,j,k)))
                      * dxi[1];
            }
            if (J_nodal) {
                // -E*.J_e over all three components (perp components do work at
                // B != 0), minus the dissipative work D.J_p that Faraday drains
                // from W_B (Joule + hyper-resistive heating -> electrons)
                W += Ex_arr(i,j,k) * (Jx(i,j,k) - Jpx(i,j,k))
                   + Ey_arr(i,j,k) * (Jy(i,j,k) - Jpy(i,j,k))
                   + Ez_arr(i,j,k) * (Jz(i,j,k) - Jpz(i,j,k))
                   - (Dx(i,j,k) * Jpx(i,j,k)
                    + Dy(i,j,k) * Jpy(i,j,k)
                    + Dz(i,j,k) * Jpz(i,j,k));
            } else {
                // Yee: per-component full-Ohm pairing, staggered components
                // half-split to the node (Ey is nodal in XZ)
                auto qx = [&] (int ie) {
                    return Ex_arr(ie,j,k) * (Jx(ie,j,k) - Jpx(ie,j,k))
                         - Dx(ie,j,k) * Jpx(ie,j,k);
                };
                auto qz = [&] (int je) {
                    return Ez_arr(i,je,k) * (Jz(i,je,k) - Jpz(i,je,k))
                         - Dz(i,je,k) * Jpz(i,je,k);
                };
                W += 0.5_rt * (qx(i-1) + qx(i))
                   + Ey_arr(i,j,k) * (Jy(i,j,k) - Jpy(i,j,k)) - Dy(i,j,k) * Jpy(i,j,k)
                   + 0.5_rt * (qz(j-1) + qz(j));
            }
#elif defined(WARPX_DIM_RZ)
            // m = 0, Yee-staggered RZ (asserted at first use). The enthalpy flux
            // uses the conservative flux form, so the volume-weighted grid sum
            // telescopes exactly (including through the axis, where [r F] -> 0).
            // W carries the full-Ohm pairing -E*.J_e + eta*Jp^2 evaluated per
            // staggered component at its own location (matching the push-field
            // subtraction), volume-split from the edges to the nodes so that
            // sum_nodes(A W) = sum_edges(A q). The Hall channel and the RZ
            // Faraday ledger pair at interpolation order (not machine-exact,
            // unlike the collocated Cartesian grid).
            amrex::ignore_unused(Fx, Fz, ue_x, ue_z);
            if ((j <= dlo.y || j >= dhi.y) && !is_per[1]) { return; }
            if (i >= dhi.x) { return; }                        // outer r boundary
            const amrex::Real r = rmin + i*dr;
            const bool on_axis = (r < 0.5_rt*dr);
            if (i <= dlo.x && !on_axis) { return; }            // annular inner boundary
            const amrex::Real A_i  = on_axis ? 0.125_rt*dr*dr : r*dr;
            const amrex::Real A_ip = (r + 0.5_rt*dr)*dr;       // r-edge at i+1/2
            const amrex::Real A_im = (r - 0.5_rt*dr)*dr;       // r-edge at i-1/2

            // enthalpy flux, conservative form
            auto ue_redge = [&] (int ie) {
                const amrex::Real rho_e = amrex::max(
                    0.5_rt*(rho_arr(ie,j,k,0) + rho_arr(ie+1,j,k,0)), rho_floor);
                return uclamp((Jx(ie,j,k) - Jpx(ie,j,k)) / rho_e);
            };
            auto ue_zedge = [&] (int je) {
                const amrex::Real rho_e = amrex::max(
                    0.5_rt*(rho_arr(i,je,k,0) + rho_arr(i,je+1,k,0)), rho_floor);
                return uclamp((Jz(i,je,k) - Jpz(i,je,k)) / rho_e);
            };
            {
                const amrex::Real Frp =
                    ue_redge(i) * 0.5_rt*(pe_it_arr(i,j,k) + pe_it_arr(i+1,j,k));
                const amrex::Real rFrm = on_axis ? 0.0_rt :
                    (r - 0.5_rt*dr) *
                    ue_redge(i-1) * 0.5_rt*(pe_it_arr(i-1,j,k) + pe_it_arr(i,j,k));
                divF += ((r + 0.5_rt*dr)*Frp - rFrm) / A_i;
                divF += (ue_zedge(j) * 0.5_rt*(pe_it_arr(i,j,k) + pe_it_arr(i,j+1,k))
                       - ue_zedge(j-1) * 0.5_rt*(pe_it_arr(i,j-1,k) + pe_it_arr(i,j,k)))
                      * dxi[1];
            }

            // full-Ohm work pairing per staggered component
            auto q_redge = [&] (int ie) {
                return Ex_arr(ie,j,k) * (Jx(ie,j,k) - Jpx(ie,j,k))
                     - Dx(ie,j,k) * Jpx(ie,j,k);
            };
            auto q_zedge = [&] (int je) {
                return Ez_arr(i,je,k) * (Jz(i,je,k) - Jpz(i,je,k))
                     - Dz(i,je,k) * Jpz(i,je,k);
            };
            {
                const amrex::Real q_theta =
                    Ey_arr(i,j,k) * (Jy(i,j,k) - Jpy(i,j,k)) - Dy(i,j,k) * Jpy(i,j,k);
                const amrex::Real w_r =
                    (0.5_rt * q_redge(i) * A_ip
                     + (on_axis ? 0.0_rt : 0.5_rt * q_redge(i-1) * A_im)) / A_i;
                W += w_r
                   + 0.5_rt * (q_zedge(j-1) + q_zedge(j))
                   + q_theta;
            }
#elif defined(WARPX_DIM_3D)
            auto ue_y = [&] (int ii, int jj, int kk) {
                const amrex::Real n_ = amrex::max(rho_arr(ii,jj,kk,0), rho_floor);
                return uclamp((Interp(Jy, Jy_stag, nodal, coarsen, ii, jj, kk, 0)
                      - Interp(Jpy, Jy_stag, nodal, coarsen, ii, jj, kk, 0)) / n_);
            };
            auto Fy = [&] (int ii, int jj, int kk) { return ue_y(ii,jj,kk) * pec(ii,jj,kk); };
            if (((i <= dlo.x || i >= dhi.x) && !is_per[0]) ||
                ((j <= dlo.y || j >= dhi.y) && !is_per[1]) ||
                ((k <= dlo.z || k >= dhi.z) && !is_per[2])) { return; }
            if (J_nodal) {
                divF += (Fx(i+1,j,k) - Fx(i-1,j,k)) * 0.5_rt * dxi[0]
                      + (Fy(i,j+1,k) - Fy(i,j-1,k)) * 0.5_rt * dxi[1]
                      + (Fz(i,j,k+1) - Fz(i,j,k-1)) * 0.5_rt * dxi[2];
            } else {
                // Yee: conservative edge flux; see the 1D branch for why the
                // central node form cannot be used with the deposited current.
                auto ue_xedge = [&] (int ie) {
                    const amrex::Real rho_e = amrex::max(
                        0.5_rt*(rho_arr(ie,j,k,0) + rho_arr(ie+1,j,k,0)), rho_floor);
                    return uclamp((Jx(ie,j,k) - Jpx(ie,j,k)) / rho_e);
                };
                auto ue_yedge = [&] (int je) {
                    const amrex::Real rho_e = amrex::max(
                        0.5_rt*(rho_arr(i,je,k,0) + rho_arr(i,je+1,k,0)), rho_floor);
                    return uclamp((Jy(i,je,k) - Jpy(i,je,k)) / rho_e);
                };
                auto ue_zedge = [&] (int ke) {
                    const amrex::Real rho_e = amrex::max(
                        0.5_rt*(rho_arr(i,j,ke,0) + rho_arr(i,j,ke+1,0)), rho_floor);
                    return uclamp((Jz(i,j,ke) - Jpz(i,j,ke)) / rho_e);
                };
                divF += (ue_xedge(i)   * 0.5_rt*(pe_it_arr(i,j,k)   + pe_it_arr(i+1,j,k))
                       - ue_xedge(i-1) * 0.5_rt*(pe_it_arr(i-1,j,k) + pe_it_arr(i,j,k)))
                      * dxi[0]
                      + (ue_yedge(j)   * 0.5_rt*(pe_it_arr(i,j,k)   + pe_it_arr(i,j+1,k))
                       - ue_yedge(j-1) * 0.5_rt*(pe_it_arr(i,j-1,k) + pe_it_arr(i,j,k)))
                      * dxi[1]
                      + (ue_zedge(k)   * 0.5_rt*(pe_it_arr(i,j,k)   + pe_it_arr(i,j,k+1))
                       - ue_zedge(k-1) * 0.5_rt*(pe_it_arr(i,j,k-1) + pe_it_arr(i,j,k)))
                      * dxi[2];
            }
            if (J_nodal) {
                // -E*.J_e over all three components (perp components do work at
                // B != 0), minus the dissipative work D.J_p that Faraday drains
                // from W_B (Joule + hyper-resistive heating -> electrons)
                W += Ex_arr(i,j,k) * (Jx(i,j,k) - Jpx(i,j,k))
                   + Ey_arr(i,j,k) * (Jy(i,j,k) - Jpy(i,j,k))
                   + Ez_arr(i,j,k) * (Jz(i,j,k) - Jpz(i,j,k))
                   - (Dx(i,j,k) * Jpx(i,j,k)
                    + Dy(i,j,k) * Jpy(i,j,k)
                    + Dz(i,j,k) * Jpz(i,j,k));
            } else {
                // Yee: per-component full-Ohm pairing, each component half-split
                // to the node along its own staggered direction
                auto qx = [&] (int ie) {
                    return Ex_arr(ie,j,k) * (Jx(ie,j,k) - Jpx(ie,j,k))
                         - Dx(ie,j,k) * Jpx(ie,j,k);
                };
                auto qy = [&] (int je) {
                    return Ey_arr(i,je,k) * (Jy(i,je,k) - Jpy(i,je,k))
                         - Dy(i,je,k) * Jpy(i,je,k);
                };
                auto qz = [&] (int ke) {
                    return Ez_arr(i,j,ke) * (Jz(i,j,ke) - Jpz(i,j,ke))
                         - Dz(i,j,ke) * Jpz(i,j,ke);
                };
                W += 0.5_rt * (qx(i-1) + qx(i))
                   + 0.5_rt * (qy(j-1) + qy(j))
                   + 0.5_rt * (qz(k-1) + qz(k));
            }
#else
            amrex::ignore_unused(Fx, Fz, divF, dlo, dhi, is_per, dxi);
#endif
            const amrex::Real pe_new = pe0(i,j,k)
                - theta_dt * (gamma * divF + (gamma - 1._rt) * W);
            pe_arr(i,j,k) = amrex::max(pe_new, 0._rt);
        });
    }

    // Duplicated nodal points on box faces are written redundantly by each box;
    // force them single-valued before they are consumed (Ohm's law, next iterate).
    pe->OverrideSync(geom.periodicity());
    pe->FillBoundary(geom.periodicity());
    if (pe_it < n_pe_iters - 1) {
        amrex::MultiFab::Copy(*m_pe_scratch, *pe, 0, 0, pe->nComp(), pe->nGrowVect());
    }
    } // pe fixed-point iterations

#if defined(WARPX_DIM_RZ)
    {
        // Re-phase the axis-row pressure to the local density through a
        // radially averaged entropy s = pe/rho^gamma (rings 0..2): the axis
        // row's J_e is deposition-noise dominated and the advected pe there
        // dephases from n, closing an anti-restoring feedback loop. Slaving
        // the axis-row pe to n (QDSMC-style rebuild) preserves the entropy
        // evolution while restoring the pressure-density phase lock.
        const amrex::Real gam = m_hybrid_pic_model->m_gamma;
        const amrex::Real rfloor = m_hybrid_pic_model->m_n_floor * PhysConst::q_e;
        const amrex::MultiFab* rho_mf = m_WarpX->m_fields.get(FieldType::rho_fp, lev);
        const amrex::Box dom_n =
            amrex::convert(geom.Domain(), amrex::IntVect::TheNodeVector());
        const int iax = dom_n.smallEnd(0);
        const bool has_axis = (std::abs(geom.ProbLo(0)) < 0.5_rt*geom.CellSize(0));
        if (has_axis) {
            for (amrex::MFIter mfi(*pe, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box tb = mfi.tilebox(amrex::IntVect::TheNodeVector());
                if (tb.smallEnd(0) > iax) { continue; }
                auto const& pe_a = pe->array(mfi);
                auto const& rh   = rho_mf->const_array(mfi);
                const amrex::Box tb0(amrex::IntVect(iax, tb.smallEnd(1)),
                                     amrex::IntVect(iax, tb.bigEnd(1)),
                                     tb.ixType());
                amrex::ParallelFor(tb0, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    amrex::Real sbar = 0._rt;
                    for (int ii = 0; ii <= 2; ++ii) {
                        const amrex::Real nr =
                            amrex::max(rh(i+ii,j,k,0), rfloor);
                        sbar += pe_a(i+ii,j,k) / std::pow(nr, gam);
                    }
                    sbar *= (1._rt/3._rt);
                    const amrex::Real n0 = amrex::max(rh(i,j,k,0), rfloor);
                    pe_a(i,j,k) = sbar * std::pow(n0, gam);
                });
            }
            pe->OverrideSync(geom.periodicity());
            pe->FillBoundary(geom.periodicity());
        }
    }
#endif

    if (!a_from_jacobian) {
        amrex::MultiFab::Copy(*m_pe_theta, *pe, 0, 0, pe->nComp(), pe->nGrowVect());
    }
}

void ThetaImplicitHybrid::SetMassMatricesForPC ( const amrex::Real /*a_theta_dt*/ )
{
    using ablastr::fields::Direction;

    // Skip c²μ₀θΔt EM scaling from the base class: it is derived from
    // Ampere's law and is physically incorrect for the hybrid solver, where
    // E is determined by Ohm's law (not by the displacement current).
    //
    // For curl-curl MLMG: still add +1 to the diagonal so that
    // beta = 1 + (particle response) >= 1 everywhere, including in vacuum
    // cells where the particle response is zero.  Without this, the curl-curl
    // operator is singular in vacuum and GMRES stagnates.
    const PreconditionerType pc_type = m_nlsolver->GetPreconditionerType();
    if (pc_type == PreconditionerType::pc_curl_curl_mlmg) {
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            amrex::MultiFab* MMxx_PC = m_WarpX->m_fields.get(FieldType::MassMatrices_PC, Direction{0}, lev);
            amrex::MultiFab* MMyy_PC = m_WarpX->m_fields.get(FieldType::MassMatrices_PC, Direction{1}, lev);
            amrex::MultiFab* MMzz_PC = m_WarpX->m_fields.get(FieldType::MassMatrices_PC, Direction{2}, lev);
            const int diag_Mxx = (MMxx_PC->nComp()-1)/2;
            const int diag_Myy = (MMyy_PC->nComp()-1)/2;
            const int diag_Mzz = (MMzz_PC->nComp()-1)/2;
            MMxx_PC->plus(1.0_rt, diag_Mxx, 1, 0);
            MMyy_PC->plus(1.0_rt, diag_Myy, 1, 0);
            MMzz_PC->plus(1.0_rt, diag_Mzz, 1, 0);
        }
    }
}

void ThetaImplicitHybrid::UpdateWarpXFields ( const WarpXSolverVec&  a_E,
                                                amrex::Real start_time )
{
    BL_PROFILE("ThetaImplicitHybrid::UpdateWarpXFields()");

    const amrex::Real theta_time = start_time + m_theta * m_dt;

    // Set E^{n+θ} in WarpX
    m_WarpX->SetElectricFieldAndApplyBCs( a_E, theta_time );

    // Compute B^{n+θ} = B^n - θ·dt·curl(E^{n+θ}) via Faraday's law
    ablastr::fields::MultiLevelVectorField const& B_old = 
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, m_num_amr_levels - 1);
    m_WarpX->UpdateMagneticFieldAndApplyBCs( B_old, m_theta * m_dt, start_time );
}

void ThetaImplicitHybrid::FinishFieldUpdate( amrex::Real end_time )
{
    BL_PROFILE("ThetaImplicitHybrid::FinishFieldUpdate()");

    // Extrapolate from t^{n+θ} to t^{n+1}:
    // F^{n+1} = (1/θ)·F^{n+θ} + (1 - 1/θ)·F^n
    const amrex::Real c0 = 1.0_rt / m_theta;
    const amrex::Real c1 = 1.0_rt - c0;

    // E^{n+1}  (or E_T^{n+1} if Darwin split)
    m_E.linComb( c0, m_E, c1, m_Eold );
    m_WarpX->SetElectricFieldAndApplyBCs( m_E, end_time );
    // Restore full E^{n+1} = E_T^{n+1} + E_L^{n+θ}  (E_L from final Newton iter)
    if (m_use_darwin_split) { AddLongitudinalE(); }

    // B^{n+1}
    ablastr::fields::MultiLevelVectorField const& B_old = 
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, 0);
    m_WarpX->FinishMagneticFieldAndApplyBCs( B_old, m_theta, end_time );

    // Add external fields to get total fields at t^{n+1}
    if (m_add_external_fields) {
        m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
            end_time, 0.5_rt * m_dt);
        AddExternalBfield();
        AddExternalEfield();
    }
    // pe^{n+1} = (pe^{n+theta} - (1-theta) pe^n) / theta, then roll the state
    if (!m_pe_theta) { return; }
    using namespace amrex::literals;
    using warpx::fields::FieldType;

    const int lev = 0;
    amrex::MultiFab* pe = m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    amrex::MultiFab::LinComb(*pe, 1._rt/m_theta, *m_pe_theta, 0,
                             -(1._rt - m_theta)/m_theta, *m_pe_old, 0,
                             0, pe->nComp(), pe->nGrowVect());
    // Guard against negative overshoot from the extrapolation, and pin
    // below-floor cells to the floored-adiabat constant (the algebraic
    // closure at n_floor -- continuous with the plasma edge and the value
    // the explicit QDSMC's insulating halo holds). Without the pin, the
    // energy-paired work term stores the vacuum-resistivity dissipation
    // eta*J^2 in the halo pe step after step, and T_e = pe/(n_floor k_B)
    // grows without bound. Applied POST-STEP only: pinning inside the
    // Newton residual makes the residual discontinuous in the cells whose
    // re-deposited rho straddles the floor, and Newton stalls.
    {
        const amrex::Real gam = m_hybrid_pic_model->m_gamma;
        const amrex::Real rho_floor = m_hybrid_pic_model->m_n_floor * PhysConst::q_e;
        const amrex::Real pe_vac = m_hybrid_pic_model->m_n_floor
            * m_hybrid_pic_model->m_elec_temp
            * std::pow(m_hybrid_pic_model->m_n_floor / m_hybrid_pic_model->m_n0_ref,
                       gam - 1._rt);
        const amrex::MultiFab* rho = m_WarpX->m_fields.get(FieldType::rho_fp, lev);
        for (amrex::MFIter mfi(*pe); mfi.isValid(); ++mfi) {
            const amrex::Box bx = mfi.growntilebox();
            auto const& a = pe->array(mfi);
            auto const& r = rho->const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                a(i,j,k) = (r(i,j,k,0) <= rho_floor)
                    ? pe_vac : amrex::max(a(i,j,k), 0._rt);
            });
        }
    }
    amrex::MultiFab::Copy(*m_pe_old, *pe, 0, 0, pe->nComp(), pe->nGrowVect());

}

void ThetaImplicitHybrid::AddExternalBfield ()
{
    using ablastr::fields::Direction;

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        for (int idim = 0; idim < 3; ++idim) {
            amrex::MultiFab::Add(
                *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{idim}, lev),
                *m_WarpX->m_fields.get(FieldType::hybrid_B_fp_external, Direction{idim}, lev),
                0, 0, 1,
                m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{idim}, lev)->nGrowVect());
        }
    }
}

void ThetaImplicitHybrid::SubtractExternalBfield ()
{
    using ablastr::fields::Direction;

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        for (int idim = 0; idim < 3; ++idim) {
            amrex::MultiFab::Subtract(
                *m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{idim}, lev),
                *m_WarpX->m_fields.get(FieldType::hybrid_B_fp_external, Direction{idim}, lev),
                0, 0, 1,
                m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{idim}, lev)->nGrowVect());
        }
    }
}

void ThetaImplicitHybrid::AddExternalEfield ()
{
    using ablastr::fields::Direction;

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        for (int idim = 0; idim < 3; ++idim) {
            amrex::MultiFab::Add(
                *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{idim}, lev),
                *m_WarpX->m_fields.get(FieldType::hybrid_E_fp_external, Direction{idim}, lev),
                0, 0, 1,
                m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{idim}, lev)->nGrowVect());
        }
    }
}

void ThetaImplicitHybrid::SubtractExternalEfield ()
{
    using ablastr::fields::Direction;

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        for (int idim = 0; idim < 3; ++idim) {
            amrex::MultiFab::Subtract(
                *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{idim}, lev),
                *m_WarpX->m_fields.get(FieldType::hybrid_E_fp_external, Direction{idim}, lev),
                0, 0, 1,
                m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{idim}, lev)->nGrowVect());
        }
    }
}

void ThetaImplicitHybrid::AddLongitudinalE ()
{
    using ablastr::fields::Direction;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        for (int idim = 0; idim < 3; ++idim) {
            amrex::MultiFab::Add(
                *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{idim}, lev),
                *m_WarpX->m_fields.get(FieldType::hybrid_E_fp_long, Direction{idim}, lev),
                0, 0, 1,
                m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{idim}, lev)->nGrowVect());
        }
    }
}

void ThetaImplicitHybrid::SubtractLongitudinalE ()
{
    using ablastr::fields::Direction;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        for (int idim = 0; idim < 3; ++idim) {
            amrex::MultiFab::Subtract(
                *m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{idim}, lev),
                *m_WarpX->m_fields.get(FieldType::hybrid_E_fp_long, Direction{idim}, lev),
                0, 0, 1,
                m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{idim}, lev)->nGrowVect());
        }
    }
}

void ThetaImplicitHybrid::SubtractLongitudinalE (WarpXSolverVec& a_vec)
{
    using ablastr::fields::Direction;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        for (int idim = 0; idim < 3; ++idim) {
            amrex::MultiFab::Subtract(
                *a_vec.getArrayVec()[lev][idim],
                *m_WarpX->m_fields.get(FieldType::hybrid_E_fp_long, Direction{idim}, lev),
                0, 0, 1,
                a_vec.getArrayVec()[lev][idim]->nGrowVect());
        }
    }
}

const amrex::Vector<amrex::MultiFab*>*
ThetaImplicitHybrid::GetAlphaCoeff () const
{
    BL_PROFILE("ThetaImplicitHybrid::GetAlphaCoeff()");
    using namespace amrex;
    using namespace ablastr::coarsen::sample;

    // If fields aren't initialized yet (called from Define()),
    // return the pointer without computing — alpha_mf was
    // initialized to 0 and will be filled properly in Update().
    if (!m_is_defined) {
        return &m_alpha_mfarrvec;
    }

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {

        const ablastr::fields::VectorField Bfield_fp =
            m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, lev);
        const MultiFab& rho_fp =
            *m_WarpX->m_fields.get(FieldType::rho_fp, lev);

        const GpuArray<int, 3> Bx_stag = m_hybrid_pic_model->Bx_IndexType;
        const GpuArray<int, 3> By_stag = m_hybrid_pic_model->By_IndexType;
        const GpuArray<int, 3> Bz_stag = m_hybrid_pic_model->Bz_IndexType;
        const GpuArray<int, 3> nodal   = {1, 1, 1};
        const GpuArray<int, 3> coarsen = {1, 1, 1};

        const Real rho_floor    = m_hybrid_pic_model->m_n_floor * PhysConst::q_e;
        const Real theta_dt     = m_theta * m_dt;
        const Real one_over_mu0 = 1._rt / PhysConst::mu0;
        const auto eta          = m_hybrid_pic_model->m_eta;
        const Real t_new        = m_WarpX->gett_new(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(m_alpha_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {

            Array4<Real>       const& alpha = m_alpha_mf[lev].array(mfi);
            Array4<Real const> const& Bx    = Bfield_fp[0]->const_array(mfi);
            Array4<Real const> const& By    = Bfield_fp[1]->const_array(mfi);
            Array4<Real const> const& Bz    = Bfield_fp[2]->const_array(mfi);
            Array4<Real const> const& rho   = rho_fp.const_array(mfi);

            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                // Interpolate B from Yee grid to nodal grid
                const Real Bx_n = Interp(Bx, Bx_stag, nodal, coarsen, i, j, k, 0);
                const Real By_n = Interp(By, By_stag, nodal, coarsen, i, j, k, 0);
                const Real Bz_n = Interp(Bz, Bz_stag, nodal, coarsen, i, j, k, 0);

                // |B| at nodal point
                const Real Bmag = std::sqrt(Bx_n*Bx_n + By_n*By_n + Bz_n*Bz_n);

                // rho is nodal in the hybrid model - read directly, apply floor
                const Real rho_limited = std::max(rho(i, j, k), rho_floor);

                // alpha = theta*dt * sqrt((|B|/rho)^2 + eta^2) / mu0
                //
                // The hybrid Jacobian on the transverse (Er, Etheta) block has eigenvalues
                //   -(theta*dt*k^2/mu0) * (eta ± i*|B|/rho)
                // (J×B/rho gives the imaginary skew part; eta*J gives the real diagonal).
                // The curl-curl PC clusters J·P^{-1} eigenvalues at magnitude ~1 when
                //   alpha * k^2 = k^2 * sqrt(eta^2 + (|B|/rho)^2),
                // i.e., alpha = sqrt((|B|/rho)^2 + eta^2) / mu0.
                // For eta=0, this reduces to the original |B|/(mu0*rho).
                // For eta >> |B|/rho (high resistivity), alpha → eta/mu0.
                const Real eta_val       = eta(rho_limited, 0._rt, t_new);
                const Real Bmag_over_rho = Bmag / rho_limited;
                alpha(i, j, k) = theta_dt * one_over_mu0
                                //  * std::sqrt(Bmag_over_rho*Bmag_over_rho + eta_val*eta_val);
                                 * std::sqrt(Bmag_over_rho*Bmag_over_rho);
            });
        }
    }

    return &m_alpha_mfarrvec;
}