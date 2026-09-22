/* Copyright 2026 Prabhat Kumar
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Fields.H"
#include "ThetaImplicitHybrid.H"
#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ElectronPressureFlux.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/ExternalVectorPotential.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"
#include <ablastr/utils/Communication.H>
#include <ablastr/coarsen/sample.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_GpuContainers.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <string>
#include <vector>

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

    {
        const amrex::ParmParse pp_impl("implicit_evolve");
        pp_impl.query("pe_newton_unknown", m_pe_unknown);
        // enthalpy-flux discretization of the in-loop pe advance: the van Albada MUSCL
        // face flux needs the collocated (nodal J) Cartesian grid and is the default
        // there; central differences elsewhere
        // (collocated Cartesian and collocated RZ m = 0; the Yee branches keep
        // the central edge flux)
        const bool pe_adv_muscl_ok = m_WarpX->m_fields.get(
            FieldType::current_fp, ablastr::fields::Direction{2}, 0)->ixType().nodeCentered();
        std::string pe_adv_name = pe_adv_muscl_ok ? "vanalbada" : "central";
        pp_impl.query("pe_advection", pe_adv_name);
        if (pe_adv_name == "central") { m_pe_advection = 0; }
        else if (pe_adv_name == "vanalbada") { m_pe_advection = 1; }
        else {
            WARPX_ABORT_WITH_MESSAGE(
                "implicit_evolve.pe_advection = " + pe_adv_name +
                " is not valid; options: central, vanalbada");
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_pe_advection == 0 || pe_adv_muscl_ok,
            "implicit_evolve.pe_advection = vanalbada requires the collocated grid");
        pp_impl.query("filter_push_fields", m_filter_push_fields);
        if (m_filter_push_fields) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::use_filter,
                "implicit_evolve.filter_push_fields requires warpx.use_filter = 1");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_add_external_fields,
                "implicit_evolve.filter_push_fields: external fields are not "
                "supported (the external-field work ledger is unfiltered)");
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_WarpX->Geom(0).isPeriodic(d),
                    "implicit_evolve.filter_push_fields requires a fully "
                    "periodic domain (binomial-filter self-adjointness at "
                    "walls is not handled)");
            }
        }
        pp_impl.query("pe_ue_cap_fac", m_pe_ue_cap_fac);
        pp_impl.query("pe_wall_mirror", m_pe_wall_mirror);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_pe_wall_mirror >= 0 && m_pe_wall_mirror <= 2,
            "implicit_evolve.pe_wall_mirror must be 0, 1 or 2");
#if !defined(WARPX_DIM_RZ)
        // The mirror closure is written for the RZ half-device deck, whose axial (index-1)
        // lower face is the device mid-plane. In 3D index 1 is y (a wall face outside the
        // EB), so the mirror would advance the y-face pressure planes with reflected
        // stencils (2026-09-17: runaway face pe, 16-it Newton, guard-cell abort at step 4).
        if (m_pe_wall_mirror != 0) {
            ablastr::warn_manager::WMRecordWarning("ThetaImplicitHybrid",
                "implicit_evolve.pe_wall_mirror is RZ-only; using the legacy face treatment",
                ablastr::warn_manager::WarnPriority::medium);
            m_pe_wall_mirror = 0;
        }
#endif
        pp_impl.query("joule_Te_cutoff", m_joule_Te_cutoff_eV);
        pp_impl.query("joule_Te_cutoff_width", m_joule_Te_cutoff_width);
        pp_impl.query("joule_redirect_to_ions", m_joule_redirect);
        pp_impl.query("joule_redirect_verbose", m_joule_redirect_verbose);
        if (m_joule_redirect) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_joule_Te_cutoff_eV > 0.0_rt,
                "implicit_evolve.joule_redirect_to_ions requires implicit_evolve.joule_Te_cutoff > 0");
        }
        pp_impl.queryarr("pe_debug_nodes", m_pe_debug_nodes);
        pp_impl.query("predictor", m_predictor);
        pp_impl.query("predictor_max_iters", m_predictor_max_iters);
        pp_impl.query("predictor_max_step", m_predictor_max_step);
        pp_impl.query("predictor_rho_factor", m_predictor_rho_factor);
        pp_impl.query("dt_halving_max_levels", m_dt_halving_max_levels);
        pp_impl.query("dt_halving_test_step", m_dt_halving_test_step);
        pp_impl.query("dt_halving_max_rel", m_dt_halving_max_rel);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_dt_halving_max_levels >= 0,
            "implicit_evolve.dt_halving_max_levels must be >= 0");
        if (m_joule_Te_cutoff_eV > 0.0_rt) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_hybrid_pic_model->m_include_joule_heating &&
                m_hybrid_pic_model->m_solve_electron_energy_equation,
                "implicit_evolve.joule_Te_cutoff requires the in-loop electron energy "
                "equation with hybrid_pic_model.include_joule_heating = 1");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_joule_Te_cutoff_width > 0.0_rt,
                "implicit_evolve.joule_Te_cutoff_width must be positive");
        }
        if (m_pe_unknown) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_hybrid_pic_model->m_solve_electron_energy_equation &&
                !m_hybrid_pic_model->m_implicit_use_algebraic_closure,
                "implicit_evolve.pe_newton_unknown requires the in-loop electron "
                "energy equation (not the algebraic closure)");
            // pe-row scale: E_pe = -grad(pe)/(e n) ~ pe/(e n0 dx), so 1/(q_e n0_ref dx_min)
            // makes the pe residual commensurate with E in the solver norms
            amrex::Real dx_min = m_WarpX->Geom(0).CellSize(0);
            for (int d = 1; d < AMREX_SPACEDIM; ++d) {
                dx_min = std::min(dx_min, m_WarpX->Geom(0).CellSize(d));
            }
            m_pe_scale = 1.0_rt /
                (PhysConst::q_e * m_hybrid_pic_model->m_n0_ref * dx_min);
        }
        // the preconditioner reads the pe-row scale through the hybrid model
        m_hybrid_pic_model->m_pe_newton_scale = m_pe_unknown ? m_pe_scale : 1.0_rt;
    }
    if (m_pe_unknown) {
        m_E.Define( m_WarpX, "Efield_fp", "hybrid_electron_pressure_fp" );
    } else {
        m_E.Define( m_WarpX, "Efield_fp" );
    }
    m_Eold.Define( m_E );
    m_dE_prev.Define( m_E );

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

    parseNonlinearSolverParams( pp );

    if (m_use_mass_matrices) {
        // lagged mass matrices (see DepositMassMatricesThisIter)
        pp.query("mass_matrices_deposit_interval", m_mass_matrices_deposit_interval);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_mass_matrices_deposit_interval >= 0,
            "implicit_evolve.mass_matrices_deposit_interval must be >= 0");
        pp.query("mass_matrices_step_interval", m_mass_matrices_step_interval);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_mass_matrices_step_interval >= 1,
            "implicit_evolve.mass_matrices_step_interval must be >= 1");
        pp.query("mass_matrices_rho_response_factor", m_mm_rho_response_factor);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_mm_rho_response_factor >= 1.0_rt,
            "implicit_evolve.mass_matrices_rho_response_factor must be >= 1");
    }

    m_nlsolver->Define(m_E, this);

    if (m_pe_unknown) {
        // the pe row must pass through the preconditioner (identity at
        // minimum) or the right-preconditioned operator is singular in it
        const PreconditionerType pc_type = m_nlsolver->GetPreconditionerType();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            pc_type == PreconditionerType::none ||
            pc_type == PreconditionerType::pc_hybrid_pic,
            "implicit_evolve.pe_newton_unknown: only pc_type none or "
            "pc_hybrid_pic pass the pe row through the preconditioner");
    }

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
    if (m_dt_halving_max_levels > 0) {
        amrex::Print() << "dt halving on unconverged Newton:    up to "
                       << m_dt_halving_max_levels << " level(s)\n";
    }
    if (m_use_mass_matrices) {
        amrex::Print() << "mass matrices deposit interval:      "
                       << m_mass_matrices_deposit_interval << "\n";
        amrex::Print() << "mass matrices step interval:         "
                       << m_mass_matrices_step_interval << "\n";
    }
    PrintBaseImplicitSolverParameters();
    m_nlsolver->PrintParams();
    amrex::Print() << "-----------------------------------------------------------\n\n";
}

int ThetaImplicitHybrid::OneStep ( const amrex::Real  start_time,
                                    const amrex::Real  a_dt,
                                    const int          a_step )
{
    BL_PROFILE("ThetaImplicitHybrid::OneStep()");

    const int status = AdvanceStep(start_time, a_dt, a_step, 0);
    // sub-steps set WarpX::dt to their own dt (the particle push reads it):
    // leave it at the outer step for the caller
    for (int lev = 0; lev < m_num_amr_levels; ++lev) { m_WarpX->setdt(a_dt, lev); }
    return status;
}

int ThetaImplicitHybrid::AdvanceStep ( const amrex::Real  start_time,
                                        const amrex::Real  a_dt,
                                        const int          a_step,
                                        const int          a_level )
{
    using namespace amrex::literals;

    const bool can_halve = (a_level < m_dt_halving_max_levels);
    if (can_halve) { SaveStepStartState(a_level); }

    const int status = SolveStep(start_time, a_dt, a_step);
    // testing hook: force the rejection path once (outer level of one step)
    const bool forced_fail = (a_level == 0 && a_step == m_dt_halving_test_step);
    const bool converged = (status >= 0) && m_nlsolver->GetLastConverged() && !forced_fail;

    if (!converged && can_halve) {
        // Reject the step: back to the state at start_time and redo it as two
        // half-steps, each again subject to halving up to the level cap.
        RestoreStepStartState(a_level);
        m_have_dE_prev = false;   // the rejected increment is no predictor
        ++m_dt_halvings;
        amrex::Print() << "ThetaImplicitHybrid: Newton did not converge (exit status "
                       << status << ") at t = " << start_time << ", halving level "
                       << a_level << ": redoing the step as two half-steps of dt = "
                       << 0.5_rt*a_dt << " (" << m_dt_halvings << " halving(s) so far)\n";
        const amrex::Real dt_half = 0.5_rt*a_dt;
        const int s1 = AdvanceStep(start_time, dt_half, a_step, a_level + 1);
        if (s1 < 0) { return s1; }
        const int s2 = AdvanceStep(start_time + dt_half, dt_half, a_step, a_level + 1);
        m_have_dE_prev = false;   // a half-step increment does not predict a full step
        return s2;
    }

    // -5 = Newton did not converge (iteration cap or exhausted line search) with
    // require_convergence: not fatal by itself here, the halving-limit policy below decides;
    // any other negative status (divergence, giving up) ends the run
    if (status < 0 && status != -5) { return status; }
    if (!converged && m_dt_halving_max_levels > 0
        && m_nlsolver->GetLastRelNorm() > m_dt_halving_max_rel) {
        // deepest halving level and still far from converged: do not advance
        // the state from here (the next step would start from a corrupted
        // state and spiral); a negative status aborts the run cleanly
        amrex::Print() << "ThetaImplicitHybrid: unconverged at the halving limit (relative "
                       << "residual " << m_nlsolver->GetLastRelNorm() << " > "
                       << m_dt_halving_max_rel << ", implicit_evolve.dt_halving_max_rel) at t = "
                       << start_time << ": giving up the step\n";
        return -5;
    }
    FinishStep(start_time, a_step);
    return status;
}

int ThetaImplicitHybrid::SolveStep ( const amrex::Real  start_time,
                                      const amrex::Real  a_dt,
                                      const int          a_step )
{
    BL_PROFILE("ThetaImplicitHybrid::SolveStep()");

    m_dt = a_dt;
    // the particle push (WarpX::PushParticlesandDeposit) reads WarpX::dt
    for (int lev = 0; lev < m_num_amr_levels; ++lev) { m_WarpX->setdt(a_dt, lev); }

    // Handle external field splitting: work with internal fields during the solve
    if (m_add_external_fields) {
        m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
            start_time, 0.5_rt * m_dt);
        SubtractExternalEfield();
        SubtractExternalBfield();
    }

    // Save particle state at t^n
    m_WarpX->SaveParticlesAtImplicitStepStart();

    // Save E^n (and pe^n when pe is a Newton unknown)
    if (m_pe_unknown) {
        amrex::MultiFab* pe =
            m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, 0);
        if (!m_pe_old) {
            // first use: the pressure field holds its (seeded) initialization
            m_pe_old = std::make_unique<amrex::MultiFab>(
                pe->boxArray(), pe->DistributionMap(), pe->nComp(), pe->nGrowVect());
            amrex::MultiFab::Copy(*m_pe_old, *pe, 0, 0, pe->nComp(), pe->nGrowVect());
            m_pe_old->FillBoundary(m_WarpX->Geom(0).periodicity());
        }
        m_Eold.Copy(FieldType::Efield_fp,
                    FieldType::hybrid_electron_pressure_fp);
        m_Eold.getScalarVec()[0]->mult(m_pe_scale, 0, 1);
    } else {
        m_Eold.Copy(FieldType::Efield_fp);
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

    // Initial guess: E^{n+θ} = E^n, or with implicit_evolve.predictor the previous
    // step's theta-increment added: E^n + (E^{n-1+θ} - E^{n-1}). The residual at E^n
    // is dominated by the per-step field change and its first Newton direction
    // typically needs a line search (formation: alpha 0.5 at iteration 0 even with
    // the exact Jacobian); the extrapolated guess starts inside the quadratic basin.
    m_E.Copy(m_Eold);
    // step-limited predictor (see m_predictor_max_step)
    const bool predictor_on = m_predictor
        && (m_predictor_max_step < 0 || a_step < m_predictor_max_step);
    if (predictor_on && m_have_dE_prev) {
        m_E.linComb(1.0_rt, m_Eold, 1.0_rt, m_dE_prev);
    }

    // Lagged mass matrices across steps (see DepositMassMatricesThisIter):
    // deposit on the first step of the run/restart and on every k-th step.
    if (m_use_mass_matrices) {
        // also whenever the (sub)step size differs from the last deposit's: the kernels
        // carry dt through the gyro factors, so half-step solves must not inherit
        // full-step matrices (nor the next full step the half-step ones)
        m_mm_deposit_this_step = (m_mass_matrices_step_interval <= 1) || !m_mm_deposited_once
                                 || (a_step % m_mass_matrices_step_interval == 0)
                                 || (m_dt != m_mm_deposit_dt);
    }

    // Solve nonlinear system for E^{n+θ} (and eventually Pe^{n+θ})
    m_nlsolver->Solve( m_E, m_Eold, start_time, m_dt, a_step );

    if (m_mm_deposit_this_step) { m_mm_deposited_once = true; m_mm_deposit_dt = m_dt; }

    const int exit_status = m_nlsolver->GetExitStatus();
    if (exit_status < 0) { return exit_status; }

    if (predictor_on) {
        // theta-increment of this step, the next step's predictor -- unless this step
        // struggled (>= predictor_max_iters Newton iterations: line-search-limited steps
        // leave an increment that is not a smooth continuation; extrapolating it fed a
        // diverging first linear solve in the formation run), then fall back to E^n
        m_dE_prev.linComb(1.0_rt, m_E, -1.0_rt, m_Eold);
        if (m_predictor_rho_factor > 0.0_rt) { WeightPredictorIncrement(); }
        m_have_dE_prev = (m_nlsolver->GetLastIterations() < m_predictor_max_iters);
    }

    return exit_status;
}

namespace {
/** Volume-weighted (RZ: 2 pi r dr dz) global sum of a nodal scalar field, for ledger prints */
amrex::Real VolumeWeightedSum (const amrex::MultiFab& mf, const amrex::Geometry& geom)
{
    using namespace amrex::literals;
    const auto dx = geom.CellSizeArray();
    const amrex::Real rmin = geom.ProbLo(0);
    amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& a = mf.const_array(mfi);
        reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
#if defined(WARPX_DIM_RZ)
                const amrex::Real r = amrex::max(rmin + i*dx[0], 0.25_rt*dx[0]);
                const amrex::Real w = 2.0_rt*MathConst::pi*r*dx[0]*dx[1];
#elif defined(WARPX_DIM_3D)
                const amrex::Real w = dx[0]*dx[1]*dx[2];
                amrex::ignore_unused(rmin);
#elif defined(WARPX_DIM_XZ)
                const amrex::Real w = dx[0]*dx[1];
                amrex::ignore_unused(rmin);
#else
                const amrex::Real w = dx[0];
                amrex::ignore_unused(rmin);
#endif
                return {a(i,j,k)*w};
            });
    }
    amrex::Real s = amrex::get<0>(reduce_data.value(reduce_op));
    amrex::ParallelDescriptor::ReduceRealSum(s);
    return s;
}
}

void ThetaImplicitHybrid::SlaveWallPressure ( const int lev )
{
    // Entropy slave of the non-periodic axial face planes (pe_wall_mirror = 2, collocated RZ):
    // pe = s n^gamma with s the mean pe/n^gamma of the three planes inside (d s/dz = 0 at a
    // symmetry plane, the RZ axis-row treatment), applied to the ACCEPTED pe^{n+1} after the
    // solve. Inside the solve the face nodes follow the mirror advance (pe_wall_mirror = 1),
    // whose flux convergence has no work-term counterpart on the face (E_z = J_e,z = 0 there)
    // and no density response, so left alone the face pressure e-folds every few steps
    // (formation v2, 25 us: 28 keV on the plane, 70 eV one plane in). Imposing the slave as
    // a Newton constraint instead (v3) broke the solve (7 rejections in 112 steps, death).
    // EB-covered face nodes keep their value.
#if defined(WARPX_DIM_RZ)
    using namespace amrex::literals;
    const amrex::Geometry& geom = m_WarpX->Geom(lev);
    if (geom.isPeriodic(1)) { return; }
    amrex::MultiFab* pe = m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    const amrex::MultiFab* rho_mf = m_WarpX->m_fields.get(FieldType::rho_fp, lev);
    const amrex::MultiFab* jz = m_WarpX->m_fields.get(FieldType::hybrid_current_fp_plasma,
                                                      ablastr::fields::Direction{2}, lev);
    if (!jz->ixType().nodeCentered()) { return; }   // collocated grids only
    const amrex::Real gam = m_hybrid_pic_model->m_gamma;
    const amrex::Real rfloor = m_hybrid_pic_model->m_n_floor * PhysConst::q_e;
    const amrex::Box dom_n = amrex::convert(geom.Domain(), amrex::IntVect::TheNodeVector());
    const int jlo = dom_n.smallEnd(1), jhi = dom_n.bigEnd(1);
    const amrex::iMultiFab* eb_flag = EB::enabled()
        ? m_WarpX->GetEBUpdateEFlag()[lev][0].get() : nullptr;
    for (amrex::MFIter mfi(*pe, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box tb = mfi.tilebox(amrex::IntVect::TheNodeVector());
        auto const& pe_a = pe->array(mfi);
        auto const& rh   = rho_mf->const_array(mfi);
        amrex::Array4<int const> ebq;
        if (eb_flag) { ebq = eb_flag->const_array(mfi); }
        const bool has_eb = (eb_flag != nullptr);
        for (const int jw : {jlo, jhi}) {
            if (jw < tb.smallEnd(1) || jw > tb.bigEnd(1)) { continue; }
            const int dir = (jw == jlo) ? 1 : -1;
            const amrex::Box tbw(amrex::IntVect(tb.smallEnd(0), jw),
                                 amrex::IntVect(tb.bigEnd(0), jw), tb.ixType());
            amrex::ParallelFor(tbw, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                if (has_eb && ebq(i,j,k) == 0) { return; }
                amrex::Real sbar = 0._rt;
                for (int jj = 1; jj <= 3; ++jj) {
                    const amrex::Real nn = amrex::max(rh(i,j+dir*jj,k,0), rfloor);
                    sbar += pe_a(i,j+dir*jj,k) / std::pow(nn, gam);
                }
                sbar *= (1._rt/3._rt);
                const amrex::Real nw = amrex::max(rh(i,j,k,0), rfloor);
                pe_a(i,j,k) = sbar * std::pow(nw, gam);
            });
        }
    }
    pe->OverrideSync(geom.periodicity());
    pe->FillBoundary(geom.periodicity());
#else
    amrex::ignore_unused(lev);
#endif
}

void ThetaImplicitHybrid::FinishStep ( const amrex::Real  start_time,
                                        const int          a_step )
{
    BL_PROFILE("ThetaImplicitHybrid::FinishStep()");

    // Update WarpX fields to t^{n+θ}
    UpdateWarpXFields( m_E, start_time );
    if (m_pe_unknown) {
        // the accepted Newton iterate is pe^{n+theta}; snapshot it for the t^{n+1}
        // extrapolation in FinishFieldUpdate (the in-loop path snapshots in the advance)
        AMREX_ALWAYS_ASSERT(m_pe_theta);
        const amrex::MultiFab* pe =
            m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, 0);
        amrex::MultiFab::Copy(*m_pe_theta, *pe, 0, 0,
                              pe->nComp(), pe->nGrowVect());
    }
    m_WarpX->reduced_diags->ComputeDiagsMidStep(a_step);

    // Advance particles from t^{n+1/2} to t^{n+1}
    m_WarpX->FinishImplicitParticleUpdate( start_time + m_dt );

    // Advance fields (including the electron pressure state) from t^{n+θ} to t^{n+1}
    FinishFieldUpdate( start_time + m_dt );
    // symmetry-plane entropy pin of the axial face planes on the accepted pe^{n+1}; the
    // start-of-step copy pe^n that FinishFieldUpdate just rolled must see the pinned value
    // (the Q_ei block below re-rolls it only when relaxation or the redirect is on)
    if (m_pe_wall_mirror == 2) {
        for (int lev = 0; lev < m_num_amr_levels; ++lev) { SlaveWallPressure(lev); }
        if (m_pe_old) {
            const amrex::MultiFab* pe =
                m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, 0);
            amrex::MultiFab::Copy(*m_pe_old, *pe, 0, 0, pe->nComp(), pe->nGrowVect());
            m_pe_old->FillBoundary(m_WarpX->Geom(0).periodicity());
        }
    }

    // Electron energy equation: the transport/compression/Joule part of the
    // update already happened inside the Newton solve (in-loop pe advance,
    // energy-paired). What remains is the symmetric Q_ei ion-electron
    // collisional exchange, which must kick the ion particles and therefore
    // runs once per step here, on the t^{n+1} state. The Joule redirect (energy the Te
    // gate withheld from pe, staged in m_ion_redirect_E by the last residual
    // evaluation) rides on the same ion-heating kernel.
    const bool redirect_active = m_joule_redirect && m_ion_redirect_E &&
        m_hybrid_pic_model->m_solve_electron_energy_equation;
    if (m_hybrid_pic_model->m_solve_electron_energy_equation &&
        (m_hybrid_pic_model->m_include_temperature_relaxation || redirect_active)) {
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
            // Rebuild the species sum the Q_ei (and QDSMC Joule) kernels divide by,
            // f_s = rho_s / Sigma_t rho_t. Only the explicit HybridPICDepositRhoAndJ
            // refreshed hybrid_rho_species_sum_fp, so implicitly it held the
            // initialization deposit for the whole run (f_s = n(t)/n(0): the relaxation
            // 8x too fast in the compressed FRC core, arbitrary where plasma moved into
            // initially empty cells) and ZERO after a restart (floored: f_s ~ 300, the
            // electrons relaxed toward T_i at ~1 % per step = +28 J/step of spurious
            // electron heating, 10x the physical rate; formation restarts 2026-09-17).
            for (int lev = 0; lev < m_num_amr_levels; ++lev) {
                amrex::MultiFab* rsum =
                    m_WarpX->m_fields.get("hybrid_rho_species_sum_fp", lev);
                rsum->setVal(0.0_rt);
                for (auto const & spec : mypc.GetSpeciesNames()) {
                    auto & pc = mypc.GetParticleContainerFromName(spec);
                    if (pc.getCharge() == 0._prt) { continue; }
                    const amrex::MultiFab* rs = m_WarpX->m_fields.get("rho_fp_" + spec, lev);
                    amrex::MultiFab::Add(*rsum, *rs, 0, 0, 1, amrex::IntVect::TheZeroVector());
                }
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
            m_hybrid_pic_model->ApplyIonElectronEnergyExchange(
                lev, m_dt, (redirect_active && lev == 0) ? m_ion_redirect_E.get() : nullptr);
            m_hybrid_pic_model->FillPeFromTe(lev);
        }
        if (redirect_active) {
            if (m_joule_redirect_verbose > 0 && m_Q_diss && m_Q_redir) {
                const amrex::Real p_pe = VolumeWeightedSum(*m_Q_diss, m_WarpX->Geom(0));
                const amrex::Real p_i  = VolumeWeightedSum(*m_Q_redir, m_WarpX->Geom(0));
                amrex::Print() << "  Joule deposit: pe " << p_pe
                               << " W, redirected to ions " << p_i << " W\n";
            }
            // consumed: a step whose residual evaluations never wrote it must not reuse it
            m_ion_redirect_E->setVal(0.0_rt);
            m_Q_redir->setVal(0.0_rt);
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
}

void ThetaImplicitHybrid::SaveStepStartState ( const int a_level )
{
    // Total fields (external part included: called before the split of
    // SolveStep), the electron pressure and, through the x_n/u_n particle
    // attributes written by SolveStep, the particles: everything a rejected
    // step must be rewound to. One set of copies per halving level.
    using ablastr::fields::Direction;
    AMREX_ALWAYS_ASSERT(m_num_amr_levels == 1);
    const int lev = 0;
    if (static_cast<int>(m_E_save.size()) <= a_level) {
        m_E_save.resize(a_level + 1);
        m_B_save.resize(a_level + 1);
        m_pe_save.resize(a_level + 1);
    }
    for (int n = 0; n < 3; ++n) {
        const amrex::MultiFab* E = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{n}, lev);
        const amrex::MultiFab* B = m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{n}, lev);
        if (!m_E_save[a_level][n]) {
            m_E_save[a_level][n] = std::make_unique<amrex::MultiFab>(
                E->boxArray(), E->DistributionMap(), E->nComp(), E->nGrowVect());
            m_B_save[a_level][n] = std::make_unique<amrex::MultiFab>(
                B->boxArray(), B->DistributionMap(), B->nComp(), B->nGrowVect());
        }
        amrex::MultiFab::Copy(*m_E_save[a_level][n], *E, 0, 0, E->nComp(), E->nGrowVect());
        amrex::MultiFab::Copy(*m_B_save[a_level][n], *B, 0, 0, B->nComp(), B->nGrowVect());
    }
    const amrex::MultiFab* pe =
        m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    if (!m_pe_save[a_level]) {
        m_pe_save[a_level] = std::make_unique<amrex::MultiFab>(
            pe->boxArray(), pe->DistributionMap(), pe->nComp(), pe->nGrowVect());
    }
    amrex::MultiFab::Copy(*m_pe_save[a_level], *pe, 0, 0, pe->nComp(), pe->nGrowVect());
}

void ThetaImplicitHybrid::RestoreStepStartState ( const int a_level )
{
    // Inverse of SaveStepStartState. The densities, currents, mass matrices
    // and push-field work arrays are rebuilt from these by the next solve;
    // m_pe_old is only rolled in FinishStep and therefore still holds pe^n.
    using ablastr::fields::Direction;
    const int lev = 0;
    for (int n = 0; n < 3; ++n) {
        amrex::MultiFab* E = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{n}, lev);
        amrex::MultiFab* B = m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{n}, lev);
        amrex::MultiFab::Copy(*E, *m_E_save[a_level][n], 0, 0, E->nComp(), E->nGrowVect());
        amrex::MultiFab::Copy(*B, *m_B_save[a_level][n], 0, 0, B->nComp(), B->nGrowVect());
    }
    amrex::MultiFab* pe = m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, lev);
    amrex::MultiFab::Copy(*pe, *m_pe_save[a_level], 0, 0, pe->nComp(), pe->nGrowVect());
    m_WarpX->RestoreParticlesAtImplicitStepStart();
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
        if (WarpX::use_filter) {
            // PreRHSOp filters Efield_fp in place before the linear stage contracts the
            // mass matrices with it, so E0 must be the filtered push field as well
            m_WarpX->ApplyFilterMF(
                m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp_save, 0), 0);
        }
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

    if (m_filter_push_fields) {
        // conservative smoothing: the particles gather the filtered push field, everything
        // downstream keeps the unfiltered registry (Jacobian probes included)
        FilterPushFieldsSwap(true);
    }
    // A trial iterate may push particles beyond the deposition guard range (a wild
    // linear-solve direction, the stiff wall sheath of the 3D formation case): have the
    // deposits count and skip them instead of asserting, and hand Newton a residual it
    // must reject. Every evaluation re-pushes from the step-start state, so nothing of
    // the skipped state survives.
    WarpXParticleContainer::SetDepositOutOfRangeTolerant(true);
    PreRHSOp( theta_time, a_nl_iter, a_from_jacobian );
    WarpXParticleContainer::SetDepositOutOfRangeTolerant(false);
    if (m_filter_push_fields) {
        FilterPushFieldsSwap(false);
    }
    {
        amrex::Long n_out = WarpXParticleContainer::NumDepositOutOfRange();
        amrex::ParallelDescriptor::ReduceLongSum(n_out);
        m_rhs_invalid = (n_out > 0);
        if (m_rhs_invalid) {
            ++m_rhs_invalid_count;
            amrex::Print() << "ThetaImplicitHybrid: " << n_out << " particle(s) beyond the "
                           << "deposition guard range in this residual evaluation (Newton "
                           << "iteration " << a_nl_iter
                           << (a_from_jacobian ? ", Jacobian probe" : "")
                           << "): the trial is rejected (" << m_rhs_invalid_count
                           << " such evaluation(s) so far)\n";
            a_RHS.setVal(m_invalid_rhs_value);
            return;
        }
    }

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

    if (m_use_mass_matrices_jacobian) {
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
    // the ion push and Faraday ledgers -- the implicit scheme's
    // implementation of the gamma-law closure with a discrete electron
    // energy ledger. With implicit_use_algebraic_closure on, the pressure
    // is instead re-evaluated algebraically from the iterate's rho (the
    // exact counterpart of the explicit path's default closure): no pe
    // state, no work pairing, and none of the pairing's floored-edge
    // artifacts (Yee Hall-work residual, Cartesian m=4 separatrix layer).
    if (m_hybrid_pic_model->m_implicit_use_algebraic_closure) {
        m_hybrid_pic_model->CalculateElectronPressure();
    } else {
        AdvanceElectronPressure( a_from_jacobian, theta_time );
    }

    m_hybrid_pic_model->HybridPICSolveE(
        Efield_fp, current_fp, Bfield_fp, rho_fp,
        m_WarpX->GetEBUpdateEFlag(),
        true, true   // with resistivity and ∇Pe included (∇Pe is curl-free so doesn't affect Faraday, but needed for self-consistent Newton residual)
    );

    // EB: the masked Ohm solve leaves the Newton iterate at covered locations (a zero,
    // singular residual row). Stamp E = 0 there, the explicit path's convention, so the
    // covered rows become identity rows F = E of the Jacobian.
    if (EB::enabled()) {
        using warpx::fields::FieldType;
        auto const& eb_flags = m_WarpX->GetEBUpdateEFlag()[0];
        for (int dim = 0; dim < 3; ++dim) {
            amrex::MultiFab* Ed = m_WarpX->m_fields.get(
                FieldType::Efield_fp, ablastr::fields::Direction{dim}, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(*Ed, amrex::TilingIfNotGPU());
                 mfi.isValid(); ++mfi) {
                const amrex::Box bx = mfi.tilebox();
                auto const& e = Ed->array(mfi);
                auto const& f = eb_flags[dim]->const_array(mfi);
                amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        if (f(i,j,k) == 0) { e(i,j,k) = 0.0_rt; }
                    });
            }
        }
    }

    m_WarpX->ApplyFillBoundaryE();

    // RHS = E_ohm - E_old
    if (m_pe_unknown) {
        a_RHS.Copy(FieldType::Efield_fp, FieldType::hybrid_electron_pressure_fp);
        // pe row: the pressure field holds the iterate (read by Ohm's law above); the
        // residual uses the one-evaluation theta update m_pe_rhs, RHS_pe = kappa*(pe_rhs - pe^n)
        amrex::MultiFab::Copy(*a_RHS.getScalarVec()[0], *m_pe_rhs, 0, 0, 1,
                              amrex::IntVect::TheZeroVector());
        a_RHS.getScalarVec()[0]->mult(m_pe_scale, 0, 1);
    } else {
        a_RHS.Copy(FieldType::Efield_fp);
    }
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
    const amrex::Real rho_resp_min = m_mm_rho_response_factor * rho_floor;

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
            if (rho0v > rho_resp_min) {
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

void ThetaImplicitHybrid::WeightPredictorIncrement ()
{
    // Predictor increment restricted to the plasma: at near-floor nodes the Ohm's-law
    // field is set by a handful of macro-ions through E ~ J/rho_floor and flips between
    // steps; extrapolating it seeded a first-evaluation particle kick beyond the guard
    // cells in the formation halo phase (prod_dt4_peu_v5, step 902). The weight uses
    // the same staggered rho interpolation as the Ohm solve and the pairing.
    using namespace amrex::literals;
    using warpx::fields::FieldType;
    using namespace ablastr::coarsen::sample;
    const int lev = 0;
    const amrex::MultiFab* rho = m_WarpX->m_fields.get(FieldType::rho_fp, lev);
    const amrex::Real rho_f = m_hybrid_pic_model->m_n_floor * PhysConst::q_e;
    const amrex::Real r0 = m_predictor_rho_factor * rho_f;
    const amrex::GpuArray<int, 3> nodal3 = {1, 1, 1};
    const amrex::GpuArray<int, 3> crs3   = {1, 1, 1};
    const std::array<amrex::GpuArray<int, 3>, 4> stags = {
        m_hybrid_pic_model->Ex_IndexType, m_hybrid_pic_model->Ey_IndexType,
        m_hybrid_pic_model->Ez_IndexType, nodal3};
    for (int n = 0; n < (m_pe_unknown ? 4 : 3); ++n) {
        amrex::MultiFab& dE = (n < 3) ? *m_dE_prev.getArrayVec()[lev][n]
                                      : *m_dE_prev.getScalarVec()[lev];
        const amrex::GpuArray<int, 3> stag = stags[n];
        const int ng = amrex::max(0, amrex::min(dE.nGrowVect().min(),
                                                rho->nGrowVect().min() - 1));
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(dE, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box bx = mfi.growntilebox(ng);
            auto const& d  = dE.array(mfi);
            auto const& rr = rho->const_array(mfi);
            const int nc = dE.nComp();
            amrex::ParallelFor(bx, nc, [=] AMREX_GPU_DEVICE (int i, int j, int k, int c) {
                const amrex::Real rho_e = Interp(rr, nodal3, stag, crs3, i, j, k, 0);
                d(i,j,k,c) *= 0.5_rt*(1.0_rt + std::tanh((rho_e - r0)/rho_f));
            });
        }
    }
}

void ThetaImplicitHybrid::FilterPushFieldsSwap (const bool a_apply)
{
    // Conservative smoothing: PreRHSOp binomial-filters Efield_fp in place for the gather;
    // save E* before and restore it after so Ohm's law and the pe work pairing keep the
    // unfiltered field (B needs no treatment, v x B does no work).
    using warpx::fields::FieldType;
    const int lev = 0;
    const ablastr::fields::VectorField E =
        m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, lev);
    for (int n = 0; n < 3; ++n) {
        amrex::MultiFab& Emf = *E[n];
        if (a_apply) {
            if (!m_E_unfiltered[n]) {
                m_E_unfiltered[n] = std::make_unique<amrex::MultiFab>(
                    Emf.boxArray(), Emf.DistributionMap(),
                    Emf.nComp(), Emf.nGrowVect());
            }
            amrex::MultiFab::Copy(*m_E_unfiltered[n], Emf, 0, 0,
                                  Emf.nComp(), Emf.nGrowVect());
        } else {
            amrex::MultiFab::Copy(Emf, *m_E_unfiltered[n], 0, 0,
                                  Emf.nComp(), Emf.nGrowVect());
        }
    }
}

void ThetaImplicitHybrid::AdvanceElectronPressure ( const bool a_from_jacobian,
                                                    const amrex::Real a_theta_time )
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
    // Efield_fp holds the internal push field E* = a_E - D (external-field
    // contributions have already been subtracted again). theta-Faraday consumes
    // a_E = E* + D, whose dissipative part D enters the pairing as the -D.J_p
    // heating term below.
    ablastr::fields::VectorField Efld =
        m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, lev);
    // With external (vector-potential) fields, Ohm's law -- hence the identity
    // E.J_e = u_e.grad(pe) + eta J.J_e that the reversible pairing rests on -- holds
    // for the Ohm's-law total field E_Ohm, of which the registry keeps the internal
    // part a_E = E_Ohm - w(rho) E_ext (HybridExtSubWeight: w = 1 in the plasma, 0
    // below the density floor, where the vacuum Ohm's law is read as the plasma-
    // induced part and E_ext adds to it). The pairing must therefore read
    // E* + w(rho) E_ext = E_Ohm - D. Pairing against the internal E* alone drops
    // E_ext.J_e in the plasma: with a time-dependent coil drive (theta-pinch
    // formation) that is the inductive work on the skin-current electrons, O(1)
    // of the compression work (observed as a shell cooled to the Te floor inside
    // the edge current and a hot rim where the current reverses); adding the full
    // E_ext instead double counts it below the floor, where the skin current
    // peaks in rarefied nodes (rim 5 -> 100 eV in 1 us, Joule-gate independent).
    // Static external fields (E_ext = 0) are unaffected. hybrid_E_fp_external
    // holds E_ext at the theta time here (UpdateHybridExternalFields in
    // ComputeRHS), the same field the ions were pushed with; the weight uses the
    // same staggered rho interpolation and floor parameters as the Ohm solve.
    if (m_add_external_fields) {
        const amrex::Real rho_f  = m_hybrid_pic_model->m_n_floor * PhysConst::q_e;
        const amrex::Real flw    = m_hybrid_pic_model->m_n_floor_smooth_width * rho_f;
        const amrex::GpuArray<int, 3> nodal3 = {1, 1, 1};
        const amrex::GpuArray<int, 3> crs3   = {1, 1, 1};
        const std::array<amrex::GpuArray<int, 3>, 3> Estag = {
            m_hybrid_pic_model->Ex_IndexType, m_hybrid_pic_model->Ey_IndexType,
            m_hybrid_pic_model->Ez_IndexType};
        for (int n = 0; n < 3; ++n) {
            const amrex::MultiFab& Eext = *m_WarpX->m_fields.get(
                FieldType::hybrid_E_fp_external, Direction{n}, lev);
            if (!m_E_pair[n]) {
                m_E_pair[n] = std::make_unique<amrex::MultiFab>(
                    Efld[n]->boxArray(), Efld[n]->DistributionMap(),
                    Efld[n]->nComp(), Efld[n]->nGrowVect());
            }
            amrex::MultiFab::Copy(*m_E_pair[n], *Efld[n], 0, 0,
                                  Efld[n]->nComp(), Efld[n]->nGrowVect());
            // valid region plus the ghosts rho can serve (its stagger interpolation
            // reads one neighbour); the pairing reads E at its own node
            const int ng = amrex::min(Efld[n]->nGrowVect().min(),
                                      rho->nGrowVect().min() - 1);
            const amrex::GpuArray<int, 3> stag = Estag[n];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(*m_E_pair[n], amrex::TilingIfNotGPU());
                 mfi.isValid(); ++mfi) {
                const amrex::Box bx = mfi.growntilebox(amrex::max(ng, 0));
                auto const& ep = m_E_pair[n]->array(mfi);
                auto const& ee = Eext.const_array(mfi);
                auto const& rr = rho->const_array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    const amrex::Real rho_e = Interp(rr, nodal3, stag, crs3, i, j, k, 0);
                    ep(i,j,k) += HybridExtSubWeight(rho_e, rho_f, flw) * ee(i,j,k);
                });
            }
        }
        Efld = {m_E_pair[0].get(), m_E_pair[1].get(), m_E_pair[2].get()};
    }
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
        // first use: the pressure field holds its (seeded) initialization
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
    if (m_pe_unknown && !m_pe_rhs) {
        m_pe_rhs = std::make_unique<amrex::MultiFab>(
            pe->boxArray(), pe->DistributionMap(), pe->nComp(), pe->nGrowVect());
    }

    const auto dxi = geom.InvCellSizeArray();
    [[maybe_unused]] const amrex::Real rmin = geom.ProbLo(0);
    [[maybe_unused]] const amrex::Real dr = geom.CellSize(0);
    const amrex::Real theta_dt = m_theta * m_dt;
    const amrex::Real gamma = m_hybrid_pic_model->m_gamma;
    const amrex::Real rho_floor = m_hybrid_pic_model->m_n_floor * PhysConst::q_e;
    const amrex::Real q_e = PhysConst::q_e;
    // width of the C1 positivity floor (HybridPeFloor) as a fraction of the floored-adiabat
    // pe: a hard max(pe, 0) in the residual is non-differentiable where floored cells ride
    // it and stalls Newton; the hard clamp remains in the post-step halo pin
    constexpr amrex::Real pe_eps_fac = 0.01_rt;
    const amrex::Real pe_eps = pe_eps_fac * m_hybrid_pic_model->m_n_floor
        * m_hybrid_pic_model->m_elec_temp
        * std::pow(m_hybrid_pic_model->m_n_floor / m_hybrid_pic_model->m_n0_ref,
                   gamma - 1._rt);
    // marker-CFL cap on u_e: the theta-centered fixed point contracts only where
    // |u_e| k_max theta dt < 1, which the fictitious floored-edge u_e = J/(e n_floor)
    // violates (the physical interior u_e sits well below the cap)
    amrex::Real dx_min = geom.CellSize(0);
    for (int d = 1; d < AMREX_SPACEDIM; ++d) {
        dx_min = std::min(dx_min, geom.CellSize(d));
    }
    const amrex::Real ue_cap = m_pe_ue_cap_fac * dx_min / theta_dt;
    // symmetry-plane closure of the non-periodic axial faces (RZ collocated), see the kernel
    const bool wall_mirror = (m_pe_wall_mirror >= 1);

    // Holmstrom vacuum region: blend the advanced pressure toward the floored adiabat with
    // the same (statistics-aware) vacuum weight Ohm's law uses. In near-empty cells u_e =
    // (J_i - J)/rho is noise and the transport/compression terms ran away (formation
    // 21.6 us: pe x100 in one step at an axis node with 0-2 macro-ions, grad-pe E 1e6 V/m,
    // then a B/J spike the solver could not follow). Linear in pe, smooth in rho.
    const bool vac_blend = m_hybrid_pic_model->m_holmstrom_vacuum_region;
    const amrex::Real vac_floor_w = m_hybrid_pic_model->m_n_floor_smooth_width * rho_floor;
    const amrex::Real vac_rref = m_hybrid_pic_model->m_vacuum_weight_r_ref;
    const amrex::Real vac_maxf = m_hybrid_pic_model->m_vacuum_weight_max_factor;
    const amrex::Real ad_n0 = m_hybrid_pic_model->m_n0_ref;
    const amrex::Real ad_T0 = m_hybrid_pic_model->m_elec_temp;

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

    // Joule heating on the collocated grid: deposit the positive-definite Q (HybridPeJouleQ,
    // see m_Q_diss) instead of -D.Jp, whose hyper-resistive part is sign-indefinite pointwise
    const bool q_posdef = jheat && J_nodal;
    // Te-threshold Joule gate (implicit counterpart of the explicit QDSMC redirect above
    // joule_redirect_Te_threshold): the anomalous (Chodura ~ 1/sqrt(n)) resistivity times
    // the edge skin current heats the rarefied rim nodes at rates the few electrons there
    // cannot absorb (formation: 5 -> 80 eV by 0.5 us with the gate off, unbounded), so
    // above Tc the dissipative deposit is withheld from pe. Frozen at pe^n: no Newton
    // dependence. Positive-definite (collocated) deposit only, see m_joule_Te_cutoff_eV.
    const bool jgate = jheat && m_joule_Te_cutoff_eV > 0.0_rt;
    if (jgate) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(q_posdef,
            "implicit_evolve.joule_Te_cutoff requires the collocated grid");
    }
    if (q_posdef) {
        if (!m_Q_diss) {
            m_Q_diss = std::make_unique<amrex::MultiFab>(
                pe->boxArray(), pe->DistributionMap(), 1,
                amrex::IntVect::TheZeroVector());
            m_Q_diss->setVal(0.0_rt);
        }
        const ablastr::fields::VectorField Bf =
            m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, lev);
        const auto eta_ex  = m_hybrid_pic_model->m_eta;
        const auto etah_ex = m_hybrid_pic_model->m_eta_h;
        const bool inc_hyp = m_hybrid_pic_model->m_include_hyper_resistivity_term;
        const amrex::Real t_now = a_theta_time;
        const amrex::Dim3 qlo = dlo, qhi = dhi;
        const amrex::GpuArray<bool, 3> qper = is_per;
        const auto dxiq = geom.InvCellSizeArray();
        const amrex::Real jcut   = m_joule_Te_cutoff_eV;
        const amrex::Real jcut_w = m_joule_Te_cutoff_width * m_joule_Te_cutoff_eV;
        // Joule redirect to the ions: the gated-out fraction (1 - sJ) Q dt is staged per
        // charged species as the (2/3)-scaled per-ion energy E_s = (2/3) Z_s (1 - sJ) Q dt / n_e
        // [J] (explicit QDSMC convention) for QDSMCApplyIonHeating in FinishStep. Written by
        // residual evaluations only (the last one is the accepted state), never by Jacobian
        // probes. Q is the theta-state power density, dt the (sub)step, matching the sJ Q dt
        // that pe receives over the step.
        const bool do_redir = m_joule_redirect && jgate && !a_from_jacobian;
        if (do_redir && !m_ion_redirect_E) {
            auto & mypc = m_WarpX->GetPartContainer();
            m_n_ion_species = 0;
            for (auto const & nm : mypc.GetSpeciesNames()) {
                auto & pc = mypc.GetParticleContainerFromName(nm);
                if (pc.getCharge() == 0._prt) { continue; }
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_n_ion_species < kMaxRedirectSpecies,
                    "implicit_evolve.joule_redirect_to_ions: too many charged species");
                m_ion_Z[m_n_ion_species++] =
                    static_cast<amrex::Real>(pc.getCharge() / PhysConst::q_e);
            }
            m_ion_redirect_E = std::make_unique<amrex::MultiFab>(
                pe->boxArray(), pe->DistributionMap(), std::max(m_n_ion_species, 1),
                amrex::IntVect::TheZeroVector());
            m_ion_redirect_E->setVal(0.0_rt);
            m_Q_redir = std::make_unique<amrex::MultiFab>(
                pe->boxArray(), pe->DistributionMap(), 1, amrex::IntVect::TheZeroVector());
            m_Q_redir->setVal(0.0_rt);
        }
        const int n_ion = do_redir ? m_n_ion_species : 0;
        const amrex::GpuArray<amrex::Real, kMaxRedirectSpecies> Zs = m_ion_Z;
        const amrex::Real dt_full = m_dt;
        // the redirect is weighted by the Holmstrom vacuum weight of the node (the same
        // C1 weight that blends the pe advance to the adiabat): in the floored halo the
        // Joule power is E = eta J of the vacuum model, not a plasma process, and per ion
        // it would be divided by the floored n_e (2026-09-17, 3D formation: keV ions off
        // the face planes, guard-cell abort at step 4)
        const amrex::Real drq = geom.CellSize(0);
        const amrex::Real rminq = geom.ProbLo(0);
        for (amrex::MFIter mfi(*m_Q_diss, amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            const amrex::Box tb = mfi.tilebox();
            auto const& q   = m_Q_diss->array(mfi);
            auto const& jpx = Jp[0]->const_array(mfi);
            auto const& jpy = Jp[1]->const_array(mfi);
            auto const& jpz = Jp[2]->const_array(mfi);
            auto const& bxa = Bf[0]->const_array(mfi);
            auto const& bya = Bf[1]->const_array(mfi);
            auto const& bza = Bf[2]->const_array(mfi);
            auto const& rr  = rho->const_array(mfi);
            auto const& pen = m_pe_old->const_array(mfi);
            amrex::Array4<amrex::Real> rd, qr;
            if (do_redir) {
                rd = m_ion_redirect_E->array(mfi);
                qr = m_Q_redir->array(mfi);
            }
            // shared with the explicit fluid pe solver (ElectronPressureFlux.H)
            amrex::ParallelFor(tb,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    amrex::Real sJ = 1.0_rt;
                    if (jgate) {
                        // Te^n in eV = pe^n / (e n) = pe^n / rho (floored rho)
                        const amrex::Real Te_eV =
                            pen(i,j,k) / amrex::max(rr(i,j,k,0), rho_floor);
                        sJ = 0.5_rt*(1.0_rt - std::tanh((Te_eV - jcut)/jcut_w));
                    }
                    const amrex::Real Q = HybridPeJouleQ(i, j, k, jpx, jpy, jpz, bxa, bya, bza,
                                                         rr, eta_ex, etah_ex, inc_hyp, t_now,
                                                         dxiq, qlo, qhi, qper);
                    q(i,j,k) = sJ * Q;
                    if (do_redir) {
                        amrex::Real wv = 1.0_rt;
                        if (vac_blend) {
#if defined(WARPX_DIM_RZ)
                            const amrex::Real r_w = rminq + i*drq;
#else
                            const amrex::Real r_w = 0.0_rt;
#endif
                            const amrex::Real rf = HybridVacuumFloor(rho_floor, r_w, vac_rref,
                                                                     vac_maxf, drq);
                            wv = HybridExtSubWeight(rr(i,j,k,0), rf, vac_floor_w*(rf/rho_floor));
                        }
                        const amrex::Real W = wv * (1.0_rt - sJ) * Q;   // withheld power density
                        qr(i,j,k) = W;
                        const amrex::Real ne =
                            amrex::max(rr(i,j,k,0), rho_floor) / PhysConst::q_e;
                        for (int s = 0; s < n_ion; ++s) {
                            rd(i,j,k,s) = (2.0_rt/3.0_rt) * Zs[s] * W * dt_full / ne;
                        }
                    }
                });
        }
    }

    // electron conduction (numeric kappa_e or a kappa_e(rho,Te) expression); pe and rho are
    // nodal on every grid, RZ m = 0 uses the area-weighted radial divergence below
    const bool has_kappa = m_hybrid_pic_model->m_has_kappa_e;
    const bool has_kexpr = m_hybrid_pic_model->m_has_kappa_e_expression;
    const amrex::Real kappa_e = m_hybrid_pic_model->m_kappa_e;
    const auto kappa_ex = m_hybrid_pic_model->m_kappa;
    if (has_kappa) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            pe->nGrowVect().min() >= 1 && rho->nGrowVect().min() >= 1,
            "hybrid_pic_model.kappa_e (electron conduction) needs >= 1 "
            "ghost cell on pe and rho");
    }

    const int pe_adv = m_pe_advection;
    if (pe_adv != 0) {
        // the MUSCL face reconstruction reads the second node beyond a periodic box face
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(pe->nGrowVect().min() >= 2,
            "implicit_evolve.pe_advection = vanalbada needs >= 2 pe ghost cells");
    }
    // face reconstruction parameters, shared with the explicit fluid pe solver
    HybridPeFluxParams pfp;
    pfp.adv = pe_adv;

    // The update must be theta-centered in pe as well: an explicit (forward) pe in the
    // RHS integrates the pressure side of the ion-acoustic oscillation with forward
    // Euler and is numerically unstable (growth ~ exp(omega^2 dt t / 2)). The update is
    // linear in pe, so a short fixed-point iteration (contraction ~ theta*omega*dt)
    // converges the theta-centered value: pe_rhs = (1-theta)*pe^n + theta*pe_iter.
    // Contraction per cycle is ~theta*k_max*Cs*dt; 4 cycles verified sufficient
    // (8 cycles bit-reproduces the energy history on the cold-beam FGI test).
    const int n_pe_iters = m_pe_unknown ? 1 : 4;
    amrex::MultiFab* pe_out = pe;
    if (m_pe_unknown) {
        // pe is a Newton unknown: evaluate the theta update once at the iterate (held by the
        // pressure field for Ohm's law) into m_pe_rhs for the residual row. Non-periodic
        // domain-face nodes the kernel skips keep pe^n, so their rows are identities
        // kappa*(pe - pe^n); the RZ collocated axial faces are advanced (mirror closure).
        amrex::MultiFab::Copy(*m_pe_scratch, *pe, 0, 0, pe->nComp(), pe->nGrowVect());
        amrex::MultiFab::Copy(*m_pe_rhs, *m_pe_old, 0, 0, pe->nComp(), pe->nGrowVect());
        // freeze the flux stencil's wall values at pe^n as the in-loop path does: a wall
        // value slaved to the interior iterate feeds the near-wall flux back onto itself
        // (unstable wall layer); the wall rows themselves stay identities. Exception: the
        // RZ collocated axial faces are advanced as symmetry planes (mirror closure in the
        // kernel), so their nodes are regular unknowns and must not be frozen here.
#if defined(WARPX_DIM_RZ)
        const bool z_mirror = J_nodal && (m_pe_wall_mirror >= 1);
#else
        const bool z_mirror = false;
#endif
        for (amrex::MFIter mfi(*m_pe_scratch); mfi.isValid(); ++mfi) {
            const amrex::Box vbx = mfi.validbox();
            auto const& s  = m_pe_scratch->array(mfi);
            auto const& p0 = m_pe_old->const_array(mfi);
            const amrex::Dim3 dl = dlo, dh = dhi;
            const amrex::GpuArray<bool, 3> per = is_per;
            amrex::ParallelFor(vbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    amrex::ignore_unused(j, k);
                    const bool face =
                        ((i == dl.x || i == dh.x) && !per[0])
#if (AMREX_SPACEDIM >= 2)
                        || ((j == dl.y || j == dh.y) && !per[1] && !z_mirror)
#endif
#if (AMREX_SPACEDIM == 3)
                        || ((k == dl.z || k == dh.z) && !per[2])
#endif
                        ;
                    if (face) { s(i,j,k) = p0(i,j,k); }
                });
        }
        pe_out = m_pe_rhs.get();
    } else {
        amrex::MultiFab::Copy(*m_pe_scratch, *m_pe_old, 0, 0, pe->nComp(), pe->nGrowVect());
    }
    // optional per-node term breakdown (implicit_evolve.pe_debug_nodes), nonlinear
    // evaluations only; slots of 16 reals per node, filled inside the kernel
    const int n_dbg = static_cast<int>(m_pe_debug_nodes.size()) / 2;
    amrex::Gpu::DeviceVector<amrex::Real> dbg_dev(std::max(1, 16*n_dbg), 0.0_rt);
    amrex::Gpu::DeviceVector<int> dbg_ij(std::max(1, 2*n_dbg), -1);
    if (n_dbg > 0) {
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, m_pe_debug_nodes.begin(),
                         m_pe_debug_nodes.begin() + 2*n_dbg, dbg_ij.begin());
    }
    amrex::Real* const dbgp = (n_dbg > 0 && !a_from_jacobian) ? dbg_dev.data() : nullptr;
    const int* const dbgij = dbg_ij.data();

    for (int pe_it = 0; pe_it < n_pe_iters; ++pe_it) {

    // EB: freeze covered nodes at pe^n (identity rows with pe_unknown, no-op otherwise) so
    // the wall-adjacent Ohm rows do not read an advected floor-density state through
    // grad(pe); the nodal flag is the component-0 E flag of the masked Ohm solves
    const amrex::iMultiFab* eb_pe_flag = EB::enabled()
        ? m_WarpX->GetEBUpdateEFlag()[0][0].get() : nullptr;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*pe, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        amrex::Array4<amrex::Real>       const& pe_arr  = pe_out->array(mfi);
        amrex::Array4<amrex::Real const> const& pe_it_arr = m_pe_scratch->const_array(mfi);
        amrex::Array4<amrex::Real const> const& pe0     = m_pe_old->const_array(mfi);
        amrex::Array4<int const> ebp;
        if (eb_pe_flag) { ebp = eb_pe_flag->const_array(mfi); }
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
        amrex::Array4<amrex::Real const> Qd;
        if (q_posdef) { Qd = m_Q_diss->const_array(mfi); }

        const amrex::Box tb = mfi.tilebox(amrex::IntVect::TheNodeVector());

        amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k) {

            // EB-covered node: freeze at pe^n (see eb_pe_flag above)
            if (ebp && ebp(i,j,k) == 0) {
                pe_arr(i,j,k) = pe0(i,j,k);
                return;
            }

            // smooth (tanh) marker-CFL cap on u_e, see ue_cap above: a hard min/max makes
            // the residual non-differentiable where floored cells ride the cap and Newton
            // grinds there; tanh keeps the bound and the small-u identity response
            auto uclamp = [&] (amrex::Real u) {
                return ue_cap * std::tanh(u / ue_cap);
            };
            // Non-periodic axial faces are symmetry planes for the collocated RZ advance
            // (PMC fields + reflecting ions at z_min; the PMC parity fold treats z_max the
            // same): axial index reflected about the face, axial electron flux odd (zero
            // through the wall), scalars even. Before 2026-09-17 the boundary planes were
            // skipped, leaving pe there frozen at pe(t=0) while Ohm's law kept differencing
            // it: with the corrected Polaris coils the open-field column reached z = 0 at
            // ~20 us and the wall plane carried Te 50-100 eV (= pe(0)/n(t)) against 15-50
            // eV one plane in, |E_r| 2.6e5 V/m at the wall, solver death at 22.4 us.
            auto jm = [&] (int m) {
                if (is_per[1] || !wall_mirror) { return m; }
                if (m < dlo.y) { return 2*dlo.y - m; }
                if (m > dhi.y) { return 2*dhi.y - m; }
                return m;
            };
            auto sz = [&] (int m) {
                return (!is_per[1] && wall_mirror && (m < dlo.y || m > dhi.y))
                    ? -1.0_rt : 1.0_rt;
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
            // pe at the face between the p0 and p1 nodes (pm1/p2 = next nodes outward),
            // reconstructed from the upwind side (ElectronPressureFlux.H, shared with the
            // explicit fluid pe solver)
            auto pe_face = [&] (amrex::Real uf, amrex::Real pm1, amrex::Real p0,
                                amrex::Real p1, amrex::Real p2) {
                return HybridPeFace(pfp, uf, pm1, p0, p1, p2);
            };
            // electron heat conduction +div(kappa_e grad(pe/n)) in conservative face-flux
            // form, face conductivity = average of the adjacent nodal values; RZ m = 0 uses
            // the area-weighted radial divergence (axis: r F -> 0)
            auto Tn = [&] (int ii, int jj, int kk) {
                return pec(ii,jj,kk) * q_e
                    / amrex::max(rho_arr(ii,jj,kk,0), rho_floor);
            };
            auto kapn = [&] (int ii, int jj, int kk) {
                if (!has_kexpr) { return kappa_e; }
                return kappa_ex(amrex::max(rho_arr(ii,jj,kk,0), rho_floor),
                                Tn(ii,jj,kk) / q_e);   // Te in eV
            };
            // conduction face flux between two adjacent nodes
            auto Fk = [&] (amrex::Real T0, amrex::Real T1,
                           amrex::Real k0, amrex::Real k1,
                           amrex::Real dxi_f) {
                return 0.5_rt*(k0 + k1) * (T1 - T0) * dxi_f;
            };
            amrex::Real cond = 0.0_rt;
            if (has_kappa) {
#if defined(WARPX_DIM_1D_Z)
                cond = (Fk(Tn(i,j,k),   Tn(i+1,j,k),
                           kapn(i,j,k), kapn(i+1,j,k), dxi[0])
                      - Fk(Tn(i-1,j,k), Tn(i,j,k),
                           kapn(i-1,j,k), kapn(i,j,k), dxi[0])) * dxi[0];
#elif defined(WARPX_DIM_XZ)
                cond = (Fk(Tn(i,j,k),   Tn(i+1,j,k),
                           kapn(i,j,k), kapn(i+1,j,k), dxi[0])
                      - Fk(Tn(i-1,j,k), Tn(i,j,k),
                           kapn(i-1,j,k), kapn(i,j,k), dxi[0])) * dxi[0]
                     + (Fk(Tn(i,j,k),   Tn(i,j+1,k),
                           kapn(i,j,k), kapn(i,j+1,k), dxi[1])
                      - Fk(Tn(i,j-1,k), Tn(i,j,k),
                           kapn(i,j-1,k), kapn(i,j,k), dxi[1])) * dxi[1];
#elif defined(WARPX_DIM_3D)
                cond = (Fk(Tn(i,j,k),   Tn(i+1,j,k),
                           kapn(i,j,k), kapn(i+1,j,k), dxi[0])
                      - Fk(Tn(i-1,j,k), Tn(i,j,k),
                           kapn(i-1,j,k), kapn(i,j,k), dxi[0])) * dxi[0]
                     + (Fk(Tn(i,j,k),   Tn(i,j+1,k),
                           kapn(i,j,k), kapn(i,j+1,k), dxi[1])
                      - Fk(Tn(i,j-1,k), Tn(i,j,k),
                           kapn(i,j-1,k), kapn(i,j,k), dxi[1])) * dxi[1]
                     + (Fk(Tn(i,j,k),   Tn(i,j,k+1),
                           kapn(i,j,k), kapn(i,j,k+1), dxi[2])
                      - Fk(Tn(i,j,k-1), Tn(i,j,k),
                           kapn(i,j,k-1), kapn(i,j,k), dxi[2])) * dxi[2];
#elif defined(WARPX_DIM_RZ)
                const amrex::Real rC = rmin + i*dr;
                const bool onax = (rC < 0.5_rt*dr);
                const amrex::Real Ak = onax ? 0.125_rt*dr*dr : rC*dr;
                const amrex::Real Fkrp = Fk(Tn(i,j,k), Tn(i+1,j,k),
                                            kapn(i,j,k), kapn(i+1,j,k),
                                            dxi[0]);
                const amrex::Real rFkrm = onax ? 0.0_rt :
                    (rC - 0.5_rt*dr) * Fk(Tn(i-1,j,k), Tn(i,j,k),
                                          kapn(i-1,j,k), kapn(i,j,k),
                                          dxi[0]);
                // axial neighbours reflected about non-periodic z faces (jm): the heat
                // flux through a symmetry plane vanishes
                cond = ((rC + 0.5_rt*dr)*Fkrp - rFkrm) / Ak
                     + (Fk(Tn(i,j,k),   Tn(i,jm(j+1),k),
                           kapn(i,j,k), kapn(i,jm(j+1),k), dxi[1])
                      - Fk(Tn(i,jm(j-1),k), Tn(i,j,k),
                           kapn(i,jm(j-1),k), kapn(i,j,k), dxi[1])) * dxi[1];
#endif
            }

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
            if (J_nodal && pe_adv != 0) {
                // limited upwind face flux (conservative: telescopes exactly)
                auto Fz_face = [&] (int m) {
                    const amrex::Real uf = 0.5_rt*(ue_z(m,j,k) + ue_z(m+1,j,k));
                    return uf * pe_face(uf, pec(m-1,j,k), pec(m,j,k),
                                        pec(m+1,j,k), pec(m+2,j,k));
                };
                divF += (Fz_face(i) - Fz_face(i-1)) * dxi[0];
            } else if (J_nodal) {
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
                   - (q_posdef ? Qd(i,j,k)
                      : (Dx(i,j,k) * Jpx(i,j,k)
                       + Dy(i,j,k) * Jpy(i,j,k)
                       + Dz(i,j,k) * Jpz(i,j,k)));
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
            if (J_nodal && pe_adv != 0) {
                // limited upwind face fluxes (conservative: telescope exactly)
                auto Fx_face = [&] (int m) {
                    const amrex::Real uf = 0.5_rt*(ue_x(m,j,k) + ue_x(m+1,j,k));
                    return uf * pe_face(uf, pec(m-1,j,k), pec(m,j,k),
                                        pec(m+1,j,k), pec(m+2,j,k));
                };
                auto Fz_face = [&] (int m) {
                    const amrex::Real uf = 0.5_rt*(ue_z(i,m,k) + ue_z(i,m+1,k));
                    return uf * pe_face(uf, pec(i,m-1,k), pec(i,m,k),
                                        pec(i,m+1,k), pec(i,m+2,k));
                };
                divF += (Fx_face(i) - Fx_face(i-1)) * dxi[0]
                      + (Fz_face(j) - Fz_face(j-1)) * dxi[1];
            } else if (J_nodal) {
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
                   - (q_posdef ? Qd(i,j,k)
                      : (Dx(i,j,k) * Jpx(i,j,k)
                       + Dy(i,j,k) * Jpy(i,j,k)
                       + Dz(i,j,k) * Jpz(i,j,k)));
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
            // m = 0 RZ. The enthalpy flux uses the conservative area-weighted
            // form, so the volume-weighted grid sum telescopes exactly
            // (including through the axis, where [r F] -> 0).
            //   collocated (J_nodal): face velocities averaged from the nodal
            //   ue, limited upwind face pe (pe_adv != 0) or the central average,
            //   and the nodal full-Ohm work pairing of the Cartesian collocated
            //   branches. (Before 2026-09-14 this branch applied the Yee edge
            //   formulas to nodal data, i.e. half-cell-shifted flux velocities:
            //   an O(1) flux error at sharp density edges that cooled the
            //   formation's plasma edge to 0.4 eV within 0.5 us.)
            //   Yee: W carries the full-Ohm pairing -E*.J_e + eta*Jp^2 evaluated
            //   per staggered component at its own location (matching the
            //   push-field subtraction), volume-split from the edges to the
            //   nodes so that sum_nodes(A W) = sum_edges(A q). The Hall channel
            //   and the RZ Faraday ledger pair at interpolation order.
            amrex::ignore_unused(Fx, Fz);
            // collocated (J_nodal): the axial symmetry-plane closure (jm/sz) advances the
            // boundary planes too; the Yee branch keeps skipping them
            if ((j <= dlo.y || j >= dhi.y) && !is_per[1] && !(J_nodal && wall_mirror)) { return; }
            if (i >= dhi.x) { return; }                        // outer r boundary
            const amrex::Real r = rmin + i*dr;
            const bool on_axis = (r < 0.5_rt*dr);
            if (i <= dlo.x && !on_axis) { return; }            // annular inner boundary
            const amrex::Real A_i  = on_axis ? 0.125_rt*dr*dr : r*dr;
            const amrex::Real A_ip = (r + 0.5_rt*dr)*dr;       // r-edge at i+1/2
            const amrex::Real A_im = (r - 0.5_rt*dr)*dr;       // r-edge at i-1/2

            if (J_nodal) {
                // Flux-consistent enthalpy transport: F = Gamma_face * (pe/rho)_face,
                // Gamma = J_i - J_p the nodal electron charge flux (no division by
                // the rho of a rarefied node) and pe/rho the advected quantity
                // (kB Te per charge), limited-upwind (pe_adv != 0) or central. The
                // pressure then travels with the particle flux: a near-floor rim
                // receives pressure only in proportion to the particles it
                // receives. (u_face * pe_upwind, with u = Gamma/rho averaged between
                // a bulk node and a rarefied node whose few outgoing tail ions
                // give u ~ v_th, carried the BULK pressure into the rim at ~v_th/2
                // while the density did not follow: Te_rim 33 eV by 0.4 us, 450 eV
                // by 1.2 us in the formation runs.) The marker-CFL cap on u_e does
                // not apply to this form.
                // axial indices reflected about non-periodic z faces (jm), the axial
                // charge flux odd there (sz): F_z(face beyond the wall) = -F_z(face inside)
                auto Gr = [&] (int m) { return Jx(m,j,k) - Jpx(m,j,k); };
                auto Gz = [&] (int m) {
                    const int mm = jm(m);
                    return sz(m) * (Jz(i,mm,k) - Jpz(i,mm,k));
                };
                auto Tq = [&] (int ii, int jj) {
                    const int jr = jm(jj);
                    return pec(ii,jr,k) / amrex::max(rho_arr(ii,jr,k,0), rho_floor);
                };
                // face between nodes m and m+1 (radial) / m and m+1 (axial);
                // the axis-adjacent radial face keeps the central average (the
                // limiter would read the axis ghost)
                auto Fr_face = [&] (int m) {
                    const amrex::Real gf = 0.5_rt*(Gr(m) + Gr(m+1));
                    const amrex::Real tf = (pe_adv != 0 && m > dlo.x)
                        ? pe_face(gf, Tq(m-1,j), Tq(m,j), Tq(m+1,j), Tq(m+2,j))
                        : 0.5_rt*(Tq(m,j) + Tq(m+1,j));
                    return gf * tf;
                };
                auto Fz_face = [&] (int m) {
                    const amrex::Real gf = 0.5_rt*(Gz(m) + Gz(m+1));
                    const amrex::Real tf = (pe_adv != 0)
                        ? pe_face(gf, Tq(i,m-1), Tq(i,m), Tq(i,m+1), Tq(i,m+2))
                        : 0.5_rt*(Tq(i,m) + Tq(i,m+1));
                    return gf * tf;
                };
                amrex::ignore_unused(ue_x, ue_z);
                const amrex::Real rFrm = on_axis ? 0.0_rt : (r - 0.5_rt*dr) * Fr_face(i-1);
                divF += ((r + 0.5_rt*dr) * Fr_face(i) - rFrm) / A_i
                      + (Fz_face(j) - Fz_face(j-1)) * dxi[1];
                W += Ex_arr(i,j,k) * (Jx(i,j,k) - Jpx(i,j,k))
                   + Ey_arr(i,j,k) * (Jy(i,j,k) - Jpy(i,j,k))
                   + Ez_arr(i,j,k) * (Jz(i,j,k) - Jpz(i,j,k))
                   - (q_posdef ? Qd(i,j,k)
                      : (Dx(i,j,k) * Jpx(i,j,k)
                       + Dy(i,j,k) * Jpy(i,j,k)
                       + Dz(i,j,k) * Jpz(i,j,k)));
                amrex::ignore_unused(A_ip, A_im);
            } else {
            amrex::ignore_unused(ue_x, ue_z, pe_face);
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
            }   // Yee branch
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
            if (J_nodal && pe_adv != 0) {
                // limited upwind face fluxes (conservative: telescope exactly)
                auto Fx_face = [&] (int m) {
                    const amrex::Real uf = 0.5_rt*(ue_x(m,j,k) + ue_x(m+1,j,k));
                    return uf * pe_face(uf, pec(m-1,j,k), pec(m,j,k),
                                        pec(m+1,j,k), pec(m+2,j,k));
                };
                auto Fy_face = [&] (int m) {
                    const amrex::Real uf = 0.5_rt*(ue_y(i,m,k) + ue_y(i,m+1,k));
                    return uf * pe_face(uf, pec(i,m-1,k), pec(i,m,k),
                                        pec(i,m+1,k), pec(i,m+2,k));
                };
                auto Fz_face = [&] (int m) {
                    const amrex::Real uf = 0.5_rt*(ue_z(i,j,m) + ue_z(i,j,m+1));
                    return uf * pe_face(uf, pec(i,j,m-1), pec(i,j,m),
                                        pec(i,j,m+1), pec(i,j,m+2));
                };
                divF += (Fx_face(i) - Fx_face(i-1)) * dxi[0]
                      + (Fy_face(j) - Fy_face(j-1)) * dxi[1]
                      + (Fz_face(k) - Fz_face(k-1)) * dxi[2];
            } else if (J_nodal) {
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
                   - (q_posdef ? Qd(i,j,k)
                      : (Dx(i,j,k) * Jpx(i,j,k)
                       + Dy(i,j,k) * Jpy(i,j,k)
                       + Dz(i,j,k) * Jpz(i,j,k)));
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
            amrex::ignore_unused(Fx, Fz, divF, dlo, dhi, is_per, dxi,
                                 pe_face, Tn, kapn, Fk);
#endif
            const amrex::Real pe_new = pe0(i,j,k)
                - theta_dt * (gamma * divF + (gamma - 1._rt) * (W - cond));
            if (dbgp) {
                for (int q = 0; q < n_dbg; ++q) {
                    if (i == dbgij[2*q] && j == dbgij[2*q+1]) {
                        amrex::Real* d = dbgp + 16*q;
                        d[0] = Ex_arr(i,j,k); d[1] = Ey_arr(i,j,k); d[2] = Ez_arr(i,j,k);
                        d[3] = Jx(i,j,k) - Jpx(i,j,k);
                        d[4] = Jy(i,j,k) - Jpy(i,j,k);
                        d[5] = Jz(i,j,k) - Jpz(i,j,k);
                        d[6] = q_posdef ? Qd(i,j,k)
                                        : (Dx(i,j,k)*Jpx(i,j,k) + Dy(i,j,k)*Jpy(i,j,k)
                                           + Dz(i,j,k)*Jpz(i,j,k));
                        d[7] = divF; d[8] = W; d[9] = cond;
                        d[10] = pe0(i,j,k); d[11] = pe_new; d[12] = rho_arr(i,j,k,0);
                        d[13] = pe_it_arr(i,j,k);
                        d[14] = Jpx(i,j,k); d[15] = Jpy(i,j,k);
                    }
                }
            }
            amrex::Real pe_fin = pe_new;
            if (vac_blend) {
#if defined(WARPX_DIM_RZ)
                const amrex::Real r_w = rmin + i*dr;
#else
                const amrex::Real r_w = 0.0_rt;   // no radial statistics scaling
#endif
                const amrex::Real rf = HybridVacuumFloor(rho_floor, r_w, vac_rref, vac_maxf, dr);
                const amrex::Real wv = HybridExtSubWeight(rho_arr(i,j,k,0), rf,
                                                          vac_floor_w*(rf/rho_floor));
                const amrex::Real pe_ad = ElectronPressure::get_pressure(
                    ad_n0, ad_T0, gamma, amrex::max(rho_arr(i,j,k,0), rho_floor));
                pe_fin = wv*pe_new + (1.0_rt - wv)*pe_ad;
            }
            // C1 positivity floor with compact support (exact identity for pe >= 2*eps,
            // hence energy-neutral there), shared with the explicit fluid solver
            pe_arr(i,j,k) = HybridPeFloor(pe_fin, pe_eps);
        });
    }
    if (dbgp && pe_it == n_pe_iters - 1) {
        amrex::Gpu::streamSynchronize();
        std::vector<amrex::Real> h(16*n_dbg);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, dbg_dev.begin(),
                         dbg_dev.begin() + 16*n_dbg, h.begin());
        for (int q = 0; q < n_dbg; ++q) {
            const amrex::Real* d = h.data() + 16*q;
            const amrex::Real EG = d[0]*d[3] + d[1]*d[4] + d[2]*d[5];
            amrex::AllPrint() << "pe_debug (" << m_pe_debug_nodes[2*q] << ","
                << m_pe_debug_nodes[2*q+1] << ") t=" << a_theta_time
                << " Epair=(" << d[0] << "," << d[1] << "," << d[2] << ")"
                << " G=Ji-Jp=(" << d[3] << "," << d[4] << "," << d[5] << ")"
                << " Jp=(" << d[14] << "," << d[15] << ")"
                << " rho=" << d[12] << " pe0=" << d[10] << " pe_it=" << d[13]
                << " pe_new=" << d[11]
                << " | dpe: E.G " << -theta_dt*(gamma - 1._rt)*EG
                << " Qd " << theta_dt*(gamma - 1._rt)*d[6]
                << " flux " << -theta_dt*gamma*d[7]
                << " cond " << theta_dt*(gamma - 1._rt)*d[9]
                << " (E.G=" << EG << " Qd=" << d[6] << " divF=" << d[7] << ")\n";
        }
    }

    // Duplicated nodal points on box faces are written redundantly by each box;
    // force them single-valued before they are consumed (Ohm's law, next iterate).
    pe_out->OverrideSync(geom.periodicity());
    pe_out->FillBoundary(geom.periodicity());
    if (pe_it < n_pe_iters - 1) {
        amrex::MultiFab::Copy(*m_pe_scratch, *pe_out, 0, 0, pe->nComp(), pe->nGrowVect());
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
            for (amrex::MFIter mfi(*pe_out, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box tb = mfi.tilebox(amrex::IntVect::TheNodeVector());
                if (tb.smallEnd(0) > iax) { continue; }
                auto const& pe_a = pe_out->array(mfi);
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
            pe_out->OverrideSync(geom.periodicity());
            pe_out->FillBoundary(geom.periodicity());
        }
    }
#endif

    if (!a_from_jacobian && !m_pe_unknown) {
        amrex::MultiFab::Copy(*m_pe_theta, *pe, 0, 0, pe->nComp(), pe->nGrowVect());
    }
}

void ThetaImplicitHybrid::UpdateWarpXFields ( const WarpXSolverVec&  a_E,
                                                amrex::Real start_time )
{
    BL_PROFILE("ThetaImplicitHybrid::UpdateWarpXFields()");

    const amrex::Real theta_time = start_time + m_theta * m_dt;

    // Set E^{n+θ} in WarpX
    m_WarpX->SetElectricFieldAndApplyBCs( a_E, theta_time );

    // pe as Newton unknown: write the iterate's pe row into the pressure field and fill the
    // domain ghosts (PEC sides included) for Ohm's law. The wall nodes are not rewritten:
    // they are unknowns pinned to pe^n by identity rows (static walls, as in the in-loop path).
    if (m_pe_unknown) {
        amrex::MultiFab* pe =
            m_WarpX->m_fields.get(FieldType::hybrid_electron_pressure_fp, 0);
        amrex::MultiFab::Copy(*pe, *a_E.getScalarVec()[0], 0, 0, 1,
                              amrex::IntVect::TheZeroVector());
        pe->mult(1.0_rt / m_pe_scale, 0, 1);
        pe->FillBoundary(m_WarpX->Geom(0).periodicity());
        m_WarpX->ApplyElectronPressureBoundary(0, PatchType::fine,
                                               /*rewrite_pec_nodes=*/false);
    }

    // Compute B^{n+θ} = B^n - θ·dt·curl(E^{n+θ}) via Faraday's law
    ablastr::fields::MultiLevelVectorField const& B_old =
        m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::B_old, m_num_amr_levels - 1);
    m_WarpX->UpdateMagneticFieldAndApplyBCs( B_old, m_theta * m_dt, start_time );
}


amrex::Array<const amrex::MultiFab*, 3>
ThetaImplicitHybrid::GetBfieldThetaForPC ( const int lev ) const
{
    // During the nonlinear solve, UpdateWarpXFields (called from every
    // residual evaluation) leaves the Bfield_fp registry holding the TOTAL
    // theta-midpoint field B^{n+theta} of the current iterate. Valid only
    // after the first residual evaluation of the current Newton iterate;
    // before that (and between steps) the registry holds the end-of-step
    // totals B^{n+1} (= B^n at the next entry).
    using ablastr::fields::Direction;
    amrex::Array<const amrex::MultiFab*, 3> B = {
        m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{0}, lev),
        m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{1}, lev),
        m_WarpX->m_fields.get(FieldType::Bfield_fp, Direction{2}, lev) };
    if (!m_add_external_fields) { return B; }
    // assemble the TOTAL field (see the header doc): Bfield_fp is internal
    // between residual evaluations on this branch
    for (int d = 0; d < 3; ++d) {
        const amrex::MultiFab* Bext = m_WarpX->m_fields.get(
            FieldType::hybrid_B_fp_external, Direction{d}, lev);
        auto& tot = m_B_tot_pc[d];
        if (!tot || tot->boxArray() != B[d]->boxArray()
            || tot->DistributionMap() != B[d]->DistributionMap()) {
            tot = std::make_unique<amrex::MultiFab>(
                B[d]->boxArray(), B[d]->DistributionMap(),
                B[d]->nComp(), B[d]->nGrowVect());
        }
        amrex::MultiFab::Copy(*tot, *B[d], 0, 0,
                              B[d]->nComp(), B[d]->nGrowVect());
        amrex::MultiFab::Add(*tot, *Bext, 0, 0,
                             Bext->nComp(), tot->nGrowVect());
        B[d] = tot.get();
    }
    return B;
}

const amrex::MultiFab*
ThetaImplicitHybrid::GetRhoMidForPC ( const int lev ) const
{
    // The rho_fp registry carries two time slots of WarpX::ncomps
    // components each; consumers read component nComp()/2 (the
    // midpoint-position deposit rho^{n+1/2} of the current iterate,
    // written by every residual evaluation). Valid only after the first
    // residual evaluation of the current Newton iterate.
    return m_WarpX->m_fields.get(FieldType::rho_fp, lev);
}

amrex::Array<const amrex::MultiFab*, 3>
ThetaImplicitHybrid::GetIonCurrentForPC ( const int lev ) const
{
    // The Ohm solve consumes current_fp as the ion (particle) current;
    // it is deposited each residual evaluation and frozen during Jacobian
    // probes, so between preconditioner updates and the GMRES solve it
    // holds exactly the frozen drift-leg coefficient (J - J_i) x delta_B.
    using ablastr::fields::Direction;
    return { m_WarpX->m_fields.get(FieldType::current_fp, Direction{0}, lev),
             m_WarpX->m_fields.get(FieldType::current_fp, Direction{1}, lev),
             m_WarpX->m_fields.get(FieldType::current_fp, Direction{2}, lev) };
}

void ThetaImplicitHybrid::FinishFieldUpdate( amrex::Real end_time )
{
    BL_PROFILE("ThetaImplicitHybrid::FinishFieldUpdate()");

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
    if (m_pe_theta) {
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
    if (m_pe_unknown) {
        // static walls (see UpdateWarpXFields) held pe^n through the solve; refresh only
        // the domain ghosts for the end-of-step Ohm's-law E below
        pe->FillBoundary(m_WarpX->Geom(lev).periodicity());
        m_WarpX->ApplyElectronPressureBoundary(lev, PatchType::fine,
                                               /*rewrite_pec_nodes=*/false);
    }
    amrex::MultiFab::Copy(*m_pe_old, *pe, 0, 0, pe->nComp(), pe->nGrowVect());
    }

    // E^{n+1}: E is algebraic in the generalized Ohm's law, so evaluate it
    // at the delivered end-of-step state -- total B^{n+1} (externals
    // included above), pe^{n+1}, and the same ion-deposit family the theta
    // stage used. Per-level calls: the multi-level HybridPICSolveE wrapper
    // fires the afterEpush python callback and this must not add a firing.
    {
        using warpx::fields::FieldType;
        if (m_hybrid_pic_model->m_implicit_use_algebraic_closure) {
            m_hybrid_pic_model->CalculateElectronPressure();
        }
        ablastr::fields::MultiLevelVectorField E_fp =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, m_num_amr_levels - 1);
        ablastr::fields::MultiLevelVectorField J_fp =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::current_fp, m_num_amr_levels - 1);
        ablastr::fields::MultiLevelVectorField B_fp =
            m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, m_num_amr_levels - 1);
        ablastr::fields::MultiLevelScalarField r_fp =
            m_WarpX->m_fields.get_mr_levels(FieldType::rho_fp, m_num_amr_levels - 1);
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            m_hybrid_pic_model->HybridPICSolveE(
                E_fp[lev], J_fp[lev], B_fp[lev], *r_fp[lev],
                m_WarpX->GetEBUpdateEFlag()[lev], lev,
                true  /* solve_for_Faraday: include resistivity, eta_H */,
                true  /* keep_grad_pe */);
        }
        // apply the standard E boundary conditions and keep the solver
        // vector consistent with the delivered field
        if (m_pe_unknown) {
            m_E.Copy(FieldType::Efield_fp, FieldType::hybrid_electron_pressure_fp);
            m_E.getScalarVec()[0]->mult(m_pe_scale, 0, 1);
        } else {
            m_E.Copy(FieldType::Efield_fp);
        }
        m_WarpX->SetElectricFieldAndApplyBCs( m_E, end_time );
    }
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
