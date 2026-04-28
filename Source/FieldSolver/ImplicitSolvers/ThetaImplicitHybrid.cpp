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
#include "WarpX.H"
#include <ablastr/utils/Communication.H>
#include <ablastr/coarsen/sample.H>

using warpx::fields::FieldType;
using namespace amrex::literals;

void ThetaImplicitHybrid::Define ( WarpX* const a_WarpX )
{
    BL_PROFILE("ThetaImplicitHybrid::Define()");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_is_defined,
        "ThetaImplicitHybrid object is already defined!");

    m_WarpX = a_WarpX;
    m_num_amr_levels = 1;

    m_hybrid_pic_model = m_WarpX->get_pointer_HybridPICModel();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_hybrid_pic_model != nullptr,
        "ThetaImplicitHybrid solver requires hybrid PIC model to be defined");

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

void ThetaImplicitHybrid::OneStep ( const amrex::Real  start_time,
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

    // Save E^n
    m_Eold.Copy(FieldType::Efield_fp);
    
    // Save B^n
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const ablastr::fields::VectorField Bfp = m_WarpX->m_fields.get_alldirs(FieldType::Bfield_fp, lev);
        ablastr::fields::VectorField B_old = m_WarpX->m_fields.get_alldirs(FieldType::B_old, lev);
        for (int n = 0; n < 3; ++n) {
            amrex::MultiFab::Copy(*B_old[n], *Bfp[n], 0, 0, 
                                  B_old[n]->nComp(), B_old[n]->nGrowVect());
        }
    }

    // Initial guess: E^{n+θ} = E^n
    m_E.Copy(m_Eold);
    
    // Solve nonlinear system for E^{n+θ} (and eventually Pe^{n+θ})
    m_nlsolver->Solve( m_E, m_Eold, start_time, m_dt, a_step );

    // Update WarpX fields to t^{n+θ}
    UpdateWarpXFields( m_E, start_time );
    m_WarpX->reduced_diags->ComputeDiagsMidStep(a_step);

    // Advance particles from t^{n+1/2} to t^{n+1}
    m_WarpX->FinishImplicitParticleUpdate();

    // Advance fields from t^{n+θ} to t^{n+1}
    FinishFieldUpdate( start_time + m_dt );
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

    // --- Compute resistivity-free E for particle push ---
    m_hybrid_pic_model->CalculatePlasmaCurrent(Bfield_fp, m_WarpX->GetEBUpdateEFlag());
    m_hybrid_pic_model->CalculateElectronPressure();

    m_hybrid_pic_model->HybridPICSolveE(
        Efield_fp, current_fp, Bfield_fp, rho_fp,
        m_WarpX->GetEBUpdateEFlag(),
        false  // no resistivity; ∇Pe included
    );

    m_WarpX->ApplyFillBoundaryE();

    if (m_add_external_fields) {
        m_hybrid_pic_model->m_external_vector_potential->UpdateHybridExternalFields(
            theta_time, 0.5_rt * m_dt);
        AddExternalBfield();
        AddExternalEfield();
    }

    PreRHSOp( theta_time, a_nl_iter, a_from_jacobian );

    if (m_add_external_fields) {
        SubtractExternalBfield();
        SubtractExternalEfield();
    }

    // --- Compute full Ohm's law E for Faraday update ---
    // J_plasma unchanged (same B), but Pe depends on newly deposited ρ
    m_hybrid_pic_model->CalculateElectronPressure();

    m_hybrid_pic_model->HybridPICSolveE(
        Efield_fp, current_fp, Bfield_fp, rho_fp,
        m_WarpX->GetEBUpdateEFlag(),
        true, true   // with resistivity and ∇Pe included (∇Pe is curl-free so doesn't affect Faraday, but needed for self-consistent Newton residual)
    );

    m_WarpX->ApplyFillBoundaryE();

    // RHS = E_ohm - E_old
    a_RHS.Copy(FieldType::Efield_fp);
    a_RHS.linComb(1.0, a_RHS, -1.0, m_Eold);
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

    // E^{n+1}
    m_E.linComb( c0, m_E, c1, m_Eold );
    m_WarpX->SetElectricFieldAndApplyBCs( m_E, end_time );

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

                // alpha = theta*dt*|B| / (mu0 * rho_limited)
                // derived from: alpha = theta*dt*|B| / (mu0 * n_e0 * e)
                //               with rho = n_e0 * e
                alpha(i, j, k) = theta_dt * Bmag * one_over_mu0 / rho_limited;
            });
        }
    }

    return &m_alpha_mfarrvec;
}