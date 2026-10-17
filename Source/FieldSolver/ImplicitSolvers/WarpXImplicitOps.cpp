/* Copyright 2022 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#include "BoundaryConditions/PML.H"
#include "Diagnostics/MultiDiagnostics.H"
#include "Diagnostics/ReducedDiags/MultiReducedDiags.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H"
#include "FieldSolver/ImplicitSolvers/ImplicitOptions.H"
#include "Fields.H"
#include "Parallelization/GuardCellManager.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/ParticleBoundaryBuffer.H"
#include "Python/callbacks.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXUtil.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/utils/SignalHandling.H>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <array>
#include <memory>
#include <ostream>
#include <vector>

void
WarpX::SetElectricFieldAndApplyBCs ( const WarpXSolverVec& a_E, amrex::Real a_time )
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_E.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "WarpX::SetElectricFieldAndApplyBCs() must be called with Efield_fp type");

    using warpx::fields::FieldType;

    ablastr::fields::MultiLevelVectorField Efield_fp = m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level);
    const ablastr::fields::MultiLevelVectorField& Evec = a_E.getArrayVec();
    for (int lev = 0; lev <= finest_level; ++lev) {
        for (int n = 0; n < 3; ++n) {
            amrex::MultiFab::Copy(*Efield_fp[lev][n], *Evec[lev][n], 0, 0, ncomps, 0);
        }
    }
    for (auto const& block : a_E.getExtraFields()) {
        auto const field = m_fields.get_alldirs(block.type, block.level);
        for (int n = 0; n < 3; ++n) {
            amrex::MultiFab::Copy(*field[n], *block.values[n], 0, 0, field[n]->nComp(), 0);
        }
    }
    FillBoundaryE(guard_cells.ng_alloc_EB, WarpX::sync_nodal_points);
    // Physical-boundary reflections can read neighboring grids' guard cells.
    // Refresh those cells from the trial field before applying the reflections.
    for (int lev = 0; lev <= finest_level; ++lev) {
        ApplyEfieldBoundary(lev, PatchType::fine, a_time);
        if (lev > 0) {
            ApplyEfieldBoundary(lev, PatchType::coarse, a_time);
        }
    }
}

void
WarpX::UpdateMagneticFieldAndApplyBCs( ablastr::fields::MultiLevelVectorField const& a_Bn,
                                       amrex::Real a_thetadt, amrex::Real start_time )
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    for (int lev = 0; lev <= finest_level; ++lev) {
        ablastr::fields::VectorField Bfp = m_fields.get_alldirs(FieldType::Bfield_fp, lev);
        amrex::MultiFab::Copy(*Bfp[0], *a_Bn[lev][0], 0, 0, ncomps, a_Bn[lev][0]->nGrowVect());
        amrex::MultiFab::Copy(*Bfp[1], *a_Bn[lev][1], 0, 0, ncomps, a_Bn[lev][1]->nGrowVect());
        amrex::MultiFab::Copy(*Bfp[2], *a_Bn[lev][2], 0, 0, ncomps, a_Bn[lev][2]->nGrowVect());
    }
    EvolveB(a_thetadt, SubcyclingHalf::None, start_time);
    FillBoundaryB(guard_cells.ng_alloc_EB, WarpX::sync_nodal_points);
}

void
WarpX::FinishElectricFieldAndApplyBCs(amrex::Real a_theta, amrex::Real a_time)
{
    using warpx::fields::FieldType;

    ablastr::fields::MultiLevelVectorField const& En =
        m_fields.get_mr_levels_alldirs(FieldType::E_old, finest_level);
    FinishImplicitField(m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level), En,
                        a_theta);
    for (int lev = 0; lev <= finest_level; ++lev) {
        ApplyEfieldBoundary(lev, PatchType::fine, a_time);
        if (lev > 0) {
            ApplyEfieldBoundary(lev, PatchType::coarse, a_time);
        }
    }
    FillBoundaryE(guard_cells.ng_alloc_EB, WarpX::sync_nodal_points);
}

void
WarpX::FinishMagneticFieldAndApplyBCs(amrex::Real a_theta, amrex::Real a_time)
{
    using warpx::fields::FieldType;

    ablastr::fields::MultiLevelVectorField const& Bn =
        m_fields.get_mr_levels_alldirs(FieldType::B_old, finest_level);
    FinishImplicitField(m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level), Bn,
                        a_theta);
    for (int lev = 0; lev <= finest_level; ++lev) {
        ApplyBfieldBoundary(lev, PatchType::fine, SubcyclingHalf::None, a_time);
        ApplyCurrentControlledPort(lev, PatchType::fine, a_time);
        if (lev > 0) {
            ApplyBfieldBoundary(lev, PatchType::coarse, SubcyclingHalf::None, a_time);
            ApplyCurrentControlledPort(lev, PatchType::coarse, a_time);
        }
    }
    FillBoundaryB(guard_cells.ng_alloc_EB, WarpX::sync_nodal_points);
    ApplyCompositeCurrentControlledPorts(a_time);
}

void
WarpX::SpectralSourceFreeFieldAdvance (amrex::Real start_time)
{
    using namespace amrex::literals;
    using warpx::fields::FieldType;
    // Do the first piece of the Strang splitting, source free advance of E and B
    // It would be more efficient to write a specialized PSATD advance that does not use J,
    // but this works for now.

    // Create temporary MultiFabs to hold J
    int const lev = 0;
    ablastr::fields::VectorField current_fp = m_fields.get_alldirs(FieldType::current_fp, lev);
    amrex::MultiFab* rho_fp = m_fields.get(FieldType::rho_fp, lev);
    amrex::MultiFab j0(current_fp[0]->boxArray(), current_fp[0]->DistributionMap(),
                       current_fp[0]->nComp(), current_fp[0]->nGrowVect());
    amrex::MultiFab j1(current_fp[1]->boxArray(), current_fp[1]->DistributionMap(),
                       current_fp[1]->nComp(), current_fp[1]->nGrowVect());
    amrex::MultiFab j2(current_fp[2]->boxArray(), current_fp[2]->DistributionMap(),
                       current_fp[2]->nComp(), current_fp[2]->nGrowVect());
    amrex::MultiFab::Copy(j0, *(current_fp[0]), 0, 0, current_fp[0]->nComp(), current_fp[0]->nGrowVect());
    amrex::MultiFab::Copy(j1, *(current_fp[1]), 0, 0, current_fp[1]->nComp(), current_fp[1]->nGrowVect());
    amrex::MultiFab::Copy(j2, *(current_fp[2]), 0, 0, current_fp[2]->nComp(), current_fp[2]->nGrowVect());

    current_fp[0]->setVal(0._rt);
    current_fp[1]->setVal(0._rt);
    current_fp[2]->setVal(0._rt);
    if (rho_fp) { rho_fp->setVal(0._rt); }
    PushPSATD(start_time); // Note that this does dt/2
    FillBoundaryE(guard_cells.ng_alloc_EB, WarpX::sync_nodal_points);
    FillBoundaryB(guard_cells.ng_alloc_EB, WarpX::sync_nodal_points);

    // Restore the current_fp MultiFab. Note that this is only needed for diagnostics when
    // J is being written out (since current_fp is not otherwise used).
    amrex::MultiFab::Copy(*(current_fp[0]), j0, 0, 0, current_fp[0]->nComp(), current_fp[0]->nGrowVect());
    amrex::MultiFab::Copy(*(current_fp[1]), j1, 0, 0, current_fp[1]->nComp(), current_fp[1]->nGrowVect());
    amrex::MultiFab::Copy(*(current_fp[2]), j2, 0, 0, current_fp[2]->nComp(), current_fp[2]->nGrowVect());
}

void
WarpX::SaveParticlesAtImplicitStepStart ( )
{
    // The implicit advance routines require the particle velocity
    // and position values at the beginning of the step to compute the
    // time-centered position and velocity needed for the implicit stencil.
    // Thus, we need to save this information.

    for (auto const& pc : *mypc) {

        for (int lev = 0; lev <= finest_level; ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
            {

            auto particle_comps = pc->GetRealSoANames();

            for (WarpXParIter pti(*pc, lev); pti.isValid(); ++pti) {

                const auto getPosition = GetParticlePosition(pti);

                auto& attribs = pti.GetAttribs();
                amrex::ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr();
                amrex::ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr();
                amrex::ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr();

#if !defined(WARPX_DIM_1D_Z)
                amrex::ParticleReal* x_n = pti.GetAttribs("x_n").dataPtr();
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                amrex::ParticleReal* y_n = pti.GetAttribs("y_n").dataPtr();
#endif
#if !defined(WARPX_DIM_RCYLINDER)
                amrex::ParticleReal* z_n = pti.GetAttribs("z_n").dataPtr();
#endif
                amrex::ParticleReal* ux_n = pti.GetAttribs("ux_n").dataPtr();
                amrex::ParticleReal* uy_n = pti.GetAttribs("uy_n").dataPtr();
                amrex::ParticleReal* uz_n = pti.GetAttribs("uz_n").dataPtr();

                // Check if nsuborbits is present, and if so it is set to 1
                int *nsuborbits = (pc->HasiAttrib("nsuborbits") ? pti.GetiAttribs("nsuborbits").dataPtr() : nullptr);

                const long np = pti.numParticles();

                amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
                {
                    amrex::ParticleReal xp, yp, zp;
                    getPosition(ip, xp, yp, zp);

#if !defined(WARPX_DIM_1D_Z)
                    x_n[ip] = xp;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
                    y_n[ip] = yp;
#endif
#if !defined(WARPX_DIM_RCYLINDER)
                    z_n[ip] = zp;
#endif

                    ux_n[ip] = ux[ip];
                    uy_n[ip] = uy[ip];
                    uz_n[ip] = uz[ip];

                    if (nsuborbits) {
                        nsuborbits[ip] = 1;
                    }

                });

            }
            }

        }

    }

}

void
WarpX::FinishImplicitParticleUpdate (amrex::Real time)
{
    using namespace amrex::literals;

    // The implicit advance routines use the time-centered position and
    // momentum to advance the system in time. Thus, at the end of the
    // step we need to transform the particle postion and momentum from
    // time n+1/2 to time n+1. This is done here via virtual dispatch:
    // photon particles call Evolve; all other particles extrapolate position
    // and momentum.

    for (int lev = 0; lev <= finest_level; ++lev) {
        for (auto const& pc : *mypc) {
            pc->FinishImplicitParticleUpdate(m_fields, lev, time, dt[lev]);
        }
    }

}

void
WarpX::ResetImplicitParticleData ()
{
    using namespace amrex::literals;

    // Reset the particle data when starting substepping.
    for (int lev = 0; lev <= finest_level; ++lev) {
        for (auto const& pc : *mypc) {
            pc->ResetImplicitParticleData(lev);
        }
    }

}

void
WarpX::FinishImplicitField( ablastr::fields::MultiLevelVectorField const& Field_fp,
                            ablastr::fields::MultiLevelVectorField const& Field_n,
                            amrex::Real  theta )
{
    using namespace amrex::literals;

    // The implicit field advance routines use the fields at time n+theta
    // with 0.5 <= theta <= 1.0. Thus, at the end of the step we need to
    // transform the fields from time n+theta to time n+1. This is done here.

    for (int lev = 0; lev <= finest_level; ++lev) {

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
       for ( amrex::MFIter mfi(*Field_fp[lev][0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
        {

            amrex::Array4<amrex::Real> const& Fx = Field_fp[lev][0]->array(mfi);
            amrex::Array4<amrex::Real> const& Fy = Field_fp[lev][1]->array(mfi);
            amrex::Array4<amrex::Real> const& Fz = Field_fp[lev][2]->array(mfi);

            amrex::Array4<amrex::Real> const& Fx_n = Field_n[lev][0]->array(mfi);
            amrex::Array4<amrex::Real> const& Fy_n = Field_n[lev][1]->array(mfi);
            amrex::Array4<amrex::Real> const& Fz_n = Field_n[lev][2]->array(mfi);

            amrex::Box const& tbx = mfi.tilebox(Field_n[lev][0]->ixType().toIntVect());
            amrex::Box const& tby = mfi.tilebox(Field_n[lev][1]->ixType().toIntVect());
            amrex::Box const& tbz = mfi.tilebox(Field_n[lev][2]->ixType().toIntVect());

            const amrex::Real c0 = 1._rt/theta;
            const amrex::Real c1 = 1._rt - c0;

            amrex::ParallelFor(
            tbx, ncomps, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                Fx(i,j,k,n) = c0*Fx(i,j,k,n) + c1*Fx_n(i,j,k,n);
            },
            tby, ncomps, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                Fy(i,j,k,n) = c0*Fy(i,j,k,n) + c1*Fy_n(i,j,k,n);
            },
            tbz, ncomps, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                Fz(i,j,k,n) = c0*Fz(i,j,k,n) + c1*Fz_n(i,j,k,n);
            });
        }
    }
}

void
WarpX::DepositMassMatrices (amrex::Real a_dt, ImplicitOptions const& implicit_options)
{
    ABLASTR_PROFILE("WarpX::DepositMassMatrices()");

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            implicit_options.ncomp_xx[dir] > 0 &&
            implicit_options.ncomp_yy[dir] > 0 &&
            implicit_options.ncomp_zz[dir] > 0,
            "DepositMassMatrices requires initialized mass-matrix component counts. "
            "Call InitData before depositing mass matrices.");
    }

    amrex::Real const dt_scale = a_dt/dt[0];

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        mypc->DepositMassMatrices(
            m_fields,
            lev,
            dt[lev]*dt_scale,
            implicit_options
        );
    }

}

void
WarpX::ImplicitComputeRHSE (amrex::Real a_dt, WarpXSolverVec& a_Erhs_vec)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        ImplicitComputeRHSE(lev, a_dt, a_Erhs_vec);
    }
}

void
WarpX::ImplicitComputeRHSE (int lev, amrex::Real a_dt, WarpXSolverVec& a_Erhs_vec)
{
    ABLASTR_PROFILE("WarpX::ImplicitComputeRHSE()");
    ImplicitComputeRHSE(lev, PatchType::fine, a_dt, a_Erhs_vec);
    if (lev > 0)
    {
        ImplicitComputeRHSE(lev, PatchType::coarse, a_dt, a_Erhs_vec);
    }
}

void
WarpX::ImplicitComputeRHSE (int lev, PatchType patch_type, amrex::Real a_dt, WarpXSolverVec& a_Erhs_vec)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_Erhs_vec.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "WarpX::ImplicitComputeRHSE() must be called with Efield_fp type");

    using warpx::fields::FieldType;
    bool const fine = patch_type == PatchType::fine;
    auto const rhs =
        fine ? a_Erhs_vec.getArrayVec()[lev] : a_Erhs_vec.getExtraField(FieldType::Efield_cp, lev);
    for (auto* field : rhs) {
        field->setVal(0.0);
    }
    auto* const solver = fine ? m_fdtd_solver_fp[lev].get() : m_fdtd_solver_cp[lev].get();
    if (m_em_solver_medium == MediumForEM::Macroscopic) {
        auto const field =
            m_fields.get_alldirs(fine ? FieldType::Efield_fp : FieldType::Efield_cp, lev);
        auto const magnetic =
            m_fields.get_alldirs(fine ? FieldType::Bfield_fp : FieldType::Bfield_cp, lev);
        auto const current =
            m_fields.get_alldirs(fine ? FieldType::current_fp : FieldType::current_cp, lev);
        auto const& material = m_macroscopic_properties[lev][fine ? 0 : 1];
        // With zero E, the existing material curl gives dt*(curl(H)-J)/(epsilon+dt*sigma).
        // Undo that denominator and include the conductivity of the trial E^{n+theta}.
        solver->MacroscopicEvolveE(MacroscopicSolverAlgo::BackwardEuler, rhs, magnetic, current,
                                   GetEBUpdateEFlag(patch_type)[lev], a_dt, material, lev);
        for (int n = 0; n < 3; ++n) {
            auto const staggering = n == 0   ? material->Ex_IndexType
                                    : n == 1 ? material->Ey_IndexType
                                             : material->Ez_IndexType;
            auto const sigma_staggering = material->sigma_IndexType;
            auto const epsilon_staggering = material->epsilon_IndexType;
            auto const ratio = material->macro_cr_ratio;
            for (amrex::MFIter mfi(*rhs[n], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                auto const sigma = material->getsigma_mf().const_array(mfi);
                auto const epsilon = material->getepsilon_mf().const_array(mfi);
                auto const values = rhs[n]->array(mfi);
                auto const trial = field[n]->const_array(mfi);
                amrex::ParallelFor(
                    mfi.tilebox(), ncomps, [=] AMREX_GPU_DEVICE(int i, int j, int k, int mode) {
                        amrex::Real const conductivity = ablastr::coarsen::sample::Interp(
                            sigma, sigma_staggering, staggering, ratio, i, j, k, 0);
                        amrex::Real const permittivity = ablastr::coarsen::sample::Interp(
                            epsilon, epsilon_staggering, staggering, ratio, i, j, k, 0);
                        amrex::Real const damping = a_dt * conductivity / permittivity;
                        values(i, j, k, mode) =
                            (1 + damping) * values(i, j, k, mode) - damping * trial(i, j, k, mode);
                    });
            }
        }
    } else {
        solver->EvolveE(m_fields, lev, patch_type, rhs, GetEBUpdateEFlag(patch_type)[lev], a_dt);
    }

    if (do_pml && pml[lev] && pml[lev]->ok()) {
        auto const type = fine ? FieldType::pml_E_fp : FieldType::pml_E_cp;
        auto const pml_rhs = a_Erhs_vec.getExtraField(type, lev);
        for (auto* field : pml_rhs) {
            field->setVal(0.0);
        }
        solver->EvolveEPML(m_fields, patch_type, lev, pml[lev]->GetEBUpdateEFlag(patch_type),
                           fine ? pml[lev]->GetMultiSigmaBox_fp() : pml[lev]->GetMultiSigmaBox_cp(),
                           a_dt, false, pml_rhs);
        pml[lev]->ApplyImplicitSigma(pml_rhs, m_fields.get_alldirs(type, lev), a_dt, false,
                                     patch_type);
    }
}
