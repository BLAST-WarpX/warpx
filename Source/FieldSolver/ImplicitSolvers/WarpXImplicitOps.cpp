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
#include "Evolve/WarpXDtType.H"
#include "Evolve/WarpXPushType.H"
#include "Fields.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Parallelization/GuardCellManager.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/ParticleBoundaryBuffer.H"
#include "Python/callbacks.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXUtil.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <ablastr/utils/SignalHandling.H>
#include <ablastr/warn_manager/WarnManager.H>

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
WarpX::ImplicitPreRHSOp ( amrex::Real  a_cur_time,
                          amrex::Real  a_full_dt,
                          int          a_nl_iter,
                          bool         a_from_jacobian )
{
    using namespace amrex::literals;
    using warpx::fields::FieldType;
    amrex::ignore_unused( a_full_dt, a_nl_iter, a_from_jacobian );

    if (use_filter) { ApplyFilterMF(m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level), 0); }

    // Advance the particle positions by 1/2 dt,
    // particle velocities by dt, then take average of old and new v,
    // deposit currents, giving J at n+1/2
    // This uses Efield_fp and Bfield_fp, the field at n+1/2 from the previous iteration.
    const bool skip_current = false;
    const PushType push_type = PushType::Implicit;
    PushParticlesandDeposit(a_cur_time, skip_current, push_type);

    SyncCurrentAndRho();

}

void
WarpX::SetElectricFieldAndApplyBCs ( const WarpXSolverVec& a_E, amrex::Real a_time )
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        a_E.getArrayVecType()==warpx::fields::FieldType::Efield_fp,
        "WarpX::SetElectricFieldAndApplyBCs() must be called with Efield_fp type");

    using warpx::fields::FieldType;

    ablastr::fields::MultiLevelVectorField Efield_fp = m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level);
    const ablastr::fields::MultiLevelVectorField& Evec = a_E.getArrayVec();
    amrex::MultiFab::Copy(*Efield_fp[0][0], *Evec[0][0], 0, 0, ncomps, Evec[0][0]->nGrowVect());
    amrex::MultiFab::Copy(*Efield_fp[0][1], *Evec[0][1], 0, 0, ncomps, Evec[0][1]->nGrowVect());
    amrex::MultiFab::Copy(*Efield_fp[0][2], *Evec[0][2], 0, 0, ncomps, Evec[0][2]->nGrowVect());
    FillBoundaryE(guard_cells.ng_alloc_EB, WarpX::sync_nodal_points);
    ApplyEfieldBoundary(0, PatchType::fine, a_time);
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
    EvolveB(a_thetadt, DtType::Full, start_time);
    FillBoundaryB(guard_cells.ng_alloc_EB, WarpX::sync_nodal_points);
}

void
WarpX::FinishMagneticFieldAndApplyBCs( ablastr::fields::MultiLevelVectorField const& a_Bn,
                                       amrex::Real a_theta, amrex::Real a_time )
{
    using warpx::fields::FieldType;

    FinishImplicitField(m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, 0), a_Bn, a_theta);
    ApplyBfieldBoundary(0, PatchType::fine, DtType::Full, a_time);
    FillBoundaryB(guard_cells.ng_alloc_EB, WarpX::sync_nodal_points);
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

            auto particle_comps = pc->getParticleComps();

            for (WarpXParIter pti(*pc, lev); pti.isValid(); ++pti) {

                const auto getPosition = GetParticlePosition(pti);

                auto& attribs = pti.GetAttribs();
                amrex::ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr();
                amrex::ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr();
                amrex::ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr();

#if (AMREX_SPACEDIM >= 2)
                amrex::ParticleReal* x_n = pti.GetAttribs(particle_comps["x_n"]).dataPtr();
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
                amrex::ParticleReal* y_n = pti.GetAttribs(particle_comps["y_n"]).dataPtr();
#endif
                amrex::ParticleReal* z_n = pti.GetAttribs(particle_comps["z_n"]).dataPtr();
                amrex::ParticleReal* ux_n = pti.GetAttribs(particle_comps["ux_n"]).dataPtr();
                amrex::ParticleReal* uy_n = pti.GetAttribs(particle_comps["uy_n"]).dataPtr();
                amrex::ParticleReal* uz_n = pti.GetAttribs(particle_comps["uz_n"]).dataPtr();

                const long np = pti.numParticles();

                amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
                {
                    amrex::ParticleReal xp, yp, zp;
                    getPosition(ip, xp, yp, zp);

#if (AMREX_SPACEDIM >= 2)
                    x_n[ip] = xp;
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
                    y_n[ip] = yp;
#endif
                    z_n[ip] = zp;

                    ux_n[ip] = ux[ip];
                    uy_n[ip] = uy[ip];
                    uz_n[ip] = uz[ip];

                });

            }
            }

        }

    }

}

void
WarpX::FinishImplicitParticleUpdate ()
{
    using namespace amrex::literals;

    // The implicit advance routines use the time-centered position and
    // momentum to advance the system in time. Thus, at the end of the
    // step we need to transform the particle postion and momentum from
    // time n+1/2 to time n+1. This is done here.

    for (auto const& pc : *mypc) {

        for (int lev = 0; lev <= finest_level; ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
            {

            auto particle_comps = pc->getParticleComps();

            for (WarpXParIter pti(*pc, lev); pti.isValid(); ++pti) {

                const auto getPosition = GetParticlePosition(pti);
                const auto setPosition = SetParticlePosition(pti);

                auto& attribs = pti.GetAttribs();
                amrex::ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr();
                amrex::ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr();
                amrex::ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr();

#if (AMREX_SPACEDIM >= 2)
                amrex::ParticleReal* x_n = pti.GetAttribs(particle_comps["x_n"]).dataPtr();
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
                amrex::ParticleReal* y_n = pti.GetAttribs(particle_comps["y_n"]).dataPtr();
#endif
                amrex::ParticleReal* z_n = pti.GetAttribs(particle_comps["z_n"]).dataPtr();
                amrex::ParticleReal* ux_n = pti.GetAttribs(particle_comps["ux_n"]).dataPtr();
                amrex::ParticleReal* uy_n = pti.GetAttribs(particle_comps["uy_n"]).dataPtr();
                amrex::ParticleReal* uz_n = pti.GetAttribs(particle_comps["uz_n"]).dataPtr();

                const long np = pti.numParticles();

                amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (long ip)
                {
                    amrex::ParticleReal xp, yp, zp;
                    getPosition(ip, xp, yp, zp);

#if (AMREX_SPACEDIM >= 2)
                    xp = 2._rt*xp - x_n[ip];
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
                    yp = 2._rt*yp - y_n[ip];
#endif
                    zp = 2._rt*zp - z_n[ip];

                    ux[ip] = 2._rt*ux[ip] - ux_n[ip];
                    uy[ip] = 2._rt*uy[ip] - uy_n[ip];
                    uz[ip] = 2._rt*uz[ip] - uz_n[ip];

                    setPosition(ip, xp, yp, zp);
                });

            }
            }

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
    WARPX_PROFILE("WarpX::ImplicitComputeRHSE()");
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

    // set RHS to zero value
    a_Erhs_vec.getArrayVec()[lev][0]->setVal(0.0);
    a_Erhs_vec.getArrayVec()[lev][1]->setVal(0.0);
    a_Erhs_vec.getArrayVec()[lev][2]->setVal(0.0);

    // Compute Efield_rhs in regular cells by calling EvolveE. Because
    // a_Erhs_vec is set to zero above, calling EvolveE below results in
    // a_Erhs_vec storing only the RHS of the update equation. I.e.,
    // c^2*dt*(curl(B^{n+theta} - mu0*J^{n+1/2})
    if (patch_type == PatchType::fine) {
        m_fdtd_solver_fp[lev]->EvolveE( m_fields,
                                        lev,
                                        patch_type,
                                        a_Erhs_vec.getArrayVec()[lev],
                                        a_dt );
    } else {
        m_fdtd_solver_cp[lev]->EvolveE( m_fields,
                                        lev,
                                        patch_type,
                                        a_Erhs_vec.getArrayVec()[lev],
                                        a_dt );
    }

    // Compute Efield_rhs in PML cells by calling EvolveEPML
    if (do_pml && pml[lev]->ok()) {
        amrex::Abort("PML not yet implemented with implicit solvers.");
    }

}
