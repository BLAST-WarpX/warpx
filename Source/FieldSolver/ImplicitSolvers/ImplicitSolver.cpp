#include "ImplicitSolver.H"
#include "Fields.H"
#include "WarpX.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXAlgorithmSelection.H"

using namespace amrex;
using namespace amrex::literals;

void ImplicitSolver::CreateParticleAttributes () const
{
    // Set comm to false to that the attributes are not communicated
    // nor written to the checkpoint files
    int const comm = 0;

    // Add space to save the positions and velocities at the start of the time steps
    for (auto const& pc : m_WarpX->GetPartContainer()) {
#if !defined(WARPX_DIM_1D_Z)
        pc->AddRealComp("x_n", comm);
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        pc->AddRealComp("y_n", comm);
#endif
#if !defined(WARPX_DIM_RCYLINDER)
        pc->AddRealComp("z_n", comm);
#endif
        pc->AddRealComp("ux_n", comm);
        pc->AddRealComp("uy_n", comm);
        pc->AddRealComp("uz_n", comm);

        if (m_particle_suborbits) {
            pc->AddIntComp("nsuborbits", comm);
        }
    }
}

const Geometry& ImplicitSolver::GetGeometry (const int a_lvl) const
{
    AMREX_ASSERT((a_lvl >= 0) && (a_lvl < m_num_amr_levels));
    return m_WarpX->Geom(a_lvl);
}

const Array<FieldBoundaryType,AMREX_SPACEDIM>& ImplicitSolver::GetFieldBoundaryLo () const
{
    return m_WarpX->GetFieldBoundaryLo();
}

const Array<FieldBoundaryType,AMREX_SPACEDIM>& ImplicitSolver::GetFieldBoundaryHi () const
{
    return m_WarpX->GetFieldBoundaryHi();
}

Array<LinOpBCType,AMREX_SPACEDIM> ImplicitSolver::GetLinOpBCLo () const
{
    return convertFieldBCToLinOpBC(m_WarpX->GetFieldBoundaryLo(),/*bdry_side=*/0);
}

Array<LinOpBCType,AMREX_SPACEDIM> ImplicitSolver::GetLinOpBCHi () const
{
    return convertFieldBCToLinOpBC(m_WarpX->GetFieldBoundaryHi(),/*bdry_side=*/1);
}

Array<LinOpBCType,AMREX_SPACEDIM> ImplicitSolver::convertFieldBCToLinOpBC (const Array<FieldBoundaryType,AMREX_SPACEDIM>& a_fbc, const int bdry_side) const
{
    Array<LinOpBCType, AMREX_SPACEDIM> lbc;
    for (auto& bc : lbc) { bc = LinOpBCType::interior; }
    for (int i = 0; i < AMREX_SPACEDIM; i++) {
        if (a_fbc[i] == FieldBoundaryType::PML) {
            WARPX_ABORT_WITH_MESSAGE("LinOpBCType not set for this FieldBoundaryType");
        } else if (a_fbc[i] == FieldBoundaryType::Periodic) {
            lbc[i] = LinOpBCType::Periodic;
        } else if (a_fbc[i] == FieldBoundaryType::PEC) {
            lbc[i] = LinOpBCType::Dirichlet;
        } else if (a_fbc[i] == FieldBoundaryType::Damped) {
            WARPX_ABORT_WITH_MESSAGE("LinOpBCType not set for this FieldBoundaryType");
        } else if (a_fbc[i] == FieldBoundaryType::Absorbing_SilverMueller) {
            ablastr::warn_manager::WMRecordWarning("Implicit solver",
                "With SilverMueller, in the Curl-Curl preconditioner Symmetry boundary will be used since the full boundary is not yet implemented.",
                ablastr::warn_manager::WarnPriority::medium);
            lbc[i] = LinOpBCType::symmetry;
        } else if (a_fbc[i] == FieldBoundaryType::Neumann) {
            // Also for FieldBoundaryType::PMC
            lbc[i] = LinOpBCType::symmetry;
        } else if (a_fbc[i] == FieldBoundaryType::PECInsulator) {
            const int voltage_driven = m_WarpX->GetPECInsulator_IsESet(i,bdry_side);
            if (voltage_driven) { // Dirichlet for E
                lbc[i] = LinOpBCType::Dirichlet;
            } else { // Dirichlet for B
                ablastr::warn_manager::WMRecordWarning("Implicit solver with current-driven PECInsulator",
                    "in the Curl-Curl preconditioner. Symmetry boundary will be used since the full boundary is not yet implemented.",
                    ablastr::warn_manager::WarnPriority::medium);
                lbc[i] = LinOpBCType::symmetry;
            }
        } else if (a_fbc[i] == FieldBoundaryType::None) {
            WARPX_ABORT_WITH_MESSAGE("LinOpBCType not set for this FieldBoundaryType");
        } else if (a_fbc[i] == FieldBoundaryType::Open) {
            WARPX_ABORT_WITH_MESSAGE("LinOpBCType not set for this FieldBoundaryType");
        } else {
            WARPX_ABORT_WITH_MESSAGE("Invalid value for FieldBoundaryType");
        }
    }
    return lbc;
}

void ImplicitSolver::CumulateJ ()
{

    // Add J0, which contains J from particles included in the mass matrices (MM) to current_fp, which
    // is either zero or contains J from suborbit particles that are not inclued in the MM.
    // Do this BEFORE call to SyncCurrentAndRho().
    //
    // J during the linear stage of JFNK is computed as J(E=E0+dE) = J_suborbit + J0 + MM*(E - E0),
    // where MM are the mass matrices (i.e., dJ/dE), E0 is the electric field at the start of the Newton
    // step (see SaveE function), J0 is the current associated with particles that are included in the MM
    // using E0, and J_suborbit is the current associated with particles that have suborbits.

    using warpx::fields::FieldType;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        ablastr::fields::VectorField J = m_WarpX->m_fields.get_alldirs(FieldType::current_fp, lev);
        const ablastr::fields::VectorField J0 = m_WarpX->m_fields.get_alldirs(FieldType::current_fp_non_suborbit, lev);
        amrex::MultiFab::Add(*J[0], *J0[0], 0, 0, J0[0]->nComp(), J0[0]->nGrowVect());
        amrex::MultiFab::Add(*J[1], *J0[1], 0, 0, J0[1]->nComp(), J0[1]->nGrowVect());
        amrex::MultiFab::Add(*J[2], *J0[2], 0, 0, J0[2]->nComp(), J0[2]->nGrowVect());
    }

}

void ImplicitSolver::SaveE ()
{

    // Copy Efield_fp to E0.

    using warpx::fields::FieldType;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const ablastr::fields::VectorField E = m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, lev);
        ablastr::fields::VectorField E0 = m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp_save, lev);
        amrex::MultiFab::Copy(*E0[0], *E[0], 0, 0, E[0]->nComp(), E[0]->nGrowVect());
        amrex::MultiFab::Copy(*E0[1], *E[1], 0, 0, E[1]->nComp(), E[1]->nGrowVect());
        amrex::MultiFab::Copy(*E0[2], *E[2], 0, 0, E[2]->nComp(), E[2]->nGrowVect());
    }

}

void ImplicitSolver::ComputeJfromMassMatrices (const bool  a_J_from_MM_only)
{
    BL_PROFILE("ImplicitSolver::ComputeJfromMassMatrices()");
    using namespace amrex::literals;

    using warpx::fields::FieldType;
    using ablastr::fields::Direction;
    const int ncomps = 1;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {

        ablastr::fields::VectorField J = m_WarpX->m_fields.get_alldirs(FieldType::current_fp, lev);
        ablastr::fields::VectorField E = m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, lev);
        ablastr::fields::VectorField J0 = m_WarpX->m_fields.get_alldirs(FieldType::current_fp_non_suborbit, lev);
        ablastr::fields::VectorField E0 = m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp_save, lev);

        ablastr::fields::VectorField SX = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_X, lev);
        ablastr::fields::VectorField SY = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_Y, lev);
        ablastr::fields::VectorField SZ = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_Z, lev);

        const amrex::IntVect Jx_nodal = J[0]->ixType().toIntVect();
        const amrex::IntVect Jy_nodal = J[1]->ixType().toIntVect();
        const amrex::IntVect Jz_nodal = J[2]->ixType().toIntVect();

        if (a_J_from_MM_only) {
            // Initialize comps of J to zero before adding J from MM
            J[0]->setVal(0.0);
            J[1]->setVal(0.0);
            J[2]->setVal(0.0);
        }

        // Compute the component offset in each direction (careful with staggering)
        amrex::IntVect offset_xx, offset_xy, offset_xz;
        amrex::IntVect offset_yx, offset_yy, offset_yz;
        amrex::IntVect offset_zx, offset_zy, offset_zz;
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
            offset_xx[dir] = (m_ncomp_xx[dir]-1)/2;
            offset_xy[dir] = (Jx_nodal[dir] > Jy_nodal[dir]) ?  (m_ncomp_xy[dir]/2)
                                                             : ((m_ncomp_xy[dir]-1)/2);
            offset_xz[dir] = (Jx_nodal[dir] > Jz_nodal[dir]) ?  (m_ncomp_xz[dir]/2)
                                                             : ((m_ncomp_xz[dir]-1)/2);
            offset_yx[dir] = (Jy_nodal[dir] > Jx_nodal[dir]) ?  (m_ncomp_yx[dir]/2)
                                                             : ((m_ncomp_yx[dir]-1)/2);
            offset_yy[dir] = (m_ncomp_yy[dir]-1)/2;
            offset_yz[dir] = (Jy_nodal[dir] > Jz_nodal[dir]) ?  (m_ncomp_yz[dir]/2)
                                                             : ((m_ncomp_yz[dir]-1)/2);
            offset_zx[dir] = (Jz_nodal[dir] > Jx_nodal[dir]) ?  (m_ncomp_zx[dir]/2)
                                                             : ((m_ncomp_zx[dir]-1)/2);
            offset_zy[dir] = (Jz_nodal[dir] > Jy_nodal[dir]) ?  (m_ncomp_zy[dir]/2)
                                                             : ((m_ncomp_zy[dir]-1)/2);
            offset_zz[dir] = (m_ncomp_zz[dir]-1)/2;
        }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( amrex::MFIter mfi(*J[0], false); mfi.isValid(); ++mfi )
        {

            amrex::Array4<amrex::Real> const& Jx = J[0]->array(mfi);
            amrex::Array4<amrex::Real> const& Jy = J[1]->array(mfi);
            amrex::Array4<amrex::Real> const& Jz = J[2]->array(mfi);

            amrex::Array4<const amrex::Real> const& Ex = E[0]->array(mfi);
            amrex::Array4<const amrex::Real> const& Ey = E[1]->array(mfi);
            amrex::Array4<const amrex::Real> const& Ez = E[2]->array(mfi);

            amrex::Array4<const amrex::Real> const& Jx0 = J0[0]->array(mfi);
            amrex::Array4<const amrex::Real> const& Jy0 = J0[1]->array(mfi);
            amrex::Array4<const amrex::Real> const& Jz0 = J0[2]->array(mfi);

            amrex::Array4<const amrex::Real> const& Ex0 = E0[0]->array(mfi);
            amrex::Array4<const amrex::Real> const& Ey0 = E0[1]->array(mfi);
            amrex::Array4<const amrex::Real> const& Ez0 = E0[2]->array(mfi);

            amrex::Array4<const amrex::Real> const& Sxx = SX[0]->array(mfi);
            amrex::Array4<const amrex::Real> const& Sxy = SX[1]->array(mfi);
            amrex::Array4<const amrex::Real> const& Sxz = SX[2]->array(mfi);

            amrex::Array4<const amrex::Real> const& Syx = SY[0]->array(mfi);
            amrex::Array4<const amrex::Real> const& Syy = SY[1]->array(mfi);
            amrex::Array4<const amrex::Real> const& Syz = SY[2]->array(mfi);

            amrex::Array4<const amrex::Real> const& Szx = SZ[0]->array(mfi);
            amrex::Array4<const amrex::Real> const& Szy = SZ[1]->array(mfi);
            amrex::Array4<const amrex::Real> const& Szz = SZ[2]->array(mfi);

            // Use grown boxes here with all J guard cells
            amrex::Box Jbx = amrex::convert(mfi.validbox(),J[0]->ixType());
            amrex::Box Jby = amrex::convert(mfi.validbox(),J[1]->ixType());
            amrex::Box Jbz = amrex::convert(mfi.validbox(),J[2]->ixType());
            Jbx.grow(J[0]->nGrowVect());
            Jby.grow(J[1]->nGrowVect());
            Jbz.grow(J[2]->nGrowVect());

            // Use same box for E as for J (requires ngE >= ngJ)
            amrex::Box Ebx = Jbx;
            amrex::Box Eby = Jby;
            amrex::Box Ebz = Jbz;

            const amrex::IntVect ncomp_xx = m_ncomp_xx;
            const amrex::IntVect ncomp_xy = m_ncomp_xy;
            const amrex::IntVect ncomp_xz = m_ncomp_xz;
            const amrex::IntVect ncomp_yx = m_ncomp_yx;
            const amrex::IntVect ncomp_yy = m_ncomp_yy;
            const amrex::IntVect ncomp_yz = m_ncomp_yz;
            const amrex::IntVect ncomp_zx = m_ncomp_zx;
            const amrex::IntVect ncomp_zy = m_ncomp_zy;
            const amrex::IntVect ncomp_zz = m_ncomp_zz;

            amrex::ParallelFor(
            Jbx, ncomps, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                const int idx[3] = {i, j, k};
                amrex::GpuArray<int, 3> index_min = {0, 0, 0};
                amrex::GpuArray<int, 3> index_max = {0, 0, 0};

                // Compute Sxx*dEx
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_xx[dim],Ebx.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_xx[dim]-1-offset_xx[dim],Ebx.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SxxdEx = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            int Nc = AMREX_D_TERM( ii+offset_xx[0],
                                   + ncomp_xx[0]*( jj+offset_xx[1] ),
                                   + ncomp_xx[0]*ncomp_xx[1]*( kk+offset_xx[2] ) );
                            SxxdEx += Sxx(i,j,k,Nc)*( Ex(i+ii,j+jj,k+kk,n)
                                                  -  Ex0(i+ii,j+jj,k+kk,n) );
                        }
                    }
                }

                // Compute Sxy*dEy
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_xy[dim],Eby.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_xy[dim]-1-offset_xy[dim],Eby.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SxydEy = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            int Nc = AMREX_D_TERM( ii+offset_xy[0],
                                   + ncomp_xy[0]*( jj+offset_xy[1] ),
                                   + ncomp_xy[0]*ncomp_xy[1]*( kk+offset_xy[2] ) );
                            SxydEy += Sxy(i,j,k,Nc)*( Ey(i+ii,j+jj,k+kk,n)
                                                   - Ey0(i+ii,j+jj,k+kk,n) );
                        }
                    }
                }

                // Compute Sxz*dEz
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_xz[dim],Ebz.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_xz[dim]-1-offset_xz[dim],Ebz.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SxzdEz = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            int Nc = AMREX_D_TERM( ii+offset_xz[0],
                                   + ncomp_xz[0]*( jj+offset_xz[1] ),
                                   + ncomp_xz[0]*ncomp_xz[1]*( kk+offset_xz[2] ) );
                            SxzdEz += Sxz(i,j,k,Nc)*( Ez(i+ii,j+jj,k+kk,n)
                                                   - Ez0(i+ii,j+jj,k+kk,n) );
                        }
                    }
                }

                Jx(i,j,k,n) += Jx0(i,j,k,n) + SxxdEx + SxydEy + SxzdEz;
            });
            amrex::ParallelFor(
            Jby, ncomps, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                const int idx[3] = {i, j, k};
                amrex::GpuArray<int, 3> index_min = {0, 0, 0};
                amrex::GpuArray<int, 3> index_max = {0, 0, 0};

                // Compute Syx*dEx
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_yx[dim],Ebx.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_yx[dim]-1-offset_yx[dim],Ebx.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SyxdEx = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            int Nc = AMREX_D_TERM( ii+offset_yx[0],
                                   + ncomp_yx[0]*( jj+offset_yx[1] ),
                                   + ncomp_yx[0]*ncomp_yx[1]*( kk+offset_yx[2] ) );
                            SyxdEx += Syx(i,j,k,Nc)*( Ex(i+ii,j+jj,k+kk,n)
                                                  -  Ex0(i+ii,j+jj,k+kk,n) );
                        }
                    }
                }

                // Compute Syy*dEy
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_yy[dim],Eby.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_yy[dim]-1-offset_yy[dim],Eby.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SyydEy = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            int Nc = AMREX_D_TERM( ii+offset_yy[0],
                                   + ncomp_yy[0]*( jj+offset_yy[1] ),
                                   + ncomp_yy[0]*ncomp_yy[1]*( kk+offset_yy[2] ) );
                            SyydEy += Syy(i,j,k,Nc)*( Ey(i+ii,j+jj,k+kk,n)
                                                  -  Ey0(i+ii,j+jj,k+kk,n) );
                        }
                    }
                }

                // Compute Syz*dEz
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_yz[dim],Ebz.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_yz[dim]-1-offset_yz[dim],Ebz.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SyzdEz = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            int Nc = AMREX_D_TERM( ii+offset_yz[0],
                                   + ncomp_yz[0]*( jj+offset_yz[1] ),
                                   + ncomp_yz[0]*ncomp_yz[1]*( kk+offset_yz[2] ) );
                            SyzdEz += Syz(i,j,k,Nc)*( Ez(i+ii,j+jj,k+kk,n)
                                                  -  Ez0(i+ii,j+jj,k+kk,n) );
                        }
                    }
                }

                Jy(i,j,k,n) += Jy0(i,j,k,n) + SyxdEx + SyydEy + SyzdEz;
            });
            amrex::ParallelFor(
            Jbz, ncomps, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                const int idx[3] = {i, j, k};
                amrex::GpuArray<int, 3> index_min = {0, 0, 0};
                amrex::GpuArray<int, 3> index_max = {0, 0, 0};

                // Compute Szx*dEx
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_zx[dim],Ebx.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_zx[dim]-1-offset_zx[dim],Ebx.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SzxdEx = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            int Nc = AMREX_D_TERM( ii+offset_zx[0],
                                   + ncomp_zx[0]*( jj+offset_zx[1] ),
                                   + ncomp_zx[0]*ncomp_zx[1]*( kk+offset_zx[2] ) );
                            SzxdEx += Szx(i,j,k,Nc)*( Ex(i+ii,j+jj,k+kk,n)
                                                  -  Ex0(i+ii,j+jj,k+kk,n) );
                        }
                    }
                }

                // Compute Szy*dEy
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_zy[dim],Eby.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_zy[dim]-1-offset_zy[dim],Eby.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SzydEy = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            int Nc = AMREX_D_TERM( ii+offset_zy[0],
                                   + ncomp_zy[0]*( jj+offset_zy[1] ),
                                   + ncomp_zy[0]*ncomp_zy[1]*( kk+offset_zy[2] ) );
                            SzydEy += Szy(i,j,k,Nc)*( Ey(i+ii,j+jj,k+kk,n)
                                                  -  Ey0(i+ii,j+jj,k+kk,n) );
                        }
                    }
                }

                // Compute Szz*dEz
                for (int dim=0; dim<AMREX_SPACEDIM; ++dim) {
                    index_min[dim] = std::max(-offset_zz[dim],Ebz.smallEnd(dim)-idx[dim]);
                    index_max[dim] = std::min(ncomp_zz[dim]-1-offset_zz[dim],Ebz.bigEnd(dim)-idx[dim]);
                }
                amrex::Real SzzdEz = 0.0;
                for (int ii = index_min[0]; ii <= index_max[0]; ++ii) {
                    for (int jj = index_min[1]; jj <= index_max[1]; ++jj) {
                        for (int kk = index_min[2]; kk <= index_max[2]; ++kk) {
                            int Nc = AMREX_D_TERM( ii+offset_zz[0],
                                   + ncomp_zz[0]*( jj+offset_zz[1] ),
                                   + ncomp_zz[0]*ncomp_zz[1]*( kk+offset_zz[2] ) );
                            SzzdEz += Szz(i,j,k,Nc)*( Ez(i+ii,j+jj,k+kk,n)
                                                  -  Ez0(i+ii,j+jj,k+kk,n) );
                        }
                    }
                }

                Jz(i,j,k,n) += Jz0(i,j,k,n) + SzxdEx + SzydEy + SzzdEz;
            });
        }

    }
}


void ImplicitSolver::parseNonlinearSolverParams ( const amrex::ParmParse&  pp )
{

    pp.get("nonlinear_solver", m_nlsolver_type);

    if (m_nlsolver_type == NonlinearSolverType::picard) {

        // Picard
        m_nlsolver = std::make_unique<PicardSolver<WarpXSolverVec,ImplicitSolver>>();
        m_max_particle_iterations = 1;
        m_particle_tolerance = 0.0;

    }
    else if (      (m_nlsolver_type == NonlinearSolverType::newton)
                || (m_nlsolver_type == NonlinearSolverType::petsc_snes) ) {

        // JFNK solvers
        if (m_nlsolver_type == NonlinearSolverType::newton) {
            m_nlsolver = std::make_unique<NewtonSolver<WarpXSolverVec,ImplicitSolver>>();
        } else {
#ifdef AMREX_USE_PETSC
            m_nlsolver = std::make_unique<PETScSNES<WarpXSolverVec,ImplicitSolver>>();
#else
            WARPX_ABORT_WITH_MESSAGE("ImplicitSolver::parseNonlinearSolverParams(): must compile with PETSc to use petsc_snes (AMREX_USE_PETSC must be defined)");
#endif
        }
        pp.query("max_particle_iterations", m_max_particle_iterations);
        pp.query("particle_tolerance", m_particle_tolerance);
        pp.query("particle_suborbits", m_particle_suborbits);
        pp.query("print_unconverged_particle_details", m_print_unconverged_particle_details);
        pp.query("use_mass_matrices_jacobian", m_use_mass_matrices_jacobian);
        pp.query("use_mass_matrices_pc", m_use_mass_matrices_pc);
        if (m_use_mass_matrices_jacobian || m_use_mass_matrices_pc) {
            m_use_mass_matrices = true;
        }
        if (m_use_mass_matrices_jacobian) {
            // Default m_skip_particle_picard_init to true if using suborbits
            if (m_particle_suborbits) { m_skip_particle_picard_init = true; }
            pp.query("skip_particle_picard_init", m_skip_particle_picard_init);
        }
        if (m_use_mass_matrices_pc) {
            m_mass_matrices_pc_width = 0;
#if AMREX_SPACEDIM != 3
            pp.query("mass_matrices_pc_width", m_mass_matrices_pc_width);
#endif
        }
#if defined(WARPX_DIM_RCYLINDER)
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_use_mass_matrices,
            "Using mass matrices is not setup for DIM = RCYLINDER!");
#endif
#if defined(WARPX_DIM_RSPHERE)
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_use_mass_matrices,
            "Using mass matrices is not setup for DIM = RSHERE!");
#endif
#if defined(WARPX_DIM_RZ)
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_use_mass_matrices,
            "Using mass matrices is not setup for DIM = RZ");
#endif
#if defined(WARPX_DIM_3D)
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !m_use_mass_matrices_jacobian,
            "Using mass matrices for jacobian can not be used for DIM = 3");
#endif
        if ( (m_WarpX->current_deposition_algo == CurrentDepositionAlgo::Villasenor ||
              m_WarpX->current_deposition_algo == CurrentDepositionAlgo::Esirkepov) &&
             (m_WarpX->nox < 2) ) {
            std::stringstream warningMsg;
            warningMsg << "Particle-suppressed JFNK (e.g., theta-implicit evolve with newton nonlinear solver) ";
            warningMsg << "is being used with a charge-conserving deposition (esirkepov or villasenor) and particle_shape = 1.\n";
            warningMsg << "Some particle orbits may not converge!!!\n";
            warningMsg << "Consider using particle_shape > 1.\n";
            ablastr::warn_manager::WMRecordWarning("ImplicitSolver", warningMsg.str());
        }
    }
    else {
        WARPX_ABORT_WITH_MESSAGE(
            "invalid nonlinear_solver specified. Valid options are picard and newton.");
    }

}

void ImplicitSolver::InitializeMassMatrices ()
{

    // Initializes the MassMatrices and MassMatrices_PC containers
    // The latter has a reduced number of elements that is used for the preconditioner.
    //
    // dJx = MassMatrices_xx*dEx + MassMatrices_xy*dEy + MassMatrices_xz*dEz
    // dJy = MassMatrices_yx*dEx + MassMatrices_yy*dEy + MassMatrices_yz*dEz
    // dJz = MassMatrices_zx*dEx + MassMatrices_zy*dEy + MassMatrices_zz*dEz

    // check that PC is being used by nonlinear solver
    if (m_use_mass_matrices_pc) {
        const PreconditionerType pc_type = m_nlsolver->GetPreconditionerType();
        if (pc_type == PreconditionerType::none) {
            m_use_mass_matrices_pc = false;
        }
        if (pc_type == PreconditionerType::pc_curl_curl_mlmg) {
            // This PC does not yet support off-diagonal mass matrix terms
            if (m_use_mass_matrices_pc) { m_mass_matrices_pc_width = 0; }
            else { m_mass_matrices_pc_width = -1; }
        }
    }

    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    const int shape = m_WarpX->nox;
    const amrex::IntVect ngJ = m_WarpX->m_fields.get(FieldType::current_fp, Direction{0}, 0)->nGrowVect();
    const amrex::IntVect ngE = m_WarpX->m_fields.get(FieldType::Efield_fp, Direction{0}, 0)->nGrowVect();

    // Get nodal flags for each component of J
    const ablastr::fields::VectorField J = m_WarpX->m_fields.get_alldirs(FieldType::current_fp, 0);
    const amrex::IntVect Jx_nodal = J[0]->ixType().toIntVect();
    const amrex::IntVect Jy_nodal = J[1]->ixType().toIntVect();
    const amrex::IntVect Jz_nodal = J[2]->ixType().toIntVect();

    // Compute the total number of components for each mass matrices container.
    // This depends on the particle shape factor and the type of current deposition.
    int Nc_tot_xx = 1, Nc_tot_xy = 1, Nc_tot_xz = 1;
    int Nc_tot_yx = 1, Nc_tot_yy = 1, Nc_tot_yz = 1;
    int Nc_tot_zx = 1, Nc_tot_zy = 1, Nc_tot_zz = 1;
    if (m_use_mass_matrices_jacobian) {

        for (int dir=0; dir<AMREX_SPACEDIM; dir++) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE( ngE[dir]>=ngJ[dir],
                "Mass Matrices for Jacobian requires guard cells for E "
                "to be at least as many as those for J.");
        }

        if (m_WarpX->current_deposition_algo == CurrentDepositionAlgo::Direct) {
            for (int dir=0; dir<AMREX_SPACEDIM; dir++) {
                m_ncomp_xx[dir] = 1 + 2*shape;
                m_ncomp_xy[dir] = 1 + 2*shape + ( (Jx_nodal[dir] + Jy_nodal[dir]) % 2 );
                m_ncomp_xz[dir] = 1 + 2*shape + ( (Jx_nodal[dir] + Jz_nodal[dir]) % 2 );
                m_ncomp_yy[dir] = 1 + 2*shape;
                m_ncomp_yx[dir] = 1 + 2*shape + ( (Jy_nodal[dir] + Jx_nodal[dir]) % 2 );
                m_ncomp_yz[dir] = 1 + 2*shape + ( (Jy_nodal[dir] + Jz_nodal[dir]) % 2 );
                m_ncomp_zz[dir] = 1 + 2*shape;
                m_ncomp_zx[dir] = 1 + 2*shape + ( (Jz_nodal[dir] + Jx_nodal[dir]) % 2 );
                m_ncomp_zy[dir] = 1 + 2*shape + ( (Jz_nodal[dir] + Jy_nodal[dir]) % 2 );
                //
                Nc_tot_xx *= m_ncomp_xx[dir];
                Nc_tot_xy *= m_ncomp_xy[dir];
                Nc_tot_xz *= m_ncomp_xz[dir];
                Nc_tot_yx *= m_ncomp_yx[dir];
                Nc_tot_yy *= m_ncomp_yy[dir];
                Nc_tot_yz *= m_ncomp_yz[dir];
                Nc_tot_zx *= m_ncomp_zx[dir];
                Nc_tot_zy *= m_ncomp_zy[dir];
                Nc_tot_zz *= m_ncomp_zz[dir];
            }
        }
        else if (m_WarpX->current_deposition_algo == CurrentDepositionAlgo::Villasenor) {
#ifndef WARPX_DIM_3D
            int max_crossings = ngJ[0] - shape + 1;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(max_crossings > 0,
                "Mass Matrices for Jacobian with Villasenor deposition requires particles.max_grid_crossings > 0.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(max_crossings == m_WarpX->particle_max_grid_crossings,
                "Guard cells for J are not consistent with particle_max_grid_crossings.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(max_crossings <= 2,
                "Mass Matrices for Jacobian with Villasenor deposition requires particles.max_grid_crossings <= 2.");
#endif
            // Comment on direction-dependent number of mass matrices components
            // set below for charge-conserving Villasenor deposition:
            // 1 + 2*(shape - 1) (both comps centered)
            // 0 + 2*shape       (mixed nodal/centered comps)
            // 1 + 2*shape       (both comps nodal)
#if defined(WARPX_DIM_1D_Z)
            // x and y are nodal, z is centered
            m_ncomp_xx[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_xy[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_xz[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_yx[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_yy[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_yz[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_zx[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_zy[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_zz[0] = 1 + 2*(shape-1) + 2*max_crossings;
#elif defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
            // x is centered, y and z are nodal
            m_ncomp_xx[0] = 1 + 2*(shape-1) + 2*max_crossings;
            m_ncomp_xy[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_xz[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_yx[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_yy[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_yz[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_zx[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_zy[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_zz[0] = 1 + 2*shape + 2*max_crossings;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            // dir = 0: x is centered, y and z are nodal
            m_ncomp_xx[0] = 1 + 2*(shape-1) + 2*max_crossings;
            m_ncomp_xy[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_xz[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_yx[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_yy[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_yz[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_zx[0] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_zy[0] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_zz[0] = 1 + 2*shape + 2*max_crossings;
            // dir = 1: x and y are nodal, z is centered
            m_ncomp_xx[1] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_xy[1] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_xz[1] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_yx[1] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_yy[1] = 1 + 2*shape + 2*max_crossings;
            m_ncomp_yz[1] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_zx[1] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_zy[1] = 0 + 2*shape + 2*max_crossings;
            m_ncomp_zz[1] = 1 + 2*(shape-1) + 2*max_crossings;
#endif
            for (int dir=0; dir<AMREX_SPACEDIM; dir++) {
                Nc_tot_xx *= m_ncomp_xx[dir];
                Nc_tot_xy *= m_ncomp_xy[dir];
                Nc_tot_xz *= m_ncomp_xz[dir];
                Nc_tot_yx *= m_ncomp_yx[dir];
                Nc_tot_yy *= m_ncomp_yy[dir];
                Nc_tot_yz *= m_ncomp_yz[dir];
                Nc_tot_zx *= m_ncomp_zx[dir];
                Nc_tot_zy *= m_ncomp_zy[dir];
                Nc_tot_zz *= m_ncomp_zz[dir];
            }
        }
        else {
            WARPX_ABORT_WITH_MESSAGE("Mass matrices can only be used with Direct and Villasenor depositions.");
        }
    }
    else { // Mass matrices used for PC only
        for (int dir=0; dir<AMREX_SPACEDIM; dir++) {
            m_ncomp_xx[dir] = 1;
            m_ncomp_xy[dir] = 0;
            m_ncomp_xz[dir] = 0;
            m_ncomp_yx[dir] = 0;
            m_ncomp_yy[dir] = 1;
            m_ncomp_yz[dir] = 0;
            m_ncomp_zx[dir] = 0;
            m_ncomp_zy[dir] = 0;
            m_ncomp_zz[dir] = 1;
            //
            Nc_tot_xx *= m_ncomp_xx[dir];
            Nc_tot_xy *= m_ncomp_xy[dir];
            Nc_tot_xz *= m_ncomp_xz[dir];
            Nc_tot_yx *= m_ncomp_yx[dir];
            Nc_tot_yy *= m_ncomp_yy[dir];
            Nc_tot_yz *= m_ncomp_yz[dir];
            Nc_tot_zx *= m_ncomp_zx[dir];
            Nc_tot_zy *= m_ncomp_zy[dir];
            Nc_tot_zz *= m_ncomp_zz[dir];
        }
    }

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const auto& ba_Jx = m_WarpX->m_fields.get(FieldType::current_fp, Direction{0}, lev)->boxArray();
        const auto& ba_Jy = m_WarpX->m_fields.get(FieldType::current_fp, Direction{1}, lev)->boxArray();
        const auto& ba_Jz = m_WarpX->m_fields.get(FieldType::current_fp, Direction{2}, lev)->boxArray();
        const auto& dm = m_WarpX->m_fields.get(FieldType::current_fp, Direction{0}, lev)->DistributionMap();
        //
        if (m_use_mass_matrices_jacobian) {
            m_WarpX->m_fields.alloc_init(FieldType::Efield_fp_save, Direction{0}, lev, ba_Jx, dm, 1, ngE, 0.0_rt);
            m_WarpX->m_fields.alloc_init(FieldType::Efield_fp_save, Direction{1}, lev, ba_Jy, dm, 1, ngE, 0.0_rt);
            m_WarpX->m_fields.alloc_init(FieldType::Efield_fp_save, Direction{2}, lev, ba_Jz, dm, 1, ngE, 0.0_rt);
        }
        //
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_X, Direction{0}, lev, ba_Jx, dm, Nc_tot_xx, ngJ, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_X, Direction{1}, lev, ba_Jx, dm, Nc_tot_xy, ngJ, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_X, Direction{2}, lev, ba_Jx, dm, Nc_tot_xz, ngJ, 0.0_rt);
        //
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_Y, Direction{0}, lev, ba_Jy, dm, Nc_tot_yx, ngJ, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_Y, Direction{1}, lev, ba_Jy, dm, Nc_tot_yy, ngJ, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_Y, Direction{2}, lev, ba_Jy, dm, Nc_tot_yz, ngJ, 0.0_rt);
        //
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_Z, Direction{0}, lev, ba_Jz, dm, Nc_tot_zx, ngJ, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_Z, Direction{1}, lev, ba_Jz, dm, Nc_tot_zy, ngJ, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_Z, Direction{2}, lev, ba_Jz, dm, Nc_tot_zz, ngJ, 0.0_rt);
        //
        if (m_use_mass_matrices_pc) {
            int ncomp_tot_pc_xx = 1;
            int ncomp_tot_pc_yy = 1;
            int ncomp_tot_pc_zz = 1;
            if (m_use_mass_matrices_jacobian) { // Additional MM components in PC not setup yet
                                                // for when MM is only used for the PC
                const int ncomp_dir_pc = 1 + 2*m_mass_matrices_pc_width;
                for (int dir=0; dir<AMREX_SPACEDIM; dir++) {
                    m_ncomp_pc_xx[dir] = std::min(m_ncomp_xx[dir],ncomp_dir_pc);
                    m_ncomp_pc_yy[dir] = std::min(m_ncomp_yy[dir],ncomp_dir_pc);
                    m_ncomp_pc_zz[dir] = std::min(m_ncomp_zz[dir],ncomp_dir_pc);
                    ncomp_tot_pc_xx *= m_ncomp_pc_xx[dir];
                    ncomp_tot_pc_yy *= m_ncomp_pc_yy[dir];
                    ncomp_tot_pc_zz *= m_ncomp_pc_zz[dir];
                }
            }
            m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_PC, Direction{0}, lev, ba_Jx, dm, ncomp_tot_pc_xx, ngJ, 0.0_rt);
            m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_PC, Direction{1}, lev, ba_Jy, dm, ncomp_tot_pc_yy, ngJ, 0.0_rt);
            m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_PC, Direction{2}, lev, ba_Jz, dm, ncomp_tot_pc_zz, ngJ, 0.0_rt);
        }
    }

    // Set the pointer to mass matrix MultiFab
    if (m_use_mass_matrices_pc) {
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            m_mmpc_mfarrvec.push_back(m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_PC, lev));
        }
    }

}

void ImplicitSolver::PreLinearSolve ()
{
    BL_PROFILE("ImplicitSolver::PreLinearSolve()");

    if (m_use_mass_matrices) {

        m_WarpX->DepositMassMatrices();

        if (m_use_mass_matrices_jacobian) {
            FinishMassMatrices();
            SaveE();
        }

        if (m_use_mass_matrices_pc) {
            SyncMassMatricesPCAndApplyBCs();
            const amrex::Real theta_dt = m_theta*m_dt;
            SetMassMatricesForPC( theta_dt );
        }

    }

}

void ImplicitSolver::PreRHSOp ( const amrex::Real  a_cur_time,
                                const int          a_nl_iter,
                                const bool         a_from_jacobian )
{
    BL_PROFILE("ImplicitSolver::PreRHSOp()");

    using warpx::fields::FieldType;

    if (m_WarpX->use_filter) {
        int finest_level = 0;
        m_WarpX->ApplyFilterMF(m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level), 0);
    }

    // Advance the particle positions by 1/2 dt,
    // particle velocities by dt, then take average of old and new v,
    // deposit currents, giving J at n+1/2
    // This uses Efield_fp and Bfield_fp, the field at n+1/2 from the previous iteration.
    const bool skip_deposition = false;

    // Set the implict solver options for particles and setting the current density
    ImplicitOptions options;
    options.linear_stage_of_jfnk = a_from_jacobian;
    options.use_mass_matrices_pc = m_use_mass_matrices_pc;
    options.use_mass_matrices_jacobian = m_use_mass_matrices_jacobian;
    options.evolve_suborbit_particles_only = false;

    if (a_nl_iter == 0 && !a_from_jacobian &&
        m_use_mass_matrices_jacobian && m_skip_particle_picard_init) {
        // Only do a single Picard iteration for particles on the initial Newton step
        options.max_particle_iterations = 1;
        options.particle_tolerance = 0.0;
    }
    else {
        options.max_particle_iterations = m_max_particle_iterations;
        options.particle_tolerance = m_particle_tolerance;
    }

    if (m_use_mass_matrices_jacobian && a_from_jacobian) { // Called from linear stage of JFNK and using mass matrices for Jacobian
        if (m_particle_suborbits) {
            options.evolve_suborbit_particles_only = true;
            m_WarpX->PushParticlesandDeposit(a_cur_time, skip_deposition, PositionPushType::Full, MomentumPushType::Full, &options);
        }
        const bool J_from_MM_only = !options.evolve_suborbit_particles_only;
        ComputeJfromMassMatrices( J_from_MM_only );
    }
    else { // Conventional particle-suppressed JFNK
        m_WarpX->PushParticlesandDeposit(a_cur_time, skip_deposition, PositionPushType::Full, MomentumPushType::Full, &options);
        CumulateJ();
    }

#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        ablastr::fields::VectorField J = m_WarpX->m_fields.get_alldirs(FieldType::current_fp, lev);
        m_WarpX->ApplyInverseVolumeScalingToCurrentDensity(J[0], J[1], J[2], lev);
    }
#endif

    // Apply BCs to J and communicate
    m_WarpX->SyncCurrentAndRho();

    if (m_nlsolver_type == NonlinearSolverType::petsc_snes && !a_from_jacobian) {
        // The native Newton solver calls this routine immediately before the linear solve,
        // and only when a linear solve is required (i.e., the system is not converged).
        // PETSc's SNES solver does not provide this optimization, so we must call it here.
        PreLinearSolve();
    }

}

void ImplicitSolver::SyncMassMatricesPCAndApplyBCs ()
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // Add select mass matrices elements to the preconditioner containers,
    // which may alread include contributions from suborbit particles that
    // are not included in the mass matrices.

    const int diag_comp_xx = (AMREX_D_TERM(m_ncomp_xx[0],*m_ncomp_xx[1],*m_ncomp_xx[2])-1)/2;
    const int diag_comp_yy = (AMREX_D_TERM(m_ncomp_yy[0],*m_ncomp_yy[1],*m_ncomp_yy[2])-1)/2;
    const int diag_comp_zz = (AMREX_D_TERM(m_ncomp_zz[0],*m_ncomp_zz[1],*m_ncomp_zz[2])-1)/2;
    int MM_PC_ncomp_xx[3] = {1, 1, 1};
    int MM_PC_ncomp_yy[3] = {1, 1, 1};
    int MM_PC_ncomp_zz[3] = {1, 1, 1};
    int MM_PC_width_xx[3] = {0, 0, 0};
    int MM_PC_width_yy[3] = {0, 0, 0};
    int MM_PC_width_zz[3] = {0, 0, 0};
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
        MM_PC_ncomp_xx[dir]  = m_ncomp_pc_xx[dir];
        MM_PC_ncomp_yy[dir]  = m_ncomp_pc_yy[dir];
        MM_PC_ncomp_zz[dir]  = m_ncomp_pc_zz[dir];
        MM_PC_width_xx[dir]  = (m_ncomp_pc_xx[dir] - 1)/2;
        MM_PC_width_yy[dir]  = (m_ncomp_pc_yy[dir] - 1)/2;
        MM_PC_width_zz[dir]  = (m_ncomp_pc_zz[dir] - 1)/2;
    }

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {

        const amrex::MultiFab* MM_xx = m_WarpX->m_fields.get(FieldType::MassMatrices_X, Direction{0}, lev);
        const amrex::MultiFab* MM_yy = m_WarpX->m_fields.get(FieldType::MassMatrices_Y, Direction{1}, lev);
        const amrex::MultiFab* MM_zz = m_WarpX->m_fields.get(FieldType::MassMatrices_Z, Direction{2}, lev);
        ablastr::fields::VectorField MM_PC = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_PC, lev);

        // Below is general for 1D and 2D. It works for 3D because for now we limit width = 0 in 3D.

        const int diag_comp_pc_xx = (MM_PC[0]->nComp() - 1)/2;
        for (int comp1 = 0; comp1 < MM_PC_ncomp_xx[1]; comp1++) {
            const int jj0 = comp1 - MM_PC_width_xx[1]; // -2 -1, 0, 1, 2
            const int mm_comp_start    = diag_comp_xx    - MM_PC_width_xx[0] + m_ncomp_xx[0]*jj0;
            const int mm_pc_comp_start = diag_comp_pc_xx - MM_PC_width_xx[0] + m_ncomp_pc_xx[0]*jj0;
            amrex::MultiFab::Add(*MM_PC[0], *MM_xx, mm_comp_start, mm_pc_comp_start, m_ncomp_pc_xx[0], MM_xx->nGrowVect());
        }
        const int diag_comp_pc_yy = (MM_PC[1]->nComp() - 1)/2;
        for (int comp1 = 0; comp1 < MM_PC_ncomp_yy[1]; comp1++) {
            const int jj0 = comp1 - MM_PC_width_yy[1]; // -2 -1, 0, 1, 2
            const int mm_comp_start    = diag_comp_yy    - MM_PC_width_yy[0] + m_ncomp_yy[0]*jj0;
            const int mm_pc_comp_start = diag_comp_pc_yy - MM_PC_width_yy[0] + m_ncomp_pc_yy[0]*jj0;
            amrex::MultiFab::Add(*MM_PC[1], *MM_yy, mm_comp_start, mm_pc_comp_start, m_ncomp_pc_yy[0], MM_yy->nGrowVect());
        }
        const int diag_comp_pc_zz = (MM_PC[2]->nComp() - 1)/2;
        for (int comp1 = 0; comp1 < MM_PC_ncomp_zz[1]; comp1++) {
            const int jj0 = comp1 - MM_PC_width_zz[1]; // -2 -1, 0, 1, 2
            const int mm_comp_start    = diag_comp_zz    - MM_PC_width_zz[0] + m_ncomp_zz[0]*jj0;
            const int mm_pc_comp_start = diag_comp_pc_zz - MM_PC_width_zz[0] + m_ncomp_pc_zz[0]*jj0;
            amrex::MultiFab::Add(*MM_PC[2], *MM_zz, mm_comp_start, mm_pc_comp_start, m_ncomp_pc_zz[0], MM_zz->nGrowVect());
        }

    }

    // Do addOp Exchange on MassMatrices_PC
    m_WarpX->SyncMassMatricesPC();

    // Apply BCs to MassMatrices_PC
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        m_WarpX->ApplyJfieldBoundary(lev,
            m_WarpX->m_fields.get(FieldType::MassMatrices_PC, Direction{0}, lev),
            m_WarpX->m_fields.get(FieldType::MassMatrices_PC, Direction{1}, lev),
            m_WarpX->m_fields.get(FieldType::MassMatrices_PC, Direction{2}, lev),
            PatchType::fine);
    }
}

void ImplicitSolver::SetMassMatricesForPC ( const amrex::Real a_theta_dt )
{

    using namespace amrex::literals;
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // Scale mass matrices used by preconditioner by c^2*mu0*theta*dt.
    // Add one to diagonal terms when using the curl_curl_mlmg pc_type.
    // The pc_type petsc already has the one from the curl curl operator
    // Note: This should be done after Sync/communication has been called

    const amrex::Real pc_factor = PhysConst::c * PhysConst::c * PhysConst::mu0 * a_theta_dt;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        amrex::MultiFab* MMxx_PC = m_WarpX->m_fields.get(FieldType::MassMatrices_PC, Direction{0}, lev);
        amrex::MultiFab* MMyy_PC = m_WarpX->m_fields.get(FieldType::MassMatrices_PC, Direction{1}, lev);
        amrex::MultiFab* MMzz_PC = m_WarpX->m_fields.get(FieldType::MassMatrices_PC, Direction{2}, lev);
        MMxx_PC->mult(pc_factor, 0, MMxx_PC->nComp());
        MMyy_PC->mult(pc_factor, 0, MMyy_PC->nComp());
        MMzz_PC->mult(pc_factor, 0, MMzz_PC->nComp());
        const PreconditionerType pc_type = m_nlsolver->GetPreconditionerType();
        if (pc_type == PreconditionerType::pc_curl_curl_mlmg) {
            // Need to add 1 to the diagonal terms for the curl_curl pc
            const int diag_comp_Mxx = (MMxx_PC->nComp()-1)/2;
            const int diag_comp_Myy = (MMyy_PC->nComp()-1)/2;
            const int diag_comp_Mzz = (MMzz_PC->nComp()-1)/2;
            MMxx_PC->plus(1.0_rt, diag_comp_Mxx, 1, 0);
            MMyy_PC->plus(1.0_rt, diag_comp_Myy, 1, 0);
            MMzz_PC->plus(1.0_rt, diag_comp_Mzz, 1, 0);
        }
    }

}

void ImplicitSolver::FinishMassMatrices ()
{
    BL_PROFILE("ImplicitSolver::FinishMassMatrices()");

    // The MM deposit routine takes advantage of symmetry for the diagonal mass
    // matrices to only deposit roughly half of the values. The remainder are
    // computed via copy here in this routine.

#if AMREX_SPACEDIM < 3
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

#if AMREX_SPACEDIM > 1
    const int ncomp_tot_xx = AMREX_D_TERM(m_ncomp_xx[0],*m_ncomp_xx[1],*m_ncomp_xx[2]);
    const int ncomp_tot_yy = AMREX_D_TERM(m_ncomp_yy[0],*m_ncomp_yy[1],*m_ncomp_yy[2]);
    const int ncomp_tot_zz = AMREX_D_TERM(m_ncomp_zz[0],*m_ncomp_zz[1],*m_ncomp_zz[2]);
#endif

    amrex::GpuArray<int,3> ncomp_xx = {1,1,1};
    amrex::GpuArray<int,3> ncomp_yy = {1,1,1};
    amrex::GpuArray<int,3> ncomp_zz = {1,1,1};
    amrex::GpuArray<int,3> Sxx_width = {0,0,0};
    amrex::GpuArray<int,3> Syy_width = {0,0,0};
    amrex::GpuArray<int,3> Szz_width = {0,0,0};
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
        ncomp_xx[dir] = m_ncomp_xx[dir];
        ncomp_yy[dir] = m_ncomp_yy[dir];
        ncomp_zz[dir] = m_ncomp_zz[dir];
        Sxx_width[dir] = (ncomp_xx[dir] - 1)/2;
        Syy_width[dir] = (ncomp_yy[dir] - 1)/2;
        Szz_width[dir] = (ncomp_zz[dir] - 1)/2;
    }

    for (int lev = 0; lev < m_num_amr_levels; ++lev) {

        ablastr::fields::VectorField SX = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_X, lev);
        ablastr::fields::VectorField SY = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_Y, lev);
        ablastr::fields::VectorField SZ = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_Z, lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for ( amrex::MFIter mfi(*SX[0], false); mfi.isValid(); ++mfi )
        {

            amrex::Array4<amrex::Real> const& Sxx = SX[0]->array(mfi);
            amrex::Array4<amrex::Real> const& Syy = SY[1]->array(mfi);
            amrex::Array4<amrex::Real> const& Szz = SZ[2]->array(mfi);

            // Use grown boxes here with all S guard cells
            amrex::Box Sbx = amrex::convert(mfi.validbox(),SX[0]->ixType());
            amrex::Box Sby = amrex::convert(mfi.validbox(),SY[1]->ixType());
            amrex::Box Sbz = amrex::convert(mfi.validbox(),SZ[2]->ixType());
            Sbx.grow(SX[0]->nGrowVect());
            Sbz.grow(SZ[2]->nGrowVect());
            Sby.grow(SY[1]->nGrowVect());

#if AMREX_SPACEDIM == 1
            amrex::ParallelFor( Sbx, Sby, Sbz,

                [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // Sxx(i,d + n) = Sxx(i + n,d - n), where d = Sxx_width[0]
                const int width = amrex::min(Sxx_width[0],Sbx.bigEnd(0)-i);
                for (int n = 1; n <= width; ++n) {
                    const int dst_comp = Sxx_width[0] + n;
                    const int src_comp = Sxx_width[0] - n;
                    Sxx(i,j,k,dst_comp) = Sxx(i + n,j,k,src_comp);
                }
            },

                [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // Syy(i,d + n) = Syy(i + n,d - n), where d = Syy_width[0]
                const int width = std::min(Syy_width[0], Sby.bigEnd(0) - i);
                for (int n = 1; n <= width; n++) {
                    const int dst_comp = Syy_width[0] + n;
                    const int src_comp = Syy_width[0] - n;
                    Syy(i,j,k,dst_comp) = Syy(i + n,j,k,src_comp);
                }
            },

                [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // Szz(i,d + n) = Szz(i + n,d - n), where d = Szz_width[0]
                const int width_zz = std::min(Szz_width[0],Sbz.bigEnd(0) - i);
                for (int n = 1; n <= width_zz; n++) {
                    const int dst_comp = Szz_width[0] + n;
                    const int src_comp = Szz_width[0] - n;
                    Szz(i,j,k,dst_comp) = Szz(i + n,j,k,src_comp);
                }
            });

#elif AMREX_SPACEDIM == 2
            amrex::ParallelFor( Sbx, Sby, Sbz,

                [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                ignore_unused(k);
                const amrex::IntVect iv_dst = amrex::IntVect(AMREX_D_DECL(i,j,k));

                const int row_start = amrex::max(0,ncomp_xx[1] - ncomp_xx[0]);

                for (int m = row_start; m < ncomp_xx[1]; ++m) {
                    const int jj = m - Sxx_width[1];

                    const int above_diag = (m > Sxx_width[1]) ? 1 : 0;
                    const int width0 = amrex::min(m + above_diag - row_start + 1, ncomp_xx[0]);

                    for (int n = 0; n < width0; ++n) {
                        const int ii = Sxx_width[0] - n;

                        const amrex::IntVect iv_src = iv_dst + amrex::IntVect(AMREX_D_DECL(ii,jj,0));
                        if (!Sbx.contains(iv_src)) { continue; }

                        const int dst_comp = ncomp_xx[0]*(m + 1) - (n + 1);
                        const int src_comp = ncomp_tot_xx - 1 - dst_comp;

                        Sxx(iv_dst,dst_comp) = Sxx(iv_src,src_comp);
                    }

                }

            },

                [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                ignore_unused(k);
                const amrex::IntVect iv_dst = amrex::IntVect(AMREX_D_DECL(i,j,k));

                const int row_start = 1;

                for (int m = row_start; m < ncomp_yy[1]; m++) {
                    const int jj = m - Syy_width[1];

                    const int above_diag = (m > Syy_width[1]) ? 1 : 0;
                    const int width0 = std::min(m + above_diag - row_start + 1, ncomp_yy[0]);

                    for (int n = 0; n < width0; n++) {
                        const int ii = Syy_width[0] - n;

                        const amrex::IntVect iv_src = iv_dst + amrex::IntVect(AMREX_D_DECL(ii,jj,0));
                        if (!Sby.contains(iv_src)) { continue; }

                        const int dst_comp = ncomp_yy[0]*(m + 1) - (n + 1);
                        const int src_comp = ncomp_tot_yy - 1 - dst_comp;

                        Syy(iv_dst,dst_comp) = Syy(iv_src,src_comp);
                    }
                }

            },

                [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                ignore_unused(k);
                const amrex::IntVect iv_dst = amrex::IntVect(AMREX_D_DECL(i,j,k));

                const int row_start = std::max(0,ncomp_zz[1] - ncomp_zz[0]);

                for (int m = row_start; m < ncomp_zz[1]; m++) {
                    const int jj = m - Szz_width[1];

                    const int above_diag = (m > Szz_width[1]) ? 1 : 0;
                    const int width0 = std::min(m - row_start + above_diag + 1, ncomp_zz[0]);

                    for (int n = 0; n < width0; n++) {
                        const int ii = Szz_width[0] - n;

                        const amrex::IntVect iv_src = iv_dst + amrex::IntVect(AMREX_D_DECL(ii,jj,0));
                        if (!Sbz.contains(iv_src)) { continue; }

                        const int dst_comp = ncomp_zz[0]*(m + 1) - (n + 1);
                        const int src_comp = ncomp_tot_zz - 1 - dst_comp;

                        Szz(iv_dst,dst_comp) = Szz(iv_src,src_comp);
                    }
                }

            });
#endif
        }
    }
#endif
}

void ImplicitSolver::PrintBaseImplicitSolverParameters () const
{
    amrex::Print() << "max particle iterations:             " << m_max_particle_iterations << "\n";
    amrex::Print() << "particle relative tolerance:         " << m_particle_tolerance << "\n";
    amrex::Print() << "use particle suborbits:              " << (m_particle_suborbits ? "true":"false") << "\n";
    amrex::Print() << "print unconverged particle details:  " << (m_print_unconverged_particle_details ? "true":"false") << "\n";
    amrex::Print() << "Nonlinear solver type:               " << amrex::getEnumNameString(m_nlsolver_type) << "\n";
    if ( (m_nlsolver_type == NonlinearSolverType::newton)
      || (m_nlsolver_type == NonlinearSolverType::petsc_snes) ) {
        amrex::Print() << "use mass matrices:                   " << (m_use_mass_matrices ? "true":"false") << "\n";
        if (m_use_mass_matrices) {
            amrex::Print() << "    for jacobian calc:   " << (m_use_mass_matrices_jacobian ? "true":"false") << "\n";
            if (m_use_mass_matrices_jacobian) {
                amrex::Print() << "        skip particle picard init:  " << (m_skip_particle_picard_init ? "true":"false") << "\n";
            }
            amrex::Print() << "    for preconditioner:  " << (m_use_mass_matrices_pc ? "true":"false") << "\n";
            if (m_use_mass_matrices_pc) {
                amrex::Print() << "    mass matrices pc width:  " << m_mass_matrices_pc_width << "\n";
            }
            amrex::Print() << "    ncomp_xx:  " << m_ncomp_xx << ";  ncomp_pc_xx:  " << m_ncomp_pc_xx << "\n";
            amrex::Print() << "    ncomp_xy:  " << m_ncomp_xy << ";  ncomp_pc_xy:  " << amrex::IntVect(0) << "\n";
            amrex::Print() << "    ncomp_xz:  " << m_ncomp_xz << ";  ncomp_pc_xz:  " << amrex::IntVect(0) << "\n";
            amrex::Print() << "    ncomp_yx:  " << m_ncomp_yx << ";  ncomp_pc_yx:  " << amrex::IntVect(0) << "\n";
            amrex::Print() << "    ncomp_yy:  " << m_ncomp_yy << ";  ncomp_pc_yy:  " << m_ncomp_pc_yy << "\n";
            amrex::Print() << "    ncomp_yz:  " << m_ncomp_yz << ";  ncomp_pc_yz:  " << amrex::IntVect(0) << "\n";
            amrex::Print() << "    ncomp_zx:  " << m_ncomp_zx << ";  ncomp_pc_zx:  " << amrex::IntVect(0) << "\n";
            amrex::Print() << "    ncomp_zy:  " << m_ncomp_zy << ";  ncomp_pc_zy:  " << amrex::IntVect(0) << "\n";
            amrex::Print() << "    ncomp_zz:  " << m_ncomp_zz << ";  ncomp_pc_zz:  " << m_ncomp_pc_zz << "\n";
        }
    }
}
