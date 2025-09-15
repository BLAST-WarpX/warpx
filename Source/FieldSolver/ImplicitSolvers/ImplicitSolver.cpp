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
    return convertFieldBCToLinOpBC(m_WarpX->GetFieldBoundaryLo());
}

Array<LinOpBCType,AMREX_SPACEDIM> ImplicitSolver::GetLinOpBCHi () const
{
    return convertFieldBCToLinOpBC(m_WarpX->GetFieldBoundaryHi());
}

Array<LinOpBCType,AMREX_SPACEDIM> ImplicitSolver::convertFieldBCToLinOpBC (const Array<FieldBoundaryType,AMREX_SPACEDIM>& a_fbc) const
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
                "With SilverMueller, in the Curl-Curl preconditioner Neumann boundary will be used since the full boundary is not yet implemented.",
                ablastr::warn_manager::WarnPriority::medium);
            lbc[i] = LinOpBCType::symmetry;
        } else if (a_fbc[i] == FieldBoundaryType::Neumann) {
            // Also for FieldBoundaryType::PMC
            lbc[i] = LinOpBCType::symmetry;
        } else if (a_fbc[i] == FieldBoundaryType::PECInsulator) {
            ablastr::warn_manager::WMRecordWarning("Implicit solver",
                "With PECInsulator, in the Curl-Curl preconditioner Neumann boundary will be used since the full boundary is not yet implemented.",
                ablastr::warn_manager::WarnPriority::medium);
            lbc[i] = LinOpBCType::symmetry;
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

void ImplicitSolver::SaveEandJ ()
{

    // Copy Efield_fp and current_fp to Efield_fp_save and current_fp_save
    // Do this BEFORE call to SyncCurrentAndRho()

    using warpx::fields::FieldType;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        ablastr::fields::VectorField E = m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, lev);
        ablastr::fields::VectorField E0 = m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp_save, lev);
        amrex::MultiFab::Copy(*E0[0], *E[0], 0, 0, E[0]->nComp(), E[0]->nGrowVect());
        amrex::MultiFab::Copy(*E0[1], *E[1], 0, 0, E[1]->nComp(), E[1]->nGrowVect());
        amrex::MultiFab::Copy(*E0[2], *E[2], 0, 0, E[2]->nComp(), E[2]->nGrowVect());

        ablastr::fields::VectorField J = m_WarpX->m_fields.get_alldirs(FieldType::current_fp, lev);
        ablastr::fields::VectorField J0 = m_WarpX->m_fields.get_alldirs(FieldType::current_fp_save, lev);
        amrex::MultiFab::Copy(*J0[0], *J[0], 0, 0, J[0]->nComp(), J[0]->nGrowVect());
        amrex::MultiFab::Copy(*J0[1], *J[1], 0, 0, J[1]->nComp(), J[1]->nGrowVect());
        amrex::MultiFab::Copy(*J0[2], *J[2], 0, 0, J[2]->nComp(), J[2]->nGrowVect());
    }

}

void ImplicitSolver::ComputeJfromMassMatrices ()
{
    BL_PROFILE("ImplicitSolver::ComputeJfromMassMatrices()");
    using namespace amrex::literals;

    using warpx::fields::FieldType;
    using ablastr::fields::Direction;
    const int ncomps = 1;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {

        ablastr::fields::VectorField J = m_WarpX->m_fields.get_alldirs(FieldType::current_fp, lev);
        ablastr::fields::VectorField E = m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp, lev);
        ablastr::fields::VectorField J0 = m_WarpX->m_fields.get_alldirs(FieldType::current_fp_save, lev);
        ablastr::fields::VectorField E0 = m_WarpX->m_fields.get_alldirs(FieldType::Efield_fp_save, lev);

        ablastr::fields::VectorField SX = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_X, lev);
        ablastr::fields::VectorField SY = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_Y, lev);
        ablastr::fields::VectorField SZ = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_Z, lev);

        const amrex::IntVect Jx_nodal = J[0]->ixType().toIntVect();
        const amrex::IntVect Jy_nodal = J[1]->ixType().toIntVect();
        const amrex::IntVect Jz_nodal = J[2]->ixType().toIntVect();

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

                Jx(i,j,k,n) = Jx0(i,j,k,n) + SxxdEx + SxydEy + SxzdEz;
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

                Jy(i,j,k,n) = Jy0(i,j,k,n) + SyxdEx + SyydEy + SyzdEz;
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

                Jz(i,j,k,n) = Jz0(i,j,k,n) + SzxdEx + SzydEy + SzzdEz;
            });
        }

    }
}


void ImplicitSolver::parseNonlinearSolverParams ( const amrex::ParmParse&  pp )
{

    std::string nlsolver_type_str;
    pp.get("nonlinear_solver", nlsolver_type_str);

    if (nlsolver_type_str=="picard") {
        m_nlsolver_type = NonlinearSolverType::Picard;
        m_nlsolver = std::make_unique<PicardSolver<WarpXSolverVec,ImplicitSolver>>();
        m_max_particle_iterations = 1;
        m_particle_tolerance = 0.0;
    }
    else if (nlsolver_type_str=="newton") {
        m_nlsolver_type = NonlinearSolverType::Newton;
        m_nlsolver = std::make_unique<NewtonSolver<WarpXSolverVec,ImplicitSolver>>();
        pp.query("max_particle_iterations", m_max_particle_iterations);
        pp.query("particle_tolerance", m_particle_tolerance);
        pp.query("particle_suborbits", m_particle_suborbits);
        pp.query("print_unconverged_particle_details", m_print_unconverged_particle_details);
        pp.query("use_mass_matrices_jacobian", m_use_mass_matrices_jacobian);
        pp.query("use_mass_matrices_pc", m_use_mass_matrices_pc);
        if (m_use_mass_matrices_jacobian || m_use_mass_matrices_pc) {
            m_use_mass_matrices = true;
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
        if (!m_nlsolver->UsePreconditioner()) {
            m_use_mass_matrices_pc = false;
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
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE( max_crossings>0,
                "Mass Matrices for Jacobian with Villasenor deposition requires particles.max_grid_crossings > 0.");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE( max_crossings==m_WarpX->particle_max_grid_crossings,
                "Guard cells for J are not consistent with particle_max_grid_crossings.");
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
            //
            m_WarpX->m_fields.alloc_init(FieldType::current_fp_save, Direction{0}, lev, ba_Jx, dm, 1, ngJ, 0.0_rt);
            m_WarpX->m_fields.alloc_init(FieldType::current_fp_save, Direction{1}, lev, ba_Jy, dm, 1, ngJ, 0.0_rt);
            m_WarpX->m_fields.alloc_init(FieldType::current_fp_save, Direction{2}, lev, ba_Jz, dm, 1, ngJ, 0.0_rt);
        }
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
            m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_PC, Direction{0}, lev, ba_Jx, dm, 1, ngJ, 0.0_rt);
            m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_PC, Direction{1}, lev, ba_Jy, dm, 1, ngJ, 0.0_rt);
            m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_PC, Direction{2}, lev, ba_Jz, dm, 1, ngJ, 0.0_rt);
        }
    }

    // Set the pointer to mass matrix MultiFab
    if (m_use_mass_matrices_pc) {
        for (int lev = 0; lev < m_num_amr_levels; ++lev) {
            m_mmpc_mfarrvec.push_back(m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_PC, 0));
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
    const bool skip_current = false;

    // Set the implict solver options for particles and setting the current density
    ImplicitOptions options;
    options.nonlinear_iteration = a_nl_iter;
    options.max_particle_iterations = m_max_particle_iterations;
    options.particle_tolerance = m_particle_tolerance;
    options.linear_stage_of_jfnk = a_from_jacobian;

    if (m_use_mass_matrices && !a_from_jacobian) { // Called from non-linear stage of JFNK and using mass matrices
        options.deposit_mass_matrices = true;
        m_WarpX->PushParticlesandDeposit(a_cur_time, skip_current, &options);
        if (m_use_mass_matrices_jacobian) { SaveEandJ(); }
        if (m_use_mass_matrices_pc) {
           SyncMassMatricesPCAndApplyBCs();
           const amrex::Real theta_dt = m_theta*m_dt;
           SetMassMatricesForPC( theta_dt );
        }
    }
    else if (m_use_mass_matrices_jacobian) { // Called from linear stage of JFNK and using mass matrices
        ComputeJfromMassMatrices();
    }
    else {  // Conventional particle-suppressed JFNK
        options.deposit_mass_matrices = false;
        m_WarpX->PushParticlesandDeposit(a_cur_time, skip_current, &options);
    }

    // Apply BCs to J and communicate
    m_WarpX->SyncCurrentAndRho();

}

void ImplicitSolver::SyncMassMatricesPCAndApplyBCs ()
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // Copy mass matrices elements used for the preconditioner
    const int diag_comp_xx = (AMREX_D_TERM(m_ncomp_xx[0],*m_ncomp_xx[1],*m_ncomp_xx[2])-1)/2;
    const int diag_comp_yy = (AMREX_D_TERM(m_ncomp_yy[0],*m_ncomp_yy[1],*m_ncomp_yy[2])-1)/2;
    const int diag_comp_zz = (AMREX_D_TERM(m_ncomp_zz[0],*m_ncomp_zz[1],*m_ncomp_zz[2])-1)/2;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        amrex::MultiFab* MM_xx = m_WarpX->m_fields.get(FieldType::MassMatrices_X, Direction{0}, lev);
        amrex::MultiFab* MM_yy = m_WarpX->m_fields.get(FieldType::MassMatrices_Y, Direction{1}, lev);
        amrex::MultiFab* MM_zz = m_WarpX->m_fields.get(FieldType::MassMatrices_Z, Direction{2}, lev);
        ablastr::fields::VectorField MM_PC = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_PC, lev);
        amrex::MultiFab::Copy(*MM_PC[0], *MM_xx, diag_comp_xx, 0, 1, MM_xx->nGrowVect());
        amrex::MultiFab::Copy(*MM_PC[1], *MM_yy, diag_comp_yy, 0, 1, MM_yy->nGrowVect());
        amrex::MultiFab::Copy(*MM_PC[2], *MM_zz, diag_comp_zz, 0, 1, MM_zz->nGrowVect());
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

    // Scale mass matrices used by preconditioner by c^2*mu0*theta*dt and add 1 to diagonal terms
    // Note: This should be done after Sync/communication has been called

    const amrex::Real pc_factor = PhysConst::c * PhysConst::c * PhysConst::mu0 * a_theta_dt;
    const int diag_comp = 0;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        for (int dir = 0 ; dir < 3 ; dir++) {
            amrex::MultiFab* MM_PC = m_WarpX->m_fields.get(FieldType::MassMatrices_PC, Direction{dir}, lev);
            MM_PC->mult(pc_factor, 0, MM_PC->nComp());
            MM_PC->plus(1.0_rt, diag_comp, 1, 0);
        }
    }

}

void ImplicitSolver::PrintBaseImplicitSolverParameters () const
{
    amrex::Print() << "max particle iterations:             " << m_max_particle_iterations << "\n";
    amrex::Print() << "particle relative tolerance:         " << m_particle_tolerance << "\n";
    amrex::Print() << "use particle suborbits:              " << (m_particle_suborbits ? "true":"false") << "\n";
    amrex::Print() << "print unconverged particle details:  " << (m_print_unconverged_particle_details ? "true":"false") << "\n";
    amrex::Print() << "Nonlinear solver type:               " << amrex::getEnumNameString(m_nlsolver_type) << "\n";
    if (m_nlsolver_type==NonlinearSolverType::Newton) {
        amrex::Print() << "use mass matrices:                   " << (m_use_mass_matrices ? "true":"false") << "\n";
        if (m_use_mass_matrices) {
            amrex::Print() << "    for jacobian calc:   " << (m_use_mass_matrices_jacobian ? "true":"false") << "\n";
            amrex::Print() << "    for preconditioner:  " << (m_use_mass_matrices_pc ? "true":"false") << "\n";
            amrex::Print() << "    ncomp_xx:  " << m_ncomp_xx << "\n";
            amrex::Print() << "    ncomp_xy:  " << m_ncomp_xy << "\n";
            amrex::Print() << "    ncomp_xz:  " << m_ncomp_xz << "\n";
            amrex::Print() << "    ncomp_yx:  " << m_ncomp_yx << "\n";
            amrex::Print() << "    ncomp_yy:  " << m_ncomp_yy << "\n";
            amrex::Print() << "    ncomp_yz:  " << m_ncomp_yz << "\n";
            amrex::Print() << "    ncomp_zx:  " << m_ncomp_zx << "\n";
            amrex::Print() << "    ncomp_zy:  " << m_ncomp_zy << "\n";
            amrex::Print() << "    ncomp_zz:  " << m_ncomp_zz << "\n";
        }
    }
}
