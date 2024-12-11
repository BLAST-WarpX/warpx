/* Copyright 2022 S. Eric Clark, LLNL
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#include "Fields.H"
#include "FieldSolver/MagnetostaticSolver/MagnetostaticSolver.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Parallelization/GuardCellManager.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Python/callbacks.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXUtil.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "Parallelization/WarpXComm_K.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/utils/Communication.H>
#include <ablastr/warn_manager/WarnManager.H>
#include <ablastr/fields/VectorPoissonSolver.H>

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_MFIter.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_SPACE.H>
#include <AMReX_Vector.H>
#include <AMReX_MFInterp_C.H>
#ifdef AMREX_USE_EB
#   include <AMReX_EBFabFactory.H>
#endif

#include <array>
#include <memory>
#include <string>

using namespace amrex;

void
WarpX::ComputeMagnetostaticField()
{
    WARPX_PROFILE("WarpX::ComputeMagnetostaticField");
    // Fields have been reset in Electrostatic solver for this time step, these fields
    // are added into the B fields after electrostatic solve

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(this->max_level == 0,
        "Magnetostatic solver not implemented with mesh refinement.");

    AddMagnetostaticFieldLabFrame();
}

void
WarpX::AddMagnetostaticFieldLabFrame()
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    WARPX_PROFILE("WarpX::AddMagnetostaticFieldLabFrame");

    // Store the boundary conditions for the field solver if they haven't been
    // stored yet
    if (!m_vector_poisson_boundary_handler.bcs_set) {
        m_vector_poisson_boundary_handler.defineVectorPotentialBCs();
    }

#ifdef WARPX_DIM_RZ
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(n_rz_azimuthal_modes == 1,
                                     "Error: RZ magnetostatic only implemented for a single mode");
#endif

    // reset current_fp before depositing current density for this step
    for (int lev = 0; lev <= max_level; lev++) {
        for (int dim=0; dim < 3; dim++) {
            m_fields.get(FieldType::current_fp, Direction{dim}, lev)->setVal(0.);
        }
    }

    // Deposit current density (source of Poisson solver)
    for (int ispecies=0; ispecies<mypc->nSpecies(); ispecies++){
        WarpXParticleContainer& species = mypc->GetParticleContainer(ispecies);
        if (!species.do_not_deposit) {
            species.DepositCurrent(
                m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level),
                dt[0], 0.);
        }
    }

#ifdef WARPX_DIM_RZ
    for (int lev = 0; lev <= max_level; lev++) {
        ApplyInverseVolumeScalingToCurrentDensity(
            m_fields.get(FieldType::current_fp, Direction{0}, lev),
            m_fields.get(FieldType::current_fp, Direction{1}, lev),
            m_fields.get(FieldType::current_fp, Direction{2}, lev),
            lev );
    }
#endif

    SyncCurrent("current_fp");

    // set the boundary and current density potentials
    setVectorPotentialBC(m_fields.get_mr_levels_alldirs(FieldType::vector_potential_fp_nodal, finest_level));

    // Compute the vector potential A, by solving the Poisson equation
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE( !IsPythonCallbackInstalled("poissonsolver"),
        "Python Level Poisson Solve not supported for Magnetostatic implementation.");

    // const amrex::Real magnetostatic_absolute_tolerance = self_fields_absolute_tolerance*PhysConst::c;
    // temporary fix!!!
    const amrex::Real magnetostatic_absolute_tolerance = 0.0;
    amrex::Real self_fields_required_precision;
    if constexpr (std::is_same<Real, float>::value) {
        self_fields_required_precision = 1e-5;
    }
    else {
        self_fields_required_precision = 1e-11;
    }
    const int self_fields_max_iters = 200;
    const int self_fields_verbosity = 2;

    computeVectorPotential(
        m_fields.get_mr_levels_alldirs(FieldType::current_fp, finest_level),
        m_fields.get_mr_levels_alldirs(FieldType::vector_potential_fp_nodal, finest_level),
        self_fields_required_precision, magnetostatic_absolute_tolerance, self_fields_max_iters,
        self_fields_verbosity);
}

/* Compute the vector potential `A` by solving the Poisson equation with `J` as
   a source.
   This uses the amrex solver.

    More specifically, this solves the equation
    \f[
        \vec{\nabla}^2 r \vec{A} = - r \mu_0 \vec{J}
 \f]

   \param[in] curr The current density
   \param[out] A The vector potential to be computed by this function
   \param[in] required_precision The relative convergence threshold for the MLMG solver
   \param[in] absolute_tolerance The absolute convergence threshold for the MLMG solver
   \param[in] max_iters The maximum number of iterations allowed for the MLMG solver
   \param[in] verbosity The verbosity setting for the MLMG solver
*/
void
WarpX::computeVectorPotential (ablastr::fields::MultiLevelVectorField const& curr,
                               ablastr::fields::MultiLevelVectorField const& A,
                               Real const required_precision,
                               Real absolute_tolerance,
                               int const max_iters,
                               int const verbosity) // const // This breaks non-const m_fields.get_mr_levels_alldirs
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // create a vector to our fields, sorted by level
    amrex::Vector<amrex::Array<amrex::MultiFab*,3>> sorted_curr;
    amrex::Vector<amrex::Array<amrex::MultiFab*,3>> sorted_A;
    for (int lev = 0; lev <= finest_level; ++lev) {
        sorted_curr.emplace_back(amrex::Array<amrex::MultiFab*,3> ({curr[lev][0],
                                                                    curr[lev][1],
                                                                    curr[lev][2]}));
        sorted_A.emplace_back(amrex::Array<amrex::MultiFab*,3> ({A[lev][0],
                                                                 A[lev][1],
                                                                 A[lev][2]}));
    }

    const ablastr::fields::MultiLevelVectorField Bfield_fp = m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, finest_level);
    const std::optional<MagnetostaticSolver::EBCalcBfromVectorPotentialPerLevel> post_A_calculation(
    {
        Bfield_fp,
        m_fields.get_mr_levels_alldirs(FieldType::vector_potential_grad_buf_e_stag, finest_level),
        m_fields.get_mr_levels_alldirs(FieldType::vector_potential_grad_buf_b_stag, finest_level)
    });

#if defined(AMREX_USE_EB)
    std::optional<amrex::Vector<amrex::EBFArrayBoxFactory const *> > eb_farray_box_factory;
    auto &warpx = WarpX::GetInstance();

    if (EB::enabled()) {
        amrex::Vector<amrex::EBFArrayBoxFactory const *> factories;
        for (int lev = 0; lev <= finest_level; ++lev) {
            factories.push_back(&warpx.fieldEBFactory(lev));
        }
        eb_farray_box_factory = factories;
    }
#else
    const std::optional<amrex::Vector<amrex::FArrayBoxFactory const *> > eb_farray_box_factory;
#endif

    ablastr::fields::computeVectorPotential(
        sorted_curr,
        sorted_A,
        required_precision,
        absolute_tolerance,
        max_iters,
        verbosity,
        this->geom,
        this->dmap,
        this->grids,
        this->m_vector_poisson_boundary_handler,
        EB::enabled(),
        WarpX::do_single_precision_comms,
        this->ref_ratio,
        post_A_calculation,
        gett_new(0),
        eb_farray_box_factory
    );

}


/* \brief Set Dirichlet/Neumann boundary conditions for the magnetostatic solver.

    The given potential's values are fixed on the boundaries of the given
    dimension according to the desired values from the simulation input file,
    boundary.potential_lo and boundary.potential_hi.

   \param[inout] A The vector potential
*/
void
WarpX::setVectorPotentialBC (ablastr::fields::MultiLevelVectorField const& A) const
{
    using ablastr::fields::Direction;

    // check if any dimension has non-periodic boundary conditions
    if (!m_vector_poisson_boundary_handler.has_non_periodic) { return; }

    auto dirichlet_flag = m_vector_poisson_boundary_handler.dirichlet_flag;

    // loop over all mesh refinement levels and set the boundary values
    for (int lev=0; lev <= max_level; lev++) {

        amrex::Box domain = Geom(lev).Domain();
        domain.surroundingNodes();
        for (int adim=0; adim < 3; adim++) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for ( MFIter mfi(*A[lev][Direction{adim}], TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
                // Extract the vector potential
                auto A_arr = A[lev][Direction{adim}]->array(mfi);
                // Extract tileboxes for which to loop
                const Box& tb  = mfi.tilebox( A[lev][Direction{adim}]->ixType().toIntVect());

                // loop over dimensions
                for (int idim=0; idim<AMREX_SPACEDIM; idim++){
                    // check if neither boundaries in this dimension should be set
                    if (!(dirichlet_flag[adim][2*idim] || dirichlet_flag[adim][2*idim+1])) { continue; }

                    // a check can be added below to test if the boundary values
                    // are already correct, in which case the ParallelFor over the
                    // cells can be skipped

                    if (!domain.strictly_contains(tb)) {
                        amrex::ParallelFor( tb,
                            [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                                IntVect iv(AMREX_D_DECL(i,j,k));

                                if (dirichlet_flag[adim][2*idim] && iv[idim] == domain.smallEnd(idim)) {
                                    A_arr(i,j,k) = 0.;
                                }

                                if (dirichlet_flag[adim][2*idim+1] && iv[idim] == domain.bigEnd(idim)) {
                                    A_arr(i,j,k) = 0.;
                                }
                            } // loop ijk
                        );
                    }
                } // idim
            } // MFIter
        }
    } // lev
}

void MagnetostaticSolver::VectorPoissonBoundaryHandler::defineVectorPotentialBCs ( )
{
    for (int adim = 0; adim < 3; adim++) {
#ifdef WARPX_DIM_RZ
        int dim_start = 0;
        WarpX& warpx = WarpX::GetInstance();
        auto geom = warpx.Geom(0);
        if (geom.ProbLo(0) == 0){
            lobc[adim][0] = LinOpBCType::Neumann;
            dirichlet_flag[adim][0] = false;
            dim_start = 1;

            // handle the r_max boundary explicitly
            if (WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC) {
                if (adim == 0) {
                    hibc[adim][0] = LinOpBCType::Neumann;
                    dirichlet_flag[adim][1] = false;
                } else{
                    hibc[adim][0] = LinOpBCType::Dirichlet;
                    dirichlet_flag[adim][1] = true;
                }
            }
            else if (WarpX::field_boundary_hi[0] == FieldBoundaryType::Neumann) {
                hibc[adim][0] = LinOpBCType::Neumann;
                dirichlet_flag[adim][1] = false;
            }
        }
#else
        const int dim_start = 0;
#endif
        bool ndotA = false;
        for (int idim=dim_start; idim<AMREX_SPACEDIM; idim++){
            ndotA = (adim == idim);

#if defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            if (idim == 1) { ndotA = (adim == 2); }
#endif

            if ( WarpX::field_boundary_lo[idim] == FieldBoundaryType::Periodic
                && WarpX::field_boundary_hi[idim] == FieldBoundaryType::Periodic ) {
                lobc[adim][idim] = LinOpBCType::Periodic;
                hibc[adim][idim] = LinOpBCType::Periodic;
                dirichlet_flag[adim][idim*2] = false;
                dirichlet_flag[adim][idim*2+1] = false;
            }
            else {
                has_non_periodic = true;
                if ( WarpX::field_boundary_lo[idim] == FieldBoundaryType::PEC ) {
                    if (ndotA) {
                        lobc[adim][idim] = LinOpBCType::Neumann;
                        dirichlet_flag[adim][idim*2] = false;
                    } else {
                        lobc[adim][idim] = LinOpBCType::Dirichlet;
                        dirichlet_flag[adim][idim*2] = true;
                    }
                }

                else if ( WarpX::field_boundary_lo[idim] == FieldBoundaryType::Neumann ) {
                    lobc[adim][idim] = LinOpBCType::Neumann;
                    dirichlet_flag[adim][idim*2] = false;
                }
                else {
                    WARPX_ABORT_WITH_MESSAGE(
                        "Field boundary conditions have to be either periodic, PEC or neumann "
                        "when using the magnetostatic solver,  but they are " + amrex::getEnumNameString(WarpX::field_boundary_lo[idim])
                    );
                }

                if ( WarpX::field_boundary_hi[idim] == FieldBoundaryType::PEC ) {
                    if (ndotA) {
                        hibc[adim][idim] = LinOpBCType::Neumann;
                        dirichlet_flag[adim][idim*2+1] = false;
                    } else {
                        hibc[adim][idim] = LinOpBCType::Dirichlet;
                        dirichlet_flag[adim][idim*2+1] = true;
                    }
                }
                else if ( WarpX::field_boundary_hi[idim] == FieldBoundaryType::Neumann ) {
                    hibc[adim][idim] = LinOpBCType::Neumann;
                    dirichlet_flag[adim][idim*2+1] = false;
                }
                else {
                    WARPX_ABORT_WITH_MESSAGE(
                        "Field boundary conditions have to be either periodic, PEC or neumann "
                        "when using the magnetostatic solver,  but they are " + amrex::getEnumNameString(WarpX::field_boundary_lo[idim])
                    );
                }
            }
        }
    }
    bcs_set = true;
}

void MagnetostaticSolver::EBCalcBfromVectorPotentialPerLevel::doInterp (amrex::MultiFab & src,
                                                                        amrex::MultiFab & dst)
{
    WarpX &warpx = WarpX::GetInstance();

    // Grab Interpolation Coefficients
    // Order of finite-order centering of fields
    const int fg_nox = WarpX::field_centering_nox;
    const int fg_noy = WarpX::field_centering_noy;
    const int fg_noz = WarpX::field_centering_noz;

    // Device vectors of stencil coefficients used for finite-order centering of fields
    amrex::Real const * stencil_coeffs_x = warpx.device_field_centering_stencil_coeffs_x.data();
    amrex::Real const * stencil_coeffs_y = warpx.device_field_centering_stencil_coeffs_y.data();
    amrex::Real const * stencil_coeffs_z = warpx.device_field_centering_stencil_coeffs_z.data();

    // Synchronize the ghost cells, do halo exchange
    ablastr::utils::communication::FillBoundary(src,
                                                src.nGrowVect(),
                                                WarpX::do_single_precision_comms);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(dst, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        IntVect const src_stag = src.ixType().toIntVect();
        IntVect const dst_stag = dst.ixType().toIntVect();

        Array4<amrex::Real const> const& src_arr = src.const_array(mfi);
        Array4<amrex::Real> const& dst_arr = dst.array(mfi);

        const Box bx = mfi.tilebox();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int j, int k, int l) noexcept
        {
            warpx_interp(j, k, l, dst_arr, src_arr, dst_stag, src_stag, fg_nox, fg_noy, fg_noz,
                stencil_coeffs_x, stencil_coeffs_y, stencil_coeffs_z);
        });
    }
}

void MagnetostaticSolver::EBCalcBfromVectorPotentialPerLevel::operator()(amrex::Array<std::unique_ptr<amrex::MLMG>,3> & mlmg, int const lev)
{
    using namespace amrex::literals;

    // This operator gets the gradient solution on the cell edges, aligned with E field staggered grid
    // This routine interpolates to the B-field staggered grid,

    const amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> buf_ptr =
    {
#if defined(WARPX_DIM_3D)
        m_grad_buf_e_stag[lev][0],
        m_grad_buf_e_stag[lev][1],
        m_grad_buf_e_stag[lev][2]
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        m_grad_buf_e_stag[lev][0],
        m_grad_buf_e_stag[lev][2]
#endif
    };

    // This will grab the gradient values for Ax
    mlmg[0]->getGradSolution({buf_ptr});

    // Interpolate dAx/dz to By grid buffer, then add to By
    this->doInterp(*m_grad_buf_e_stag[lev][2],
                   *m_grad_buf_b_stag[lev][1]);
    MultiFab::Add(*(m_b_field[lev][1]), *(m_grad_buf_b_stag[lev][1]), 0, 0, 1, 0 );

    // Interpolate dAx/dy to Bz grid buffer, then subtract from Bz
    this->doInterp(*m_grad_buf_e_stag[lev][1],
                   *m_grad_buf_b_stag[lev][2]);
    m_grad_buf_b_stag[lev][2]->mult(-1._rt);
    MultiFab::Add(*(m_b_field[lev][2]), *(m_grad_buf_b_stag[lev][2]), 0, 0, 1, 0 );

    // This will grab the gradient values for Ay
    mlmg[1]->getGradSolution({buf_ptr});

    // Interpolate dAy/dx to Bz grid buffer, then add to Bz
    this->doInterp(*m_grad_buf_e_stag[lev][0],
                   *m_grad_buf_b_stag[lev][2]);
    MultiFab::Add(*(m_b_field[lev][2]), *(m_grad_buf_b_stag[lev][2]), 0, 0, 1, 0 );

    // Interpolate dAy/dz to Bx grid buffer, then subtract from Bx
    this->doInterp(*m_grad_buf_e_stag[lev][2],
                   *m_grad_buf_b_stag[lev][0]);
    m_grad_buf_b_stag[lev][0]->mult(-1._rt);
    MultiFab::Add(*(m_b_field[lev][0]), *(m_grad_buf_b_stag[lev][0]), 0, 0, 1, 0 );

    // This will grab the gradient values for Az
    mlmg[2]->getGradSolution({buf_ptr});

    // Interpolate dAz/dy to Bx grid buffer, then add to Bx
    this->doInterp(*m_grad_buf_e_stag[lev][1],
                   *m_grad_buf_b_stag[lev][0]);
    MultiFab::Add(*(m_b_field[lev][0]), *(m_grad_buf_b_stag[lev][0]), 0, 0, 1, 0 );

    // Interpolate dAz/dx to By grid buffer, then subtract from By
    this->doInterp(*m_grad_buf_e_stag[lev][0],
                   *m_grad_buf_b_stag[lev][1]);
    m_grad_buf_b_stag[lev][1]->mult(-1._rt);
    MultiFab::Add(*(m_b_field[lev][1]), *(m_grad_buf_b_stag[lev][1]), 0, 0, 1, 0 );
}
