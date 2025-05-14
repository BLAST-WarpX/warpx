#include "ImplicitSolver.H"
#include "Fields.H"
#include "WarpX.H"
#include "Particles/MultiParticleContainer.H"

using namespace amrex;
using namespace amrex::literals;

void ImplicitSolver::CreateParticleAttributes () const
{
    // Set comm to false to that the attributes are not communicated
    // nor written to the checkpoint files
    int const comm = 0;

    // Add space to save the positions and velocities at the start of the time steps
    for (auto const& pc : m_WarpX->GetPartContainer()) {
#if (AMREX_SPACEDIM >= 2)
        pc->AddRealComp("x_n", comm);
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
        pc->AddRealComp("y_n", comm);
#endif
        pc->AddRealComp("z_n", comm);
        pc->AddRealComp("ux_n", comm);
        pc->AddRealComp("uy_n", comm);
        pc->AddRealComp("uz_n", comm);
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
            WARPX_ABORT_WITH_MESSAGE("LinOpBCType not set for this FieldBoundaryType");
        } else if (a_fbc[i] == FieldBoundaryType::Damped) {
            WARPX_ABORT_WITH_MESSAGE("LinOpBCType not set for this FieldBoundaryType");
        } else if (a_fbc[i] == FieldBoundaryType::Absorbing_SilverMueller) {
            WARPX_ABORT_WITH_MESSAGE("LinOpBCType not set for this FieldBoundaryType");
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

void ImplicitSolver::InitializeMassMatrices ()
{

    // Initializes the MassMatrices and MassMatrices_PC containers
    // The latter has a reduced number of elements that is used for the preconditioner.
    // They are the same for now as we only include the diagonal elements of the diagonal matrices.
    // Off-diagonal matrices (e.g. MassMatrices_xy) are not yet included.
    //
    // dJx = MassMatrices_xx*dEx + MassMatrices_xy*dEy + MassMatrices_xz*dEz
    // dJy = MassMatrices_yx*dEx + MassMatrices_yy*dEy + MassMatrices_yz*dEz
    // dJz = MassMatrices_zx*dEx + MassMatrices_zy*dEy + MassMatrices_zz*dEz

    using ablastr::fields::Direction;
    using warpx::fields::FieldType;
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        const auto& ba_Jx = m_WarpX->m_fields.get(FieldType::current_fp, Direction{0}, lev)->boxArray();
        const auto& ba_Jy = m_WarpX->m_fields.get(FieldType::current_fp, Direction{1}, lev)->boxArray();
        const auto& ba_Jz = m_WarpX->m_fields.get(FieldType::current_fp, Direction{2}, lev)->boxArray();
        const auto& dm = m_WarpX->m_fields.get(FieldType::current_fp, Direction{0}, lev)->DistributionMap();
        const amrex::IntVect ngb = m_WarpX->m_fields.get(FieldType::current_fp, Direction{0}, lev)->nGrowVect();
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices, Direction{0}, lev, ba_Jx, dm, 1, ngb, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices, Direction{1}, lev, ba_Jy, dm, 1, ngb, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices, Direction{2}, lev, ba_Jz, dm, 1, ngb, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_PC, Direction{0}, lev, ba_Jx, dm, 1, ngb, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_PC, Direction{1}, lev, ba_Jy, dm, 1, ngb, 0.0_rt);
        m_WarpX->m_fields.alloc_init(FieldType::MassMatrices_PC, Direction{2}, lev, ba_Jz, dm, 1, ngb, 0.0_rt);
    }

    // Set the pointer to mass matrix MultiFab
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        m_mmpc_mfarrvec.push_back(m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_PC, 0));
    }

}

void ImplicitSolver::PreRHSOp ( const amrex::Real  a_cur_time,
                                const int          a_nl_iter,
                                const bool         a_from_jacobian )
{
    using warpx::fields::FieldType;
    amrex::ignore_unused( a_nl_iter );

    if (m_WarpX->use_filter) {
        int finest_level = 0;
        m_WarpX->ApplyFilterMF(m_WarpX->m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level), 0);
    }

    // Advance the particle positions by 1/2 dt,
    // particle velocities by dt, then take average of old and new v,
    // deposit currents, giving J at n+1/2
    // This uses Efield_fp and Bfield_fp, the field at n+1/2 from the previous iteration.
    const PushType push_type = PushType::Implicit;
    const bool skip_current = false;
    bool deposit_mass_matrices = false;
    if (m_use_mass_matrices && !a_from_jacobian) { deposit_mass_matrices = true; }
    m_WarpX->PushParticlesandDeposit(a_cur_time, skip_current, deposit_mass_matrices, push_type);

    m_WarpX->SyncCurrentAndRho();
    if (deposit_mass_matrices) {
        SyncMassMatricesPCAndApplyBCs();
        const amrex::Real theta_dt = m_theta*m_dt;
        SetMassMatricesForPC( theta_dt );
    }

}

void ImplicitSolver::SyncMassMatricesPCAndApplyBCs ()
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // Copy mass matrices elements used for the preconditioner
    for (int lev = 0; lev < m_num_amr_levels; ++lev) {
        ablastr::fields::VectorField MM = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices, lev);
        ablastr::fields::VectorField MM_PC = m_WarpX->m_fields.get_alldirs(FieldType::MassMatrices_PC, lev);
        amrex::MultiFab::Copy(*MM_PC[0], *MM[0], 0, 0, MM[0]->nComp(), MM[0]->nGrowVect());
        amrex::MultiFab::Copy(*MM_PC[1], *MM[1], 0, 0, MM[1]->nComp(), MM[1]->nGrowVect());
        amrex::MultiFab::Copy(*MM_PC[2], *MM[2], 0, 0, MM[2]->nComp(), MM[2]->nGrowVect());
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
