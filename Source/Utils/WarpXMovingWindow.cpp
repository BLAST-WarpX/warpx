/* Copyright 2019-2020 Andrew Myers, Axel Huebl, Maxence Thevenet
 * Remi Lehe, Revathi Jambunathan, Weiqun Zhang
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#include "BoundaryConditions/PML.H"
#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
#   include "BoundaryConditions/PML_RZ.H"
#endif
#include "Initialization/ExternalField.H"
#include "Particles/MultiParticleContainer.H"
#include "Fields.H"
#include "Fluids/MultiFluidContainer.H"
#include "Fluids/WarpXFluidContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H"

#include <ablastr/utils/Communication.H>

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_Dim3.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_INT.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_SPACE.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <cmath>
#include <memory>
#include <string>

using namespace amrex;

void
WarpX::UpdateInjectionPosition (const amrex::Real a_dt)
{
    const int dir = moving_window_dir;

    // Loop over species (particles and lasers)
    const int n_containers = mypc->nContainers();
    for (int i=0; i<n_containers; i++)
    {
        WarpXParticleContainer& pc = mypc->GetParticleContainer(i);

        // Continuously inject plasma in new cells (by default only on level 0)
        if (pc.doContinuousInjection())
        {
            // Get bulk momentum and velocity of plasma
            // 1D: dir=0 is z
            // 2D: dir=0 is x, dir=1 is z
            // 3D: dir=0 is x, dir=1 is y, dir=2 is z
            amrex::Vector<amrex::Real> current_injection_position = {0._rt, 0._rt, 0._rt};
#if defined(WARPX_DIM_1D_Z)
            current_injection_position[2] = pc.m_current_injection_position;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
            current_injection_position[dir*2] = pc.m_current_injection_position;
#else // 3D
            current_injection_position[dir] = pc.m_current_injection_position;
#endif

            // This only uses the base plasma injector
            PlasmaInjector* plasma_injector = pc.GetPlasmaInjector(0);

            amrex::Real v_shift = 0._rt;
            if (plasma_injector != nullptr)
            {
                const amrex::XDim3 u_bulk = plasma_injector->getInjectorMomentumHost()->getBulkMomentum(
                    current_injection_position[0],
                    current_injection_position[1],
                    current_injection_position[2]);
#if defined(WARPX_DIM_1D_Z)
                amrex::Vector<amrex::Real> u_bulk_vec = {u_bulk.z};
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                amrex::Vector<amrex::Real> u_bulk_vec = {u_bulk.x, u_bulk.z};
#else // 3D
                amrex::Vector<amrex::Real> u_bulk_vec = {u_bulk.x, u_bulk.y, u_bulk.z};
#endif
                v_shift = PhysConst::c * u_bulk_vec[dir] / std::sqrt(1._rt + u_bulk_vec[dir]*u_bulk_vec[dir]);
            }

            // In boosted-frame simulations, the plasma has moved since the last
            // call to this function, and injection position needs to be updated.
            // Note that the bulk velocity v, obtained from getBulkMomentum, is
            // transformed to the boosted frame velocity v' via the formula
            // v' = (v-c*beta)/(1-v*beta/c)
            if (WarpX::gamma_boost > 1._rt)
            {
                v_shift = (v_shift - PhysConst::c*WarpX::beta_boost)
                          / (1._rt - v_shift*WarpX::beta_boost/PhysConst::c);
#if defined(WARPX_DIM_3D)
                v_shift *= WarpX::boost_direction[dir];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                // In 2D, dir=0 corresponds to x and dir=1 corresponds to z.
                // This needs to be converted to access boost_direction,
                // which has always 3 components.
                v_shift *= WarpX::boost_direction[2*dir];
#elif defined(WARPX_DIM_1D_Z)
                // In 1D, dir=0 corresponds to z.
                // This needs to be converted to access boost_direction,
                // which has always 3 components.
                v_shift *= WarpX::boost_direction[2];
                amrex::ignore_unused(dir);
#endif
            }

            // Update current injection position
            pc.m_current_injection_position += v_shift * a_dt;
        }
    }
}

int
WarpX::MoveWindow (const int step, bool move_j)
{
    WARPX_PROFILE("WarpX::MoveWindow");

    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    if (step == start_moving_window_step) {
        amrex::Print() << Utils::TextMsg::Info("Starting moving window");
    }
    if (step == end_moving_window_step) {
        amrex::Print() << Utils::TextMsg::Info("Stopping moving window");
    }
    if (!moving_window_active(step)) { return 0; }

    // Update the continuous position of the moving window,
    // and of the plasma injection
    moving_window_x += (moving_window_v - WarpX::beta_boost * PhysConst::c)/(1 - moving_window_v * WarpX::beta_boost / PhysConst::c) * dt[0];
    const int dir = moving_window_dir;

    // Update current injection position for all containers
    UpdateInjectionPosition(dt[0]);
    // Update antenna position for all lasers
    // TODO Make this specific to lasers only
    mypc->UpdateAntennaPosition(dt[0]);

    // compute the number of cells to shift on the base level
    amrex::Real new_lo[AMREX_SPACEDIM];
    amrex::Real new_hi[AMREX_SPACEDIM];
    const amrex::Real* current_lo = geom[0].ProbLo();
    const amrex::Real* current_hi = geom[0].ProbHi();
    const amrex::Real* cdx = geom[0].CellSize();
    const int num_shift_base = static_cast<int>((moving_window_x - current_lo[dir]) / cdx[dir]);

    if (num_shift_base == 0) { return 0; }

    // update the problem domain. Note the we only do this on the base level because
    // amrex::Geometry objects share the same, static RealBox.
    for (int i=0; i<AMREX_SPACEDIM; i++) {
        new_lo[i] = current_lo[i];
        new_hi[i] = current_hi[i];
    }
    new_lo[dir] = current_lo[dir] + num_shift_base * cdx[dir];
    new_hi[dir] = current_hi[dir] + num_shift_base * cdx[dir];

    ResetProbDomain(amrex::RealBox(new_lo, new_hi));

    // Moving slice coordinates - lo and hi - with moving window //
    // slice box is modified only if slice diagnostics is initialized in input //
    if ( slice_plot_int > 0 )
    {
        amrex::Real new_slice_lo[AMREX_SPACEDIM];
        amrex::Real new_slice_hi[AMREX_SPACEDIM];
        const amrex::Real* current_slice_lo = slice_realbox.lo();
        const amrex::Real* current_slice_hi = slice_realbox.hi();
        for ( int i = 0; i < AMREX_SPACEDIM; i++) {
            new_slice_lo[i] = current_slice_lo[i];
            new_slice_hi[i] = current_slice_hi[i];
        }
        const int num_shift_base_slice = static_cast<int> ((moving_window_x -
                                   current_slice_lo[dir]) / cdx[dir]);
        new_slice_lo[dir] = current_slice_lo[dir] + num_shift_base_slice*cdx[dir];
        new_slice_hi[dir] = current_slice_hi[dir] + num_shift_base_slice*cdx[dir];
        slice_realbox.setLo(new_slice_lo);
        slice_realbox.setHi(new_slice_hi);
    }

    int num_shift      = num_shift_base;
    int num_shift_crse = num_shift;

    constexpr auto do_update_cost = true;
    constexpr auto dont_update_cost = false; //We can't update cost for PML

    // Shift the mesh fields
    for (int lev = 0; lev <= finest_level; ++lev) {

        if (lev > 0) {
            num_shift_crse = num_shift;
            num_shift *= refRatio(lev-1)[dir];
        }

        // Shift each component of vector fields (E, B, j)
        for (int dim = 0; dim < 3; ++dim) {
            // Fine grid
            amrex::ParserExecutor<3> Bfield_parser;
            amrex::ParserExecutor<3> Efield_parser;
            bool use_Bparser = false;
            bool use_Eparser = false;
            if (m_p_ext_field_params->B_ext_grid_type ==
                    ExternalFieldType::parse_ext_grid_function) {
                use_Bparser = true;
                if (dim == 0) { Bfield_parser = m_p_ext_field_params->Bxfield_parser->compile<3>(); }
                if (dim == 1) { Bfield_parser = m_p_ext_field_params->Byfield_parser->compile<3>(); }
                if (dim == 2) { Bfield_parser = m_p_ext_field_params->Bzfield_parser->compile<3>(); }
            }
            if (m_p_ext_field_params->E_ext_grid_type ==
                    ExternalFieldType::parse_ext_grid_function) {
                use_Eparser = true;
                if (dim == 0) { Efield_parser = m_p_ext_field_params->Exfield_parser->compile<3>(); }
                if (dim == 1) { Efield_parser = m_p_ext_field_params->Eyfield_parser->compile<3>(); }
                if (dim == 2) { Efield_parser = m_p_ext_field_params->Ezfield_parser->compile<3>(); }
            }
            shiftMF(*m_fields.get(FieldType::Bfield_fp, Direction{dim}, lev), geom[lev], num_shift, dir, lev, do_update_cost, m_safe_guard_cells,
                m_p_ext_field_params->B_external_grid[dim], use_Bparser, Bfield_parser);
            shiftMF(*m_fields.get(FieldType::Efield_fp, Direction{dim}, lev), geom[lev], num_shift, dir, lev, do_update_cost, m_safe_guard_cells,
                m_p_ext_field_params->E_external_grid[dim], use_Eparser, Efield_parser);
            if (fft_do_time_averaging) {
                ablastr::fields::MultiLevelVectorField Efield_avg_fp = m_fields.get_mr_levels_alldirs(FieldType::Efield_avg_fp, finest_level);
                ablastr::fields::MultiLevelVectorField Bfield_avg_fp = m_fields.get_mr_levels_alldirs(FieldType::Bfield_avg_fp, finest_level);
                shiftMF(*Bfield_avg_fp[lev][dim], geom[lev], num_shift, dir, lev, do_update_cost, m_safe_guard_cells,
                    m_p_ext_field_params->B_external_grid[dim], use_Bparser, Bfield_parser);
                shiftMF(*Efield_avg_fp[lev][dim], geom[lev], num_shift, dir, lev, do_update_cost, m_safe_guard_cells,
                   m_p_ext_field_params-> E_external_grid[dim], use_Eparser, Efield_parser);
            }
            if (move_j) {
                shiftMF(*m_fields.get(FieldType::current_fp, Direction{dim}, lev), geom[lev], num_shift, dir, lev, do_update_cost, m_safe_guard_cells);
            }
            if (pml[lev] && pml[lev]->ok()) {
                amrex::MultiFab* pml_B = m_fields.get(FieldType::pml_B_fp, Direction{dim}, lev);
                amrex::MultiFab* pml_E = m_fields.get(FieldType::pml_E_fp, Direction{dim}, lev);
                shiftMF(*pml_B, geom[lev], num_shift, dir, lev, dont_update_cost, m_safe_guard_cells);
                shiftMF(*pml_E, geom[lev], num_shift, dir, lev, dont_update_cost, m_safe_guard_cells);
            }
#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
            if (pml_rz[lev] && dim < 2) {
                amrex::MultiFab* pml_rz_B = m_fields.get(FieldType::pml_B_fp, Direction{dim}, lev);
                amrex::MultiFab* pml_rz_E = m_fields.get(FieldType::pml_E_fp, Direction{dim}, lev);
                shiftMF(*pml_rz_B, geom[lev], num_shift, dir, lev, dont_update_cost, m_safe_guard_cells);
                shiftMF(*pml_rz_E, geom[lev], num_shift, dir, lev, dont_update_cost, m_safe_guard_cells);
            }
#endif
            if (lev > 0) {
                // coarse grid
                shiftMF(*m_fields.get(FieldType::Bfield_cp, Direction{dim}, lev), geom[lev-1], num_shift_crse, dir, lev, do_update_cost, m_safe_guard_cells,
                    m_p_ext_field_params->B_external_grid[dim], use_Bparser, Bfield_parser);
                shiftMF(*m_fields.get(FieldType::Efield_cp, Direction{dim}, lev), geom[lev-1], num_shift_crse, dir, lev, do_update_cost, m_safe_guard_cells,
                    m_p_ext_field_params->E_external_grid[dim], use_Eparser, Efield_parser);
                shiftMF(*m_fields.get(FieldType::Bfield_aux, Direction{dim}, lev), geom[lev], num_shift, dir, lev, do_update_cost, m_safe_guard_cells);
                shiftMF(*m_fields.get(FieldType::Efield_aux, Direction{dim}, lev), geom[lev], num_shift, dir, lev, do_update_cost, m_safe_guard_cells);
                if (fft_do_time_averaging) {
                    ablastr::fields::MultiLevelVectorField Efield_avg_cp = m_fields.get_mr_levels_alldirs(FieldType::Efield_avg_cp, finest_level);
                    ablastr::fields::MultiLevelVectorField Bfield_avg_cp = m_fields.get_mr_levels_alldirs(FieldType::Bfield_avg_cp, finest_level);
                    shiftMF(*Bfield_avg_cp[lev][dim], geom[lev-1], num_shift_crse, dir, lev, do_update_cost, m_safe_guard_cells,
                        m_p_ext_field_params->B_external_grid[dim], use_Bparser, Bfield_parser);
                    shiftMF(*Efield_avg_cp[lev][dim], geom[lev-1], num_shift_crse, dir, lev, do_update_cost, m_safe_guard_cells,
                        m_p_ext_field_params->E_external_grid[dim], use_Eparser, Efield_parser);
                }
                if (move_j) {
                    shiftMF(*m_fields.get(FieldType::current_cp, Direction{dim}, lev), geom[lev-1], num_shift_crse, dir, lev, do_update_cost, m_safe_guard_cells);
                }
                if (do_pml && pml[lev]->ok()) {
                    amrex::MultiFab* pml_B_cp = m_fields.get(FieldType::pml_B_cp, Direction{dim}, lev);
                    amrex::MultiFab* pml_E_cp = m_fields.get(FieldType::pml_E_cp, Direction{dim}, lev);
                    shiftMF(*pml_B_cp, geom[lev-1], num_shift_crse, dir, lev, dont_update_cost, m_safe_guard_cells);
                    shiftMF(*pml_E_cp, geom[lev-1], num_shift_crse, dir, lev, dont_update_cost, m_safe_guard_cells);
                }
            }
        }

        // Shift scalar field F with div(E) cleaning in valid domain
        // TODO: shift F from pml_rz for RZ geometry with PSATD, once implemented
        if (m_fields.has(FieldType::F_fp, lev))
        {
            // Fine grid
            shiftMF(*m_fields.get(FieldType::F_fp, lev), geom[lev], num_shift, dir, lev, do_update_cost, m_safe_guard_cells);
            if (lev > 0)
            {
                // Coarse grid
                shiftMF(*m_fields.get(FieldType::F_cp, lev), geom[lev-1], num_shift_crse, dir, lev, do_update_cost, m_safe_guard_cells);
            }
        }

        // Shift scalar field F with div(E) cleaning in pml region
        if (do_pml_dive_cleaning)
        {
            // Fine grid
            if (do_pml && pml[lev]->ok())
            {
                amrex::MultiFab* pml_F = m_fields.get(FieldType::pml_F_fp, lev);
                shiftMF(*pml_F, geom[lev], num_shift, dir, lev, dont_update_cost, m_safe_guard_cells);
            }
            if (lev > 0)
            {
                // Coarse grid
                if (do_pml && pml[lev]->ok())
                {
                    amrex::MultiFab* pml_F = m_fields.get(FieldType::pml_F_cp, lev);
                    shiftMF(*pml_F, geom[lev-1], num_shift_crse, dir, lev, dont_update_cost, m_safe_guard_cells);
                }
            }
        }

        // Shift scalar field G with div(B) cleaning in valid domain
        // TODO: shift G from pml_rz for RZ geometry with PSATD, once implemented
        if (m_fields.has(FieldType::G_fp, lev))
        {
            // Fine grid
            shiftMF(*m_fields.get(FieldType::G_fp, lev), geom[lev], num_shift, dir, lev, do_update_cost, m_safe_guard_cells);
            if (lev > 0)
            {
                // Coarse grid
                shiftMF(*m_fields.get(FieldType::G_cp, lev), geom[lev-1], num_shift_crse, dir, lev, do_update_cost, m_safe_guard_cells);
            }
        }

        // Shift scalar field G with div(B) cleaning in pml region
        if (do_pml_divb_cleaning)
        {
            // Fine grid
            if (do_pml && pml[lev]->ok())
            {
                amrex::MultiFab* pml_G = m_fields.get(FieldType::pml_G_fp, lev);
                shiftMF(*pml_G, geom[lev], num_shift, dir, lev, dont_update_cost, m_safe_guard_cells);
            }
            if (lev > 0)
            {
                // Coarse grid
                if (do_pml && pml[lev]->ok())
                {
                    amrex::MultiFab* pml_G = m_fields.get(FieldType::pml_G_cp, lev);
                    shiftMF(*pml_G, geom[lev-1], num_shift_crse, dir, lev, dont_update_cost, m_safe_guard_cells);
                }
            }
        }

        // Shift scalar component rho
        if (move_j) {
            if (m_fields.has(FieldType::rho_fp, lev)) {
                // Fine grid
                shiftMF(*m_fields.get(FieldType::rho_fp,lev),   geom[lev], num_shift, dir, lev, do_update_cost, m_safe_guard_cells);
                if (lev > 0){
                    // Coarse grid
                    shiftMF(*m_fields.get(FieldType::rho_cp,lev), geom[lev-1], num_shift_crse, dir, lev, do_update_cost, m_safe_guard_cells);
                }
            }
        }

        // Shift values of N, NU for each fluid species
        if (do_fluid_species) {
            const int n_fluid_species = myfl->nSpecies();
            for (int i=0; i<n_fluid_species; i++) {
                WarpXFluidContainer const& fl = myfl->GetFluidContainer(i);
                shiftMF( *m_fields.get(fl.name_mf_N, lev), geom[lev], num_shift, dir, lev, do_update_cost, m_safe_guard_cells );
                shiftMF( *m_fields.get(fl.name_mf_NU, Direction{0}, lev), geom[lev], num_shift, dir, lev, do_update_cost, m_safe_guard_cells );
                shiftMF( *m_fields.get(fl.name_mf_NU, Direction{1}, lev), geom[lev], num_shift, dir, lev, do_update_cost, m_safe_guard_cells );
                shiftMF( *m_fields.get(fl.name_mf_NU, Direction{2}, lev), geom[lev], num_shift, dir, lev, do_update_cost, m_safe_guard_cells );
            }
        }
    }

    // Loop over species (particles and lasers)
    const int n_containers = mypc->nContainers();
    for (int i=0; i<n_containers; i++)
    {
        WarpXParticleContainer& pc = mypc->GetParticleContainer(i);

        // Continuously inject plasma in new cells (by default only on level 0)
        if (pc.doContinuousInjection())
        {
            const int lev = 0;

            // particleBox encloses the cells where we generate particles
            // (only injects particles in an integer number of cells,
            // for correct particle spacing)
            amrex::RealBox particleBox = geom[lev].ProbDomain();
            amrex::Real new_injection_position = pc.m_current_injection_position;
            if (moving_window_v > 0._rt)
            {
                // Forward-moving window
                const amrex::Real dx = geom[lev].CellSize(dir);
                new_injection_position = pc.m_current_injection_position +
                    std::floor( (geom[lev].ProbHi(dir) - pc.m_current_injection_position)/dx ) * dx;
            }
            else if (moving_window_v < 0._rt)
            {
                // Backward-moving window
                const amrex::Real dx = geom[lev].CellSize(dir);
                new_injection_position = pc.m_current_injection_position -
                    std::floor( (pc.m_current_injection_position - geom[lev].ProbLo(dir))/dx) * dx;
            }
            // Modify the corresponding bounds of the particleBox
            if (moving_window_v > 0._rt)
            {
                particleBox.setLo( dir, pc.m_current_injection_position );
                particleBox.setHi( dir, new_injection_position );
            }
            else if (moving_window_v < 0._rt)
            {
                particleBox.setLo( dir, new_injection_position );
                particleBox.setHi( dir, pc.m_current_injection_position );
            }

            if (particleBox.ok() and (pc.m_current_injection_position != new_injection_position)){
                // Performs continuous injection of all WarpXParticleContainer
                // in mypc.
                pc.ContinuousInjection(particleBox);
                pc.m_current_injection_position = new_injection_position;
            }
        }
    }

    // Continuously inject fluid species in new cells (by default only on level 0)
    const int lev = 0;
    // Find box in which to initialize new fluid cells
    amrex::Box injection_box = geom[lev].Domain();
    injection_box.surroundingNodes(); // get nodal box
    // Restrict box in the direction of the moving window, to only include the new cells
    if (moving_window_v > 0._rt)
    {
        injection_box.setSmall( dir, injection_box.bigEnd(dir) - num_shift_base + 1 );
    }
    else if (moving_window_v < 0._rt)
    {
        injection_box.setBig( dir, injection_box.smallEnd(dir) + num_shift_base - 1 );
    }
    // Loop over fluid species, and fill the values of the new cells
    if (do_fluid_species) {
        const int n_fluid_species = myfl->nSpecies();
        const amrex::Real cur_time = t_new[0];
        for (int i=0; i<n_fluid_species; i++) {
            WarpXFluidContainer& fl = myfl->GetFluidContainer(i);
            fl.InitData( m_fields, injection_box, cur_time, lev );
        }
    }

    // Recompute macroscopic properties of the medium
    if (WarpX::em_solver_medium == MediumForEM::Macroscopic) {
        const int lev_zero = 0;
        m_macroscopic_properties->InitData(
            Geom(lev_zero),
            m_fields.get(FieldType::Efield_fp, Direction{0}, lev_zero)->ixType().toIntVect(),
            m_fields.get(FieldType::Efield_fp, Direction{1}, lev_zero)->ixType().toIntVect(),
            m_fields.get(FieldType::Efield_fp, Direction{2}, lev_zero)->ixType().toIntVect()
        );
    }

    return num_shift_base;
}

void
WarpX::shiftMF (amrex::MultiFab& mf, const amrex::Geometry& geom,
                int num_shift, int dir, const int lev, bool update_cost_flag,
                const bool safe_guard_cells,
                amrex::Real external_field, bool useparser,
                amrex::ParserExecutor<3> const& field_parser)
{
    using namespace amrex::literals;
    WARPX_PROFILE("WarpX::shiftMF()");
    const amrex::BoxArray& ba = mf.boxArray();
    const amrex::DistributionMapping& dm = mf.DistributionMap();
    const int nc = mf.nComp();
    const amrex::IntVect& ng = mf.nGrowVect();

    AMREX_ALWAYS_ASSERT(ng[dir] >= num_shift);

    amrex::MultiFab tmpmf(ba, dm, nc, ng);
    amrex::MultiFab::Copy(tmpmf, mf, 0, 0, nc, ng);

    if ( safe_guard_cells ) {
        // Fill guard cells.
        ablastr::utils::communication::FillBoundary(tmpmf, WarpX::do_single_precision_comms, geom.periodicity());
    } else {
        amrex::IntVect ng_mw = amrex::IntVect::TheUnitVector();
        // Enough guard cells in the MW direction
        ng_mw[dir] = num_shift;
        // Make sure we don't exceed number of guard cells allocated
        ng_mw = ng_mw.min(ng);
        // Fill guard cells.
        ablastr::utils::communication::FillBoundary(tmpmf, ng_mw, WarpX::do_single_precision_comms, geom.periodicity());
    }

    // Make a box that covers the region that the window moved into
    const amrex::IndexType& typ = ba.ixType();
    const amrex::Box& domainBox = geom.Domain();
    amrex::Box adjBox;
    if (num_shift > 0) {
        adjBox = adjCellHi(domainBox, dir, ng[dir]);
    } else {
        adjBox = adjCellLo(domainBox, dir, ng[dir]);
    }
    adjBox = amrex::convert(adjBox, typ);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (idim == dir and typ.nodeCentered(dir)) {
            if (num_shift > 0) {
                adjBox.growLo(idim, -1);
            } else {
                adjBox.growHi(idim, -1);
            }
        } else if (idim != dir) {
            adjBox.growLo(idim, ng[idim]);
            adjBox.growHi(idim, ng[idim]);
        }
    }

    amrex::IntVect shiftiv(0);
    shiftiv[dir] = num_shift;
    const amrex::Dim3 shift = shiftiv.dim3();

    const amrex::RealBox& real_box = geom.ProbDomain();
    const auto dx = geom.CellSizeArray();

    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif

    for (amrex::MFIter mfi(tmpmf, TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        auto const& dstfab = mf.array(mfi);
        auto const& srcfab = tmpmf.array(mfi);

        const amrex::Box& outbox = mfi.growntilebox() & adjBox;

        if (outbox.ok()) {
            if (!useparser) {
                AMREX_PARALLEL_FOR_4D ( outbox, nc, i, j, k, n,
                {
                    srcfab(i,j,k,n) = external_field;
                })
            } else {
                // index type of the src mf
                auto const& mf_IndexType = (tmpmf).ixType();
                amrex::IntVect mf_type(AMREX_D_DECL(0,0,0));
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                    mf_type[idim] = mf_IndexType.nodeCentered(idim);
                }

                amrex::ParallelFor (outbox, nc,
                      [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                      // Compute x,y,z co-ordinates based on index type of mf
#if defined(WARPX_DIM_1D_Z)
                      const amrex::Real x = 0.0_rt;
                      const amrex::Real y = 0.0_rt;
                      const amrex::Real fac_z = (1.0_rt - mf_type[0]) * dx[0]*0.5_rt;
                      const amrex::Real z = i*dx[0] + real_box.lo(0) + fac_z;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                      const amrex::Real fac_x = (1.0_rt - mf_type[0]) * dx[0]*0.5_rt;
                      const amrex::Real x = i*dx[0] + real_box.lo(0) + fac_x;
                      const amrex::Real y = 0.0;
                      const amrex::Real fac_z = (1.0_rt - mf_type[1]) * dx[1]*0.5_rt;
                      const amrex::Real z = j*dx[1] + real_box.lo(1) + fac_z;
#else
                      const amrex::Real fac_x = (1.0_rt - mf_type[0]) * dx[0]*0.5_rt;
                      const amrex::Real x = i*dx[0] + real_box.lo(0) + fac_x;
                      const amrex::Real fac_y = (1.0_rt - mf_type[1]) * dx[1]*0.5_rt;
                      const amrex::Real y = j*dx[1] + real_box.lo(1) + fac_y;
                      const amrex::Real fac_z = (1.0_rt - mf_type[2]) * dx[2]*0.5_rt;
                      const amrex::Real z = k*dx[2] + real_box.lo(2) + fac_z;
#endif
                      srcfab(i,j,k,n) = field_parser(x,y,z);
                });
            }

        }

        amrex::Box dstBox = mf[mfi].box();
        if (num_shift > 0) {
            dstBox.growHi(dir, -num_shift);
        } else {
            dstBox.growLo(dir,  num_shift);
        }
        AMREX_PARALLEL_FOR_4D ( dstBox, nc, i, j, k, n,
        {
            dstfab(i,j,k,n) = srcfab(i+shift.x,j+shift.y,k+shift.z,n);
        })

        if (cost && update_cost_flag &&
            WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }

#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
    if (WarpX::GetInstance().getPMLRZ()) {
        // This does the exchange of data in the corner guard cells, the cells that are in the
        // guard region both radially and longitudinally. These are the PML cells in the overlapping
        // longitudinal region. FillBoundary normally does not update these cells.
        // This update is needed so that the cells at the end of the FABs are updated appropriately
        // with the data shifted from the neighboring FAB. Without this update, the RZ PML becomes
        // unstable with the moving grid.
        // This code creates a temporary MultiFab using a BoxList where the radial size of all of
        // its boxes is increased so that the radial guard cells are included in the boxes valid domain.
        // The temporary MultiFab is setup to refer to the data of the original Multifab (this can
        // be done since the shape of the data is all the same, just the indexing is different).
        amrex::BoxList bl;
        const auto ba_size = static_cast<int>(ba.size());
        for (int i = 0; i < ba_size; ++i) {
            bl.push_back(amrex::grow(ba[i], 0, mf.nGrowVect()[0]));
        }
        const amrex::BoxArray rba(std::move(bl));
        amrex::MultiFab rmf(rba, dm, mf.nComp(), IntVect(0,mf.nGrowVect()[1]), MFInfo().SetAlloc(false));

        for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
            rmf.setFab(mfi, FArrayBox(mf[mfi], amrex::make_alias, 0, mf.nComp()));
        }
        rmf.FillBoundary(false);
    }
#endif

}

void
WarpX::ShiftGalileanBoundary ()
{
    const amrex::Real cur_time = t_new[0];
    amrex::Real new_lo[AMREX_SPACEDIM];
    amrex::Real new_hi[AMREX_SPACEDIM];
    const amrex::Real* current_lo = geom[0].ProbLo();
    const amrex::Real* current_hi = geom[0].ProbHi();

    const amrex::Real time_shift = (cur_time - time_of_last_gal_shift);

#if defined(WARPX_DIM_3D)
        m_galilean_shift = {
            m_v_galilean[0]*time_shift,
            m_v_galilean[1]*time_shift,
            m_v_galilean[2]*time_shift };
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        m_galilean_shift = {
            m_v_galilean[0]*time_shift,
            std::numeric_limits<amrex::Real>::quiet_NaN(),
            m_v_galilean[2]*time_shift };
#elif defined(WARPX_DIM_1D_Z)
        m_galilean_shift = {
            std::numeric_limits<Real>::quiet_NaN(),
            std::numeric_limits<Real>::quiet_NaN(),
            m_v_galilean[2]*time_shift };
#endif

#if defined(WARPX_DIM_3D)
        for (int i=0; i<AMREX_SPACEDIM; i++) {
            new_lo[i] = current_lo[i] + m_galilean_shift[i];
            new_hi[i] = current_hi[i] + m_galilean_shift[i];
        }
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    {
        new_lo[0] = current_lo[0] + m_galilean_shift[0];
        new_hi[0] = current_hi[0] + m_galilean_shift[0];
        new_lo[1] = current_lo[1] + m_galilean_shift[2];
        new_hi[1] = current_hi[1] + m_galilean_shift[2];
    }
#elif defined(WARPX_DIM_1D_Z)
    {
        new_lo[0] = current_lo[0] + m_galilean_shift[2];
        new_hi[0] = current_hi[0] + m_galilean_shift[2];
    }
#endif
    time_of_last_gal_shift = cur_time;

    ResetProbDomain(amrex::RealBox(new_lo, new_hi));
}

void
WarpX::ResetProbDomain (const amrex::RealBox& rb)
{
    amrex::Geometry::ResetDefaultProbDomain(rb);
    for (int lev = 0; lev <= max_level; ++lev) {
        amrex::Geometry g = Geom(lev);
        g.ProbDomain(rb);
        SetGeometry(lev, g);
    }
}
