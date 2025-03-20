/* Copyright 2023 Grant Johnson, Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Fields.H"
#include "Particles/Pusher/UpdateMomentumHigueraCary.H"
#include "Utils/WarpXProfilerWrapper.H"

#include "MusclHancockUtils.H"
#include "Fluids/WarpXFluidContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXUtil.H"
#include "Utils/SpeciesUtils.H"
#include "WarpX.H"

#include <ablastr/coarsen/sample.H>
#include <ablastr/utils/Communication.H>

using namespace ablastr::utils::communication;
using namespace amrex;


WarpXFluidContainer::WarpXFluidContainer(int ispecies, const std::string &name):
    species_id{ispecies},
    species_name{name}
{
    ReadParameters();

    // Initialize injection objects
    const ParmParse pp_species_name(species_name);
    SpeciesUtils::parseDensity(species_name, "", h_inj_rho, density_parser);
    SpeciesUtils::parseMomentum(species_name, "", "none", h_inj_mom,
        ux_parser, uy_parser, uz_parser, ux_th_parser, uy_th_parser, uz_th_parser, h_mom_temp, h_mom_vel);
    if (h_inj_rho) {
#ifdef AMREX_USE_GPU
        d_inj_rho = static_cast<InjectorDensity*>
            (amrex::The_Arena()->alloc(sizeof(InjectorDensity)));
        amrex::Gpu::htod_memcpy_async(d_inj_rho, h_inj_rho.get(), sizeof(InjectorDensity));
#else
        d_inj_rho = h_inj_rho.get();
#endif
    }
    if (h_inj_mom) {
#ifdef AMREX_USE_GPU
        d_inj_mom = static_cast<InjectorMomentum*>
            (amrex::The_Arena()->alloc(sizeof(InjectorMomentum)));
        amrex::Gpu::htod_memcpy_async(d_inj_mom, h_inj_mom.get(), sizeof(InjectorMomentum));
#else
        d_inj_mom = h_inj_mom.get();
#endif
    }
    amrex::Gpu::synchronize();

}

void WarpXFluidContainer::ReadParameters()
{

    // Extract charge, mass, species type
    const std::string injection_style = "none";
    SpeciesUtils::extractSpeciesProperties(species_name, injection_style, charge, mass, physical_species);

    const ParmParse pp_species_name(species_name);
    pp_species_name.query("do_not_deposit", do_not_deposit);
    pp_species_name.query("do_not_gather", do_not_gather);
    pp_species_name.query("do_not_push", do_not_push);

    // default values of E_external and B_external
    // are used to set the E and B field when "constant" or "parser"
    // is not explicitly used in the input
    pp_species_name.query("B_ext_init_style", m_B_ext_s);
    std::transform(m_B_ext_s.begin(),
                    m_B_ext_s.end(),
                    m_B_ext_s.begin(),
                    ::tolower);
    pp_species_name.query("E_ext_init_style", m_E_ext_s);
    std::transform(m_E_ext_s.begin(),
                    m_E_ext_s.end(),
                    m_E_ext_s.begin(),
                    ::tolower);

    // Parse external fields:
    // if the input string for B_ext_s is
    // "parse_b_ext_function" then the mathematical expression
    // for the Bx_, By_, Bz_external_function(x,y,z)
    // must be provided in the input file.
    if (m_B_ext_s == "parse_b_ext_function") {
        // store the mathematical expression as string
        std::string str_Bx_ext_function;
        std::string str_By_ext_function;
        std::string str_Bz_ext_function;
        utils::parser::Store_parserString(
            pp_species_name, "Bx_external_function(x,y,z,t)",
            str_Bx_ext_function);
        utils::parser::Store_parserString(
            pp_species_name, "By_external_function(x,y,z,t)",
            str_By_ext_function);
        utils::parser::Store_parserString(
            pp_species_name, "Bz_external_function(x,y,z,t)",
            str_Bz_ext_function);

        // Parser for B_external on the fluid
        m_Bx_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(str_Bx_ext_function,{"x","y","z","t"}));
        m_By_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(str_By_ext_function,{"x","y","z","t"}));
        m_Bz_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(str_Bz_ext_function,{"x","y","z","t"}));

    }

    // if the input string for E_ext_s is
    // "parse_e_ext_function" then the mathematical expression
    // for the Ex_, Ey_, Ez_external_function(x,y,z)
    // must be provided in the input file.
    if (m_E_ext_s == "parse_e_ext_function") {
        // store the mathematical expression as string
        std::string str_Ex_ext_function;
        std::string str_Ey_ext_function;
        std::string str_Ez_ext_function;
        utils::parser::Store_parserString(
            pp_species_name, "Ex_external_function(x,y,z,t)",
            str_Ex_ext_function);
        utils::parser::Store_parserString(
            pp_species_name, "Ey_external_function(x,y,z,t)",
            str_Ey_ext_function);
        utils::parser::Store_parserString(
            pp_species_name, "Ez_external_function(x,y,z,t)",
            str_Ez_ext_function);
        // Parser for E_external on the fluid
        m_Ex_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(str_Ex_ext_function,{"x","y","z","t"}));
        m_Ey_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(str_Ey_ext_function,{"x","y","z","t"}));
        m_Ez_parser = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(str_Ez_ext_function,{"x","y","z","t"}));
    }
}

void WarpXFluidContainer::AllocateLevelMFs(ablastr::fields::MultiFabRegister& fields, const BoxArray &ba, const DistributionMapping &dm, int lev) const
{
    using ablastr::fields::Direction;
    const int ncomps = 1;
    const amrex::IntVect nguards(AMREX_D_DECL(2, 2, 2));

    fields.alloc_init(
            name_mf_N, lev, amrex::convert(ba, amrex::IntVect::TheNodeVector()), dm,
            ncomps, nguards, 0.0_rt);

    fields.alloc_init(
            name_mf_NU, Direction{0}, lev, amrex::convert(ba, amrex::IntVect::TheNodeVector()), dm,
            ncomps, nguards, 0.0_rt);

    fields.alloc_init(
            name_mf_NU, Direction{1}, lev, amrex::convert(ba, amrex::IntVect::TheNodeVector()), dm,
            ncomps, nguards, 0.0_rt);

    fields.alloc_init(
            name_mf_NU, Direction{2}, lev, amrex::convert(ba, amrex::IntVect::TheNodeVector()), dm,
            ncomps, nguards, 0.0_rt);

}

void WarpXFluidContainer::InitData(ablastr::fields::MultiFabRegister& fields, amrex::Box init_box, amrex::Real cur_time, int lev)
{
    using ablastr::fields::Direction;
    WARPX_PROFILE("WarpXFluidContainer::InitData");

    // Convert initialization box to nodal box
    init_box.surroundingNodes();

    // Create local copies of pointers for GPU kernels
    InjectorDensity* inj_rho = d_inj_rho;
    InjectorMomentum* inj_mom = d_inj_mom;

    // Extract grid geometry properties
    WarpX &warpx = WarpX::GetInstance();
    const amrex::Geometry &geom = warpx.Geom(lev);
    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();
    const amrex::Real clight = PhysConst::c;
    const amrex::Real gamma_boost = WarpX::gamma_boost;
    const amrex::Real beta_boost = WarpX::beta_boost;

    // Loop through cells and initialize their value
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*fields.get(name_mf_N, lev), TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {

        amrex::Box const tile_box  = mfi.tilebox(fields.get(name_mf_N, lev)->ixType().toIntVect());
        amrex::Array4<Real> const &N_arr = fields.get(name_mf_N, lev)->array(mfi);
        amrex::Array4<Real> const &NUx_arr = fields.get(name_mf_NU, Direction{0}, lev)->array(mfi);
        amrex::Array4<Real> const &NUy_arr = fields.get(name_mf_NU, Direction{1}, lev)->array(mfi);
        amrex::Array4<Real> const &NUz_arr = fields.get(name_mf_NU, Direction{2}, lev)->array(mfi);

        // Return the intersection of all cells and the ones we wish to update
        amrex::Box const init_box_intersection = init_box & tile_box;

        amrex::ParallelFor(init_box_intersection,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
#if defined(WARPX_DIM_3D)
                const amrex::Real x = problo[0] + i * dx[0];
                const amrex::Real y = problo[1] + j * dx[1];
                amrex::Real z = problo[2] + k * dx[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                const amrex::Real x = problo[0] + i * dx[0];
                const amrex::Real y = 0.0_rt;
                amrex::Real z = problo[1] + j * dx[1];
#else
                const amrex::Real x = 0.0_rt;
                const amrex::Real y = 0.0_rt;
                amrex::Real z = problo[0] + i * dx[0];
#endif

                // Lorentz transform z (from boosted to lab frame)
                if (gamma_boost > 1._rt){
                    z = gamma_boost*(z + beta_boost*clight*cur_time);
                }

                amrex::Real n = inj_rho->getDensity(x, y, z);
                amrex::XDim3 u = inj_mom->getBulkMomentum(x, y, z);

                // Give u the right dimensions of m/s
                u.x = u.x * clight;
                u.y = u.y * clight;
                u.z = u.z * clight;

                // Check if n > 0 and if not, don't compute the boost
                // Lorentz transform n, u (from lab to boosted frame)
                if (n > 0.0){
                    if (gamma_boost > 1._rt){
                        const amrex::Real gamma = std::sqrt(1.0_rt + (u.x*u.x + u.y*u.y + u.z*u.z)/(clight*clight));
                        const amrex::Real n_boosted = gamma_boost*n*( 1.0_rt - beta_boost*u.z/(gamma*clight) );
                        const amrex::Real uz_boosted = gamma_boost*(u.z - beta_boost*clight*gamma);
                        u.z = uz_boosted;
                        n = n_boosted;
                    }
                }

                // Multiply by clight so u is back in SI units
                N_arr(i, j, k) = n;
                NUx_arr(i, j, k) = n * u.x;
                NUy_arr(i, j, k) = n * u.y;
                NUz_arr(i, j, k) = n * u.z;

            }
        );
    }
}


void WarpXFluidContainer::Evolve(
    ablastr::fields::MultiFabRegister& fields,
    int lev,
    const std::string& current_fp_string,
    amrex::Real cur_time,
    bool skip_deposition)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    WARPX_PROFILE("WarpXFluidContainer::Evolve");

    if (fields.has(FieldType::rho_fp,lev) && ! skip_deposition && ! do_not_deposit) {
        // Deposit charge before particle push, in component 0 of MultiFab rho.
        DepositCharge(fields, *fields.get(FieldType::rho_fp,lev), lev, 0);
    }

    // Step the Lorentz Term
    if(!do_not_gather){
        GatherAndPush(fields,
                    *fields.get(FieldType::Efield_aux, Direction{0}, lev),
                    *fields.get(FieldType::Efield_aux, Direction{1}, lev),
                    *fields.get(FieldType::Efield_aux, Direction{2}, lev),
                    *fields.get(FieldType::Bfield_aux, Direction{0}, lev),
                    *fields.get(FieldType::Bfield_aux, Direction{1}, lev),
                    *fields.get(FieldType::Bfield_aux, Direction{2}, lev),
                    cur_time, lev);
    }

    // Cylindrical centrifugal term
    if(!do_not_push){
#if defined(WARPX_DIM_RZ)
        centrifugal_source_rz(fields, lev);
#endif

        // Apply (non-periodic) BC on the fluids (needed for spatial derivative),
        // and communicate N, NU at boundaries
        ApplyBcFluidsAndComms(fields, lev);

        // Step the Advective term
        AdvectivePush_Muscl(fields, lev);
    }

    // Deposit rho to the simulation mesh
    // Deposit charge (end of the step)
    if (fields.has(FieldType::rho_fp,lev) && ! skip_deposition && ! do_not_deposit) {
        DepositCharge(fields, *fields.get(FieldType::rho_fp,lev), lev, 1);
    }

    // Deposit J to the simulation mesh
    if (!skip_deposition && ! do_not_deposit) {
        DepositCurrent(fields,
                        *fields.get(current_fp_string, Direction{0}, lev),
                        *fields.get(current_fp_string, Direction{1}, lev),
                        *fields.get(current_fp_string, Direction{2}, lev),
                        lev);
    }
}

// Momentum source due to curvature
void WarpXFluidContainer::ApplyBcFluidsAndComms (ablastr::fields::MultiFabRegister& fields, int lev)
{
    using ablastr::fields::Direction;
    WARPX_PROFILE("WarpXFluidContainer::ApplyBcFluidsAndComms");

    WarpX &warpx = WarpX::GetInstance();
    const amrex::Geometry &geom = warpx.Geom(lev);
    const amrex::Periodicity &period = geom.periodicity();
    const Array<int,AMREX_SPACEDIM> periodic_directions = geom.isPeriodic();
    amrex::Box domain = geom.Domain();
    // Convert to nodal box
    domain.surroundingNodes();

    // H&C push the momentum
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*fields.get(name_mf_N, lev), TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {

        amrex::Box tile_box = mfi.tilebox(fields.get(name_mf_N, lev)->ixType().toIntVect());

        const amrex::Array4<Real> N_arr = fields.get(name_mf_N, lev)->array(mfi);
        const amrex::Array4<Real> NUx_arr = fields.get(name_mf_NU, Direction{0}, lev)->array(mfi);
        const amrex::Array4<Real> NUy_arr = fields.get(name_mf_NU, Direction{1}, lev)->array(mfi);
        const amrex::Array4<Real> NUz_arr = fields.get(name_mf_NU, Direction{2}, lev)->array(mfi);

        //Grow the tilebox
        tile_box.grow(1);

        amrex::ParallelFor(tile_box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {

                // If the cell is is first guard cell & the dimension is non
                // periodic, then copy Q_{i+1} = Q_{i-1}.
                // Don't check r-dir in Z:
#if defined(WARPX_DIM_3D)

                // Upper end (index 2)
                if ( (periodic_directions[2] != 1) && (k==domain.bigEnd(2)+1) ){
                    N_arr(i,j,k) = N_arr(i,j,k-2);
                    NUx_arr(i,j,k) = NUx_arr(i,j,k-2);
                    NUy_arr(i,j,k) = NUy_arr(i,j,k-2);
                    NUz_arr(i,j,k) = NUz_arr(i,j,k-2);

                // Lower end (index 2)
                } else if ( (periodic_directions[2] != 1) && (k==domain.smallEnd(2)-1) ) {
                    N_arr(i,j,k) = N_arr(i,j,k+2);
                    NUx_arr(i,j,k) = NUx_arr(i,j,k+2);
                    NUy_arr(i,j,k) = NUy_arr(i,j,k+2);
                    NUz_arr(i,j,k) = NUz_arr(i,j,k+2);
                }

#elif ( defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_3D) )

                // Upper end (index 1)
                if ( (periodic_directions[1] != 1) && (j==domain.bigEnd(1)+1) ){
                    N_arr(i,j,k) = N_arr(i,j-2,k);
                    NUx_arr(i,j,k) = NUx_arr(i,j-2,k);
                    NUy_arr(i,j,k) = NUy_arr(i,j-2,k);
                    NUz_arr(i,j,k) = NUz_arr(i,j-2,k);

                // Lower end (index 1`)
                } else if ( (periodic_directions[1] != 1) && (j==domain.smallEnd(1)-1) ) {
                    N_arr(i,j,k) = N_arr(i,j+2,k);
                    NUx_arr(i,j,k) = NUx_arr(i,j+2,k);
                    NUy_arr(i,j,k) = NUy_arr(i,j+2,k);
                    NUz_arr(i,j,k) = NUz_arr(i,j+2,k);

                }

#elif ( defined(WARPX_DIM_1D_Z) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_3D) )

                // Upper end (index 0)
                if ( (periodic_directions[0] != 1) && (i==domain.bigEnd(0)+1) ){
                    N_arr(i,j,k) = N_arr(i-2,j,k);
                    NUx_arr(i,j,k) = NUx_arr(i-2,j,k);
                    NUy_arr(i,j,k) = NUy_arr(i-2,j,k);
                    NUz_arr(i,j,k) = NUz_arr(i-2,j,k);

                // Lower end (index 0)
                } else if ( (periodic_directions[0] != 1) && (i==domain.smallEnd(0)-1) ) {
                    N_arr(i,j,k) = N_arr(i+2,j,k);
                    NUx_arr(i,j,k) = NUx_arr(i+2,j,k);
                    NUy_arr(i,j,k) = NUy_arr(i+2,j,k);
                    NUz_arr(i,j,k) = NUz_arr(i+2,j,k);
                }

#else

#endif
            }
        );
    }

    // Fill guard cells
    FillBoundary(*fields.get(name_mf_N, lev), fields.get(name_mf_N, lev)->nGrowVect(), WarpX::do_single_precision_comms, period);
    FillBoundary(*fields.get(name_mf_NU, Direction{0}, lev), fields.get(name_mf_NU, Direction{0}, lev)->nGrowVect(), WarpX::do_single_precision_comms, period);
    FillBoundary(*fields.get(name_mf_NU, Direction{1}, lev), fields.get(name_mf_NU, Direction{1}, lev)->nGrowVect(), WarpX::do_single_precision_comms, period);
    FillBoundary(*fields.get(name_mf_NU, Direction{2}, lev), fields.get(name_mf_NU, Direction{2}, lev)->nGrowVect(), WarpX::do_single_precision_comms, period);
}

// Muscl Advection Update
void WarpXFluidContainer::AdvectivePush_Muscl (ablastr::fields::MultiFabRegister& fields, int lev)
{
    using ablastr::fields::Direction;
    WARPX_PROFILE("WarpXFluidContainer::AdvectivePush_Muscl");

    // Grab the grid spacing
    WarpX &warpx = WarpX::GetInstance();
    const Real dt = warpx.getdt(lev);
    const amrex::Geometry &geom = warpx.Geom(lev);
    const auto dx = geom.CellSizeArray();
    const amrex::Real clight = PhysConst::c;
#if defined(WARPX_DIM_3D)
    const amrex::Real dt_over_dx = (dt/dx[0]);
    const amrex::Real dt_over_dy = (dt/dx[1]);
    const amrex::Real dt_over_dz = (dt/dx[2]);
    const amrex::Real dt_over_dx_half = 0.5_rt*(dt/dx[0]);
    const amrex::Real dt_over_dy_half = 0.5_rt*(dt/dx[1]);
    const amrex::Real dt_over_dz_half = 0.5_rt*(dt/dx[2]);
#elif defined(WARPX_DIM_XZ)
    const amrex::Real dt_over_dx_half = 0.5_rt*(dt/dx[0]);
    const amrex::Real dt_over_dz_half = 0.5_rt*(dt/dx[1]);
    const amrex::Real dt_over_dx = (dt/dx[0]);
    const amrex::Real dt_over_dz = (dt/dx[1]);
#elif defined(WARPX_DIM_RZ)
    const auto problo = geom.ProbLoArray();
    const amrex::Real dt_over_dx_half = 0.5_rt*(dt/dx[0]);
    const amrex::Real dt_over_dz_half = 0.5_rt*(dt/dx[1]);
    const amrex::Box& domain = geom.Domain();
#else
    const amrex::Real dt_over_dz = (dt/dx[0]);
    const amrex::Real dt_over_dz_half = 0.5_rt*(dt/dx[0]);
#endif

    const amrex::BoxArray ba = fields.get(name_mf_N, lev)->boxArray();

    // Temporary Half-step values
#if defined(WARPX_DIM_3D)
    amrex::MultiFab tmp_U_minus_x( amrex::convert(ba, IntVect(0,1,1)), fields.get(name_mf_N, lev)->DistributionMap(), 4, 1);
    amrex::MultiFab tmp_U_plus_x( amrex::convert(ba, IntVect(0,1,1)), fields.get(name_mf_N, lev)->DistributionMap(), 4, 1);
    amrex::MultiFab tmp_U_minus_y( amrex::convert(ba, IntVect(1,0,1)), fields.get(name_mf_N, lev)->DistributionMap(), 4, 1);
    amrex::MultiFab tmp_U_plus_y( amrex::convert(ba, IntVect(1,0,1)), fields.get(name_mf_N, lev)->DistributionMap(), 4, 1);
    amrex::MultiFab tmp_U_minus_z( amrex::convert(ba, IntVect(1,1,0)), fields.get(name_mf_N, lev)->DistributionMap(), 4, 1);
    amrex::MultiFab tmp_U_plus_z( amrex::convert(ba, IntVect(1,1,0)), fields.get(name_mf_N, lev)->DistributionMap(), 4, 1);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    amrex::MultiFab tmp_U_minus_x( amrex::convert(ba, IntVect(0,1)), fields.get(name_mf_N, lev)->DistributionMap(), 4, 1);
    amrex::MultiFab tmp_U_plus_x( amrex::convert(ba, IntVect(0,1)), fields.get(name_mf_N, lev)->DistributionMap(), 4, 1);
    amrex::MultiFab tmp_U_minus_z( amrex::convert(ba, IntVect(1,0)), fields.get(name_mf_N, lev)->DistributionMap(), 4, 1);
    amrex::MultiFab tmp_U_plus_z( amrex::convert(ba, IntVect(1,0)), fields.get(name_mf_N, lev)->DistributionMap(), 4, 1);
#else
    amrex::MultiFab tmp_U_minus_z( amrex::convert(ba, IntVect(0)), fields.get(name_mf_N, lev)->DistributionMap(), 4, 1);
    amrex::MultiFab tmp_U_plus_z( amrex::convert(ba, IntVect(0)), fields.get(name_mf_N, lev)->DistributionMap(), 4, 1);
#endif

    // Fill edge values of N and U at the half timestep for MUSCL
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*fields.get(name_mf_N, lev), TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {

        // Loop over a box with one extra gridpoint in the ghost region to avoid
        // an extra MPI communication between the edge value computation loop and
        // the flux calculation loop
        const amrex::Box tile_box = [&](){
            auto tt = mfi.growntilebox(1);
#if defined (WARPX_DIM_RZ)
            // Limit the grown box for RZ at r = 0, r_max
            const int idir = 0;
            const int n_cell = -1;
            tt.growLo(idir, n_cell);
            tt.growHi(idir, n_cell);
#endif
           return tt;
        }();

        amrex::Array4<Real> const &N_arr = fields.get(name_mf_N, lev)->array(mfi);
        amrex::Array4<Real> const &NUx_arr = fields.get(name_mf_NU, Direction{0}, lev)->array(mfi);
        amrex::Array4<Real> const &NUy_arr = fields.get(name_mf_NU, Direction{1}, lev)->array(mfi);
        amrex::Array4<Real> const &NUz_arr = fields.get(name_mf_NU, Direction{2}, lev)->array(mfi);

        // Boxes are computed to avoid going out of bounds.
        // Grow the entire domain
        amrex::Box box = mfi.validbox();
        box.grow(1);
#if defined(WARPX_DIM_3D)
        amrex::Box const box_x = amrex::convert( box, tmp_U_minus_x.ixType() );
        amrex::Box const box_y = amrex::convert( box, tmp_U_minus_y.ixType() );
        amrex::Box const box_z = amrex::convert( box, tmp_U_minus_z.ixType() );
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        amrex::Box const box_x = amrex::convert( box, tmp_U_minus_x.ixType() );
        amrex::Box const box_z = amrex::convert( box, tmp_U_minus_z.ixType() );
#else
        amrex::Box const box_z = amrex::convert( box, tmp_U_minus_z.ixType() );
#endif

        //N and NU are always defined at the nodes, the tmp_Q_* are defined
        //in between the nodes (i.e. on the staggered Yee grid) and store the
        //values of N and U at these points.
        //(i.e. the 4 components correspond to N + the 3 components of U)
        // Extract the temporary arrays for edge values
#if defined(WARPX_DIM_3D)
        const amrex::Array4<amrex::Real> U_minus_x = tmp_U_minus_x.array(mfi);
        const amrex::Array4<amrex::Real> U_plus_x = tmp_U_plus_x.array(mfi);
        const amrex::Array4<amrex::Real> U_minus_y = tmp_U_minus_y.array(mfi);
        const amrex::Array4<amrex::Real> U_plus_y = tmp_U_plus_y.array(mfi);
        const amrex::Array4<amrex::Real> U_minus_z = tmp_U_minus_z.array(mfi);
        const amrex::Array4<amrex::Real> U_plus_z = tmp_U_plus_z.array(mfi);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        const amrex::Array4<amrex::Real> U_minus_x = tmp_U_minus_x.array(mfi);
        const amrex::Array4<amrex::Real> U_plus_x = tmp_U_plus_x.array(mfi);
        const amrex::Array4<amrex::Real> U_minus_z = tmp_U_minus_z.array(mfi);
        const amrex::Array4<amrex::Real> U_plus_z = tmp_U_plus_z.array(mfi);
#else
        const amrex::Array4<amrex::Real> U_minus_z = tmp_U_minus_z.array(mfi);
        const amrex::Array4<amrex::Real> U_plus_z = tmp_U_plus_z.array(mfi);
#endif

        amrex::ParallelFor(tile_box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {

                // Density positivity check (Makes the algorithm safe from divide by zeros)
                if( N_arr(i,j,k) > 0.0){

                    // - Grab local Uz Uy Ux gamma
                    // Isolate U from NU
                    amrex::Real Ux = (NUx_arr(i, j, k) / N_arr(i,j,k));
                    amrex::Real Uy = (NUy_arr(i, j, k) / N_arr(i,j,k));
                    amrex::Real Uz = (NUz_arr(i, j, k) / N_arr(i,j,k));

                    // Compute useful quantities for J
                    const amrex::Real c_sq = clight*clight;
                    const amrex::Real gamma = std::sqrt(1.0_rt + (Ux*Ux + Uy*Uy + Uz*Uz)/(c_sq) );
                    const amrex::Real inv_c2_gamma3 = 1._rt/(c_sq*gamma*gamma*gamma);

                    // J represents are 4x4 matrices that show up in the advection
                    // equations written as a function of U = {N, Ux, Uy, Uz}:
                    // \partial_t U + Jx \partial_x U + Jy \partial_y U + Jz \partial_z U = 0
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
                    const amrex::Real Vx = Ux/gamma;
                    // Compute the non-zero element of Jx
                    const amrex::Real J00x = Vx;
                    const amrex::Real J01x = N_arr(i,j,k)*(1/gamma)*(1-Vx*Vx/c_sq);
                    const amrex::Real J02x = -N_arr(i,j,k)*Uy*Ux*inv_c2_gamma3;
                    const amrex::Real J03x = -N_arr(i,j,k)*Uz*Ux*inv_c2_gamma3;
                    const amrex::Real J11x = Vx;
                    const amrex::Real J22x = Vx;
                    const amrex::Real J33x = Vx;

                    amrex::Real dU0x, dU1x, dU2x, dU3x;

                    // Compute the cell slopes x
                    dU0x = ave( DownDx_N(N_arr,i,j,k), UpDx_N(N_arr,i,j,k) );
                    dU1x = ave( DownDx_U(N_arr,NUx_arr,Ux,i,j,k), UpDx_U(N_arr,NUx_arr,Ux,i,j,k) );
                    dU2x = ave( DownDx_U(N_arr,NUy_arr,Uy,i,j,k), UpDx_U(N_arr,NUy_arr,Uy,i,j,k) );
                    dU3x = ave( DownDx_U(N_arr,NUz_arr,Uz,i,j,k), UpDx_U(N_arr,NUz_arr,Uz,i,j,k) );

#endif

#if defined(WARPX_DIM_3D)
                    const amrex::Real Vy = Uy/gamma;
                    // Compute the non-zero element of Jy
                    const amrex::Real J00y = Vy;
                    const amrex::Real J01y = -N_arr(i,j,k)*Ux*Uy*inv_c2_gamma3;
                    const amrex::Real J02y = N_arr(i,j,k)*(1/gamma)*(1-Vy*Vy/c_sq);
                    const amrex::Real J03y = -N_arr(i,j,k)*Uz*Uy*inv_c2_gamma3;
                    const amrex::Real J11y = Vy;
                    const amrex::Real J22y = Vy;
                    const amrex::Real J33y = Vy;

                    // Compute the cell slopes y
                    const amrex::Real dU0y = ave( DownDy_N(N_arr,i,j,k), UpDy_N(N_arr,i,j,k) );
                    const amrex::Real dU1y = ave( DownDy_U(N_arr,NUx_arr,Ux,i,j,k), UpDy_U(N_arr,NUx_arr,Ux,i,j,k) );
                    const amrex::Real dU2y = ave( DownDy_U(N_arr,NUy_arr,Uy,i,j,k), UpDy_U(N_arr,NUy_arr,Uy,i,j,k) );
                    const amrex::Real dU3y = ave( DownDy_U(N_arr,NUz_arr,Uz,i,j,k), UpDy_U(N_arr,NUz_arr,Uz,i,j,k) );

#endif
                    const amrex::Real Vz = Uz/gamma;
                    // Compute the non-zero element of Jz
                    const amrex::Real J00z = Vz;
                    const amrex::Real J01z = -N_arr(i,j,k)*Ux*Uz*inv_c2_gamma3;
                    const amrex::Real J02z = -N_arr(i,j,k)*Uy*Uz*inv_c2_gamma3;
                    const amrex::Real J03z = N_arr(i,j,k)*(1/gamma)*(1-Vz*Vz/c_sq);
                    const amrex::Real J11z = Vz;
                    const amrex::Real J22z = Vz;
                    const amrex::Real J33z = Vz;

                    // Compute the cell slopes z
                    const amrex::Real dU0z = ave( DownDz_N(N_arr,i,j,k), UpDz_N(N_arr,i,j,k) );
                    const amrex::Real dU1z = ave( DownDz_U(N_arr,NUx_arr,Ux,i,j,k), UpDz_U(N_arr,NUx_arr,Ux,i,j,k) );
                    const amrex::Real dU2z = ave( DownDz_U(N_arr,NUy_arr,Uy,i,j,k), UpDz_U(N_arr,NUy_arr,Uy,i,j,k) );
                    const amrex::Real dU3z = ave( DownDz_U(N_arr,NUz_arr,Uz,i,j,k), UpDz_U(N_arr,NUz_arr,Uz,i,j,k) );


                    // Select the specific implementation depending on dimensionality
#if defined(WARPX_DIM_3D)

                    // Compute U ([ N, U]) at the halfsteps (U_tilde) using the slopes (dU)
                    const amrex::Real JdU0x = J00x*dU0x + J01x*dU1x + J02x*dU2x + J03x*dU3x;
                    const amrex::Real JdU1x = J11x*dU1x ;
                    const amrex::Real JdU2x = J22x*dU2x ;
                    const amrex::Real JdU3x = J33x*dU3x;
                    const amrex::Real JdU0y = J00y*dU0y + J01y*dU1y + J02y*dU2y + J03y*dU3y;
                    const amrex::Real JdU1y = J11y*dU1y;
                    const amrex::Real JdU2y = J22y*dU2y;
                    const amrex::Real JdU3y = J33y*dU3y;
                    const amrex::Real JdU0z = J00z*dU0z + J01z*dU1z + J02z*dU2z + J03z*dU3z;
                    const amrex::Real JdU1z = J11z*dU1z;
                    const amrex::Real JdU2z = J22z*dU2z;
                    const amrex::Real JdU3z = J33z*dU3z;
                    const amrex::Real U_tilde0 = N_arr(i,j,k)   - dt_over_dx_half*JdU0x - dt_over_dy_half*JdU0y - dt_over_dz_half*JdU0z;
                    const amrex::Real U_tilde1 = Ux - dt_over_dx_half*JdU1x - dt_over_dy_half*JdU1y - dt_over_dz_half*JdU1z;
                    const amrex::Real U_tilde2 = Uy - dt_over_dx_half*JdU2x - dt_over_dy_half*JdU2y - dt_over_dz_half*JdU2z;
                    const amrex::Real U_tilde3 = Uz - dt_over_dx_half*JdU3x - dt_over_dy_half*JdU3y - dt_over_dz_half*JdU3z;


                    // Predict U at the cell edges (x)
                    compute_U_edges(U_minus_x, U_plus_x, i, j, k, box_x, U_tilde0, U_tilde1, U_tilde2, U_tilde3, dU0x, dU1x, dU2x, dU3x,0);

                    // Positivity Limiter for density N, if N_edge < 0,
                    // then set the slope (dU) to to zero in that cell/direction
                    positivity_limiter (U_plus_x, U_minus_x,  N_arr, i, j, k, box_x, Ux, Uy, Uz, 0);

                    // Predict U at the cell edges (y)
                    compute_U_edges(U_minus_y, U_plus_y, i, j, k, box_y, U_tilde0, U_tilde1, U_tilde2, U_tilde3, dU0y, dU1y, dU2y, dU3y,1);

                    // Positivity Limiter for density N, if N_edge < 0,
                    // then set the slope (dU) to to zero in that cell/direction
                    positivity_limiter (U_plus_y, U_minus_y,  N_arr, i, j, k, box_y, Ux, Uy, Uz, 1);

                    // Predict U at the cell edges (z)
                    compute_U_edges(U_minus_z, U_plus_z, i, j, k, box_z, U_tilde0, U_tilde1, U_tilde2, U_tilde3, dU0z, dU1z, dU2z, dU3z,2);

                    // Positivity Limiter for density N, if N_edge < 0,
                    // then set the slope (dU) to to zero in that cell/direction
                    positivity_limiter (U_plus_z, U_minus_z,  N_arr, i, j, k, box_z, Ux, Uy, Uz, 2);

#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)

#if defined(WARPX_DIM_RZ)
                    const amrex::Real dr = dx[0];
                    const amrex::Real r = problo[0] + i * dr;
                    // Impose "none" boundaries
                    // Condition: dUx = 0 at r = 0
                    if  (i == domain.smallEnd(0)) {
                        // R|_{0+} -> L|_{0-}
                        // N -> N (N_arr(i-1,j,k) -> N_arr(i+1,j,k))
                        // NUr -> -NUr (NUx_arr(i-1,j,k) -> -NUx_arr(i+1,j,k))
                        // NUt -> -NUt (NUy_arr(i-1,j,k) -> -NUy_arr(i+1,j,k))
                        // NUz -> -NUz (NUz_arr(i-1,j,k) -> NUz_arr(i+1,j,k))
                        dU0x = ave( -UpDx_N(N_arr,i,j,k) , UpDx_N(N_arr,i,j,k) );
                        // First term in the ave is: U_{x,y} + U_{x,y}_p,
                        // which can be written as 2*U_{x,y} + UpDx_U(U_{x,y})
                        dU1x = ave( 2.0_rt*Ux + UpDx_U(N_arr,NUx_arr,Ux,i,j,k) , UpDx_U(N_arr,NUx_arr,Ux,i,j,k) );
                        dU2x = ave( 2.0_rt*Uy + UpDx_U(N_arr,NUy_arr,Uy,i,j,k) , UpDx_U(N_arr,NUy_arr,Uy,i,j,k) );
                        dU3x = ave( -UpDx_U(N_arr,NUz_arr,Uz,i,j,k) , UpDx_U(N_arr,NUz_arr,Uz,i,j,k) );
                    } else if (i == domain.bigEnd(0)+1) {
                        dU0x = ave( DownDx_N(N_arr,i,j,k) , 0.0_rt );
                        dU1x = ave( DownDx_U(N_arr,NUx_arr,Ux,i,j,k) , 0.0_rt );
                        dU2x = ave( DownDx_U(N_arr,NUy_arr,Uy,i,j,k) , 0.0_rt );
                        dU3x = ave( DownDx_U(N_arr,NUz_arr,Uz,i,j,k) , 0.0_rt );
                    }

                    // RZ sources:
                    const amrex::Real N_source =
                        (i != domain.smallEnd(0)) ? N_arr(i,j,k)*Vx/r : 0.0_rt;
#else
                    // Have no RZ-inertial source for primitive vars if in XZ
                    const amrex::Real N_source = 0.0;
#endif

                    // Compute U ([ N, U]) at the halfsteps (U_tilde) using the slopes (dU)
                    const amrex::Real  JdU0x = J00x*dU0x + J01x*dU1x + J02x*dU2x + J03x*dU3x;
                    const amrex::Real  JdU1x = J11x*dU1x;
                    const amrex::Real  JdU2x = J22x*dU2x;
                    const amrex::Real  JdU3x = J33x*dU3x;
                    const amrex::Real  JdU0z = J00z*dU0z + J01z*dU1z + J02z*dU2z + J03z*dU3z;
                    const amrex::Real  JdU1z = J11z*dU1z;
                    const amrex::Real  JdU2z = J22z*dU2z;
                    const amrex::Real  JdU3z = J33z*dU3z;
                    const amrex::Real  U_tilde0 = N_arr(i,j,k)   - dt_over_dx_half*JdU0x - dt_over_dz_half*JdU0z - (dt/2.0_rt)*N_source;
                    const amrex::Real  U_tilde1 = Ux - dt_over_dx_half*JdU1x - dt_over_dz_half*JdU1z;
                    const amrex::Real  U_tilde2 = Uy - dt_over_dx_half*JdU2x - dt_over_dz_half*JdU2z;
                    const amrex::Real  U_tilde3 = Uz - dt_over_dx_half*JdU3x - dt_over_dz_half*JdU3z;

                    // Predict U at the cell edges (x)
                    compute_U_edges(U_minus_x, U_plus_x, i, j, k, box_x, U_tilde0, U_tilde1, U_tilde2, U_tilde3, dU0x, dU1x, dU2x, dU3x,0);

                    // Positivity Limiter for density N, if N_edge < 0,
                    // then set the slope (dU) to to zero in that cell/direction
                    positivity_limiter (U_plus_x, U_minus_x,  N_arr, i, j, k, box_x, Ux, Uy, Uz, 0);

                    // Predict U at the cell edges (z)
                    compute_U_edges(U_minus_z, U_plus_z, i, j, k, box_z, U_tilde0, U_tilde1, U_tilde2, U_tilde3, dU0z, dU1z, dU2z, dU3z,2);

                    // Positivity Limiter for density N, if N_edge < 0,
                    // then set the slope (dU) to to zero in that cell/direction
                    positivity_limiter (U_plus_z, U_minus_z,  N_arr, i, j, k, box_z, Ux, Uy, Uz, 2);

#else

                    // Compute U ([ N, U]) at the halfsteps (U_tilde) using the slopes (dU)
                    const amrex::Real  JdU0z = J00z*dU0z + J01z*dU1z + J02z*dU2z + J03z*dU3z;
                    const amrex::Real  JdU1z = J11z*dU1z;
                    const amrex::Real  JdU2z = J22z*dU2z;
                    const amrex::Real  JdU3z = J33z*dU3z;
                    const amrex::Real  U_tilde0 = N_arr(i,j,k)   - dt_over_dz_half*JdU0z;
                    const amrex::Real  U_tilde1 = Ux - dt_over_dz_half*JdU1z;
                    const amrex::Real  U_tilde2 = Uy - dt_over_dz_half*JdU2z;
                    const amrex::Real  U_tilde3 = Uz - dt_over_dz_half*JdU3z;

                    // Predict U at the cell edges (z)
                    compute_U_edges(U_minus_z, U_plus_z, i, j, k, box_z, U_tilde0, U_tilde1, U_tilde2, U_tilde3, dU0z, dU1z, dU2z, dU3z,2);

                    // Positivity Limiter for density N, if N_edge < 0,
                    // then set the slope (dU) to to zero in that cell/direction
                    positivity_limiter (U_plus_z, U_minus_z,  N_arr, i, j, k, box_z, Ux, Uy, Uz, 2);

#endif
                // If N<= 0 then set the edge values (U_minus/U_plus) to zero
                } else {
#if defined(WARPX_DIM_3D)
                    set_U_edges_to_zero(U_minus_x, U_plus_x, i, j, k, box_x, 0);
                    set_U_edges_to_zero(U_minus_y, U_plus_y, i, j, k, box_y, 1);
                    set_U_edges_to_zero(U_minus_z, U_plus_z, i, j, k, box_z, 2);
#elif defined(WARPX_DIM_RZ) || defined(WARPX_DIM_XZ)
                    set_U_edges_to_zero(U_minus_x, U_plus_x, i, j, k, box_x, 0);
                    set_U_edges_to_zero(U_minus_z, U_plus_z, i, j, k, box_z, 2);
#else
                    set_U_edges_to_zero(U_minus_z, U_plus_z, i, j, k, box_z, 2);
#endif
                }
            }
        );
    }

    // Given the values of `U_minus` and `U_plus`, compute fluxes in between nodes, and update N, NU accordingly
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*fields.get(name_mf_N, lev), TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box tile_box = mfi.tilebox(fields.get(name_mf_N, lev)->ixType().toIntVect());
        const amrex::Array4<Real> N_arr = fields.get(name_mf_N, lev)->array(mfi);
        const amrex::Array4<Real> NUx_arr = fields.get(name_mf_NU, Direction{0}, lev)->array(mfi);
        const amrex::Array4<Real> NUy_arr = fields.get(name_mf_NU, Direction{1}, lev)->array(mfi);
        const amrex::Array4<Real> NUz_arr = fields.get(name_mf_NU, Direction{2}, lev)->array(mfi);

#if defined(WARPX_DIM_3D)
        amrex::Array4<amrex::Real> const &U_minus_x = tmp_U_minus_x.array(mfi);
        amrex::Array4<amrex::Real> const &U_plus_x = tmp_U_plus_x.array(mfi);
        amrex::Array4<amrex::Real> const &U_minus_y = tmp_U_minus_y.array(mfi);
        amrex::Array4<amrex::Real> const &U_plus_y = tmp_U_plus_y.array(mfi);
        amrex::Array4<amrex::Real> const &U_minus_z = tmp_U_minus_z.array(mfi);
        amrex::Array4<amrex::Real> const &U_plus_z = tmp_U_plus_z.array(mfi);
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        amrex::Array4<amrex::Real> const &U_minus_x = tmp_U_minus_x.array(mfi);
        amrex::Array4<amrex::Real> const &U_plus_x = tmp_U_plus_x.array(mfi);
        amrex::Array4<amrex::Real> const &U_minus_z = tmp_U_minus_z.array(mfi);
        amrex::Array4<amrex::Real> const &U_plus_z = tmp_U_plus_z.array(mfi);
#else
        amrex::Array4<amrex::Real> const &U_minus_z = tmp_U_minus_z.array(mfi);
        amrex::Array4<amrex::Real> const &U_plus_z = tmp_U_plus_z.array(mfi);
#endif

        amrex::ParallelFor(tile_box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {

                // Select the specific implementation depending on dimensionality
#if defined(WARPX_DIM_3D)

                // Update the conserved variables Q = [N, NU] from tn -> tn + dt
                N_arr(i,j,k) = N_arr(i,j,k)  - dt_over_dx*dF(U_minus_x,U_plus_x,i,j,k,clight,0,0)
                                             - dt_over_dy*dF(U_minus_y,U_plus_y,i,j,k,clight,0,1)
                                             - dt_over_dz*dF(U_minus_z,U_plus_z,i,j,k,clight,0,2);
                NUx_arr(i,j,k) = NUx_arr(i,j,k) - dt_over_dx*dF(U_minus_x,U_plus_x,i,j,k,clight,1,0)
                                                - dt_over_dy*dF(U_minus_y,U_plus_y,i,j,k,clight,1,1)
                                                - dt_over_dz*dF(U_minus_z,U_plus_z,i,j,k,clight,1,2);
                NUy_arr(i,j,k) = NUy_arr(i,j,k) - dt_over_dx*dF(U_minus_x,U_plus_x,i,j,k,clight,2,0)
                                                - dt_over_dy*dF(U_minus_y,U_plus_y,i,j,k,clight,2,1)
                                                - dt_over_dz*dF(U_minus_z,U_plus_z,i,j,k,clight,2,2);
                NUz_arr(i,j,k) = NUz_arr(i,j,k) - dt_over_dx*dF(U_minus_x,U_plus_x,i,j,k,clight,3,0)
                                                - dt_over_dy*dF(U_minus_y,U_plus_y,i,j,k,clight,3,1)
                                                - dt_over_dz*dF(U_minus_z,U_plus_z,i,j,k,clight,3,2);

#elif defined(WARPX_DIM_XZ)

                // Update the conserved variables Q = [N, NU] from tn -> tn + dt
                N_arr(i,j,k) = N_arr(i,j,k)  - dt_over_dx*dF(U_minus_x,U_plus_x,i,j,k,clight,0,0)
                                             - dt_over_dz*dF(U_minus_z,U_plus_z,i,j,k,clight,0,2);
                NUx_arr(i,j,k) = NUx_arr(i,j,k) - dt_over_dx*dF(U_minus_x,U_plus_x,i,j,k,clight,1,0)
                                                - dt_over_dz*dF(U_minus_z,U_plus_z,i,j,k,clight,1,2);
                NUy_arr(i,j,k) = NUy_arr(i,j,k) - dt_over_dx*dF(U_minus_x,U_plus_x,i,j,k,clight,2,0)
                                                - dt_over_dz*dF(U_minus_z,U_plus_z,i,j,k,clight,2,2);
                NUz_arr(i,j,k) = NUz_arr(i,j,k) - dt_over_dx*dF(U_minus_x,U_plus_x,i,j,k,clight,3,0)
                                                - dt_over_dz*dF(U_minus_z,U_plus_z,i,j,k,clight,3,2);

#elif defined(WARPX_DIM_RZ)

                // Compute the flux areas for RZ
                // Cell-centered radius
                const amrex::Real dr = dx[0];
                const amrex::Real dz = dx[1];
                const amrex::Real r = problo[0] + i * dr;
                amrex::Real Vij = 0.0_rt;
                amrex::Real S_Az = 0.0_rt;

                // Volume element and z-facing surfaces
                if (i == domain.smallEnd(0)) {
                    Vij = 2.0_rt*MathConst::pi*(dr/2.0_rt)*(dr/4.0_rt)*dz;
                    S_Az = 2.0_rt*MathConst::pi*(dr/4.0_rt)*(dr/2.0_rt);
                } else if (i == domain.bigEnd(0)+1) {
                    Vij = 2.0_rt*MathConst::pi*(r - dr/4.0_rt)*(dr/2.0_rt)*dz;
                    S_Az = 2.0_rt*MathConst::pi*(r - dr/4.0_rt)*(dr/2.0_rt);
                }  else {
                    Vij = 2.0_rt*MathConst::pi*r*dr*dz;
                    S_Az = 2.0_rt*MathConst::pi*(r)*dr;
                }

                // Radial Surfaces
                amrex::Real S_Ar_plus = 2.0_rt*MathConst::pi*(r + dr/2.0_rt)*dz;
                amrex::Real S_Ar_minus = 2.0_rt*MathConst::pi*(r - dr/2.0_rt)*dz;
                if (i == domain.smallEnd(0)) {
                    S_Ar_minus = 0.0_rt;
                }
                if (i == domain.bigEnd(0)+1) {
                    S_Ar_plus = 2.0_rt*MathConst::pi*(r)*dz;
                }

                // Impose "none" boundaries
                // Condition: Vx(r) = 0 at boundaries
                const amrex::Real Vx_I_minus = V_calc(U_minus_x,i,j,k,0,clight);
                const amrex::Real Vx_L_plus = V_calc(U_plus_x,i-1,j,k,0,clight);

                // compute the fluxes:
                // (note that _plus is shifted due to grid location)
                amrex::Real Vx_L_minus = 0.0_rt, Vx_I_plus = 0.0_rt;
                amrex::Real F0_minusx = 0.0_rt, F1_minusx = 0.0_rt, F2_minusx = 0.0_rt, F3_minusx = 0.0_rt;
                amrex::Real F0_plusx = 0.0_rt, F1_plusx = 0.0_rt, F2_plusx = 0.0_rt, F3_plusx = 0.0_rt;
                if (i != domain.smallEnd(0)) {
                    Vx_L_minus = V_calc(U_minus_x,i-1,j,k,0,clight);
                    F0_minusx = flux_N(  U_minus_x, U_plus_x, i-1, j, k, Vx_L_minus, Vx_L_plus)*S_Ar_minus;
                    F1_minusx = flux_NUx(U_minus_x, U_plus_x, i-1, j, k, Vx_L_minus, Vx_L_plus)*S_Ar_minus;
                    F2_minusx = flux_NUy(U_minus_x, U_plus_x, i-1, j, k, Vx_L_minus, Vx_L_plus)*S_Ar_minus;
                    F3_minusx = flux_NUz(U_minus_x, U_plus_x, i-1, j, k, Vx_L_minus, Vx_L_plus)*S_Ar_minus;
                }
                if (i < domain.bigEnd(0)) {
                    Vx_I_plus = V_calc(U_plus_x,i,j,k,0,clight);
                    F0_plusx  = flux_N(  U_minus_x, U_plus_x, i  , j, k, Vx_I_minus, Vx_I_plus)*S_Ar_plus;
                    F1_plusx  = flux_NUx(U_minus_x, U_plus_x, i  , j, k, Vx_I_minus, Vx_I_plus)*S_Ar_plus;
                    F2_plusx  = flux_NUy(U_minus_x, U_plus_x, i  , j, k, Vx_I_minus, Vx_I_plus)*S_Ar_plus;
                    F3_plusx  = flux_NUz(U_minus_x, U_plus_x, i  , j, k, Vx_I_minus, Vx_I_plus)*S_Ar_plus;
                }

                // Update the conserved variables from tn -> tn + dt
                N_arr(i,j,k) = N_arr(i,j,k)     - (dt/Vij)*(F0_plusx - F0_minusx + dF(U_minus_z,U_plus_z,i,j,k,clight,0,2)*S_Az);
                NUx_arr(i,j,k) = NUx_arr(i,j,k) - (dt/Vij)*(F1_plusx - F1_minusx + dF(U_minus_z,U_plus_z,i,j,k,clight,1,2)*S_Az);
                NUy_arr(i,j,k) = NUy_arr(i,j,k) - (dt/Vij)*(F2_plusx - F2_minusx + dF(U_minus_z,U_plus_z,i,j,k,clight,2,2)*S_Az);
                NUz_arr(i,j,k) = NUz_arr(i,j,k) - (dt/Vij)*(F3_plusx - F3_minusx + dF(U_minus_z,U_plus_z,i,j,k,clight,3,2)*S_Az);

#else

                // Update the conserved variables Q = [N, NU] from tn -> tn + dt
                N_arr(i,j,k) = N_arr(i,j,k) - dt_over_dz*dF(U_minus_z,U_plus_z,i,j,k,clight,0,2);
                NUx_arr(i,j,k) = NUx_arr(i,j,k) - dt_over_dz*dF(U_minus_z,U_plus_z,i,j,k,clight,1,2);
                NUy_arr(i,j,k) = NUy_arr(i,j,k) - dt_over_dz*dF(U_minus_z,U_plus_z,i,j,k,clight,2,2);
                NUz_arr(i,j,k) = NUz_arr(i,j,k) - dt_over_dz*dF(U_minus_z,U_plus_z,i,j,k,clight,3,2);
#endif
            }
        );
    }
}


// Momentum source due to curvature
#if defined(WARPX_DIM_RZ)
void WarpXFluidContainer::centrifugal_source_rz (ablastr::fields::MultiFabRegister& fields, int lev)
{
    using ablastr::fields::Direction;
    WARPX_PROFILE("WarpXFluidContainer::centrifugal_source_rz");

    WarpX &warpx = WarpX::GetInstance();
    const Real dt = warpx.getdt(lev);
    const amrex::Geometry &geom = warpx.Geom(lev);
    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();
    const amrex::Real clight = PhysConst::c;
    amrex::Box const& domain = geom.Domain();

    // H&C push the momentum
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*fields.get(name_mf_N, lev), TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {

        amrex::Box const &tile_box = mfi.tilebox(fields.get(name_mf_N, lev)->ixType().toIntVect());

        amrex::Array4<Real> const &N_arr = fields.get(name_mf_N, lev)->array(mfi);
        const amrex::Array4<Real> NUx_arr = fields.get(name_mf_NU, Direction{0}, lev)->array(mfi);
        const amrex::Array4<Real> NUy_arr = fields.get(name_mf_NU, Direction{1}, lev)->array(mfi);
        amrex::Array4<Real> const &NUz_arr = fields.get(name_mf_NU, Direction{2}, lev)->array(mfi);

        amrex::ParallelFor(tile_box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {

                // Verify density is non-zero
                if (N_arr(i,j,k)>0.0_rt) {

                    // Compute r
                    const amrex::Real r = problo[0] + i * dx[0];

                    // Isolate U from NU
                    amrex::Real u_r =     (NUx_arr(i, j, k) / (N_arr(i,j,k) * clight ));
                    amrex::Real u_theta = (NUy_arr(i, j, k) / (N_arr(i,j,k) * clight ));
                    const amrex::Real u_z =     (NUz_arr(i, j, k) / (N_arr(i,j,k) * clight ));

                    // (SSP-RK3) Push the fluid momentum (R and Theta)
                    // F_r, F_theta are first order euler pushes of our rhs operator
                    if (i != domain.smallEnd(0)) {
                        const amrex::Real u_r_1     = F_r(r,u_r,u_theta,u_z,dt);
                        const amrex::Real u_theta_1 = F_theta(r,u_r,u_theta,u_z,dt);
                        const amrex::Real u_r_2     = (0.75_rt)*(u_r)     + (0.25_rt)*F_r(r,u_r_1,u_theta_1,u_z,dt);
                        const amrex::Real u_theta_2 = (0.75_rt)*(u_theta) + (0.25_rt)*F_theta(r,u_r_1,u_theta_1,u_z,dt);
                        u_r            = (1.0_rt/3.0_rt)*(u_r)     + (2.0_rt/3.0_rt)*F_r(r,u_r_2,u_theta_2,u_z,dt);
                        u_theta        = (1.0_rt/3.0_rt)*(u_theta) + (2.0_rt/3.0_rt)*F_theta(r,u_r_2,u_theta_2,u_z,dt);

                        // Calculate NU, save NUr, NUtheta
                        NUx_arr(i,j,k) = N_arr(i,j,k)*u_r*clight;
                        NUy_arr(i,j,k) = N_arr(i,j,k)*u_theta*clight;

                    // BC r = 0, u_theta = 0, and there is no extra source terms
                    } else {
                        NUx_arr(i,j,k) = 0.0_rt;
                        NUy_arr(i,j,k) = 0.0_rt;
                    }
                }
            }
        );
    }
}
#endif

// Momentum source from fields
void WarpXFluidContainer::GatherAndPush (
    ablastr::fields::MultiFabRegister& fields,
    const amrex::MultiFab& Ex, const amrex::MultiFab& Ey, const amrex::MultiFab& Ez,
    const amrex::MultiFab& Bx, const amrex::MultiFab& By, const amrex::MultiFab& Bz,
    Real t,
    int lev)
{
    using ablastr::fields::Direction;
    WARPX_PROFILE("WarpXFluidContainer::GatherAndPush");

    WarpX &warpx = WarpX::GetInstance();
    const amrex::Real q = getCharge();
    const amrex::Real m = getMass();
    const Real dt = warpx.getdt(lev);
    const amrex::Geometry &geom = warpx.Geom(lev);
    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();
    const amrex::Real gamma_boost = WarpX::gamma_boost;
    const amrex::Real beta_boost = WarpX::beta_boost;
    //Check whether m_E_ext_s is "none"
    bool external_e_fields; // Needs intializing
    bool external_b_fields; // Needs intializing


    // Prepare interpolation of current components to cell center
    auto Nodal_type = amrex::GpuArray<int, 3>{0, 0, 0};
    auto Ex_type = amrex::GpuArray<int, 3>{0, 0, 0};
    auto Ey_type = amrex::GpuArray<int, 3>{0, 0, 0};
    auto Ez_type = amrex::GpuArray<int, 3>{0, 0, 0};
    auto Bx_type = amrex::GpuArray<int, 3>{0, 0, 0};
    auto By_type = amrex::GpuArray<int, 3>{0, 0, 0};
    auto Bz_type = amrex::GpuArray<int, 3>{0, 0, 0};
    for (int i = 0; i < AMREX_SPACEDIM; ++i)
    {
        Nodal_type[i] = fields.get(name_mf_N, lev)->ixType()[i];
        Ex_type[i] = Ex.ixType()[i];
        Ey_type[i] = Ey.ixType()[i];
        Ez_type[i] = Ez.ixType()[i];
        Bx_type[i] = Bx.ixType()[i];
        By_type[i] = By.ixType()[i];
        Bz_type[i] = Bz.ixType()[i];
    }

    // External field parsers
    external_e_fields = (m_E_ext_s == "parse_e_ext_function");
    external_b_fields = (m_B_ext_s == "parse_b_ext_function");

    amrex::ParserExecutor<4> Exfield_parser;
    amrex::ParserExecutor<4> Eyfield_parser;
    amrex::ParserExecutor<4> Ezfield_parser;
    amrex::ParserExecutor<4> Bxfield_parser;
    amrex::ParserExecutor<4> Byfield_parser;
    amrex::ParserExecutor<4> Bzfield_parser;

    if (external_e_fields){
        Exfield_parser = m_Ex_parser->compile<4>();
        Eyfield_parser = m_Ey_parser->compile<4>();
        Ezfield_parser = m_Ez_parser->compile<4>();
    }
    if (external_b_fields){
        Bxfield_parser = m_Bx_parser->compile<4>();
        Byfield_parser = m_By_parser->compile<4>();
        Bzfield_parser = m_Bz_parser->compile<4>();
    }


    // H&C push the momentum
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*fields.get(name_mf_N, lev), TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {

        amrex::Box const &tile_box = mfi.tilebox(fields.get(name_mf_N, lev)->ixType().toIntVect());

        amrex::Array4<Real> const &N_arr = fields.get(name_mf_N, lev)->array(mfi);
        const amrex::Array4<Real> NUx_arr = fields.get(name_mf_NU, Direction{0}, lev)->array(mfi);
        const amrex::Array4<Real> NUy_arr = fields.get(name_mf_NU, Direction{1}, lev)->array(mfi);
        const amrex::Array4<Real> NUz_arr = fields.get(name_mf_NU, Direction{2}, lev)->array(mfi);

        amrex::Array4<const amrex::Real> const& Ex_arr = Ex.array(mfi);
        amrex::Array4<const amrex::Real> const& Ey_arr = Ey.array(mfi);
        amrex::Array4<const amrex::Real> const& Ez_arr = Ez.array(mfi);
        amrex::Array4<const amrex::Real> const& Bx_arr = Bx.array(mfi);
        amrex::Array4<const amrex::Real> const& By_arr = By.array(mfi);
        amrex::Array4<const amrex::Real> const& Bz_arr = Bz.array(mfi);

        // Here, we do not perform any coarsening.
        const amrex::GpuArray<int, 3U> coarsening_ratio = {1, 1, 1};

        enum exte_flags : int { no_exte, has_exte };
        enum extb_flags : int { no_extb, has_extb };
        enum boost_flags : int { no_gamma_boost, has_gamma_boost };
        const int exte_runtime_flag = external_e_fields ? has_exte : no_exte;
        const int extb_runtime_flag = external_b_fields ? has_extb : no_extb;
        const int boost_runtime_flag = (gamma_boost > 1._rt) ? has_gamma_boost : no_gamma_boost;

        amrex::ParallelFor(TypeList<CompileTimeOptions<no_exte, has_exte>,
                                    CompileTimeOptions<no_extb, has_extb>,
                                    CompileTimeOptions<no_gamma_boost, has_gamma_boost>>{},
                           {exte_runtime_flag, extb_runtime_flag, boost_runtime_flag},
                           tile_box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k,
                                 auto exte_control, auto extb_control, auto boost_control) noexcept
            {

                // Only run if density is positive
                if (N_arr(i,j,k)>0.0) {

                    // Interpolate fields from tmp to Nodal points
                    // NOLINTBEGIN(misc-const-correctness)
                    amrex::Real Ex_Nodal = ablastr::coarsen::sample::Interp(Ex_arr,
                        Ex_type, Nodal_type, coarsening_ratio, i, j, k, 0);
                    amrex::Real Ey_Nodal = ablastr::coarsen::sample::Interp(Ey_arr,
                        Ey_type, Nodal_type, coarsening_ratio, i, j, k, 0);
                    amrex::Real Ez_Nodal = ablastr::coarsen::sample::Interp(Ez_arr,
                        Ez_type, Nodal_type, coarsening_ratio, i, j, k, 0);
                    amrex::Real Bx_Nodal = ablastr::coarsen::sample::Interp(Bx_arr,
                        Bx_type, Nodal_type, coarsening_ratio, i, j, k, 0);
                    amrex::Real By_Nodal = ablastr::coarsen::sample::Interp(By_arr,
                        By_type, Nodal_type, coarsening_ratio, i, j, k, 0);
                    amrex::Real Bz_Nodal = ablastr::coarsen::sample::Interp(Bz_arr,
                        Bz_type, Nodal_type, coarsening_ratio, i, j, k, 0);
                    // NOLINTEND(misc-const-correctness)

#ifdef AMREX_USE_CUDA
                    amrex::ignore_unused(Exfield_parser, Eyfield_parser, Ezfield_parser,
                                         Bxfield_parser, Byfield_parser, Bzfield_parser,
                                         gamma_boost, problo, dx, t, beta_boost);
#endif

                    if constexpr (boost_control == has_gamma_boost) { // Lorentz transform fields due to moving frame
                        if constexpr (exte_control == has_exte || extb_control == has_extb) {

                            // Lorentz transform z (from boosted to lab frame)
                            amrex::Real Ex_ext_boost, Ey_ext_boost, Ez_ext_boost;
                            amrex::Real Bx_ext_boost, By_ext_boost, Bz_ext_boost;
                            amrex::Real Ex_ext_lab, Ey_ext_lab, Ez_ext_lab;
                            amrex::Real Bx_ext_lab, By_ext_lab, Bz_ext_lab;

                            // Grab the location
#if defined(WARPX_DIM_3D)
                            const amrex::Real x = problo[0] + i * dx[0];
                            const amrex::Real y = problo[1] + j * dx[1];
                            const amrex::Real z = problo[2] + k * dx[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                            const amrex::Real x = problo[0] + i * dx[0];
                            const amrex::Real y = 0.0_rt;
                            const amrex::Real z = problo[1] + j * dx[1];
#else
                            const amrex::Real x = 0.0_rt;
                            const amrex::Real y = 0.0_rt;
                            const amrex::Real z = problo[0] + i * dx[0];
#endif

                            // Get the lab frame E and B
                            // Transform (boosted to lab)
                            const amrex::Real t_lab = gamma_boost*(t + beta_boost*z/PhysConst::c);
                            const amrex::Real z_lab = gamma_boost*(z + beta_boost*PhysConst::c*t);

                            // Grab the external fields in the lab frame:
                            if ( exte_control == has_exte ) {
                                Ex_ext_lab = Exfield_parser(x, y, z_lab, t_lab);
                                Ey_ext_lab = Eyfield_parser(x, y, z_lab, t_lab);
                                Ez_ext_lab = Ezfield_parser(x, y, z_lab, t_lab);
                            }else{
                                Ex_ext_lab = 0.0;
                                Ey_ext_lab = 0.0;
                                Ez_ext_lab = 0.0;
                            }
                            if ( extb_control == has_extb ) {
                                Bx_ext_lab = Bxfield_parser(x, y, z_lab, t_lab);
                                By_ext_lab = Byfield_parser(x, y, z_lab, t_lab);
                                Bz_ext_lab = Bzfield_parser(x, y, z_lab, t_lab);
                            }else{
                                Bx_ext_lab = 0.0;
                                By_ext_lab = 0.0;
                                Bz_ext_lab = 0.0;
                            }

                            // Transform E & B (lab to boosted frame)
                            // (Require both to for the lorentz transform)
                            // RHS m_parser
                            Ez_ext_boost = Ez_ext_lab;
                            Bz_ext_boost = Bz_ext_lab;
                            Ex_ext_boost = gamma_boost*(Ex_ext_lab - beta_boost*PhysConst::c*By_ext_lab);
                            Ey_ext_boost = gamma_boost*(Ey_ext_lab + beta_boost*PhysConst::c*Bx_ext_lab);
                            Bx_ext_boost = gamma_boost*(Bx_ext_lab + beta_boost*Ey_ext_lab/PhysConst::c);
                            By_ext_boost = gamma_boost*(By_ext_lab - beta_boost*Ex_ext_lab/PhysConst::c);

                            // Then add to Nodal quantities in the boosted frame:
                            Ex_Nodal += Ex_ext_boost;
                            Ey_Nodal += Ey_ext_boost;
                            Ez_Nodal += Ez_ext_boost;
                            Bx_Nodal += Bx_ext_boost;
                            By_Nodal += By_ext_boost;
                            Bz_Nodal += Bz_ext_boost;
                        }
                    } else {

                        // Added external e fields:
                        if constexpr ( exte_control == has_exte ){
#if defined(WARPX_DIM_3D)
                            const amrex::Real x = problo[0] + i * dx[0];
                            const amrex::Real y = problo[1] + j * dx[1];
                            const amrex::Real z = problo[2] + k * dx[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                            const amrex::Real x = problo[0] + i * dx[0];
                            const amrex::Real y = 0.0_rt;
                            const amrex::Real z = problo[1] + j * dx[1];
#else
                            const amrex::Real x = 0.0_rt;
                            const amrex::Real y = 0.0_rt;
                            const amrex::Real z = problo[0] + i * dx[0];
#endif

                            Ex_Nodal += Exfield_parser(x, y, z, t);
                            Ey_Nodal += Eyfield_parser(x, y, z, t);
                            Ez_Nodal += Ezfield_parser(x, y, z, t);
                        }

                        // Added external b fields:
                        if ( extb_control == has_extb ){
#if defined(WARPX_DIM_3D)
                            const amrex::Real x = problo[0] + i * dx[0];
                            const amrex::Real y = problo[1] + j * dx[1];
                            const amrex::Real z = problo[2] + k * dx[2];
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
                            const amrex::Real x = problo[0] + i * dx[0];
                            const amrex::Real y = 0.0_rt;
                            const amrex::Real z = problo[1] + j * dx[1];
#else
                            const amrex::Real x = 0.0_rt;
                            const amrex::Real y = 0.0_rt;
                            const amrex::Real z = problo[0] + i * dx[0];
#endif

                            Bx_Nodal += Bxfield_parser(x, y, z, t);
                            By_Nodal += Byfield_parser(x, y, z, t);
                            Bz_Nodal += Bzfield_parser(x, y, z, t);
                        }
                    }

                    // Isolate U from NU
                    amrex::Real tmp_Ux = (NUx_arr(i, j, k) / N_arr(i,j,k));
                    amrex::Real tmp_Uy = (NUy_arr(i, j, k) / N_arr(i,j,k));
                    amrex::Real tmp_Uz = (NUz_arr(i, j, k) / N_arr(i,j,k));

                    // Enforce RZ boundary conditions
#if defined(WARPX_DIM_RZ)
                    if  ( i == 0 ){
                        Ex_Nodal = 0.0;
                        Ey_Nodal = 0.0;
                        By_Nodal = 0.0;
                        Bx_Nodal = 0.0;
                    }
#endif

                    // Push the fluid momentum
                    UpdateMomentumHigueraCary(tmp_Ux, tmp_Uy, tmp_Uz,
                        Ex_Nodal, Ey_Nodal, Ez_Nodal,
                        Bx_Nodal, By_Nodal, Bz_Nodal, q, m, dt );

                    // Calculate NU
                    NUx_arr(i,j,k) = N_arr(i,j,k)*tmp_Ux;
                    NUy_arr(i,j,k) = N_arr(i,j,k)*tmp_Uy;
                    NUz_arr(i,j,k) = N_arr(i,j,k)*tmp_Uz;
                }
            }
        );
    }
}

void WarpXFluidContainer::DepositCharge (ablastr::fields::MultiFabRegister& fields, amrex::MultiFab &rho, int lev, int icomp)
{
    WARPX_PROFILE("WarpXFluidContainer::DepositCharge");

    WarpX &warpx = WarpX::GetInstance();
    const amrex::Geometry &geom = warpx.Geom(lev);
    const amrex::Periodicity &period = geom.periodicity();
    const amrex::Real q = getCharge();
    auto const &owner_mask_rho = amrex::OwnerMask(rho, period);

    // Assertion, make sure rho is at the same location as N
    AMREX_ALWAYS_ASSERT(rho.ixType().nodeCentered());

    // Loop over and deposit charge density
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*fields.get(name_mf_N, lev), TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {

        amrex::Box const &tile_box = mfi.tilebox(fields.get(name_mf_N, lev)->ixType().toIntVect());
        amrex::Array4<Real> const &N_arr = fields.get(name_mf_N, lev)->array(mfi);
        const amrex::Array4<amrex::Real> rho_arr = rho.array(mfi);
        const amrex::Array4<int> owner_mask_rho_arr = owner_mask_rho->array(mfi);

        // Deposit Rho
        amrex::ParallelFor(tile_box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                if ( owner_mask_rho_arr(i,j,k) ) { rho_arr(i,j,k,icomp) += q*N_arr(i,j,k); }
            }
        );
    }
}


void WarpXFluidContainer::DepositCurrent(
    ablastr::fields::MultiFabRegister& fields,
    amrex::MultiFab &jx, amrex::MultiFab &jy, amrex::MultiFab &jz,
    int lev)
{
    using ablastr::fields::Direction;
    WARPX_PROFILE("WarpXFluidContainer::DepositCurrent");

    // Temporary nodal currents
    amrex::MultiFab tmp_jx_fluid(fields.get(name_mf_N, lev)->boxArray(), fields.get(name_mf_N, lev)->DistributionMap(), 1, 0);
    amrex::MultiFab tmp_jy_fluid(fields.get(name_mf_N, lev)->boxArray(), fields.get(name_mf_N, lev)->DistributionMap(), 1, 0);
    amrex::MultiFab tmp_jz_fluid(fields.get(name_mf_N, lev)->boxArray(), fields.get(name_mf_N, lev)->DistributionMap(), 1, 0);

    const amrex::Real inv_clight_sq = 1.0_prt / PhysConst::c / PhysConst::c;
    const amrex::Real q = getCharge();

    // Prepare interpolation of current components to cell center
    auto j_nodal_type = amrex::GpuArray<int, 3>{0, 0, 0};
    auto jx_type = amrex::GpuArray<int, 3>{0, 0, 0};
    auto jy_type = amrex::GpuArray<int, 3>{0, 0, 0};
    auto jz_type = amrex::GpuArray<int, 3>{0, 0, 0};
    for (int i = 0; i < AMREX_SPACEDIM; ++i)
    {
        j_nodal_type[i] = tmp_jx_fluid.ixType()[i];
        jx_type[i] = jx.ixType()[i];
        jy_type[i] = jy.ixType()[i];
        jz_type[i] = jz.ixType()[i];
    }

    // We now need to create a mask to fix the double counting.
    WarpX &warpx = WarpX::GetInstance();
    const amrex::Geometry &geom = warpx.Geom(lev);
    const amrex::Periodicity &period = geom.periodicity();
    auto const &owner_mask_x = amrex::OwnerMask(jx, period);
    auto const &owner_mask_y = amrex::OwnerMask(jy, period);
    auto const &owner_mask_z = amrex::OwnerMask(jz, period);

    // Calculate j at the nodes
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*fields.get(name_mf_N, lev), TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Box const &tile_box = mfi.tilebox(fields.get(name_mf_N, lev)->ixType().toIntVect());

        amrex::Array4<Real> const &N_arr = fields.get(name_mf_N, lev)->array(mfi);
        amrex::Array4<Real> const &NUx_arr = fields.get(name_mf_NU, Direction{0}, lev)->array(mfi);
        amrex::Array4<Real> const &NUy_arr = fields.get(name_mf_NU, Direction{1}, lev)->array(mfi);
        amrex::Array4<Real> const &NUz_arr = fields.get(name_mf_NU, Direction{2}, lev)->array(mfi);

        const amrex::Array4<amrex::Real> tmp_jx_fluid_arr = tmp_jx_fluid.array(mfi);
        const amrex::Array4<amrex::Real> tmp_jy_fluid_arr = tmp_jy_fluid.array(mfi);
        const amrex::Array4<amrex::Real> tmp_jz_fluid_arr = tmp_jz_fluid.array(mfi);

        amrex::ParallelFor(tile_box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                // Calculate J from fluid quantities
                amrex::Real gamma = 1.0_rt, Ux = 0.0_rt, Uy = 0.0_rt, Uz = 0.0_rt;
                if (N_arr(i, j, k)>0.0_rt){
                    Ux = NUx_arr(i, j, k)/N_arr(i, j, k);
                    Uy = NUy_arr(i, j, k)/N_arr(i, j, k);
                    Uz = NUz_arr(i, j, k)/N_arr(i, j, k);
                    gamma = std::sqrt(1.0_rt + ( Ux*Ux + Uy*Uy + Uz*Uz) * inv_clight_sq ) ;
                }
                tmp_jx_fluid_arr(i, j, k) = q * (NUx_arr(i, j, k) / gamma);
                tmp_jy_fluid_arr(i, j, k) = q * (NUy_arr(i, j, k) / gamma);
                tmp_jz_fluid_arr(i, j, k) = q * (NUz_arr(i, j, k) / gamma);
            }
        );
    }

    // Interpolate j from the nodes to the simulation mesh (typically Yee mesh)
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*fields.get(name_mf_N, lev), TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::Box const &tile_box_x = mfi.tilebox(jx.ixType().toIntVect());
        amrex::Box const &tile_box_y = mfi.tilebox(jy.ixType().toIntVect());
        amrex::Box const &tile_box_z = mfi.tilebox(jz.ixType().toIntVect());

        const amrex::Array4<amrex::Real> jx_arr = jx.array(mfi);
        const amrex::Array4<amrex::Real> jy_arr = jy.array(mfi);
        const amrex::Array4<amrex::Real> jz_arr = jz.array(mfi);

        const amrex::Array4<amrex::Real> tmp_jx_fluid_arr = tmp_jx_fluid.array(mfi);
        const amrex::Array4<amrex::Real> tmp_jy_fluid_arr = tmp_jy_fluid.array(mfi);
        const amrex::Array4<amrex::Real> tmp_jz_fluid_arr = tmp_jz_fluid.array(mfi);

        const amrex::Array4<int> owner_mask_x_arr = owner_mask_x->array(mfi);
        const amrex::Array4<int> owner_mask_y_arr = owner_mask_y->array(mfi);
        const amrex::Array4<int> owner_mask_z_arr = owner_mask_z->array(mfi);

        // When using the `Interp` function, one needs to specify whether coarsening is desired.
        // Here, we do not perform any coarsening.
        const amrex::GpuArray<int, 3U> coarsening_ratio = {1, 1, 1};


        // Interpolate fluid current and deposit it
        // ( mask double counting )
        amrex::ParallelFor( tile_box_x, tile_box_y, tile_box_z,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const amrex::Real jx_tmp = ablastr::coarsen::sample::Interp(tmp_jx_fluid_arr,
                    j_nodal_type, jx_type, coarsening_ratio, i, j, k, 0);
                if ( owner_mask_x_arr(i,j,k) ) { jx_arr(i, j, k) += jx_tmp; }
            },
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const amrex::Real jy_tmp = ablastr::coarsen::sample::Interp(tmp_jy_fluid_arr,
                    j_nodal_type, jy_type, coarsening_ratio, i, j, k, 0);
                if ( owner_mask_y_arr(i,j,k) ) { jy_arr(i, j, k) += jy_tmp; }
            },
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                const amrex::Real jz_tmp = ablastr::coarsen::sample::Interp(tmp_jz_fluid_arr,
                    j_nodal_type, jz_type, coarsening_ratio, i, j, k, 0);
                if ( owner_mask_z_arr(i,j,k) ) { jz_arr(i, j, k) += jz_tmp; }
            }
        );
    }
}
