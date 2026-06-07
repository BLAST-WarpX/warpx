/* Copyright 2026 Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Simulation.H"

#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceAlgorithms/CartesianYeeAlgorithm.H"
#include "Particles/Deposition/CurrentDeposition.H"
#include "Particles/Gather/FieldGather.H"
#include "Particles/PIdx.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/UpdateMomentumBoris.H"
#include "Particles/Pusher/UpdatePosition.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXDim.H"
#include "Utils/WarpXDimDispatch.H"
#include "Utils/WarpXDimIndexing.H"

#include <ablastr/coarsen/sample.H>

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using namespace amrex::literals;

namespace
{
    /** Array slots of the axes present on the grid, in inputs order
     *
     * Inputs like amr.n_cell and geometry.prob_lo list one value per grid
     * axis; this maps each of them to its (unified, degenerate-3D) array slot.
     */
    amrex::Vector<int>
    ActiveSlots (warpx::Dim const dim)
    {
        switch (dim)
        {
            case warpx::Dim::Z:   return {2};
            case warpx::Dim::XZ:  return {0, 2};
            case warpx::Dim::XYZ: return {0, 1, 2};
            default:
                amrex::Abort("RuntimeDims prototype: unsupported geometry.dims = "
                             + warpx::geometry_dims_name(dim));
        }
        return {};
    }

    /** Names of the logical axes, by array slot */
    constexpr std::array<char const*, 3> axis_names {"x", "y", "z"};
}

Simulation::Simulation (warpx::Dim const dim)
    : m_dim(dim)
{
}

void
Simulation::ReadParameters ()
{
    const amrex::ParmParse pp;
    utils::parser::getWithParser(pp, "max_step", m_max_step);

    const amrex::ParmParse pp_warpx("warpx");
    utils::parser::queryWithParser(pp_warpx, "cfl", m_cfl);

    const amrex::ParmParse pp_algo("algo");
    utils::parser::queryWithParser(pp_algo, "particle_shape", m_nox);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(1 <= m_nox && m_nox <= 4,
        "algo.particle_shape must be between 1 and 4");

    std::string current_deposition = "direct";
    pp_algo.query("current_deposition", current_deposition);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(current_deposition == "direct",
        "RuntimeDims prototype: only algo.current_deposition = direct is supported "
        "(add it on the command line to override the inputs file)");

    std::string field_gathering = "energy-conserving";
    pp_algo.query("field_gathering", field_gathering);
    m_galerkin_interpolation = (field_gathering == "energy-conserving");
    // Use same shape factors in all directions with direct current deposition
    // and the electromagnetic solver, cf. WarpX::ReadParameters
    if (current_deposition == "direct") {
        m_galerkin_interpolation = false;
    }

    // simplified diagnostics: only a plotfile interval and prefix
    const amrex::ParmParse pp_diag("diag1");
    m_plot_int = m_max_step;
    utils::parser::queryWithParser(pp_diag, "intervals", m_plot_int);
}

amrex::IntVect
Simulation::UnifiedStagger (amrex::IntVect logical_flags) const
{
    auto const slots = ActiveSlots(m_dim);
    amrex::IntVect flags(0);
    for (int const s : slots) {
        flags[s] = logical_flags[s];
    }
    return flags;
}

void
Simulation::MakeGeometry ()
{
    auto const slots = ActiveSlots(m_dim);
    auto const ngrid = static_cast<int>(slots.size());

    const amrex::ParmParse pp_amr("amr");
    std::vector<int> n_cell_in;
    utils::parser::getArrWithParser(pp_amr, "n_cell", n_cell_in, 0, ngrid);

    std::vector<int> max_grid_size_in(static_cast<std::size_t>(ngrid), 32768);
    utils::parser::queryArrWithParser(pp_amr, "max_grid_size", max_grid_size_in, 0, ngrid);

    int max_level = 0;
    utils::parser::queryWithParser(pp_amr, "max_level", max_level);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_level == 0,
        "RuntimeDims prototype: mesh refinement is not supported");

    const amrex::ParmParse pp_geometry("geometry");
    std::vector<amrex::Real> prob_lo_in, prob_hi_in;
    utils::parser::getArrWithParser(pp_geometry, "prob_lo", prob_lo_in, 0, ngrid);
    utils::parser::getArrWithParser(pp_geometry, "prob_hi", prob_hi_in, 0, ngrid);

    // boundary conditions: only fully periodic domains are supported
    const amrex::ParmParse pp_boundary("boundary");
    std::vector<std::string> field_lo(static_cast<std::size_t>(ngrid), "periodic");
    std::vector<std::string> field_hi(static_cast<std::size_t>(ngrid), "periodic");
    pp_boundary.queryarr("field_lo", field_lo, 0, ngrid);
    pp_boundary.queryarr("field_hi", field_hi, 0, ngrid);
    for (int g = 0; g < ngrid; ++g) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            field_lo[g] == "periodic" && field_hi[g] == "periodic",
            "RuntimeDims prototype: only periodic boundary conditions are supported");
    }

    // degenerate-3D domain: collapsed dimensions have one cell and
    // prob_lo/hi = [-0.5, 0.5), so that their cell size is exactly 1
    amrex::IntVect n_cell(1);
    amrex::IntVect max_grid_size(1);
    amrex::RealBox rb({AMREX_D_DECL(-0.5_rt, -0.5_rt, -0.5_rt)},
                      {AMREX_D_DECL(0.5_rt, 0.5_rt, 0.5_rt)});
    for (int g = 0; g < ngrid; ++g) {
        int const s = slots[g];
        n_cell[s] = n_cell_in[g];
        max_grid_size[s] = max_grid_size_in[g];
        rb.setLo(s, prob_lo_in[g]);
        rb.setHi(s, prob_hi_in[g]);
    }

    const amrex::Box domain(amrex::IntVect(0), n_cell - amrex::IntVect(1));
    const amrex::Array<int, 3> is_periodic {1, 1, 1};
    m_geom.define(domain, rb, amrex::CoordSys::cartesian, is_periodic);

    m_ba = amrex::BoxArray(domain);
    m_ba.maxSize(max_grid_size);
    m_dm = amrex::DistributionMapping(m_ba);

    amrex::Print() << "  domain: " << domain << "\n"
                   << "  boxes: " << m_ba.size() << "\n";
}

void
Simulation::AllocateFields ()
{
    // guard cells along the simulated axes only
    amrex::IntVect ng(0);
    for (int const s : ActiveSlots(m_dim)) {
        ng[s] = 2;
    }

    for (int c = 0; c < 3; ++c) {
        // Yee staggering (logical): E and j are cell-centered along their own
        // axis and nodal along the others; B is the opposite
        amrex::IntVect e_flags(1), b_flags(0);
        e_flags[c] = 0;
        b_flags[c] = 1;

        const auto e_stag = UnifiedStagger(e_flags);
        const auto b_stag = UnifiedStagger(b_flags);

        m_E[c] = std::make_unique<amrex::MultiFab>(amrex::convert(m_ba, e_stag), m_dm, 1, ng);
        m_B[c] = std::make_unique<amrex::MultiFab>(amrex::convert(m_ba, b_stag), m_dm, 1, ng);
        m_j[c] = std::make_unique<amrex::MultiFab>(amrex::convert(m_ba, e_stag), m_dm, 1, ng);

        m_E[c]->setVal(0.0_rt);
        m_B[c]->setVal(0.0_rt);
        m_j[c]->setVal(0.0_rt);
    }
}

void
Simulation::InitData ()
{
    BL_PROFILE("Simulation::InitData()");

    ReadParameters();
    MakeGeometry();
    AllocateFields();

    // stencil coefficients and timestep
    std::array<amrex::Real, 3> cell_size {m_geom.CellSize(0), m_geom.CellSize(1), m_geom.CellSize(2)};
    amrex::Vector<amrex::Real> coefs_x, coefs_y, coefs_z;
    amrex::Real dt_max = 0.0_rt;
    warpx::dim_dispatch(m_dim, [&] (auto dim_const) {
        constexpr warpx::Dim D = dim_const();
        CartesianYeeAlgorithmND<D>::InitializeStencilCoefficients(cell_size, coefs_x, coefs_y, coefs_z);
        dt_max = CartesianYeeAlgorithmND<D>::ComputeMaxDt(m_geom.CellSize());
    });
    m_dt = m_cfl * dt_max;

    m_stencil_coefs_x.resize(coefs_x.size());
    m_stencil_coefs_y.resize(coefs_y.size());
    m_stencil_coefs_z.resize(coefs_z.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, coefs_x.begin(), coefs_x.end(), m_stencil_coefs_x.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, coefs_y.begin(), coefs_y.end(), m_stencil_coefs_y.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, coefs_z.begin(), coefs_z.end(), m_stencil_coefs_z.begin());
    amrex::Gpu::synchronize();

    amrex::Print() << "  dt = " << m_dt << " s (cfl = " << m_cfl << ")\n";

    InitParticles();
}

void
Simulation::InitParticles ()
{
    BL_PROFILE("Simulation::InitParticles()");

    auto const slots = ActiveSlots(m_dim);
    auto const ngrid = static_cast<int>(slots.size());

    const amrex::ParmParse pp_particles("particles");
    std::vector<std::string> species_names;
    pp_particles.queryarr("species_names", species_names);

    for (auto const& name : species_names)
    {
        Species sp;
        sp.name = name;
        const amrex::ParmParse pp_sp(name);

        utils::parser::getWithParser(pp_sp, "charge", sp.charge);
        utils::parser::getWithParser(pp_sp, "mass", sp.mass);

        std::string injection_style;
        pp_sp.get("injection_style", injection_style);
        std::transform(injection_style.begin(), injection_style.end(),
                       injection_style.begin(), ::tolower);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(injection_style == "nuniformpercell",
            "RuntimeDims prototype: only injection_style = NUniformPerCell is supported");

        std::string profile;
        pp_sp.get("profile", profile);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(profile == "constant",
            "RuntimeDims prototype: only profile = constant is supported");
        utils::parser::getWithParser(pp_sp, "density", sp.density);

        std::vector<int> ppc_in;
        utils::parser::getArrWithParser(pp_sp, "num_particles_per_cell_each_dim", ppc_in, 0, ngrid);
        for (int g = 0; g < ngrid; ++g) {
            sp.ppc[slots[g]] = ppc_in[g];
        }

        std::string momentum_distribution;
        pp_sp.get("momentum_distribution_type", momentum_distribution);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(momentum_distribution == "parse_momentum_function",
            "RuntimeDims prototype: only parse_momentum_function is supported");
        std::string ux_str, uy_str, uz_str;
        utils::parser::Store_parserString(pp_sp, "momentum_function_ux(x,y,z)", ux_str);
        utils::parser::Store_parserString(pp_sp, "momentum_function_uy(x,y,z)", uy_str);
        utils::parser::Store_parserString(pp_sp, "momentum_function_uz(x,y,z)", uz_str);
        sp.momentum_ux = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(ux_str, {"x", "y", "z"}));
        sp.momentum_uy = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(uy_str, {"x", "y", "z"}));
        sp.momentum_uz = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(uz_str, {"x", "y", "z"}));

        // injection bounds (default: unbounded)
        for (int s = 0; s < 3; ++s) {
            sp.inject_box.setLo(s, std::numeric_limits<amrex::Real>::lowest());
            sp.inject_box.setHi(s, std::numeric_limits<amrex::Real>::max());
        }
        for (int s = 0; s < 3; ++s) {
            amrex::Real bound;
            if (utils::parser::queryWithParser(pp_sp, (std::string(axis_names[s]) + "min").c_str(), bound)) {
                sp.inject_box.setLo(s, bound);
            }
            if (utils::parser::queryWithParser(pp_sp, (std::string(axis_names[s]) + "max").c_str(), bound)) {
                sp.inject_box.setHi(s, bound);
            }
        }

        sp.pc = std::make_unique<UnifiedParticleContainer>(m_geom, m_dm, m_ba);

        // uniform injection, following the conventions of
        // PhysicalParticleContainer::AddPlasma and InjectorPositionRegular
        int num_ppc = 1;
        amrex::Dim3 ppc_grid {1, 1, 1}; // per-cell counts in grid-axis (inputs) order
        for (int g = 0; g < ngrid; ++g) {
            num_ppc *= ppc_in[g];
            if (g == 0) { ppc_grid.x = ppc_in[g]; }
            if (g == 1) { ppc_grid.y = ppc_in[g]; }
            if (g == 2) { ppc_grid.z = ppc_in[g]; }
        }
        const amrex::Real scale_fac =
            m_geom.CellSize(0)*m_geom.CellSize(1)*m_geom.CellSize(2)/num_ppc;
        const amrex::Real weight = sp.density*scale_fac;

        auto const ux_exe = sp.momentum_ux->compile<3>();
        auto const uy_exe = sp.momentum_uy->compile<3>();
        auto const uz_exe = sp.momentum_uz->compile<3>();

        auto const problo = m_geom.ProbLoArray();
        auto const dx = m_geom.CellSizeArray();
        const int cpuid = amrex::ParallelDescriptor::MyProc();
        amrex::Long next_id = 1;

        for (amrex::MFIter mfi(*m_E[0], false); mfi.isValid(); ++mfi)
        {
            // cell-centered box of this grid (the validbox of m_E[0] is
            // staggered and would double-count cells at box boundaries)
            const amrex::Box box = m_ba[mfi.index()];
            auto& ptile = sp.pc->DefineAndReturnParticleTile(0, mfi.index(), mfi.LocalTileIndex());

            // host-side staging of the new particles, in cell-then-particle order
            std::array<std::vector<amrex::ParticleReal>, PIdx::nattribs> attribs;
            std::vector<uint64_t> idcpu;

            for (amrex::IntVect iv = box.smallEnd(); iv <= box.bigEnd(); box.next(iv))
            {
                for (int i_part = 0; i_part < num_ppc; ++i_part)
                {
                    // particle position within the unit cell, in grid-axis order,
                    // cf. InjectorPositionRegular::getPositionUnitBox
                    int const nx = ppc_grid.x;
                    int const ny = ppc_grid.y;
                    int const nz = ppc_grid.z;
                    int const ix_part = i_part / (ny*nz);
                    int const iz_part = (i_part - ix_part*(ny*nz)) / ny;
                    int const iy_part = (i_part - ix_part*(ny*nz)) - ny*iz_part;
                    const amrex::Real r_grid[3] = {
                        (0.5_rt + ix_part) / nx,
                        (0.5_rt + iy_part) / ny,
                        (0.5_rt + iz_part) / nz
                    };

                    // particle position, cf. getCellCoords (AddParticles.cpp):
                    // collapsed dimensions sit at the cell center, 0
                    amrex::Real pos[3] = {0.0_rt, 0.0_rt, 0.0_rt};
                    for (int g = 0; g < ngrid; ++g) {
                        int const s = slots[g];
                        pos[s] = problo[s] + (iv[s] + r_grid[g])*dx[s];
                    }

                    if (!sp.inject_box.contains(amrex::XDim3{pos[0], pos[1], pos[2]})) {
                        continue;
                    }

                    // momentum from the parsed functions, in units of c
                    const amrex::Real ux = ux_exe(pos[0], pos[1], pos[2])*PhysConst::c;
                    const amrex::Real uy = uy_exe(pos[0], pos[1], pos[2])*PhysConst::c;
                    const amrex::Real uz = uz_exe(pos[0], pos[1], pos[2])*PhysConst::c;

                    attribs[PIdx::x].push_back(static_cast<amrex::ParticleReal>(pos[0]));
                    attribs[PIdx::y].push_back(static_cast<amrex::ParticleReal>(pos[1]));
                    attribs[PIdx::z].push_back(static_cast<amrex::ParticleReal>(pos[2]));
                    attribs[PIdx::w].push_back(static_cast<amrex::ParticleReal>(weight));
                    attribs[PIdx::ux].push_back(static_cast<amrex::ParticleReal>(ux));
                    attribs[PIdx::uy].push_back(static_cast<amrex::ParticleReal>(uy));
                    attribs[PIdx::uz].push_back(static_cast<amrex::ParticleReal>(uz));
                    idcpu.push_back(amrex::SetParticleIDandCPU(next_id++, cpuid));
                }
            }

            auto const np = static_cast<amrex::Long>(idcpu.size());
            auto const old_np = ptile.numParticles();
            ptile.resize(old_np + np);
            auto& soa = ptile.GetStructOfArrays();
            for (int comp = 0; comp < PIdx::nattribs; ++comp) {
                amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                    attribs[comp].begin(), attribs[comp].end(),
                    soa.GetRealData(comp).begin() + old_np);
            }
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                idcpu.begin(), idcpu.end(), soa.GetIdCPUData().begin() + old_np);
            amrex::Gpu::synchronize();
        }

        amrex::Print() << "  species " << name << ": "
                       << sp.pc->TotalNumberOfParticles() << " particles\n";

        m_species.push_back(std::move(sp));
    }
}

/** Gather fields, push momenta and positions, and deposit the current of one
 *  species (cf. PhysicalParticleContainer::PushPX and DepositCurrent) */
template <warpx::Dim D>
void
PushAndDepositSpecies (Species& sp,
                       std::array<std::unique_ptr<amrex::MultiFab>, 3> const& E,
                       std::array<std::unique_ptr<amrex::MultiFab>, 3> const& B,
                       std::array<std::unique_ptr<amrex::MultiFab>, 3> const& j,
                       amrex::Geometry const& geom,
                       amrex::Real const dt, int const nox,
                       bool const galerkin_interpolation)
{
    const amrex::ParticleReal q = sp.charge;
    const amrex::ParticleReal m = sp.mass;

    auto const problo = geom.ProbLoArray();
    auto const dx_arr = geom.CellSizeArray();
    const amrex::XDim3 dinv {1.0_rt/dx_arr[0], 1.0_rt/dx_arr[1], 1.0_rt/dx_arr[2]};
    const amrex::Real invvol = dinv.x*dinv.y*dinv.z;

    for (UnifiedParIter pti(*sp.pc, 0); pti.isValid(); ++pti)
    {
        auto const np = pti.numParticles();
        auto& soa = pti.GetStructOfArrays();
        amrex::ParticleReal* const AMREX_RESTRICT wp = soa.GetRealData(PIdx::w).dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT uxp = soa.GetRealData(PIdx::ux).dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT uyp = soa.GetRealData(PIdx::uy).dataPtr();
        amrex::ParticleReal* const AMREX_RESTRICT uzp = soa.GetRealData(PIdx::uz).dataPtr();

        const auto GetPosition = GetParticlePosition<PIdx, D>(pti);
        const auto SetPosition = SetParticlePosition<PIdx, D>(pti);

        // --- field gather + momentum push + position push

        // box covering the gather region (valid + guard cells)
        amrex::Box gather_box = pti.validbox();
        gather_box.grow(E[0]->nGrowVect());
        const amrex::Dim3 glo = amrex::lbound(gather_box);
        const amrex::XDim3 gxyzmin {
            problo[0] + gather_box.smallEnd(0)*dx_arr[0],
            problo[1] + gather_box.smallEnd(1)*dx_arr[1],
            problo[2] + gather_box.smallEnd(2)*dx_arr[2]};

        amrex::Array4<amrex::Real const> const& ex_arr = (*E[0])[pti].const_array();
        amrex::Array4<amrex::Real const> const& ey_arr = (*E[1])[pti].const_array();
        amrex::Array4<amrex::Real const> const& ez_arr = (*E[2])[pti].const_array();
        amrex::Array4<amrex::Real const> const& bx_arr = (*B[0])[pti].const_array();
        amrex::Array4<amrex::Real const> const& by_arr = (*B[1])[pti].const_array();
        amrex::Array4<amrex::Real const> const& bz_arr = (*B[2])[pti].const_array();
        const amrex::IndexType ex_type = (*E[0])[pti].box().ixType();
        const amrex::IndexType ey_type = (*E[1])[pti].box().ixType();
        const amrex::IndexType ez_type = (*E[2])[pti].box().ixType();
        const amrex::IndexType bx_type = (*B[0])[pti].box().ixType();
        const amrex::IndexType by_type = (*B[1])[pti].box().ixType();
        const amrex::IndexType bz_type = (*B[2])[pti].box().ixType();

        {
        BL_PROFILE("PushAndDeposit::GatherAndPush");
        amrex::ParallelFor(np,
            [=] AMREX_GPU_DEVICE (long ip) {
                amrex::ParticleReal xp, yp, zp;
                GetPosition(ip, xp, yp, zp);

                amrex::ParticleReal Exp = 0._prt, Eyp = 0._prt, Ezp = 0._prt;
                amrex::ParticleReal Bxp = 0._prt, Byp = 0._prt, Bzp = 0._prt;

                doGatherShapeN<D>(xp, yp, zp, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                  ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                                  ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                                  dinv, gxyzmin, glo, 1, nox, galerkin_interpolation);

                UpdateMomentumBoris(uxp[ip], uyp[ip], uzp[ip],
                                    Exp, Eyp, Ezp, Bxp, Byp, Bzp, q, m, dt);

                UpdatePosition<D>(xp, yp, zp, uxp[ip], uyp[ip], uzp[ip], dt, m);
                SetPosition(ip, xp, yp, zp);
            });
        }

        // --- direct current deposition at the half step (relative_time = -dt/2)

        amrex::Box depos_box = pti.validbox();
        depos_box.grow(j[0]->nGrowVect());
        const amrex::Dim3 dlo = amrex::lbound(depos_box);
        const amrex::XDim3 dxyzmin {
            problo[0] + depos_box.smallEnd(0)*dx_arr[0],
            problo[1] + depos_box.smallEnd(1)*dx_arr[1],
            problo[2] + depos_box.smallEnd(2)*dx_arr[2]};

        // Deposit into zeroed local arrays and accumulate them into j
        // afterwards, like the CPU path of WarpXParticleContainer::
        // DepositCurrent (this also keeps the per-cell floating-point
        // summation order identical for the bitwise comparison runs)
        amrex::Box tbx = amrex::convert(pti.validbox(), j[0]->ixType().toIntVect());
        amrex::Box tby = amrex::convert(pti.validbox(), j[1]->ixType().toIntVect());
        amrex::Box tbz = amrex::convert(pti.validbox(), j[2]->ixType().toIntVect());
        tbx.grow(j[0]->nGrowVect());
        tby.grow(j[1]->nGrowVect());
        tbz.grow(j[2]->nGrowVect());
        amrex::FArrayBox jx_fab(tbx, 1);
        amrex::FArrayBox jy_fab(tby, 1);
        amrex::FArrayBox jz_fab(tbz, 1);
        jx_fab.setVal<amrex::RunOn::Device>(0.0_rt);
        jy_fab.setVal<amrex::RunOn::Device>(0.0_rt);
        jz_fab.setVal<amrex::RunOn::Device>(0.0_rt);

        {
        BL_PROFILE("PushAndDeposit::CurrentDeposition");
        if        (nox == 1) {
            doDepositionShapeN<1, D>(GetPosition, wp, uxp, uyp, uzp, nullptr,
                                     jx_fab, jy_fab, jz_fab, np, -0.5_rt*dt, dinv,
                                     dxyzmin, dlo, q, 1);
        } else if (nox == 2) {
            doDepositionShapeN<2, D>(GetPosition, wp, uxp, uyp, uzp, nullptr,
                                     jx_fab, jy_fab, jz_fab, np, -0.5_rt*dt, dinv,
                                     dxyzmin, dlo, q, 1);
        } else if (nox == 3) {
            doDepositionShapeN<3, D>(GetPosition, wp, uxp, uyp, uzp, nullptr,
                                     jx_fab, jy_fab, jz_fab, np, -0.5_rt*dt, dinv,
                                     dxyzmin, dlo, q, 1);
        } else if (nox == 4) {
            doDepositionShapeN<4, D>(GetPosition, wp, uxp, uyp, uzp, nullptr,
                                     jx_fab, jy_fab, jz_fab, np, -0.5_rt*dt, dinv,
                                     dxyzmin, dlo, q, 1);
        }

        (*j[0])[pti].lockAdd(jx_fab, tbx, tbx, 0, 0, 1);
        (*j[1])[pti].lockAdd(jy_fab, tby, tby, 0, 0, 1);
        (*j[2])[pti].lockAdd(jz_fab, tbz, tbz, 0, 0, 1);
        }
    }
}

void
Simulation::PushAndDeposit ()
{
    BL_PROFILE("Simulation::PushAndDeposit()");

    for (int c = 0; c < 3; ++c) {
        m_j[c]->setVal(0.0_rt);
    }

    for (auto& sp : m_species) {
        warpx::dim_dispatch(m_dim, [&] (auto dim_const) {
            constexpr warpx::Dim D = dim_const();
            PushAndDepositSpecies<D>(sp, m_E, m_B, m_j, m_geom, m_dt, m_nox,
                                     m_galerkin_interpolation);
        });
    }
}

/** Faraday's law, cf. FiniteDifferenceSolver::EvolveBCartesian */
template <warpx::Dim D>
void
EvolveBImpl (std::array<std::unique_ptr<amrex::MultiFab>, 3>& B,
             std::array<std::unique_ptr<amrex::MultiFab>, 3> const& E,
             amrex::Real const* coefs_x, amrex::Real const* coefs_y,
             amrex::Real const* coefs_z, amrex::Real const dt)
{
    using T_Algo = CartesianYeeAlgorithmND<D>;
    using namespace amrex;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*B[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Array4<Real> const& Bx = B[0]->array(mfi);
        Array4<Real> const& By = B[1]->array(mfi);
        Array4<Real> const& Bz = B[2]->array(mfi);
        Array4<Real const> const& Ex = E[0]->const_array(mfi);
        Array4<Real const> const& Ey = E[1]->const_array(mfi);
        Array4<Real const> const& Ez = E[2]->const_array(mfi);

        Box const& tbx = mfi.tilebox(B[0]->ixType().toIntVect());
        Box const& tby = mfi.tilebox(B[1]->ixType().toIntVect());
        Box const& tbz = mfi.tilebox(B[2]->ixType().toIntVect());

        amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Bx(i, j, k) += dt * T_Algo::UpwardDz(Ey, coefs_z, 1, i, j, k)
                             - dt * T_Algo::UpwardDy(Ez, coefs_y, 1, i, j, k);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                By(i, j, k) += dt * T_Algo::UpwardDx(Ez, coefs_x, 1, i, j, k)
                             - dt * T_Algo::UpwardDz(Ex, coefs_z, 1, i, j, k);
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k){
                Bz(i, j, k) += dt * T_Algo::UpwardDy(Ex, coefs_y, 1, i, j, k)
                             - dt * T_Algo::UpwardDx(Ey, coefs_x, 1, i, j, k);
            }
        );
    }
}

/** Ampere's law, cf. FiniteDifferenceSolver::EvolveECartesian */
template <warpx::Dim D>
void
EvolveEImpl (std::array<std::unique_ptr<amrex::MultiFab>, 3>& E,
             std::array<std::unique_ptr<amrex::MultiFab>, 3> const& B,
             std::array<std::unique_ptr<amrex::MultiFab>, 3> const& j,
             amrex::Real const* coefs_x, amrex::Real const* coefs_y,
             amrex::Real const* coefs_z, amrex::Real const dt)
{
    using T_Algo = CartesianYeeAlgorithmND<D>;
    using namespace amrex;

    Real constexpr c2 = PhysConst::c2;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*E[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Array4<Real> const& Ex = E[0]->array(mfi);
        Array4<Real> const& Ey = E[1]->array(mfi);
        Array4<Real> const& Ez = E[2]->array(mfi);
        Array4<Real const> const& Bx = B[0]->const_array(mfi);
        Array4<Real const> const& By = B[1]->const_array(mfi);
        Array4<Real const> const& Bz = B[2]->const_array(mfi);
        Array4<Real const> const& jx = j[0]->const_array(mfi);
        Array4<Real const> const& jy = j[1]->const_array(mfi);
        Array4<Real const> const& jz = j[2]->const_array(mfi);

        Box const& tex = mfi.tilebox(E[0]->ixType().toIntVect());
        Box const& tey = mfi.tilebox(E[1]->ixType().toIntVect());
        Box const& tez = mfi.tilebox(E[2]->ixType().toIntVect());

        amrex::ParallelFor(tex, tey, tez,
            [=] AMREX_GPU_DEVICE (int i, int j_, int k){
                Ex(i, j_, k) += c2 * dt * (
                    - T_Algo::DownwardDz(By, coefs_z, 1, i, j_, k)
                    + T_Algo::DownwardDy(Bz, coefs_y, 1, i, j_, k)
                    - PhysConst::mu0 * jx(i, j_, k) );
            },
            [=] AMREX_GPU_DEVICE (int i, int j_, int k){
                Ey(i, j_, k) += c2 * dt * (
                    - T_Algo::DownwardDx(Bz, coefs_x, 1, i, j_, k)
                    + T_Algo::DownwardDz(Bx, coefs_z, 1, i, j_, k)
                    - PhysConst::mu0 * jy(i, j_, k) );
            },
            [=] AMREX_GPU_DEVICE (int i, int j_, int k){
                Ez(i, j_, k) += c2 * dt * (
                    - T_Algo::DownwardDy(Bx, coefs_y, 1, i, j_, k)
                    + T_Algo::DownwardDx(By, coefs_x, 1, i, j_, k)
                    - PhysConst::mu0 * jz(i, j_, k) );
            }
        );
    }
}

void
Simulation::EvolveB (amrex::Real const a_dt)
{
    BL_PROFILE("Simulation::EvolveB()");
    warpx::dim_dispatch(m_dim, [&] (auto dim_const) {
        constexpr warpx::Dim D = dim_const();
        EvolveBImpl<D>(m_B, m_E, m_stencil_coefs_x.data(), m_stencil_coefs_y.data(),
                       m_stencil_coefs_z.data(), a_dt);
    });
}

void
Simulation::EvolveE (amrex::Real const a_dt)
{
    BL_PROFILE("Simulation::EvolveE()");
    warpx::dim_dispatch(m_dim, [&] (auto dim_const) {
        constexpr warpx::Dim D = dim_const();
        EvolveEImpl<D>(m_E, m_B, m_j, m_stencil_coefs_x.data(), m_stencil_coefs_y.data(),
                       m_stencil_coefs_z.data(), a_dt);
    });
}

void
Simulation::SumBoundaryJ ()
{
    BL_PROFILE("Simulation::SumBoundaryJ()");
    for (int c = 0; c < 3; ++c) {
        m_j[c]->SumBoundary(m_geom.periodicity());
    }
}

void
Simulation::FillBoundaryE ()
{
    BL_PROFILE("Simulation::FillBoundaryE()");
    for (int c = 0; c < 3; ++c) {
        m_E[c]->FillBoundary(m_geom.periodicity());
    }
}

void
Simulation::FillBoundaryB ()
{
    BL_PROFILE("Simulation::FillBoundaryB()");
    for (int c = 0; c < 3; ++c) {
        m_B[c]->FillBoundary(m_geom.periodicity());
    }
}

void
Simulation::RunDiagnostics (int const step, amrex::Real const time)
{
    BL_PROFILE("Simulation::RunDiagnostics()");

    // reduced diagnostics: maximum norm of each field component
    const std::array<std::string, 9> names
        {"Ex", "Ey", "Ez", "Bx", "By", "Bz", "jx", "jy", "jz"};
    std::array<amrex::Real, 9> max_abs {};
    for (int c = 0; c < 3; ++c) {
        max_abs[c] = m_E[c]->norm0();
        max_abs[3 + c] = m_B[c]->norm0();
        max_abs[6 + c] = m_j[c]->norm0();
    }
    if (amrex::ParallelDescriptor::IOProcessor()) {
        static bool first_call = true;
        std::ofstream ofs("diags/reduced.csv", first_call ? std::ios::trunc : std::ios::app);
        if (first_call) {
            ofs << "step,time";
            for (auto const& n : names) { ofs << ",max_" << n; }
            ofs << "\n";
            first_call = false;
        }
        ofs << step << "," << time;
        ofs.precision(17);
        for (auto const v : max_abs) { ofs << "," << v; }
        ofs << "\n";
    }

    // plotfile with cell-centered fields, averaged exactly like the
    // plotfile diagnostics of WarpX (ablastr::coarsen::sample)
    const bool plot = (m_plot_int > 0) &&
        ((step % m_plot_int == 0) || (step == m_max_step));
    if (plot) {
        amrex::MultiFab mf(m_ba, m_dm, 9, 0);
        const std::array<amrex::MultiFab const*, 9> fields {
            m_E[0].get(), m_E[1].get(), m_E[2].get(),
            m_B[0].get(), m_B[1].get(), m_B[2].get(),
            m_j[0].get(), m_j[1].get(), m_j[2].get()};
        for (int c = 0; c < 9; ++c) {
            const amrex::IntVect stag = fields[c]->ixType().toIntVect();
            const amrex::GpuArray<int, 3> sf {stag[0], stag[1], stag[2]};
            const amrex::GpuArray<int, 3> sc {0, 0, 0};
            const amrex::GpuArray<int, 3> cr {1, 1, 1};
            for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box bx = mfi.growntilebox();
                amrex::Array4<amrex::Real> const& dst = mf.array(mfi);
                amrex::Array4<amrex::Real const> const& src = fields[c]->const_array(mfi);
                amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        dst(i, j, k, c) = ablastr::coarsen::sample::Interp(src, sf, sc, cr, i, j, k, 0);
                    });
            }
        }
        const std::string plotname = amrex::Concatenate(m_plotfile_prefix, step, 5);
        amrex::Print() << "  writing plotfile " << plotname << "\n";
        amrex::WriteSingleLevelPlotfile(plotname, mf,
            {names.begin(), names.end()}, m_geom, time, step);
        for (auto& sp : m_species) {
            // full-precision CSV of the particle data, for debugging and
            // bitwise comparisons against native runs (serial only)
            if (amrex::ParallelDescriptor::NProcs() == 1) {
                std::ofstream ofs(plotname + "/" + sp.name + "_particles.csv");
                ofs.precision(17);
                ofs << "x,y,z,w,ux,uy,uz\n";
                for (UnifiedParIter pti(*sp.pc, 0); pti.isValid(); ++pti) {
                    auto& soa = pti.GetStructOfArrays();
                    auto const np = pti.numParticles();
                    for (int i = 0; i < np; ++i) {
                        for (int comp = 0; comp < PIdx::nattribs; ++comp) {
                            ofs << soa.GetRealData(comp)[i]
                                << (comp + 1 < PIdx::nattribs ? "," : "\n");
                        }
                    }
                }
            }
        }
    }
}

void
Simulation::Evolve ()
{
    BL_PROFILE("Simulation::Evolve()");

    amrex::UtilCreateDirectoryDestructive("diags", false);

    amrex::Real cur_time = 0.0_rt;
    RunDiagnostics(0, cur_time);

    for (int step = 1; step <= m_max_step; ++step)
    {
        // one PIC cycle, cf. WarpX::OneStep_nosub
        PushAndDeposit();
        SumBoundaryJ();

        EvolveB(0.5_rt*m_dt);
        FillBoundaryB();
        EvolveE(m_dt);
        FillBoundaryE();
        EvolveB(0.5_rt*m_dt);
        FillBoundaryB();

        for (auto& sp : m_species) {
            sp.pc->Redistribute();
        }

        cur_time += m_dt;
        if (step % 10 == 0 || step == m_max_step) {
            amrex::Print() << "step " << step << " of " << m_max_step
                           << ", t = " << cur_time << " s\n";
        }
        RunDiagnostics(step, cur_time);
    }
}
