/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"

#include "BoundaryConditions/PML.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Initialization/ExternalField.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <AMReX_Utility.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>

#ifdef WARPX_DIM_XZ
using namespace amrex::literals;
using ablastr::fields::Direction;
using warpx::fields::FieldType;

namespace
{
    constexpr char checkpoint_marker[] = "/CompositeXZSlab";

    std::string CheckpointIdentity (WarpX const& warpx)
    {
        std::ostringstream stream;
        stream << "CompositeXZSlab-V1 " << warpx.refRatio(0) << ' '
               << warpx.boxArray(1).minimalBox() << ' ' << WarpX::nox;
        // Preserve existing unfiltered checkpoints. A change in the effective
        // filter would change the particle charge represented by the saved field.
        if (WarpX::use_filter && WarpX::filter_npass_each_dir != amrex::IntVect(0)) {
            stream << " filter " << WarpX::filter_npass_each_dir;
        }
        stream << '\n';
        return stream.str();
    }

    using FieldArrays = amrex::GpuArray<amrex::Array4<amrex::Real>, 3>;

    FieldArrays Arrays (ablastr::fields::MultiFabRegister& fields,
                        FieldType const type, int const lev)
    {
        auto const mf = fields.get_alldirs(type, lev);
        amrex::MFIter const mfi(*mf[0]);
        return {mf[0]->array(mfi), mf[1]->array(mfi), mf[2]->array(mfi)};
    }
}
#endif

void WarpX::WriteCompositeXZCheckpoint (std::string const& directory) const
{
#ifdef WARPX_DIM_XZ
    if (m_composite_xz_interface && amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream marker(directory + checkpoint_marker);
        marker << CheckpointIdentity(*this);
        marker.close();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(marker.good(), "Cannot write the XZ slab checkpoint identity.");
    }
#else
    amrex::ignore_unused(directory);
#endif
}

void WarpX::CheckCompositeXZCheckpoint (std::string const& directory) const
{
#ifdef WARPX_DIM_XZ
    int compatible = 0;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        auto const name = directory + checkpoint_marker;
        if (amrex::FileExists(name)) {
            std::ifstream marker(name);
            std::string const saved{std::istreambuf_iterator<char>{marker}, {}};
            compatible = m_composite_xz_interface && marker.good()
                && max_level == 1 && finest_level == 1 && saved == CheckpointIdentity(*this);
        } else {
            compatible = !m_composite_xz_interface;
        }
    }
    amrex::ParallelDescriptor::Bcast(&compatible, 1, amrex::ParallelDescriptor::IOProcessorNumber());
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(compatible != 0,
        "The checkpoint and current input must use the same Cartesian XZ slab interface, "
        "patch, refinement ratio, particle shape and current filter.");
#else
    amrex::ignore_unused(directory);
#endif
}

void WarpX::InitCompositeXZInterface ()
{
#ifdef WARPX_DIM_XZ
    if (!m_composite_xz_interface) { return; }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        max_level == 1 && finest_level == 1 && amrex::ParallelDescriptor::NProcs() == 1 &&
        evolve_scheme == EvolveScheme::Explicit && !m_do_subcycling &&
        electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee &&
        electrostatic_solver_id == ElectrostaticSolverAlgo::None &&
        grid_type == ablastr::utils::enums::GridType::Staggered &&
        field_gathering_algo == GatheringAlgo::EnergyConserving &&
        m_em_solver_medium == MediumForEM::Vacuum && !EB::enabled() &&
        !do_moving_window && regrid_int <= 0 &&
        !load_balance_intervals.isActivated() &&
        !do_dive_cleaning && !do_divb_cleaning && !do_pml_dive_cleaning &&
        !do_pml_divb_cleaning && !m_do_initial_div_cleaning &&
        !do_pml_in_domain && !pml_has_particles && !do_current_centering &&
        !use_fdtd_nci_corr && !fft_do_time_averaging && !use_hybrid_QED &&
        gamma_boost == 1.0_rt && n_current_deposition_buffer == 0 &&
        n_field_gather_buffer == 0 && mypc->nSpeciesDepositOnMainGrid() == 0 &&
        mypc->nSpeciesGatherFromMainGrid() == 0 && mypc->nLasers() == 0 &&
        !do_fluid_species && m_num_mirrors == 0 && (mypc->nSpecies() == 0 || (nox >= 1 && nox <= 3)) &&
        m_p_ext_field_params->E_ext_grid_type == ExternalFieldType::default_zero &&
        m_p_ext_field_params->B_ext_grid_type == ExternalFieldType::default_zero,
        "composite_xz_interface requires a serial explicit Yee vacuum simulation "
        "with shape orders 1-3, energy-conserving gather and zero handoff "
        "buffers. Cleaning, materials, EB, moving grids, "
        "lasers, fluids, main-grid particle overrides and other feedback controls "
        "are unsupported.");
    auto const fine = boxArray(1).minimalBox();
    auto const coarse = Geom(0).Domain();
    auto const h = Geom(1).CellSize(0);
    auto const nz = coarse.length(1);
    auto const ratio = refRatio(0);
    int const rx = ratio[0];
    int const rz = ratio[1];
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        boxArray(0).size() == 1 && boxArray(1).size() == 1 &&
        boxArray(0)[0] == coarse && boxArray(1)[0] == fine &&
        coarse.smallEnd(0) == 0 && coarse.smallEnd(1) == 0 &&
        fine.smallEnd(1) == 0 && fine.bigEnd(1) == Geom(1).Domain().bigEnd(1) &&
        fine.smallEnd(0) > 0 && fine.bigEnd(0) < Geom(1).Domain().bigEnd(0) &&
        fine.smallEnd(0) % rx == 0 && (fine.bigEnd(0) + 1) % rx == 0 &&
        Geom(0).isPeriodic(1) && Geom(1).isPeriodic(1) &&
        (rx == 1 || rx == 2) && (rz == 1 || rz == 2) &&
        (rx == 2 || rz == 2) && nz >= 3 && nz <= 1024 &&
        std::abs(Geom(1).CellSize(1) / h - 1.0_rt) <=
            64.0_rt * std::numeric_limits<amrex::Real>::epsilon() &&
        PhysConst::c * dt[0] / h < 1.0_rt / std::sqrt(2.0_rt),
        "composite_xz_interface requires a square finest mesh and a ratio-one-or-two slab spanning "
        "periodic z, 3-1024 coarse z cells and c*dt/h < 1/sqrt(2).");
    bool const periodic_x = Geom(0).isPeriodic(0) &&
        field_boundary_lo[0] == FieldBoundaryType::Periodic &&
        field_boundary_hi[0] == FieldBoundaryType::Periodic;
    bool const pml_x = field_boundary_lo[0] == FieldBoundaryType::PML &&
        field_boundary_hi[0] == FieldBoundaryType::PML;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(periodic_x || pml_x,
        "composite_xz_interface supports periodic or PML outer x boundaries.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (!do_pml || (pml[1] && !pml[1]->ok())) && !m_fields.has_vector(FieldType::pml_E_fp, 1),
        "The physical interface must not allocate a refined absorber.");
    for (auto const& species : *mypc) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !species->doContinuousInjection() && !species->DoFieldIonization() &&
            !species->initialize_self_fields && !species->do_splitting,
            "Composite XZ particle creation, splitting and self-field initialization "
            "are not qualified.");
    }
    for (int d = 0; d < 3; ++d) {
        for (auto const type : {FieldType::Efield_fp, FieldType::Bfield_fp}) {
            auto const aux_type = type == FieldType::Efield_fp
                ? FieldType::Efield_aux : FieldType::Bfield_aux;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_fields.get(type, Direction{d}, 1)->nGrowVect() ==
                    m_fields.get(aux_type, Direction{d}, 1)->nGrowVect(),
                "Composite XZ requires matching native and auxiliary guards.");
        }
    }
    // Ey is nodal along z. Eliminating the midpoint gives the periodic
    // mass matrix M_coarse + P^T M_fine P. With no tangential refinement
    // P is the identity and the Ampere update uses a scalar dual volume.
    if (rz == 2) {
        auto const fine_half = h * h / 2.0_rt;
        auto const coarse_half = rx * rz * fine_half;
        auto const diagonal = coarse_half + 1.5_rt * fine_half;
        auto const offdiag = 0.25_rt * fine_half;
        auto const discriminant = std::sqrt(diagonal * diagonal - 4.0_rt * offdiag * offdiag);
        auto const root = -2.0_rt * offdiag / (diagonal + discriminant);
        auto const normalization = (1.0_rt - std::pow(root, nz)) * discriminant;
        amrex::Vector<amrex::Real> inverse(nz);
        for (int distance = 0; distance < nz; ++distance) {
            int const nearest = std::min(distance, nz - distance);
            inverse[distance] = (std::pow(root, nearest) + std::pow(root, nz - nearest)) /
                normalization;
        }
        m_composite_xz_inverse_y.resize(nz);
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, inverse.begin(), inverse.end(),
                         m_composite_xz_inverse_y.begin());
    }
    m_composite_xz_boundary.resize(4 * nz);
    m_composite_xz_rhs.resize(4 * nz);
    CheckCompositeXZParticles();
    MaskCompositeXZCoveredFields();
#endif
}

void WarpX::CheckCompositeXZParticles ()
{
#ifdef WARPX_DIM_XZ
    if (!m_composite_xz_interface) { return; }
    auto const fine = boxArray(1).minimalBox();
    auto const h = Geom(1).CellSize(0);
    auto const left = Geom(1).ProbLo(0) + fine.smallEnd(0) * h;
    auto const right = Geom(1).ProbLo(0) + (fine.bigEnd(0) + 1) * h;
    // A normal binomial pass extends the current by one cell. Restricted fine
    // current also passes through the parent filter. Keep its full stencil and
    // one coarse cell for restriction clear of the interface on both meshes.
    int const passes = use_filter ? filter_npass_each_dir[0] : 0;
    auto const filter_margin = passes > 0 ? (passes + 1) * Geom(0).CellSize(0) : 0.0_rt;
    amrex::ReduceOps<amrex::ReduceOpMax> operation;
    amrex::ReduceData<int> data(operation);
    using Tuple = typename decltype(data)::Type;
    operation.eval(1, data, [] AMREX_GPU_DEVICE (int) -> Tuple { return {0}; });
    for (auto const& species : *mypc) {
        for (int lev = 0; lev <= 1; ++lev) {
            // One extra cell beyond B-spline half-width, plus the entire next
            // physical-c particle trajectory. Never substitute a reduced wave c.
            auto const margin = 0.5_rt * (nox + 3) * Geom(lev).CellSize(0) +
                PhysConst::c * dt[0] + filter_margin;
            for (WarpXParIter pti(*species, lev); pti.isValid(); ++pti) {
                auto const position = GetParticlePosition(pti);
                operation.eval(pti.numParticles(), data,
                    [=] AMREX_GPU_DEVICE (int i) -> Tuple
                    {
                        amrex::ParticleReal x, y, z;
                        position(i, x, y, z);
                        bool const allowed = lev == 1
                            ? x > left + margin && x < right - margin
                            : x < left - margin || x > right + margin;
                        return {std::isfinite(x) && allowed ? 0 : 1};
                    });
            }
        }
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(amrex::get<0>(data.value()) == 0,
        "A particle entered the unsupported physical-interface gather/deposition "
        "buffer. Interface crossing is not implemented.");
#endif
}

void WarpX::MaskCompositeXZCoveredFields ()
{
#ifdef WARPX_DIM_XZ
    if (!m_composite_xz_interface) { return; }
    auto const fine = boxArray(1).minimalBox();
    int const rx = refRatio(0)[0];
    int const left = fine.smallEnd(0) / rx;
    int const right = (fine.bigEnd(0) + 1) / rx;
    for (auto const type : {FieldType::Efield_fp, FieldType::Bfield_fp}) {
        for (int d = 0; d < 3; ++d) {
            auto& field = *m_fields.get(type, Direction{d}, 0);
            int const begin = left + int(field.ixType().nodeCentered(0));
            for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
                auto const values = field.array(mfi);
                amrex::ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        if (i >= begin && i < right) { values(i,j,k) = 0.0_rt; }
                    });
            }
        }
    }
#endif
}

void WarpX::SynchronizeCompositeXZMagneticTrace ()
{
#ifdef WARPX_DIM_XZ
    if (!m_composite_xz_interface) { return; }
    // There is one independent normal-B trace. Linear interpolation of Ey
    // makes both fine z differences equal to the coarse difference in exact
    // arithmetic. Repeating those subtractions on each mesh accumulates
    // independent roundoff, particularly when the physical Bx is zero.
    // Materialize the dependent fine trace from its canonical coarse value
    // after each Faraday half-step, just as the E update materializes its
    // constrained tangential traces. This does not modify the independent DOF.
    auto const fine = boxArray(1).minimalBox();
    int const fl = fine.smallEnd(0);
    int const fr = fine.bigEnd(0) + 1;
    int const rx = refRatio(0)[0];
    int const rz = refRatio(0)[1];
    int const nz = Geom(0).Domain().length(1);
    auto const bc = Arrays(m_fields, FieldType::Bfield_fp, 0);
    auto const bf = Arrays(m_fields, FieldType::Bfield_fp, 1);
    amrex::ParallelFor(2 * nz,
        [=] AMREX_GPU_DEVICE (int index) noexcept
        {
            int const xf = index < nz ? fl : fr;
            int const j = index % nz;
            auto const value = bc[0](xf / rx,j,0);
            for (int z = 0; z < rz; ++z) { bf[0](xf,rz*j+z,0) = value; }
        });
#endif
}

void WarpX::PrepareCompositeXZInterface ()
{
#ifdef WARPX_DIM_XZ
    if (!m_composite_xz_interface) { return; }
    MaskCompositeXZCoveredFields();
    auto const fine = boxArray(1).minimalBox();
    auto const nz = Geom(0).Domain().length(1);
    auto const ec = Arrays(m_fields, FieldType::Efield_fp, 0);
    auto const ef = Arrays(m_fields, FieldType::Efield_fp, 1);
    auto const bc = Arrays(m_fields, FieldType::Bfield_fp, 0);
    auto const bf = Arrays(m_fields, FieldType::Bfield_fp, 1);
    auto* old = m_composite_xz_boundary.data();
    int const fl = fine.smallEnd(0);
    int const fr = fine.bigEnd(0) + 1;
    int const rx = refRatio(0)[0];
    int const rz = refRatio(0)[1];
    for (int d = 1; d <= 3; ++d) {
        amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax> operation;
        amrex::ReduceData<amrex::Real, amrex::Real> data(operation);
        using Tuple = typename decltype(data)::Type;
        operation.eval(2 * nz, data,
            [=] AMREX_GPU_DEVICE (int index) -> Tuple
            {
                int const side = index / nz;
                int const j = index % nz;
                int const jp = (j + 1) % nz;
                int const xf = side == 0 ? fl : fr;
                int const xc = xf / rx;
                auto const coarse = d == 3 ? bc[0](xc,j,0) : ec[d](xc,j,0);
                int const next = rz*j + rz - 1;
                auto const even = d == 3 ? bf[0](xf,rz*j,0) : ef[d](xf,rz*j,0);
                auto const odd = d == 3 ? bf[0](xf,next,0) : ef[d](xf,next,0);
                auto const expected_odd = d == 1 && rz == 2
                    ? 0.5_rt * (coarse + ec[d](xc,jp,0)) : coarse;
                if (d < 3) { old[(2 * side + d - 1) * nz + j] = coarse; }
                auto const error = amrex::max(std::abs(even - coarse),
                                              std::abs(odd - expected_odd));
                auto const scale = amrex::max(std::abs(coarse), std::abs(even),
                                              std::abs(odd), std::abs(expected_odd));
                return {error, scale};
            });
        auto const result = data.value();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            amrex::get<0>(result) <= amrex::max(2.e-12_rt, 32.0_rt *
                std::numeric_limits<amrex::Real>::epsilon()) *
                amrex::max(amrex::get<1>(result), std::numeric_limits<amrex::Real>::min()),
            "Initial or evolved XZ tangential-E/normal-B interface constraint failed.");
    }
#endif
}

void WarpX::ApplyCompositeXZInterface (amrex::Real const a_dt)
{
#ifndef WARPX_DIM_XZ
    amrex::ignore_unused(a_dt);
#else
    if (!m_composite_xz_interface) { return; }
    auto const fine = boxArray(1).minimalBox();
    int const fl = fine.smallEnd(0);
    int const fr = fine.bigEnd(0) + 1;
    int const rx = refRatio(0)[0];
    int const rz = refRatio(0)[1];
    int const nz = Geom(0).Domain().length(1);
    auto const h = Geom(1).CellSize(0);
    auto const dzc = Geom(0).CellSize(1);
    auto const mc = Geom(0).CellSize(0) * dzc / 2.0_rt;
    auto const mf = h * h / 2.0_rt;
    auto const bc = Arrays(m_fields, FieldType::Bfield_fp, 0);
    auto const bf = Arrays(m_fields, FieldType::Bfield_fp, 1);
    auto const jc = Arrays(m_fields, FieldType::current_fp, 0);
    auto const jf = Arrays(m_fields, FieldType::current_fp, 1);
    amrex::ReduceOps<amrex::ReduceOpMax> operation;
    amrex::ReduceData<amrex::Real> data(operation);
    using Tuple = typename decltype(data)::Type;
    operation.eval(2 * nz, data,
        [=] AMREX_GPU_DEVICE (int index) -> Tuple
        {
            int const side = index / nz;
            int const j = index % nz;
            int const xf = side == 0 ? fl : fr;
            int const xc = xf / rx;
            int const ic = side == 0 ? xc - 1 : xc;
            int const i_f = side == 0 ? xf : xf - 1;
            amrex::Real value = std::abs(jc[0](ic,j,0));
            for (int d = 0; d < 3; ++d) {
                int const cx = d == 0 ? ic : xc;
                int const fx = d == 0 ? i_f : xf;
                value = amrex::max(value, std::abs(jc[d](cx,j,0)),
                    std::abs(jf[d](fx,rz*j,0)), std::abs(jf[d](fx,rz*j+rz-1,0)));
            }
            return {value};
        });
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(amrex::get<0>(data.value()) == 0.0_rt,
        "Physical-interface current ownership is not implemented: sources must "
        "vanish on both tangential traces and adjacent normal-current edges.");
    auto* rhs = m_composite_xz_rhs.data();
    amrex::ParallelFor(4 * nz,
        [=] AMREX_GPU_DEVICE (int index) noexcept
        {
            int const side = index / (2 * nz);
            int const d = (index / nz) % 2 + 1;
            int const j = index % nz;
            int const jm = (j + nz - 1) % nz;
            int const k = rz * j;
            int const fnz = rz * nz;
            int const km = (k + fnz - 1) % fnz;
            int const xf = side == 0 ? fl : fr;
            int const xc = xf / rx;
            int const ic = side == 0 ? xc - 1 : xc;
            int const i_f = side == 0 ? xf : xf - 1;
            auto const sign = side == 0 ? 1.0_rt : -1.0_rt;
            if (d == 2) {
                auto by = bf[1](i_f,k,0);
                if (rz == 2) { by += bf[1](i_f,k+1,0); }
                rhs[index] = sign * (h * by - dzc * bc[1](ic,j,0));
            } else {
                auto const dc = (bc[0](xc,j,0) - bc[0](xc,jm,0)) / dzc;
                auto df = bf[0](xf,k,0) - bf[0](xf,km,0);
                auto bz = bf[2](i_f,k,0);
                if (rz == 2) {
                    int const kmm = (k + fnz - 2) % fnz;
                    df += 0.5_rt * (bf[0](xf,km,0) - bf[0](xf,kmm,0) +
                                    bf[0](xf,k+1,0) - bf[0](xf,k,0));
                    bz += 0.5_rt * (bf[2](i_f,km,0) + bf[2](i_f,k+1,0));
                }
                rhs[index] = mc * dc + mf * df / h -
                    sign * (h * bz - dzc * bc[2](ic,j,0));
            }
        });
    auto* updated = m_composite_xz_boundary.data();
    auto const* inverse = m_composite_xz_inverse_y.data();
    auto const dtc2 = a_dt * PhysConst::c * PhysConst::c;
    amrex::ParallelFor(4 * nz,
        [=] AMREX_GPU_DEVICE (int index) noexcept
        {
            int const d = (index / nz) % 2 + 1;
            int const j = index % nz;
            int const base = index - j;
            amrex::Real increment = 0.0_rt;
            if (d == 2 || rz == 1) {
                increment = rhs[index] / (mc + rz * mf);
            } else {
                for (int offset = 0; offset < nz; ++offset) {
                    increment += inverse[offset] * rhs[base + (j - offset + nz) % nz];
                }
            }
            updated[index] += dtc2 * increment;
        });
    auto const ec = Arrays(m_fields, FieldType::Efield_fp, 0);
    auto const ef = Arrays(m_fields, FieldType::Efield_fp, 1);
    amrex::ParallelFor(4 * nz,
        [=] AMREX_GPU_DEVICE (int index) noexcept
        {
            int const side = index / (2 * nz);
            int const d = (index / nz) % 2 + 1;
            int const j = index % nz;
            int const xf = side == 0 ? fl : fr;
            int const xc = xf / rx;
            auto const value = updated[index];
            ec[d](xc,j,0) = value;
            ef[d](xf,rz*j,0) = value;
            if (rz == 2) {
                ef[d](xf,rz*j+1,0) = d == 1
                    ? 0.5_rt * (value + updated[index - j + (j + 1) % nz]) : value;
            }
            if (d == 1 && j == 0) {
                ec[d](xc,nz,0) = value;
                ef[d](xf,rz*nz,0) = value;
            }
        });
    MaskCompositeXZCoveredFields();
#endif
}
