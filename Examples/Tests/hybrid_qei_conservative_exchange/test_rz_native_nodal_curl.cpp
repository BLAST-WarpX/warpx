/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "EmbeddedBoundary/WarpXFaceInfoBox.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Fields.H"
#include "Initialization/WarpXInit.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>
#include <AMReX_iMultiFab.H>

#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using warpx::fields::FieldType;
using namespace amrex::literals;

namespace
{
struct ErrorCheck
{
    amrex::Real absolute = 0;
    amrex::Real relative = 0;
    amrex::Real axis_absolute = 0;
    amrex::Real axis_relative = 0;
};

void
SetVector (ablastr::fields::VectorField const& fields, amrex::Real const value)
{
    for (auto* field : fields)
    {
        field->setVal(value);
    }
}

void
FillAzimuthal (ablastr::fields::VectorField const& fields, amrex::Geometry const& geometry,
               amrex::Real const slope)
{
    amrex::Real const dr = geometry.CellSize(0);
    for (amrex::MFIter mfi(*fields[0]); mfi.isValid(); ++mfi)
    {
        auto const fr = fields[0]->array(mfi);
        auto const ftheta = fields[1]->array(mfi);
        auto const fz = fields[2]->array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            fr(i, j, k, 0) = 0;
            ftheta(i, j, k, 0) = slope * static_cast<amrex::Real>(i) * dr;
            fz(i, j, k, 0) = 0;
        });
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
MixedPhi (amrex::Real const r, amrex::Real const z)
{
    amrex::Real const r2 = r * r;
    amrex::Real const phase = 2 * MathConst::pi * z;
    return r2 * (1 - r2) * std::sin(phase) *
           (1 + 0.25_rt * r2 * std::cos(phase));
}

void
FillCenteredGradient (ablastr::fields::VectorField const& fields, amrex::Geometry const& geometry)
{
    amrex::Real const dr = geometry.CellSize(0);
    amrex::Real const dz = geometry.CellSize(1);
    for (amrex::MFIter mfi(*fields[0]); mfi.isValid(); ++mfi)
    {
        auto const fr = fields[0]->array(mfi);
        auto const ftheta = fields[1]->array(mfi);
        auto const fz = fields[2]->array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            amrex::Real const r = static_cast<amrex::Real>(i) * dr;
            amrex::Real const z = static_cast<amrex::Real>(j) * dz;
            fr(i, j, k, 0) = (MixedPhi(r + dr, z) - MixedPhi(r - dr, z)) / (2 * dr);
            ftheta(i, j, k, 0) = 0;
            fz(i, j, k, 0) = (MixedPhi(r, z + dz) - MixedPhi(r, z - dz)) / (2 * dz);
        });
    }
}

ErrorCheck
CheckField (amrex::MultiFab const& field, amrex::Real const expected, amrex::Real scale)
{
    AMREX_ALWAYS_ASSERT(scale > 0);
    AMREX_ALWAYS_ASSERT(field.is_finite());

    amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax,
                     amrex::ReduceOpMax, amrex::ReduceOpMax> ops;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real> data(ops);
    using Tuple = typename decltype(data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(field, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const values = field.array(mfi);
        ops.eval(mfi.tilebox(), data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Tuple
            {
                amrex::Real const error =
                    amrex::Math::abs(values(i, j, k, 0) - expected);
                amrex::Real const relative = error / scale;
                amrex::Real const axis_error = i == 0 ? error : 0;
                amrex::Real const axis_relative = i == 0 ? relative : 0;
                return {error, relative, axis_error, axis_relative};
            });
    }

    auto const result = data.value();
    std::array<amrex::Real, 4> reduced{amrex::get<0>(result), amrex::get<1>(result),
                                       amrex::get<2>(result), amrex::get<3>(result)};
    amrex::ParallelDescriptor::ReduceRealMax(reduced.data(), 4);
    return {reduced[0], reduced[1], reduced[2], reduced[3]};
}

void
Report (std::string const& name, ErrorCheck const& error)
{
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        amrex::Print() << std::setprecision(17) << name
                       << " max_abs=" << error.absolute
                       << " max_rel=" << error.relative
                       << " axis_abs=" << error.axis_absolute
                       << " axis_rel=" << error.axis_relative << '\n';
    }
}

void
AssertNativeLayout (std::array<ablastr::fields::VectorField, 3> const& vectors)
{
    auto const& reference = *vectors[0][0];
    for (auto const& vector : vectors)
    {
        for (auto const* field : vector)
        {
            AMREX_ALWAYS_ASSERT(field != nullptr);
            AMREX_ALWAYS_ASSERT(field->nComp() == 1);
            AMREX_ALWAYS_ASSERT(field->ixType().nodeCentered());
            AMREX_ALWAYS_ASSERT(field->nGrowVect()[0] >= 1 && field->nGrowVect()[1] >= 1);
            AMREX_ALWAYS_ASSERT(field->boxArray() == reference.boxArray());
            AMREX_ALWAYS_ASSERT(field->DistributionMap() == reference.DistributionMap());
        }
    }
}
} // namespace

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        int refine = 32;
        amrex::ParmParse test_pp("test");
        test_pp.query("refine", refine);
        if (refine != 32)
        {
            amrex::ParmParse amr_pp("amr");
            amr_pp.addarr("n_cell", std::vector<int>{refine, refine});
        }

        auto& simulation = WarpX::GetInstance();
        simulation.InitData();

        auto const& geometry = simulation.Geom(0);
        AMREX_ALWAYS_ASSERT(simulation.finestLevel() == 0);
        AMREX_ALWAYS_ASSERT(geometry.Domain().length(0) == refine);
        AMREX_ALWAYS_ASSERT(geometry.Domain().length(1) == refine);
        AMREX_ALWAYS_ASSERT(geometry.ProbLo(0) == 0 && geometry.ProbLo(1) == 0);
        AMREX_ALWAYS_ASSERT(geometry.ProbHi(0) == 1 && geometry.ProbHi(1) == 1);
        AMREX_ALWAYS_ASSERT(geometry.isPeriodic(1) && !geometry.isPeriodic(0));
        AMREX_ALWAYS_ASSERT(WarpX::grid_type == ablastr::utils::enums::GridType::Collocated);

        auto Bfield = simulation.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
        auto Efield = simulation.m_fields.get_alldirs(FieldType::Efield_fp, 0);
        auto Jfield = simulation.m_fields.get_alldirs(FieldType::current_fp, 0);
        AssertNativeLayout({Bfield, Efield, Jfield});

        amrex::Real const dr = geometry.CellSize(0);
        amrex::Real const dz = geometry.CellSize(1);
        FiniteDifferenceSolver solver(
            ElectromagneticSolverAlgo::HybridPIC, {dr, 1.0_rt, dz},
            ablastr::utils::enums::GridType::Collocated);
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> no_eb_update;
        std::array<std::unique_ptr<amrex::LayoutData<FaceInfoBox>>, 3> no_borrowing;

        amrex::Real const slope = 1;
        SetVector(Bfield, 0);
        FillAzimuthal(Bfield, geometry, slope);
        SetVector(Efield, -123);
        SetVector(Jfield, -123);
        solver.CalculateCurrentAmpere(Jfield, Bfield, no_eb_update, 0);
        amrex::Real const expected_jz = 2 * slope / PhysConst::mu0;
        for (int direction = 0; direction < 3; ++direction)
        {
            auto const error = CheckField(
                *Jfield[direction], direction == 2 ? expected_jz : 0, expected_jz);
            Report("Ampere J" + std::to_string(direction), error);
            AMREX_ALWAYS_ASSERT(error.relative <= 1.e-12_rt);
            AMREX_ALWAYS_ASSERT(error.axis_relative <= 1.e-12_rt);
        }

        amrex::Real const dt = simulation.getdt(0);
        SetVector(Efield, 0);
        FillAzimuthal(Efield, geometry, slope);
        solver.ComputeCurlA(Bfield, Efield, no_eb_update, 0);
        for (int direction = 0; direction < 3; ++direction)
        {
            auto const error = CheckField(*Bfield[direction], direction == 2 ? 2*slope : 0, 2*slope);
            Report("Vector-potential curl B" + std::to_string(direction), error);
            AMREX_ALWAYS_ASSERT(error.relative <= 1.e-12_rt);
        }
        auto radial_field = Efield;
        std::swap(radial_field[0], radial_field[1]);
        FillAzimuthal(radial_field, geometry, slope);
        amrex::MultiFab divergence(Efield[0]->boxArray(), Efield[0]->DistributionMap(), 1, 0);
        solver.ComputeDivE(Efield, divergence);
        auto const divergence_error = CheckField(divergence, 2*slope, 2*slope);
        Report("Electric divergence", divergence_error);
        AMREX_ALWAYS_ASSERT(divergence_error.relative <= 1.e-12_rt);

        SetVector(Bfield, 0);
        SetVector(Jfield, 0);
        SetVector(Efield, 0);
        FillAzimuthal(Efield, geometry, slope);
        solver.EvolveB(simulation.m_fields, 0, PatchType::fine, no_eb_update, no_borrowing, dt);
        amrex::Real const expected_bz = -2 * slope * dt;
        for (int direction = 0; direction < 3; ++direction)
        {
            auto const error = CheckField(
                *Bfield[direction], direction == 2 ? expected_bz : 0, std::abs(expected_bz));
            Report("Faraday B" + std::to_string(direction), error);
            AMREX_ALWAYS_ASSERT(error.relative <= 1.e-12_rt);
            AMREX_ALWAYS_ASSERT(error.axis_relative <= 1.e-12_rt);
        }

        SetVector(Bfield, 0);
        SetVector(Efield, 0);
        FillCenteredGradient(Efield, geometry);
        solver.EvolveB(simulation.m_fields, 0, PatchType::fine, no_eb_update, no_borrowing, dt);
        amrex::Real const roundoff_scale = dt * (2 + 2 / dr + 2 / dz);
        for (int direction = 0; direction < 3; ++direction)
        {
            auto const error = CheckField(*Bfield[direction], 0, 1);
            Report("Curl-gradient B" + std::to_string(direction), error);
            AMREX_ALWAYS_ASSERT(error.absolute <= 1.e-12_rt * roundoff_scale);
            AMREX_ALWAYS_ASSERT(error.axis_absolute <= 1.e-12_rt * roundoff_scale);
        }

        solver.ComputeCurlA(Bfield, Efield, no_eb_update, 0);
        for (int direction = 0; direction < 3; ++direction)
        {
            auto const error = CheckField(*Bfield[direction], 0, 1);
            Report("Vector-potential curl-gradient B" + std::to_string(direction), error);
            AMREX_ALWAYS_ASSERT(error.absolute <= 1.e-12_rt * roundoff_scale / dt);
        }

        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
