/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Diagnostics/MultiDiagnostics.H"
#include "Diagnostics/ReducedDiags/FieldEnergy.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/NativeRZMagneticPressure.H"
#include "Fields.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using warpx::fields::FieldType;

namespace
{
struct Backup
{
    amrex::MultiFab& field;
    amrex::MultiFab values;
    explicit Backup (amrex::MultiFab& source)
        : field(source), values(source.boxArray(), source.DistributionMap(), source.nComp(),
                                source.nGrowVect())
    {
        amrex::MultiFab::Copy(values, source, 0, 0, source.nComp(), source.nGrowVect());
    }
    void restore () const
    {
        amrex::MultiFab::Copy(field, values, 0, 0, field.nComp(), field.nGrowVect());
    }
    bool identical () const
    {
        int different = 0;
        for (amrex::MFIter it(field); it.isValid(); ++it) {
            auto const size = static_cast<std::size_t>(field[it].size());
            std::vector<amrex::Real> a(size), b(size);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, field[it].dataPtr(),
                             field[it].dataPtr() + size, a.begin());
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, values[it].dataPtr(),
                             values[it].dataPtr() + size, b.begin());
            different |= std::memcmp(a.data(), b.data(), size * sizeof(amrex::Real)) != 0;
        }
        amrex::ParallelDescriptor::ReduceIntMax(different);
        return different == 0;
    }
};

std::vector<amrex::Real>
gather (amrex::MultiFab const& field, amrex::Geometry const& geom)
{
    auto const box = amrex::convert(geom.Domain(), field.ixType());
    amrex::MultiFab single(amrex::BoxArray(box),
        amrex::DistributionMapping(amrex::Vector<int>{0}), 1, 0);
    single.ParallelCopy(field, 0, 0, 1);
    std::vector<amrex::Real> values(static_cast<std::size_t>(box.numPts()));
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, single[0].dataPtr(),
                         single[0].dataPtr() + box.numPts(), values.begin());
    }
    amrex::ParallelDescriptor::Bcast(values.data(), values.size(), 0);
    return values;
}

double
save (WarpX& sim, std::string const& filename)
{
    auto& fields = sim.m_fields;
    auto* model = sim.get_pointer_HybridPICModel();
    auto const& geom = sim.Geom(0);
    auto const B = fields.get_alldirs(FieldType::Bfield_fp, 0);
    auto const E = fields.get_alldirs(FieldType::Efield_fp, 0);
    std::vector<std::pair<std::string, std::vector<amrex::Real>>> output;
    for (int d = 0; d < 3; ++d) {
        output.emplace_back("B" + std::to_string(d), gather(*B[d], geom));
        output.emplace_back("E" + std::to_string(d), gather(*E[d], geom));
    }
    output.emplace_back("U", gather(*fields.get("hybrid_native_pressure_energy_fp", 0), geom));
    output.emplace_back("T", gather(*fields.get(FieldType::hybrid_electron_temperature_fp, 0), geom));
    output.emplace_back("rho", gather(*fields.get("hybrid_native_pressure_charge_fp", 0), geom));
    output.emplace_back("V", gather(*fields.get("hybrid_native_pressure_volume_fp", 0), geom));
    auto const& u = output[6].second;
    auto const& b = output[2].second;
    int const nr = geom.Domain().length(0), nz = geom.Domain().length(1);
    double const dr = geom.CellSize(0), dz = geom.CellSize(1);
    bool const collocated = B[1]->ixType().nodeCentered();
    double thermal = 0, magnetic = 0;
    for (int j = 0; j < nz; ++j) {
        for (int i = 0; i <= nr; ++i) {
            double const v = i == 0 ? MathConst::pi * dr * dr * dz / 3
                : 2 * MathConst::pi * i * dr * dr * dz * (i == nr ? .5 : 1);
            thermal += u[i + (nr + 1) * j] * v;
        }
        for (int i = 0; i < nr + int(collocated); ++i) {
            double const v = 2 * MathConst::pi * dr * dr * dz *
                (collocated ? i * (i == nr ? .5 : 1) : i + .5);
            double const value = b[i + (nr + int(collocated)) * j];
            magnetic += .5 * v * value * value / PhysConst::mu0;
        }
    }
    FieldEnergy native_energy("pressure_runtime_energy");
    double const measured = .5 * native_energy.ComputeNorm2(*B[1], 0) * dr * dz / PhysConst::mu0;
    AMREX_ALWAYS_ASSERT(std::abs(measured - magnetic) <=
        128 * std::numeric_limits<double>::epsilon() * std::max(measured, magnetic));
    auto const ez = gather(*E[2], geom);
    for (int j = 0; j < nz; ++j) { AMREX_ALWAYS_ASSERT(ez[nr + (nr + 1) * j] == 0); }
    if (collocated) {
        for (int j = 0; j < nz; ++j) { AMREX_ALWAYS_ASSERT(b[(nr + 1) * j] == 0); }
    }
    double kinetic = 0;
    for (auto const& name : sim.GetPartContainer().GetSpeciesNames()) {
        kinetic += sim.GetPartContainer().GetParticleContainerFromName(name).sumParticleEnergy();
    }
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream file(filename);
        file << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
        file << "step " << sim.getistep(0) << "\nsubsteps " << model->m_substeps
             << "\nenergy " << thermal + magnetic << "\nkinetic " << kinetic
             << "\nwall_work 0\n";
        for (auto const& [name, values] : output) {
            file << name << ' ' << values.size();
            for (auto const value : values) { file << ' ' << value; }
            file << '\n';
        }
        AMREX_ALWAYS_ASSERT(file.good());
    }
    return thermal + magnetic;
}

void
rollback (WarpX& sim)
{
    auto& fields = sim.m_fields;
    auto& model = *sim.get_pointer_HybridPICModel();
    std::vector<std::unique_ptr<Backup>> saved;
    for (auto const type : {FieldType::Bfield_fp, FieldType::Efield_fp,
                           FieldType::hybrid_current_fp_plasma, FieldType::current_fp,
                           FieldType::hybrid_current_fp_temp}) {
        for (auto* field : fields.get_alldirs(type, 0)) {
            saved.push_back(std::make_unique<Backup>(*field));
        }
    }
    for (auto const type : {FieldType::hybrid_electron_temperature_fp,
                           FieldType::hybrid_electron_pressure_fp, FieldType::rho_fp,
                           FieldType::hybrid_rho_fp_temp}) {
        saved.push_back(std::make_unique<Backup>(*fields.get(type, 0)));
    }
    for (auto const* name : {"hybrid_native_pressure_energy_fp", "hybrid_native_pressure_charge_fp",
                            "hybrid_native_pressure_volume_fp"}) {
        saved.push_back(std::make_unique<Backup>(*fields.get(name, 0)));
    }
    int const initial_substeps = model.m_substeps;
    int const initial_budget = model.m_max_substep_attempts;
    auto const intervals = model.m_rkf45_intervals;
    model.m_rkf45_intervals = ablastr::utils::text::IntervalsParser({"0"});
    model.m_max_substep_attempts = 1;
    bool second_half_rejected = false;
    for (double dt = 1.e-6; dt < 1.e-3 && !second_half_rejected; dt *= 1.15) {
        for (auto const& field : saved) { field->restore(); }
        model.m_substeps = 2;
        warpx::hybrid::NativeRZMagneticPressureDiagnostics diagnostics;
        bool const accepted = model.tryNativePressureFieldEvolve(dt, 0, diagnostics);
        if (!accepted) {
            for (auto const& field : saved) { AMREX_ALWAYS_ASSERT(field->identical()); }
            AMREX_ALWAYS_ASSERT(model.m_substeps == 2);
            second_half_rejected = diagnostics.accepted > 0;
            if (second_half_rejected) {
                amrex::Print() << "complete native pressure step rollback after accepted first half dt="
                               << dt << '\n';
            }
        }
    }
    AMREX_ALWAYS_ASSERT(second_half_rejected);
    for (auto const& field : saved) { field->restore(); }
    model.m_substeps = initial_substeps;
    model.m_max_substep_attempts = initial_budget;
    model.m_rkf45_intervals = intervals;
}
}

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        auto* model = sim.get_pointer_HybridPICModel();
        AMREX_ALWAYS_ASSERT(model->m_pressure_only && model->m_native_pressure_state_initialized);
        double const initial_energy = save(sim, "initial_state.txt");
        bool initial_checkpoint = false;
        amrex::ParmParse("test").query("initial_checkpoint", initial_checkpoint);
        if (initial_checkpoint) {
            // Both continuation paths must start from this same deposited state. GPU atomics
            // need not reproduce every rounding bit in two independent particle depositions.
            sim.GetMultiDiags().NewIteration();
            sim.GetMultiDiags().FilterComputePackFlush(-1, true);
        }
        sim.Evolve();
        double const final_energy = save(sim, "state.txt");
        AMREX_ALWAYS_ASSERT(std::abs(final_energy - initial_energy) <=
            1024 * std::numeric_limits<double>::epsilon() * initial_energy);
        bool check_rollback = false;
        amrex::ParmParse("test").query("rollback", check_rollback);
        if (check_rollback) { rollback(sim); }
        amrex::Print() << "ordinary native pressure_only energy audit passed\n";
        WarpX::ResetInstance();
    }
    warpx::initialization::finalize_external_libraries();
}
