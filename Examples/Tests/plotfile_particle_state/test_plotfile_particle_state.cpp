/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Diagnostics/FlushFormats/FlushFormatPlotfile.H"
#include "Diagnostics/ParticleDiag/ParticleDiag.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_Parser.H>
#include <AMReX_Print.H>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace
{
    std::vector<amrex::ParticleReal> Snapshot (WarpXParticleContainer const& particles)
    {
        std::vector<amrex::ParticleReal> result;
        for (auto const& [key, tile] : particles.GetParticles(0)) {
            amrex::ignore_unused(key);
            auto const count = tile.numParticles();
            for (int component = 0; component < particles.NumRealComps(); ++component) {
                auto const& values = tile.GetStructOfArrays().GetRealData(component);
                auto const offset = result.size();
                result.resize(offset + count);
                amrex::Gpu::copy(amrex::Gpu::deviceToHost, values.begin(),
                    values.begin() + count, result.data() + offset);
            }
        }
        return result;
    }
}

int main (int argc, char* argv[])
try
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& simulation = WarpX::GetInstance();
        auto& ions = simulation.GetPartContainer().GetParticleContainerFromName("ions");
        ions.AddRealComp("sentinel");
        simulation.InitData();
        auto const sentinel_index = ions.GetRealCompIndex("sentinel");
        for (WarpXParIter iterator(ions, 0); iterator.isValid(); ++iterator) {
            auto& soa = iterator.GetStructOfArrays();
            auto* ux = soa.GetRealData(PIdx::ux).data();
            auto* uy = soa.GetRealData(PIdx::uy).data();
            auto* uz = soa.GetRealData(PIdx::uz).data();
            auto* sentinel = soa.GetRealData(sentinel_index).data();
            amrex::ParallelFor(iterator.numParticles(), [=] AMREX_GPU_DEVICE(long ip) {
                ux[ip] = static_cast<amrex::ParticleReal>(12345.6789123 + ip * 0.314159);
                uy[ip] = static_cast<amrex::ParticleReal>(-87654.32198 - ip * 0.192837);
                uz[ip] = static_cast<amrex::ParticleReal>(
                    (ip % 2 ? -1 : 1) * (90000.1234 + ip * 0.271828));
                sentinel[ip] = static_cast<amrex::ParticleReal>((ip + 1) * 1.e-30);
            });
        }
        auto const before = Snapshot(ions);
        AMREX_ALWAYS_ASSERT(!before.empty());
        amrex::Vector<ParticleDiag> diagnostics;
        diagnostics.emplace_back("readonly", "ions", &ions);
        diagnostics[0].m_particle_filter_parser =
            std::make_unique<amrex::Parser>("(uz>0)*(uz<0.001)");
        diagnostics[0].m_particle_filter_parser->registerVariables(
            {"t", "x", "y", "z", "ux", "uy", "uz"});
        amrex::Vector<amrex::MultiFab> fields(1);
        fields[0].define(ions.ParticleBoxArray(0), ions.ParticleDistributionMap(0), 1, 0);
        fields[0].setVal(0);
        amrex::Vector<amrex::Geometry> geometry{simulation.Geom(0)};
        FlushFormatPlotfile const writer;
        for (int mode = 0; mode < 3; ++mode) {
            diagnostics[0].m_do_uniform_filter = mode == 1;
            diagnostics[0].m_uniform_stride = 2;
            diagnostics[0].m_do_parser_filter = mode == 2;
            writer.WriteToFile({"rho"}, fields, geometry, {0}, 0, diagnostics, 1,
                "readonly_" + std::to_string(mode) + "_", 6, false, false, 0);
            auto const after = Snapshot(ions);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(before.size() == after.size() &&
                std::memcmp(before.data(), after.data(),
                    before.size() * sizeof(amrex::ParticleReal)) == 0,
                "Plotfile output must not modify any live particle real attribute.");
        }
        amrex::Print() << "Plotfile writing preserves live particle state exactly\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
catch (...)
{
    amrex::Abort("Unhandled exception in plotfile particle-state regression");
    return 1;
}
