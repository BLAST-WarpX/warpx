/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Radiation/RadiationTransport.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

/** Private cooling-benchmark driver. The standard application's checkpoint-dt
 * semantics are unchanged. Only stationary coupled radiation may explicitly
 * recompute its constant timestep after loading a nonzero-step checkpoint. */
int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& simulation = WarpX::GetInstance();
        simulation.InitData();
        AMREX_ALWAYS_ASSERT(simulation.getistep(0) > 0 && simulation.finestLevel() == 0 &&
                            simulation.GetRadiationTransport().commitsHybridMaterialState());
        for (int species = 0; species < simulation.GetPartContainer().nSpecies(); ++species)
        {
            AMREX_ALWAYS_ASSERT(
                simulation.GetPartContainer().GetParticleContainer(species).doNotPush());
        }
        auto const restored = simulation.getdt(0);
        bool keep_first_restart_step = false;
        amrex::ParmParse benchmark("benchmark");
        benchmark.query("keep_first_restart_step", keep_first_restart_step);
        if (keep_first_restart_step)
        {
            simulation.Evolve(1);
        }
        simulation.ComputeDt();
        amrex::Print() << "Stationary benchmark explicit timestep recomputation: checkpoint_dt="
                       << restored << " configured_dt=" << simulation.getdt(0) << '\n';
        simulation.Evolve();
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
