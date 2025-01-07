#include "FlushFormatInSitu.H"

#include "WarpX.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <string>
#include <algorithm>

#if defined(AMREX_USE_CONDUIT) || defined(AMREX_USE_ASCENT)
    #include "conduit_cpp_to_c.hpp"
#endif

using namespace amrex;

#if defined(AMREX_USE_CONDUIT) || defined(AMREX_USE_ASCENT)
void
FlushFormatInSitu::WriteParticles(const amrex::Vector<ParticleDiag>& particle_diags, conduit::Node& a_bp_mesh) const
{
    WARPX_PROFILE("FlushFormatInSitu::WriteParticles()");

    // wrap particle data for each species
    // we prefix the fields with "particle_{species_name}" b/c we
    // want to to uniquely name all the fields that can be plotted

    for (unsigned i = 0, n = particle_diags.size(); i < n; ++i) {
        Vector<std::string> particle_varnames;
        Vector<std::string> particle_int_varnames;
        std::string prefix = "particle_" + particle_diags[i].getSpeciesName();

        // Get pc for species
        // auto& pc = mypc->GetParticleContainer(i);
        WarpXParticleContainer* pc = particle_diags[i].getParticleContainer();

        // get names of real comps
        std::map<std::string, int> real_comps_map = pc->getParticleComps();

        // WarpXParticleContainer compile-time extra AoS attributes (Real): 0
        // WarpXParticleContainer compile-time extra AoS attributes (int): 0

        // WarpXParticleContainer compile-time extra SoA attributes (Real): PIdx::nattribs
        // not an efficient search, but N is small...
        for(int j = 0; j < PIdx::nattribs; ++j)
        {
            auto rvn_it = real_comps_map.begin();
            for (; rvn_it != real_comps_map.end(); ++rvn_it)
                if (rvn_it->second == j)
                    break;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                rvn_it != real_comps_map.end(),
                "WarpX In Situ: SoA real attribute not found");
            std::string varname = rvn_it->first;
            particle_varnames.push_back(prefix + "_" + varname);
        }
        // WarpXParticleContainer compile-time extra SoA attributes (int): 0

        // WarpXParticleContainer "runtime" SoA attributes (Real), e.g QED: to do

        // wrap pc for current species into a blueprint topology
        amrex::ParticleContainerToBlueprint(*pc,
                                            particle_varnames,
                                            particle_int_varnames,
                                            a_bp_mesh,
                                            prefix);
    }
}
#endif // defined(AMREX_USE_CONDUIT) || defined(AMREX_USE_ASCENT)
