#include "WarpX.H"

#include "Particles/ParticleSinkContainer.H"
#include "EmbeddedBoundary/DistanceToEB.H"

#include <AMReX_ParmParse.H>
#ifdef AMREX_USE_EB
#   include <AMReX_Config.H>
#   include <AMReX_EB2.H>
#endif

void WarpX::InitParticleSinks ()
{
#ifdef AMREX_USE_EB
    const amrex::ParmParse pp_sinks("particle_sinks");
    std::vector<std::string> sink_names;
    pp_sinks.queryarr("names", sink_names);
    if (!sink_names.empty()) {
        if (!m_particle_sinks) {
            m_particle_sinks = std::make_unique<ParticleSinkContainer>();
        }
        ParticleBoundaryType type = ParticleBoundaryType::Absorbing;
        for (const auto& sink_name : sink_names) {
            const amrex::ParmParse pp_sink(sink_name);
            pp_sink.query_enum_case_insensitive("type", type);
            InitEB(sink_name);
            m_particle_sinks->add_sink(sink_name, type, &amrex::EB2::IndexSpace::top());
        }
    }
#endif
}

void WarpX::AllocateParticleSinkFields (int lev,
                                        const amrex::BoxArray& ba_node,
                                        const amrex::DistributionMapping& dm,
                                        int nc_ls,
                                        const amrex::IntVect& ng_ls,
                                        amrex::Real val)
{
    for (const auto& sink_name : m_particle_sinks->get_names()) {
        std::string const field_name = "distance_to_" + sink_name;
        m_fields.alloc_init(field_name, lev, ba_node, dm, nc_ls, ng_ls, val);
    }
}

void WarpX::ComputeDistanceToParticleSinks ()
{
#ifdef AMREX_USE_EB
    if (!m_particle_sinks || m_particle_sinks->empty()) { return; }

    for (auto& sink : m_particle_sinks->get_sinks()) {
        if (sink.index_space != nullptr) {
            std::string const field_name = "distance_to_" + sink.name;

            // 1. Push index space and compute EB distance field (registers into m_fields)
            amrex::EB2::IndexSpace::push(const_cast<amrex::EB2::IndexSpace*>(sink.index_space));
            ComputeDistanceToEB(field_name);
            amrex::EB2::IndexSpace::pop();
            ablastr::fields::MultiLevelScalarField dist_field_ptrs =
                m_fields.get_mr_levels(field_name, finestLevel());
            m_particle_sinks->set_distance_field(sink.name, dist_field_ptrs);
        }
    }
#endif
}
