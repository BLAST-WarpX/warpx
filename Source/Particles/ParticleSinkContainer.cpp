#include "ParticleSinkContainer.H"

#include "Utils/WarpXAlgorithmSelection.H"

#include <ablastr/fields/MultiFabRegister.H>

#ifdef AMREX_USE_EB
#   include <AMReX_Config.H>
#   include <AMReX_EB2.H>
#endif

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#ifdef AMREX_USE_EB

void ParticleSinkContainer::add_index_space (std::string name,
                                             const amrex::EB2::IndexSpace* is)
{
    Sink sink;
    sink.name = std::move(name);
    sink.type = ParticleBoundaryType::Absorbing;
    sink.index_space = is;
    m_sinks.push_back(std::move(sink));
}

void ParticleSinkContainer::add_sink (std::string name,
               ParticleBoundaryType type,
               const amrex::EB2::IndexSpace* is)
{
    Sink sink;
    sink.name = std::move(name);
    sink.type = std::move(type);
    sink.index_space = is;
    m_sinks.push_back(std::move(sink));
}

#endif

amrex::Vector<const ParticleSinkContainer::Sink*> ParticleSinkContainer::get_active_sinks () const
{
    amrex::Vector<const Sink*> active_sinks;
    for (const auto& sink : m_sinks) {
        if (sink.type != ParticleBoundaryType::None) {
            active_sinks.push_back(&sink);
        }
    }
    return active_sinks;
}

amrex::Vector<ablastr::fields::MultiLevelScalarField const*>
ParticleSinkContainer::get_active_distance_fields () const
{
    amrex::Vector<ablastr::fields::MultiLevelScalarField const*> sink_ptrs;
    for (const auto& sink : m_sinks) {
        if (sink.type != ParticleBoundaryType::None) {
            sink_ptrs.push_back(&sink.distance_field);
        }
    }
    return sink_ptrs;
}

bool ParticleSinkContainer::has_absorbing_sinks () const
{
    return std::ranges::any_of(m_sinks,
                               [](const Sink& sink) {
                                   return sink.type == ParticleBoundaryType::Absorbing;
                               });
}

bool ParticleSinkContainer::has_reflecting_or_thermal_sinks () const
{
    return std::ranges::any_of(m_sinks,
                               [](const Sink& sink) {
                                   return sink.type == ParticleBoundaryType::Reflecting ||
                                          sink.type == ParticleBoundaryType::Thermal;
                               });
}

std::vector<std::string> ParticleSinkContainer::get_names () const
{
    std::vector<std::string> names;
    names.reserve(m_sinks.size());
    for (const auto& sink : m_sinks) {
        names.push_back(sink.name);
    }
    return names;
}

std::vector<ParticleBoundaryType> ParticleSinkContainer::get_types () const
{
    std::vector<ParticleBoundaryType> types;
    types.reserve(m_sinks.size());
    for (const auto& sink : m_sinks) {
        types.push_back(sink.type);
    }
    return types;
}

// Non-const lookup for modifying the sink
ParticleSinkContainer::Sink* ParticleSinkContainer::get_sink (const std::string& name)
{
    auto it = std::find_if(m_sinks.begin(), m_sinks.end(),
        [&name](const Sink& s) { return s.name == name; });
    return (it != m_sinks.end()) ? &(*it) : nullptr;
}

// Const lookup for read-only access
const ParticleSinkContainer::Sink* ParticleSinkContainer::get_sink (const std::string& name) const
{
    auto it = std::find_if(m_sinks.begin(), m_sinks.end(),
        [&name](const Sink& s) { return s.name == name; });
    return (it != m_sinks.end()) ? &(*it) : nullptr;
}



// Setter method using get_sink
void ParticleSinkContainer::set_distance_field (const std::string& name,
                                                 ablastr::fields::MultiLevelScalarField field)
{
    if (auto* sink = get_sink(name)) {
        sink->distance_field = std::move(field);
    } else {
        amrex::Abort("ParticleSinkContainer::set_distance_field: Sink '" + name + "' not found!");
    }
}
