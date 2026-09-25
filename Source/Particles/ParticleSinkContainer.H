
#ifndef WARPX_PARTICLE_SINK_H_
#define WARPX_PARTICLE_SINK_H_

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


class ParticleSinkContainer {
public:
    struct Sink {
        std::string name;
        ParticleBoundaryType type{ParticleBoundaryType::Absorbing};
#ifdef AMREX_USE_EB
        const amrex::EB2::IndexSpace* index_space{nullptr};
#endif
        ablastr::fields::MultiLevelScalarField  distance_field{nullptr};
    };
    void add_index_space (std::string name, const amrex::EB2::IndexSpace* is = nullptr);
    void add_sink (std::string name,
                   ParticleBoundaryType type = ParticleBoundaryType::Absorbing,
                   const amrex::EB2::IndexSpace* is = nullptr);
    void set_distance_field (const std::string& name,
                         ablastr::fields::MultiLevelScalarField field);
    [[nodiscard]] bool has_absorbing_sinks () const;

    [[nodiscard]] bool has_reflecting_or_thermal_sinks () const;

    // getters
    [[nodiscard]] amrex::Vector<const Sink*> get_active_sinks () const;

    [[nodiscard]] amrex::Vector<ablastr::fields::MultiLevelScalarField const*>
    get_active_distance_fields () const;

    [[nodiscard]] std::vector<ParticleBoundaryType> get_types () const;
    [[nodiscard]] std::vector<std::string> get_names () const;

    [[nodiscard]] Sink* get_sink (const std::string& name);
    [[nodiscard]] const Sink* get_sink (const std::string& name) const;

    [[nodiscard]] const std::vector<Sink>& get_sinks () const { return m_sinks; }
    [[nodiscard]] std::vector<Sink>& get_sinks () { return m_sinks; }

    [[nodiscard]] std::size_t size () const { return m_sinks.size(); }
    [[nodiscard]] bool empty () const { return m_sinks.empty(); }



private:
    std::vector<Sink> m_sinks;
};

#endif // WARPX_PARTICLE_SINK_H_
