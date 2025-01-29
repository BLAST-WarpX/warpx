#ifndef WARPX_PARTICLEDIAG_H_
#define WARPX_PARTICLEDIAG_H_

#include "ParticleDiag_fwd.H"

#include "Particles/WarpXParticleContainer_fwd.H"
#include "Particles/PinnedMemoryParticleContainer.H"

#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_Vector.H>

#include <memory>
#include <string>

class ParticleDiag
{
public:
    ParticleDiag(const std::string& diag_name, const std::string& name,
        WarpXParticleContainer* pc, PinnedMemoryParticleContainer *pinned_pc = nullptr);
    [[nodiscard]] WarpXParticleContainer* getParticleContainer() const { return m_pc; }
    [[nodiscard]] PinnedMemoryParticleContainer* getPinnedParticleContainer() const { return m_pinned_pc; }
    [[nodiscard]] std::string getSpeciesName() const { return m_name; }
    amrex::Vector<int> m_plot_flags;
    bool m_plot_phi = false; // Whether to output the potential phi on the particles

    bool m_do_random_filter  = false;
    bool m_do_uniform_filter = false;
    bool m_do_parser_filter  = false;
    bool m_do_geom_filter    = false;
    amrex::Real m_random_fraction = 1.0;
    int m_uniform_stride = 1;
    static constexpr int m_nvars = 7; // t, x, y, z, ux, uy, uz
    std::unique_ptr<amrex::Parser> m_particle_filter_parser;
    amrex::RealBox m_diag_domain;

private:
    std::string m_diag_name;
    std::string m_name;
    WarpXParticleContainer* m_pc;
    PinnedMemoryParticleContainer* m_pinned_pc;
};

#endif // WARPX_PARTICLEDIAG_H_
