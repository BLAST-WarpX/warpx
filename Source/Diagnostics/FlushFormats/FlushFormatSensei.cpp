#include "Utils/WarpXProfilerWrapper.H"
#include "FlushFormatSensei.H"

#include "WarpX.H"

#ifdef AMREX_USE_SENSEI_INSITU
# include <AMReX_AmrMeshParticleInSituBridge.H>
#endif

FlushFormatSensei::FlushFormatSensei (amrex::AmrMesh *amr_mesh,
    const std::string& diag_name) :
    m_amr_mesh(amr_mesh)
{
#ifndef AMREX_USE_SENSEI_INSITU
    amrex::ignore_unused(m_insitu_pin_mesh, m_insitu_bridge, m_amr_mesh, diag_name);
#else
    amrex::ParmParse pp_diag_name(diag_name);

    pp_diag_name.query("sensei_config", m_insitu_config);
    pp_diag_name.query("sensei_pin_mesh", m_insitu_pin_mesh);

    m_insitu_bridge = new amrex::AmrMeshParticleInSituBridge;
    m_insitu_bridge->setEnabled(true);
    m_insitu_bridge->setConfig(m_insitu_config);
    m_insitu_bridge->setPinMesh(m_insitu_pin_mesh);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_amr_mesh && !m_insitu_bridge->initialize(),
        "FlushFormtSensei::FlushFormatSensei : "
        "Failed to initialize the in situ bridge."
    );
    m_insitu_bridge->setFrequency(1);
#endif
}

#ifdef AMREX_USE_SENSEI_INSITU
FlushFormatSensei::~FlushFormatSensei ()
{
    delete m_insitu_bridge;
}
#else
FlushFormatSensei::~FlushFormatSensei () = default;
#endif

void
FlushFormatSensei::WriteToFile (
    const amrex::Vector<std::string>& varnames,
    const amrex::Vector<amrex::MultiFab>& mf,
    amrex::Vector<amrex::Geometry>& geom,
    const amrex::Vector<int> iteration, const double time,
    const amrex::Vector<ParticleDiag>& particle_diags,
    int nlev, const std::string prefix, int file_min_digits,
    bool plot_raw_fields, bool plot_raw_fields_guards,
    int verbose, const bool use_pinned_pc,
    bool isBTD, int /*snapshotID*/, int /*bufferID*/, int /*numBuffers*/,
    const amrex::Geometry& /*full_BTD_snapshot*/, bool /*isLastBTDFlush*/) const
{
    amrex::ignore_unused(
        geom, nlev, prefix, file_min_digits,
        plot_raw_fields, plot_raw_fields_guards,
        use_pinned_pc
    );

#ifndef AMREX_USE_SENSEI_INSITU
    amrex::ignore_unused(varnames, mf, iteration, time, particle_diags,
                         verbose, isBTD);
#else
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !isBTD,
        "In-situ visualization is not currently supported for back-transformed diagnostics.");

    WARPX_PROFILE("FlushFormatSensei::WriteToFile()");
    const std::string& filename = amrex::Concatenate(prefix, iteration[0], file_min_digits);
    if (verbose > 0) {
        amrex::Print() << Utils::TextMsg::Info("Writing Sensei file " + filename);
    }

    amrex::Vector<amrex::MultiFab> *mf_ptr =
        const_cast<amrex::Vector<amrex::MultiFab>*>(&mf);

    auto particles = particle_diags[0].getParticleContainer();
    bool didUpdate = m_insitu_bridge->update(
        iteration[0], time, m_amr_mesh,{mf_ptr}, {varnames},
        particles, {}, {}, {{"u",{0,1,2}}}, {});

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !didUpdate,
        "FlushFormatSensei::WriteToFile : "
        "Failed to update the in situ bridge."
    );
#endif
}

void
FlushFormatSensei::WriteParticles (
    const amrex::Vector<ParticleDiag>& particle_diags) const
{
    amrex::ignore_unused(particle_diags);
#ifdef AMREX_USE_SENSEI_INSITU
    WARPX_ABORT_WITH_MESSAGE(
        "FlushFormatSensei::WriteParticles : "
        "Not yet implemented."
    );
#endif
}
