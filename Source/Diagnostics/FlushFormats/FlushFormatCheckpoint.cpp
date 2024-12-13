#include "FlushFormatCheckpoint.H"

#include "BoundaryConditions/PML.H"
#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
#   include "BoundaryConditions/PML_RZ.H"
#endif
#include "Diagnostics/ParticleDiag/ParticleDiag.H"
#include "Fields.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_MultiFab.H>
#include <AMReX_ParticleIO.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>

using namespace amrex;
using warpx::fields::FieldType;

namespace
{
    const std::string default_level_prefix {"Level_"};
}

void
FlushFormatCheckpoint::WriteToFile (
        const amrex::Vector<std::string>& /*varnames*/,
        const amrex::Vector<amrex::MultiFab>& /*mf*/,
        amrex::Vector<amrex::Geometry>& geom,
        const amrex::Vector<int> iteration, const double /*time*/,
        const amrex::Vector<ParticleDiag>& particle_diags, int nlev,
        const std::string prefix, int file_min_digits,
        bool /*plot_raw_fields*/,
        bool /*plot_raw_fields_guards*/,
        int verbose,
        const bool /*use_pinned_pc*/,
        bool /*isBTD*/, int /*snapshotID*/,
        int /*bufferID*/, int /*numBuffers*/,
        const amrex::Geometry& /*full_BTD_snapshot*/,
        bool /*isLastBTDFlush*/) const
{
    using ablastr::fields::Direction;

    WARPX_PROFILE("FlushFormatCheckpoint::WriteToFile()");

    auto & warpx = WarpX::GetInstance();

    const VisMF::Header::Version current_version = VisMF::GetHeaderVersion();
    VisMF::SetHeaderVersion(amrex::VisMF::Header::NoFabHeader_v1);

    const std::string& checkpointname = amrex::Concatenate(prefix, iteration[0], file_min_digits);

    if (verbose > 0) {
        amrex::Print() << Utils::TextMsg::Info(
            "Writing checkpoint " + checkpointname);
    }

    // const int nlevels = finestLevel()+1;
    amrex::PreBuildDirectorHierarchy(checkpointname, default_level_prefix, nlev, true);

    WriteWarpXHeader(checkpointname, geom);

    WriteJobInfo(checkpointname);

    for (int lev = 0; lev < nlev; ++lev)
    {
        VisMF::Write(*warpx.m_fields.get(FieldType::Efield_fp, Direction{0}, lev),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ex_fp"));
        VisMF::Write(*warpx.m_fields.get(FieldType::Efield_fp, Direction{1}, lev),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ey_fp"));
        VisMF::Write(*warpx.m_fields.get(FieldType::Efield_fp, Direction{2}, lev),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ez_fp"));
        VisMF::Write(*warpx.m_fields.get(FieldType::Bfield_fp, Direction{0}, lev),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Bx_fp"));
        VisMF::Write(*warpx.m_fields.get(FieldType::Bfield_fp, Direction{1}, lev),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "By_fp"));
        VisMF::Write(*warpx.m_fields.get(FieldType::Bfield_fp, Direction{2}, lev),
                     amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Bz_fp"));

        if (WarpX::fft_do_time_averaging)
        {
            VisMF::Write(*warpx.m_fields.get(FieldType::Efield_avg_fp, Direction{0}, lev),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ex_avg_fp"));
            VisMF::Write(*warpx.m_fields.get(FieldType::Efield_avg_fp, Direction{1}, lev),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ey_avg_fp"));
            VisMF::Write(*warpx.m_fields.get(FieldType::Efield_avg_fp, Direction{2}, lev),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ez_avg_fp"));

            VisMF::Write(*warpx.m_fields.get(FieldType::Bfield_avg_fp, Direction{0}, lev),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Bx_avg_fp"));
            VisMF::Write(*warpx.m_fields.get(FieldType::Bfield_avg_fp, Direction{1}, lev),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "By_avg_fp"));
            VisMF::Write(*warpx.m_fields.get(FieldType::Bfield_avg_fp, Direction{2}, lev),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Bz_avg_fp"));
        }

        if (warpx.getis_synchronized()) {
            // Need to save j if synchronized because after restart we need j to evolve E by dt/2.
            VisMF::Write(*warpx.m_fields.get(FieldType::current_fp, Direction{0}, lev),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "jx_fp"));
            VisMF::Write(*warpx.m_fields.get(FieldType::current_fp, Direction{1}, lev),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "jy_fp"));
            VisMF::Write(*warpx.m_fields.get(FieldType::current_fp, Direction{2}, lev),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "jz_fp"));
        }

        if (lev > 0)
        {
            VisMF::Write(*warpx.m_fields.get(FieldType::Efield_cp, Direction{0}, lev),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ex_cp"));
            VisMF::Write(*warpx.m_fields.get(FieldType::Efield_cp, Direction{1}, lev),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ey_cp"));
            VisMF::Write(*warpx.m_fields.get(FieldType::Efield_cp, Direction{2}, lev),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ez_cp"));
            VisMF::Write(*warpx.m_fields.get(FieldType::Bfield_cp, Direction{0}, lev),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Bx_cp"));
            VisMF::Write(*warpx.m_fields.get(FieldType::Bfield_cp, Direction{1}, lev),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "By_cp"));
            VisMF::Write(*warpx.m_fields.get(FieldType::Bfield_cp, Direction{2}, lev),
                         amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Bz_cp"));

            if (WarpX::fft_do_time_averaging)
            {
                VisMF::Write(*warpx.m_fields.get(FieldType::Efield_avg_cp, Direction{0}, lev),
                             amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ex_avg_cp"));
                VisMF::Write(*warpx.m_fields.get(FieldType::Efield_avg_cp, Direction{1}, lev),
                             amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ey_avg_cp"));
                VisMF::Write(*warpx.m_fields.get(FieldType::Efield_avg_cp, Direction{2}, lev),
                             amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Ez_avg_cp"));

                VisMF::Write(*warpx.m_fields.get(FieldType::Bfield_avg_cp, Direction{0}, lev),
                             amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Bx_avg_cp"));
                VisMF::Write(*warpx.m_fields.get(FieldType::Bfield_avg_cp, Direction{1}, lev),
                             amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "By_avg_cp"));
                VisMF::Write(*warpx.m_fields.get(FieldType::Bfield_avg_cp, Direction{2}, lev),
                             amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "Bz_avg_cp"));
            }

            if (warpx.getis_synchronized()) {
                // Need to save j if synchronized because after restart we need j to evolve E by dt/2.
                VisMF::Write(*warpx.m_fields.get(FieldType::current_cp, Direction{0}, lev),
                             amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "jx_cp"));
                VisMF::Write(*warpx.m_fields.get(FieldType::current_cp, Direction{1}, lev),
                             amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "jy_cp"));
                VisMF::Write(*warpx.m_fields.get(FieldType::current_cp, Direction{2}, lev),
                             amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "jz_cp"));
            }
        }

        if (warpx.DoPML()) {
            if (warpx.GetPML(lev)) {
                warpx.GetPML(lev)->CheckPoint(
                    warpx.m_fields,
                    amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "pml"));
            }
#if (defined WARPX_DIM_RZ) && (defined WARPX_USE_FFT)
            if (warpx.GetPML_RZ(lev)) {
                warpx.GetPML_RZ(lev)->CheckPoint(
                    warpx.m_fields,
                    amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "pml_rz"));
            }
#endif
        }
    }

    CheckpointParticles(checkpointname, particle_diags);

    WriteDMaps(checkpointname, nlev);

    VisMF::SetHeaderVersion(current_version);

}

void
FlushFormatCheckpoint::CheckpointParticles (
    const std::string& dir,
    const amrex::Vector<ParticleDiag>& particle_diags) const
{
    for (const auto& part_diag: particle_diags) {
        WarpXParticleContainer* pc = part_diag.getParticleContainer();

        Vector<std::string> real_names;
        Vector<std::string> int_names;

        // note: positions skipped here, since we reconstruct a plotfile SoA from them
        real_names.push_back("weight");
        real_names.push_back("momentum_x");
        real_names.push_back("momentum_y");
        real_names.push_back("momentum_z");

#ifdef WARPX_DIM_RZ
        real_names.push_back("theta");
#endif

        // get the names of the real comps
        //   note: skips the mandatory AMREX_SPACEDIM positions for pure SoA
        real_names.resize(pc->NumRealComps() - AMREX_SPACEDIM);
        auto runtime_rnames = pc->getParticleRuntimeComps();
        for (auto const& x : runtime_rnames) {
            real_names[x.second + PIdx::nattribs - AMREX_SPACEDIM] = x.first;
        }

        // and the int comps
        int_names.resize(pc->NumIntComps());
        auto runtime_inames = pc->getParticleRuntimeiComps();
        for (auto const& x : runtime_inames) { int_names[x.second+0] = x.first; }

        pc->Checkpoint(dir, part_diag.getSpeciesName(), true,
                       real_names, int_names);
    }
}

void
FlushFormatCheckpoint::WriteDMaps (const std::string& dir, int nlev) const
{
    if (ParallelDescriptor::IOProcessor()) {
        auto & warpx = WarpX::GetInstance();
        for (int lev = 0; lev < nlev; ++lev) {
            std::string DMFileName = dir;
            if (!DMFileName.empty() && DMFileName[DMFileName.size()-1] != '/') {DMFileName += '/';}
            DMFileName = amrex::Concatenate(DMFileName.append("Level_"), lev, 1);
            DMFileName += "/DM";

            std::ofstream DMFile;
            DMFile.open(DMFileName.c_str(), std::ios::out|std::ios::trunc);

            if (!DMFile.good()) { amrex::FileOpenFailed(DMFileName); }

            DMFile << ParallelDescriptor::NProcs() << "\n";
            warpx.DistributionMap(lev).writeOn(DMFile);

            DMFile.flush();
            DMFile.close();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                DMFile.good(),
                "FlushFormatCheckpoint::WriteDMaps: problem writing DMFile"
            );
        }
    }
}
