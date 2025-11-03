/* Copyright 2019 Andrew Myers, Axel Huebl, David Grote
 * Luca Fedeli, Maxence Thevenet, Revathi Jambunathan
 * Weiqun Zhang, levinem, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Fields.H"
#include "Particles/ParticleIO.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/LaserParticleContainer.H"
#include "Particles/RigidInjectedParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include "ablastr/fields/MultiFabRegister.H"
#include <ablastr/utils/text/StreamUtils.H>

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Config.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParticleIO.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

using namespace amrex;
using warpx::fields::FieldType;

void
LaserParticleContainer::ReadHeader (std::istream& is)
{
    if (do_continuous_injection) {
        m_updated_position.resize(3);
        for (int i = 0; i < 3; ++i) {
            is >> m_updated_position[i];
            ablastr::utils::text::goto_next_line(is);
        }
    }
}

void
LaserParticleContainer::WriteHeader (std::ostream& os) const
{
    if (do_continuous_injection) {
        for (int i = 0; i < 3; ++i) {
            os << m_updated_position[i] << "\n";
        }
    }
}

void
RigidInjectedParticleContainer::ReadHeader (std::istream& is)
{
    // Call parent class
    PhysicalParticleContainer::ReadHeader( is );

    // Read quantities that are specific to rigid-injected species
    int nlevs;
    is >> nlevs;
    ablastr::utils::text::goto_next_line(is);

    AMREX_ASSERT(zinject_plane_levels.size() == 0);

    for (int i = 0; i < nlevs; ++i)
    {
        amrex::Real zinject_plane_tmp;
        is >> zinject_plane_tmp;
        zinject_plane_levels.push_back(zinject_plane_tmp);
        ablastr::utils::text::goto_next_line(is);
    }
    is >> vzbeam_ave_boosted;
    ablastr::utils::text::goto_next_line(is);
}

void
RigidInjectedParticleContainer::WriteHeader (std::ostream& os) const
{
    // Call parent class
    PhysicalParticleContainer::WriteHeader( os );

    // Write quantities that are specific to the rigid-injected species
    const auto nlevs = static_cast<int>(zinject_plane_levels.size());
    os << nlevs << "\n";
    for (int i = 0; i < nlevs; ++i)
    {
        os << zinject_plane_levels[i] << "\n";
    }
    os << vzbeam_ave_boosted << "\n";
}

void
PhysicalParticleContainer::ReadHeader (std::istream& is)
{
    is >> charge >> m_mass;
    ablastr::utils::text::goto_next_line(is);
}

void
PhysicalParticleContainer::WriteHeader (std::ostream& os) const
{
    // no need to write species_id
    os << charge << " " << m_mass << "\n";
}

void
MultiParticleContainer::Restart (const std::string& dir)
{
    // note: all containers is sorted like this
    // - species_names
    // - lasers_names
    // we don't need to read back the laser particle charge/mass
    for (unsigned i = 0, n = species_names.size(); i < n; ++i) {
        WarpXParticleContainer* pc = allcontainers.at(i).get();
        const std::string header_fn = dir + "/" + species_names[i] + "/Header";

        Vector<char> fileCharPtr;
        ParallelDescriptor::ReadAndBcastFile(header_fn, fileCharPtr);
        const std::string fileCharPtrString(fileCharPtr.dataPtr());
        std::istringstream is(fileCharPtrString, std::istringstream::in);
        is.exceptions(std::ios_base::failbit | std::ios_base::badbit);

        std::string line, word;

        std::getline(is, line); // Version
        std::getline(is, line); // SpaceDim

        int nr;
        is >> nr;

        std::vector<std::string> real_comp_names;
        for (int j = 0; j < nr; ++j) {
            std::string comp_name;
            is >> comp_name;
            real_comp_names.push_back(comp_name);
        }

        int n_rc = 0;
        for (auto const& comp : pc->GetRealSoANames()) {
            // skip compile-time components
            if (n_rc < WarpXParticleContainer::NArrayReal) { continue; }
            n_rc++;

            auto search = std::find(real_comp_names.begin(), real_comp_names.end(), comp);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                search != real_comp_names.end(),
                "Species " + species_names[i]
                + " needs runtime real component " +  comp
                + ", but it was not found in the checkpoint file."
            );
        }

        for (int j = PIdx::nattribs-AMREX_SPACEDIM; j < nr; ++j) {
            const auto& comp_name = real_comp_names[j];
            if (!pc->HasRealComp(comp_name)) {
                amrex::Print() << Utils::TextMsg::Info(
                    "Runtime real component " + comp_name
                    + " was found in the checkpoint file, but it has not been added yet. "
                    + " Adding it now."
                );
                pc->AddRealComp(comp_name);
            }
        }

        int ni;
        is >> ni;

        std::vector<std::string> int_comp_names;
        for (int j = 0; j < ni; ++j) {
            std::string comp_name;
            is >> comp_name;
            int_comp_names.push_back(comp_name);
        }

        int n_ic = 0;
        for (auto const& comp : pc->GetIntSoANames()) {
            // skip compile-time components
            if (n_ic < WarpXParticleContainer::NArrayInt) { continue; }
            n_ic++;

            auto search = std::find(int_comp_names.begin(), int_comp_names.end(), comp);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                search != int_comp_names.end(),
                "Species " + species_names[i] + " needs runtime int component " + comp
                + ", but it was not found in the checkpoint file."
            );
        }

        for (int j = 0; j < ni; ++j) {
            const auto& comp_name = int_comp_names[j];
            if (!pc->HasIntComp(comp_name)) {
                amrex::Print()<< Utils::TextMsg::Info(
                    "Runtime int component " + comp_name
                    + " was found in the checkpoint file, but it has not been added yet. "
                    + " Adding it now."
                );
                pc->AddIntComp(comp_name);
            }
        }

        pc->Restart(dir, species_names.at(i));
    }
    for (unsigned i = species_names.size(); i < species_names.size()+lasers_names.size(); ++i) {
        allcontainers.at(i)->Restart(dir, lasers_names.at(i-species_names.size()));
    }
}

void
MultiParticleContainer::ReadHeader (std::istream& is)
{
    // note: all containers is sorted like this
    // - species_names
    // - lasers_names
    for (unsigned i = 0, n = species_names.size()+lasers_names.size(); i < n; ++i) {
        allcontainers.at(i)->ReadHeader(is);
    }
}

void
MultiParticleContainer::WriteHeader (std::ostream& os) const
{
    // note: all containers is sorted like this
    // - species_names
    // - lasers_names
    for (unsigned i = 0, n = species_names.size()+lasers_names.size(); i < n; ++i) {
        allcontainers.at(i)->WriteHeader(os);
    }
}

void
storePhiOnParticles ( PinnedMemoryParticleContainer& tmp,
    ElectrostaticSolverAlgo electrostatic_solver_id, bool is_full_diagnostic ) {

    using PinnedParIter = typename PinnedMemoryParticleContainer::ParIterType;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrame) ||
        (electrostatic_solver_id == ElectrostaticSolverAlgo::LabFrameElectroMagnetostatic),
        "Output of the electrostatic potential (phi) on the particles was requested, "
        "but this is only available for `warpx.do_electrostatic=labframe` or `labframe-electromagnetostatic`.");
    // When this is not a full diagnostic, the particles are not written at the same physical time (i.e. PIC iteration)
    // that they were collected. This happens for diagnostics that use buffering (e.g. BackTransformed, BoundaryScraping).
    // Here `phi` is gathered at the iteration when particles are written (not collected) and is thus mismatched.
    // To avoid confusion, we raise an error in this case.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        is_full_diagnostic,
        "Output of the electrostatic potential (phi) on the particles was requested, "
        "but this is only available with `diag_type = Full`.");
    tmp.AddRealComp("phi");
    int const phi_index = tmp.GetRealCompIndex("phi");
    auto& warpx = WarpX::GetInstance();
    for (int lev=0; lev<=warpx.finestLevel(); lev++) {
        const amrex::Geometry& geom = warpx.Geom(lev);
        auto plo = geom.ProbLoArray();
        auto dxi = geom.InvCellSizeArray();
        amrex::MultiFab const& phi = *warpx.m_fields.get(FieldType::phi_fp, lev);

#ifdef AMREX_USE_OMP
        #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (PinnedParIter pti(tmp, lev); pti.isValid(); ++pti) {

            auto phi_grid = phi[pti].array();
            const auto getPosition = GetParticlePosition<PIdx>(pti);
            amrex::ParticleReal* phi_particle_arr = pti.GetStructOfArrays().GetRealData(phi_index).dataPtr();

            // Loop over the particles and update their position
            amrex::ParallelFor( pti.numParticles(),
                [=] AMREX_GPU_DEVICE (long ip) {

                    amrex::ParticleReal xp, yp, zp;
                    getPosition(ip, xp, yp, zp);
                    int i, j, k;
                    amrex::Real W[AMREX_SPACEDIM][2];
                    ablastr::particles::compute_weights<amrex::IndexType::NODE>(
                        xp, yp, zp, plo, dxi, i, j, k, W);
                    amrex::Real const phi_value  = ablastr::particles::interp_field_nodal(i, j, k, W, phi_grid);
                    phi_particle_arr[ip] = phi_value;
                }
            );
        }
    }
}
