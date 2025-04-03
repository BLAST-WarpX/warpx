/* Copyright 2019 Andrew Myers, Axel Huebl, David Grote
 * Luca Fedeli, Maxence Thevenet, Revathi Jambunathan
 * Weiqun Zhang, levinem, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Particles/ParticleIO.H"

#include "Fields.H"
#include "Particles/Gather/GetExternalFields.H"
#include "Particles/LaserParticleContainer.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/RigidInjectedParticleContainer.H"
#include "Particles/SpeciesPhysicalProperties.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

#include <ablastr/utils/text/StreamUtils.H>

#include <AMReX_BLassert.H>
#include <AMReX_Config.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParticleIO.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <array>
#include <istream>
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
    is >> charge >> mass;
    ablastr::utils::text::goto_next_line(is);
}

void
PhysicalParticleContainer::WriteHeader (std::ostream& os) const
{
    // no need to write species_id
    os << charge << " " << mass << "\n";
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

void
storeEMFieldsOnParticles (PinnedMemoryParticleContainer& tmp,
    ElectromagneticSolverAlgo electromagnetic_solver_id, const bool fields_to_plot[], bool is_full_diagnostic) {

    using PinnedParIter = typename PinnedMemoryParticleContainer::ParIterType;
    using Dir = ablastr::fields::Direction;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        electromagnetic_solver_id != ElectromagneticSolverAlgo::None,
        "output of the electromagnetic fields on the particles was requested, "
        "but this is only available with an electromagnetic solver.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        is_full_diagnostic,
        "Output of the electromagnetic fields on the particles was requested, "
        "but this is only available with `diag_type = Full`.");

    auto& warpx = WarpX::GetInstance();

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        warpx.finestLevel() ==0,
        "output of the electromagnetic fields on particles only works without mesh refinement"
    );

    constexpr auto lev0=0;
    const amrex::XDim3 dinv = WarpX::InvCellSize(lev0);
    const bool galerkin_interpolation = WarpX::galerkin_interpolation;
    const int nox = WarpX::nox;
    const int n_rz_azimuthal_modes = WarpX::n_rz_azimuthal_modes;

    // need to do that for constant expression for compilation
    void (*gatherE)(amrex::ParticleReal, amrex::ParticleReal, amrex::ParticleReal,
                    amrex::ParticleReal&, amrex::ParticleReal&, amrex::ParticleReal&,
                    const amrex::Array4<const double>&, const amrex::Array4<const double>&, const amrex::Array4<const double>&,
                    amrex::IndexType, amrex::IndexType, amrex::IndexType,
                    const amrex::XDim3&, const amrex::XDim3&, const amrex::Dim3&, int);

    void (*gatherB)(amrex::ParticleReal, amrex::ParticleReal, amrex::ParticleReal,
                    amrex::ParticleReal&, amrex::ParticleReal&, amrex::ParticleReal&,
                    const amrex::Array4<const double>&, const amrex::Array4<const double>&, const amrex::Array4<const double>&,
                    amrex::IndexType, amrex::IndexType, amrex::IndexType,
                    const amrex::XDim3&, const amrex::XDim3&, const amrex::Dim3&, int);

    if (galerkin_interpolation) {
        if (nox == 1) {
            gatherE = doDirectGatherVectorField<1,0>;
            gatherB = doDirectGatherVectorField<0,1>;
        } else if (nox == 2) {
            gatherE = doDirectGatherVectorField<2,1>;
            gatherB = doDirectGatherVectorField<1,2>;
        } else if (nox == 3) {
            gatherE = doDirectGatherVectorField<3,2>;
            gatherB = doDirectGatherVectorField<2,3>;
        } else { // if (nox == 4) {
            gatherE = doDirectGatherVectorField<4,3>;
            gatherB = doDirectGatherVectorField<3,4>;
        }
    } else {
        if (nox == 1) {
            gatherE = doDirectGatherVectorField<1,1>;
            gatherB = doDirectGatherVectorField<1,1>;
        } else if (nox == 2) {
            gatherE = doDirectGatherVectorField<2,2>;
            gatherB = doDirectGatherVectorField<2,2>;
        } else if (nox == 3) {
            gatherE = doDirectGatherVectorField<3,3>;
            gatherB = doDirectGatherVectorField<3,3>;
        } else { // if (nox == 4) {
            gatherE = doDirectGatherVectorField<4,4>;
            gatherB = doDirectGatherVectorField<4,4>;
        }
    }

    auto fields_names = amrex::Array<std::string, 6>{
        "Ex", "Ey", "Ez", "Bx", "By", "Bz"};

    auto fields_index = amrex::Array<int, 6>{0,0,0,0,0,0};

    enum Ex_flags { doEx, noEx };
    enum Ey_flags { doEy, noEy };
    enum Ez_flags { doEz, noEz };
    enum Bx_flags { doBx, noBx };
    enum By_flags { doBy, noBy };
    enum Bz_flags { doBz, noBz };

    const auto Ex_runtime_flag = (fields_to_plot[0]) ? doEx : noEx;
    const auto Ey_runtime_flag = (fields_to_plot[1]) ? doEy : noEy;
    const auto Ez_runtime_flag = (fields_to_plot[2]) ? doEz : noEz;
    const auto Bx_runtime_flag = (fields_to_plot[3]) ? doBx : noBx;
    const auto By_runtime_flag = (fields_to_plot[4]) ? doBy : noBy;
    const auto Bz_runtime_flag = (fields_to_plot[5]) ? doBz : noBz;

    for (int i = 0; i < 6; i++){
        if (fields_to_plot[i]){
            tmp.AddRealComp(fields_names[i]);
            fields_index[i] = tmp.GetRealCompIndex(fields_names[i]);
        }
    }

    amrex::MultiFab const& Ex = *warpx.m_fields.get(FieldType::Efield_aux, Dir{0}, lev0);
    amrex::MultiFab const& Ey = *warpx.m_fields.get(FieldType::Efield_aux, Dir{1}, lev0);
    amrex::MultiFab const& Ez = *warpx.m_fields.get(FieldType::Efield_aux, Dir{2}, lev0);
    amrex::MultiFab const& Bx = *warpx.m_fields.get(FieldType::Bfield_aux, Dir{0}, lev0);
    amrex::MultiFab const& By = *warpx.m_fields.get(FieldType::Bfield_aux, Dir{1}, lev0);
    amrex::MultiFab const& Bz = *warpx.m_fields.get(FieldType::Bfield_aux, Dir{2}, lev0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (PinnedParIter pti(tmp, lev0); pti.isValid(); ++pti) {

        const auto Ex_grid = Ex[pti].array();
        const auto Ey_grid = Ey[pti].array();
        const auto Ez_grid = Ez[pti].array();
        const auto Bx_grid = Bx[pti].array();
        const auto By_grid = By[pti].array();
        const auto Bz_grid = Bz[pti].array();

        const auto ex_type = Ex.ixType();
        const auto ey_type = Ey.ixType();
        const auto ez_type = Ez.ixType();
        const auto bx_type = Bx.ixType();
        const auto by_type = By.ixType();
        const auto bz_type = Bz.ixType();

        const auto getPosition = GetParticlePosition<PIdx>(pti);

        // const auto getExternalEB = GetExternalEBField(a_pti, lev0);
        // a_pti is a WarpXParIter, currently undefined

        amrex::ParticleReal* Ex_particle_arr = (fields_to_plot[0]) ? pti.GetStructOfArrays().GetRealData(fields_index[0]).dataPtr() : nullptr;
        amrex::ParticleReal* Ey_particle_arr = (fields_to_plot[1]) ? pti.GetStructOfArrays().GetRealData(fields_index[1]).dataPtr() : nullptr;
        amrex::ParticleReal* Ez_particle_arr = (fields_to_plot[2]) ? pti.GetStructOfArrays().GetRealData(fields_index[2]).dataPtr() : nullptr;
        amrex::ParticleReal* Bx_particle_arr = (fields_to_plot[3]) ? pti.GetStructOfArrays().GetRealData(fields_index[3]).dataPtr() : nullptr;
        amrex::ParticleReal* By_particle_arr = (fields_to_plot[4]) ? pti.GetStructOfArrays().GetRealData(fields_index[4]).dataPtr() : nullptr;
        amrex::ParticleReal* Bz_particle_arr = (fields_to_plot[5]) ? pti.GetStructOfArrays().GetRealData(fields_index[5]).dataPtr() : nullptr;

        const auto box = pti.tilebox();
        const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, lev0, 0._rt);
        const Dim3 lo = lbound(box);

        // Loop over the particles and compute the EM field using the doGatherShapeN function
        amrex::ParallelFor(
            TypeList<CompileTimeOptions<doEx, noEx>, CompileTimeOptions<doEy, noEy>, CompileTimeOptions<doEz, noEz>,
            CompileTimeOptions<doBx, noBx>, CompileTimeOptions<doBy, noBy>, CompileTimeOptions<doBz, noBz>>{},
            {Ex_runtime_flag, Ey_runtime_flag, Ez_runtime_flag, Bx_runtime_flag, By_runtime_flag, Bz_runtime_flag},
            pti.numParticles(),
            [=] AMREX_GPU_DEVICE (long ip, auto ex_control, auto ey_control, auto ez_control,
                auto bx_control, auto by_control, auto bz_control)
                {
                amrex::ParticleReal xp, yp, zp;
                getPosition(ip, xp, yp, zp);

                [[maybe_unused]] amrex::ParticleReal Ex_particle = 0._rt;
                [[maybe_unused]] amrex::ParticleReal Ey_particle = 0._rt;
                [[maybe_unused]] amrex::ParticleReal Ez_particle = 0._rt;
                [[maybe_unused]] amrex::ParticleReal Bx_particle = 0._rt;
                [[maybe_unused]] amrex::ParticleReal By_particle = 0._rt;
                [[maybe_unused]] amrex::ParticleReal Bz_particle = 0._rt;

                //getExternalEB(ip, Ex_particle, Ey_particle, Ez_particle,
                //    Bx_particle, By_particle, Bz_particle);
                // need to implement externalEB

                if (ex_control == noEx && ey_control == noEy && ez_control == noEz &&
                    bx_control == noBx && by_control == noBy && bz_control == noBz) {
                        // only for compiling the kernel where nothing is asked (but the function won't be called anyway)
                        amrex::ignore_unused(Ex_grid, Ey_grid, Ez_grid,
                            Bx_grid, By_grid, Bz_grid, ex_type, ey_type, ez_type,
                            bx_type, by_type, bz_type, dinv, xyzmin, lo, n_rz_azimuthal_modes);
                }


                if constexpr (ex_control == doEx || ey_control == doEy || ez_control == doEz)
                {
                    gatherE(
                        xp, yp, zp,
                        Ex_particle, Ey_particle, Ez_particle,
                        Ex_grid, Ey_grid, Ez_grid,
                        ex_type, ey_type, ez_type,
                        dinv, xyzmin, lo, n_rz_azimuthal_modes
                    );
                }

                if constexpr (bx_control == doBx || by_control == doBy || bz_control == doBz)
                {
                    // constexpr int depos_order_para = nox;
                    // constexpr int depos_order_perp = nox - galerkin_interpolation;
                    gatherB(
                        xp, yp, zp,
                        Bx_particle, By_particle, Bz_particle,
                        Bx_grid, By_grid, Bz_grid,
                        bx_type, by_type, bz_type,
                        dinv, xyzmin, lo, n_rz_azimuthal_modes
                    );
                }

                auto& rEx_particle = Ex_particle_arr;
                auto& rEy_particle = Ey_particle_arr;
                auto& rEz_particle = Ez_particle_arr;
                auto& rBx_particle = Bx_particle_arr;
                auto& rBy_particle = By_particle_arr;
                auto& rBz_particle = Bz_particle_arr;

                amrex::ignore_unused(rEx_particle, rEy_particle, rEz_particle,
                    rBx_particle, rBy_particle, rBz_particle);

                if constexpr (ex_control == doEx) {
                    rEx_particle[ip] = Ex_particle;
                }
                if constexpr (ey_control == doEy) {
                    rEy_particle[ip] = Ey_particle;
                }
                if constexpr (ez_control == doEz) {
                    rEz_particle[ip] = Ez_particle;
                }
                if constexpr (bx_control == doBx) {
                    rBx_particle[ip] = Bx_particle;
                }
                if constexpr (by_control == doBy) {
                    rBy_particle[ip] = By_particle;
                }
                if constexpr (bz_control == doBz) {
                    rBz_particle[ip] = Bz_particle;
                }
            });
    }

}
