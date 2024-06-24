/* Copyright 2019 David Grote, Luca Fedeli, Maxence Thevenet
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "PhotonParticleContainer.H"

#ifdef WARPX_QED
#   include "Particles/ElementaryProcess/QEDInternals/BreitWheelerEngineWrapper.H"
#endif
#include "Particles/Gather/FieldGather.H"
#include "Particles/Gather/GetExternalFields.H"
#include "Particles/PhysicalParticleContainer.H"
#include "Particles/Pusher/CopyParticleAttribs.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/Pusher/UpdatePositionPhoton.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_Dim3.H>
#include <AMReX_Extension.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Particles.H>
#include <AMReX_StructOfArrays.H>

#include <algorithm>
#include <array>
#include <map>
#include <memory>

using namespace amrex;

PhotonParticleContainer::PhotonParticleContainer (AmrCore* amr_core, int ispecies,
                                                  const std::string& name)
    : PhysicalParticleContainer(amr_core, ispecies, name)
{
    const ParmParse pp_species_name(species_name);

#ifdef WARPX_QED
        //Find out if Breit Wheeler process is enabled
        pp_species_name.query("do_qed_breit_wheeler", m_do_qed_breit_wheeler);

        //If Breit Wheeler process is enabled, look for the target electron and positron
        //species
        if(m_do_qed_breit_wheeler){
            pp_species_name.get("qed_breit_wheeler_ele_product_species", m_qed_breit_wheeler_ele_product_name);
            pp_species_name.get("qed_breit_wheeler_pos_product_species", m_qed_breit_wheeler_pos_product_name);
        }

        //Check for processes which do not make sense for photons
        bool test_quantum_sync = false;
        pp_species_name.query("do_qed_quantum_sync", test_quantum_sync);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        test_quantum_sync == 0,
        "ERROR: do_qed_quantum_sync can be 1 for species NOT listed in particles.photon_species only!");
        //_________________________________________________________
#endif

}

void PhotonParticleContainer::InitData()
{
    AddParticles(0); // Note - add on level 0

    Redistribute();  // We then redistribute

}

void
PhotonParticleContainer::PushPX (WarpXParIter& pti,
                                 amrex::FArrayBox const * exfab,
                                 amrex::FArrayBox const * eyfab,
                                 amrex::FArrayBox const * ezfab,
                                 amrex::FArrayBox const * bxfab,
                                 amrex::FArrayBox const * byfab,
                                 amrex::FArrayBox const * bzfab,
                                 const amrex::IntVect ngEB, const int /*e_is_nodal*/,
                                 const long offset,
                                 const long np_to_push,
                                 int lev, int gather_lev,
                                 amrex::Real dt, ScaleFields /*scaleFields*/, DtType a_dt_type)
{
    // Get inverse cell size on gather_lev
    const amrex::XDim3 dinv = WarpX::InvCellSize(std::max(gather_lev,0));

    // Get box from which field is gathered.
    // If not gathering from the finest level, the box is coarsened.
    amrex::Box box;
    if (lev == gather_lev) {
        box = pti.tilebox();
    } else {
        const IntVect& ref_ratio = WarpX::RefRatio(gather_lev);
        box = amrex::coarsen(pti.tilebox(),ref_ratio);
    }

    // Add guard cells to the box.
    box.grow(ngEB);

    auto& attribs = pti.GetAttribs();

    // Extract pointers to the different particle quantities
    ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr() + offset;
    ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr() + offset;

#ifdef WARPX_QED
    BreitWheelerEvolveOpticalDepth evolve_opt;
    amrex::ParticleReal* AMREX_RESTRICT p_optical_depth_BW = nullptr;
    const bool local_has_breit_wheeler = has_breit_wheeler();
    if (local_has_breit_wheeler) {
        evolve_opt = m_shr_p_bw_engine->build_evolve_functor();
        p_optical_depth_BW = pti.GetAttribs(particle_comps["opticalDepthBW"]).dataPtr() + offset;
    }
#endif

    auto copyAttribs = CopyParticleAttribs(pti, tmp_particle_data, offset);
    const int do_copy = (m_do_back_transformed_particles && (a_dt_type!=DtType::SecondHalf) );

    const auto GetPosition = GetParticlePosition<PIdx>(pti, offset);
    auto SetPosition = SetParticlePosition<PIdx>(pti, offset);

    const auto getExternalEB = GetExternalEBField(pti, offset);

    const amrex::ParticleReal Ex_external_particle = m_E_external_particle[0];
    const amrex::ParticleReal Ey_external_particle = m_E_external_particle[1];
    const amrex::ParticleReal Ez_external_particle = m_E_external_particle[2];
    const amrex::ParticleReal Bx_external_particle = m_B_external_particle[0];
    const amrex::ParticleReal By_external_particle = m_B_external_particle[1];
    const amrex::ParticleReal Bz_external_particle = m_B_external_particle[2];

    // Lower corner of tile box physical domain (take into account Galilean shift)
    const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, gather_lev, 0._rt);

    const Dim3 lo = lbound(box);

    const bool galerkin_interpolation = WarpX::galerkin_interpolation;
    const int nox = WarpX::nox;
    const int n_rz_azimuthal_modes = WarpX::n_rz_azimuthal_modes;

    amrex::Array4<const amrex::Real> const& ex_arr = exfab->array();
    amrex::Array4<const amrex::Real> const& ey_arr = eyfab->array();
    amrex::Array4<const amrex::Real> const& ez_arr = ezfab->array();
    amrex::Array4<const amrex::Real> const& bx_arr = bxfab->array();
    amrex::Array4<const amrex::Real> const& by_arr = byfab->array();
    amrex::Array4<const amrex::Real> const& bz_arr = bzfab->array();

    amrex::IndexType const ex_type = exfab->box().ixType();
    amrex::IndexType const ey_type = eyfab->box().ixType();
    amrex::IndexType const ez_type = ezfab->box().ixType();
    amrex::IndexType const bx_type = bxfab->box().ixType();
    amrex::IndexType const by_type = byfab->box().ixType();
    amrex::IndexType const bz_type = bzfab->box().ixType();

    const auto t_do_not_gather = do_not_gather;

    enum exteb_flags : int { no_exteb, has_exteb };
    enum qed_flags : int { no_qed, has_qed };

    const int exteb_runtime_flag = getExternalEB.isNoOp() ? no_exteb : has_exteb;
#ifdef WARPX_QED
    const int qed_runtime_flag = (local_has_breit_wheeler) ? has_qed : no_qed;
#else
    const int qed_runtime_flag = no_qed;
#endif

    amrex::ParallelFor(TypeList<CompileTimeOptions<no_exteb,has_exteb>,
                                CompileTimeOptions<no_qed  ,has_qed>>{},
                       {exteb_runtime_flag, qed_runtime_flag},
                       np_to_push,
                       [=] AMREX_GPU_DEVICE (long i, auto exteb_control,
                                             auto qed_control) {
            if (do_copy) { copyAttribs(i); }
            ParticleReal x, y, z;
            GetPosition(i, x, y, z);

            amrex::ParticleReal Exp = Ex_external_particle;
            amrex::ParticleReal Eyp = Ey_external_particle;
            amrex::ParticleReal Ezp = Ez_external_particle;
            amrex::ParticleReal Bxp = Bx_external_particle;
            amrex::ParticleReal Byp = By_external_particle;
            amrex::ParticleReal Bzp = Bz_external_particle;

            if(!t_do_not_gather){
                // first gather E and B to the particle positions
                doGatherShapeN(x, y, z, Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                               ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                               ex_type, ey_type, ez_type, bx_type, by_type, bz_type,
                               dinv, xyzmin, lo, n_rz_azimuthal_modes,
                               nox, galerkin_interpolation);
            }

            [[maybe_unused]] const auto& getExternalEB_tmp = getExternalEB; // workaround for nvcc
            if constexpr (exteb_control == has_exteb) {
                getExternalEB(i, Exp, Eyp, Ezp, Bxp, Byp, Bzp);
            }

#ifdef WARPX_QED
            [[maybe_unused]] const auto& evolve_opt_tmp = evolve_opt;
            [[maybe_unused]] auto *p_optical_depth_BW_tmp = p_optical_depth_BW;
            [[maybe_unused]] auto *ux_tmp = ux; // for nvhpc
            [[maybe_unused]] auto *uy_tmp = uy;
            [[maybe_unused]] auto *uz_tmp = uz;
            [[maybe_unused]] auto dt_tmp = dt;
            if constexpr (qed_control == has_qed) {
                evolve_opt(ux[i], uy[i], uz[i], Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                           dt, p_optical_depth_BW[i]);
            }
#else
            amrex::ignore_unused(qed_control);
#endif

            UpdatePositionPhoton( x, y, z, ux[i], uy[i], uz[i], dt );
            SetPosition(i, x, y, z);
        }
    );
}

void
PhotonParticleContainer::Evolve (int lev,
                                 const MultiFab& Ex, const MultiFab& Ey, const MultiFab& Ez,
                                 const MultiFab& Bx, const MultiFab& By, const MultiFab& Bz,
                                 MultiFab& jx, MultiFab& jy, MultiFab& jz,
                                 MultiFab* cjx, MultiFab* cjy, MultiFab* cjz,
                                 MultiFab* rho, MultiFab* crho,
                                 const MultiFab* cEx, const MultiFab* cEy, const MultiFab* cEz,
                                 const MultiFab* cBx, const MultiFab* cBy, const MultiFab* cBz,
                                 Real t, Real dt, DtType a_dt_type, bool skip_deposition,
                                 PushType push_type)
{
    // This does gather, push and deposit.
    // Push and deposit have been re-written for photons
    PhysicalParticleContainer::Evolve (lev,
                                       Ex, Ey, Ez,
                                       Bx, By, Bz,
                                       jx, jy, jz,
                                       cjx, cjy, cjz,
                                       rho, crho,
                                       cEx, cEy, cEz,
                                       cBx, cBy, cBz,
                                       t, dt, a_dt_type, skip_deposition, push_type);

}
