/* Copyright 2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Arianna Formenti, Yinjian Zhao
 * License: BSD-3-Clause-LBNL
 */
#include "ColliderRelevant.H"

#include "Diagnostics/ReducedDiags/ReducedDiags.H"
#include "Fields.H"
#if (defined WARPX_QED)
#   include "Particles/ElementaryProcess/QEDInternals/QedChiFunctions.H"
#endif
#include "Particles/Gather/FieldGather.H"
#include "Particles/Gather/GetExternalFields.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/SpeciesPhysicalProperties.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_Algorithm.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_Dim3.H>
#include <AMReX_Extension.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IndexType.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParticleReduce.H>
#include <AMReX_Particles.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>
#include <AMReX_Tuple.H>
#include <AMReX_Vector.H>

#include <ablastr/coarsen/sample.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <vector>

using namespace amrex;


ColliderRelevant::ColliderRelevant (const std::string& rd_name)
: ReducedDiags{rd_name}
{
    // read colliding species names - must be 2
    const amrex::ParmParse pp_rd_name(m_rd_name);
    pp_rd_name.getarr("species", m_beam_name);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_beam_name.size() == 2u,
        "Collider-relevant diagnostics must involve exactly two species");

    // RZ coordinate is not supported
#if (defined WARPX_DIM_RZ)
    WARPX_ABORT_WITH_MESSAGE(
        "Collider-relevant diagnostics do not work in RZ geometry.");
#endif

    ablastr::warn_manager::WMRecordWarning(
        "Diagnostics",
        "The collider-relevant reduced diagnostic is meant for \
        colliding species propagating along the z direction.",
        ablastr::warn_manager::WarnPriority::low);

    ablastr::warn_manager::WMRecordWarning(
        "Diagnostics",
        "The collider-relevant reduced diagnostic only considers the \
        coarsest level of refinement for the calculations involving chi.",
        ablastr::warn_manager::WarnPriority::low);

    // get WarpX class object
    auto& warpx = WarpX::GetInstance();

    // get MultiParticleContainer class object
    const MultiParticleContainer& mypc =  warpx.GetPartContainer();

    // loop over species
    for (int i_s = 0; i_s < 2; ++i_s)
    {
        // get WarpXParticleContainer class object
        const WarpXParticleContainer& myspc = mypc.GetParticleContainerFromName(m_beam_name[i_s]);

        // get charge
        amrex::ParticleReal const q = myspc.getCharge();

        // photon number density is not available yet
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            q!=amrex::Real(0.0),
            "Collider-relevant diagnostic does not work for neutral species yet");
    }

    // function to fill a vector with diags names and create corresponding entry in header
    std::vector<std::string> all_diag_names;
    auto add_diag = [&,c=0](
        const std::string& name, const std::string& header) mutable {
        m_headers_indices[name] = aux_header_index{header, c++};
        all_diag_names.push_back(name);
    };

#if (defined WARPX_DIM_3D)
    add_diag("dL_dt", "dL_dt(m^-2*s^-1)");
#elif (defined WARPX_DIM_XZ)
    add_diag("dL_dt", "dL_dt(m^-1*s^-1)");
#else
    add_diag("dL_dt", "dL_dt(s^-1)");
#endif

    // loop over species
    for (int i_s = 0; i_s < 2; ++i_s)
    {
        // get WarpXParticleContainer class object
        const WarpXParticleContainer& myspc = mypc.GetParticleContainerFromName(m_beam_name[i_s]);

        if (myspc.DoQED()){
            add_diag("chimin_"+m_beam_name[i_s], "chi_min_"+m_beam_name[i_s]+"()");
            add_diag("chiave_"+m_beam_name[i_s], "chi_ave_"+m_beam_name[i_s]+"()");
            add_diag("chimax_"+m_beam_name[i_s], "chi_max_"+m_beam_name[i_s]+"()");
        }
#if (defined WARPX_DIM_3D)
        add_diag("x_ave_"+m_beam_name[i_s], "x_ave_"+m_beam_name[i_s]+"(m)");
        add_diag("x_std_"+m_beam_name[i_s], "x_std_"+m_beam_name[i_s]+"(m)");
        add_diag("y_ave_"+m_beam_name[i_s], "y_ave_"+m_beam_name[i_s]+"(m)");
        add_diag("y_std_"+m_beam_name[i_s], "y_std_"+m_beam_name[i_s]+"(m)");
        add_diag("thetax_min_"+m_beam_name[i_s], "theta_x_min_"+m_beam_name[i_s]+"(rad)");
        add_diag("thetax_ave_"+m_beam_name[i_s], "theta_x_ave_"+m_beam_name[i_s]+"(rad)");
        add_diag("thetax_max_"+m_beam_name[i_s], "theta_x_max_"+m_beam_name[i_s]+"(rad)");
        add_diag("thetax_std_"+m_beam_name[i_s], "theta_x_std_"+m_beam_name[i_s]+"(rad)");
        add_diag("thetay_min_"+m_beam_name[i_s], "theta_y_min_"+m_beam_name[i_s]+"(rad)");
        add_diag("thetay_ave_"+m_beam_name[i_s], "theta_y_ave_"+m_beam_name[i_s]+"(rad)");
        add_diag("thetay_max_"+m_beam_name[i_s], "theta_y_max_"+m_beam_name[i_s]+"(rad)");
        add_diag("thetay_std_"+m_beam_name[i_s], "theta_y_std_"+m_beam_name[i_s]+"(rad)");

#elif (defined WARPX_DIM_XZ)
        add_diag("x_ave_"+m_beam_name[i_s], "x_ave_"+m_beam_name[i_s]+"(m)");
        add_diag("x_std_"+m_beam_name[i_s], "x_std_"+m_beam_name[i_s]+"(m)");
        add_diag("thetax_min_"+m_beam_name[i_s], "theta_x_min_"+m_beam_name[i_s]+"(rad)");
        add_diag("thetax_ave_"+m_beam_name[i_s], "theta_x_ave_"+m_beam_name[i_s]+"(rad)");
        add_diag("thetax_max_"+m_beam_name[i_s], "theta_x_max_"+m_beam_name[i_s]+"(rad)");
        add_diag("thetax_std_"+m_beam_name[i_s], "theta_x_std_"+m_beam_name[i_s]+"(rad)");
#endif
        m_data.resize(all_diag_names.size());
    }

    if (amrex::ParallelDescriptor::IOProcessor())
    {
        if ( m_write_header )
        {
            // open file
            std::ofstream ofs;
            ofs.open(m_path + m_rd_name + "." + m_extension,
                std::ofstream::out | std::ofstream::app);
            // write header row
            int off = 0;
            ofs << "#";
            ofs << "[" << off++ << "]step()";
            ofs << m_sep;
            ofs << "[" << off++ << "]time(s)";
            for (const auto& name : all_diag_names){
                const auto& el = m_headers_indices[name];
                ofs << m_sep << "[" << el.idx + off << "]" << el.header;
            }
            ofs << "\n";
            // close file
            ofs.close();
        }
    }
}

void ColliderRelevant::ComputeDiags (int step)
{
#if defined(WARPX_DIM_RZ)
    amrex::ignore_unused(step);
#else

    // Judge if the diags should be done
    if (!m_intervals.contains(step+1)) { return; }

    // get MultiParticleContainer class object
    const MultiParticleContainer& mypc = WarpX::GetInstance().GetPartContainer();

    // get a reference to WarpX instance
    auto& warpx = WarpX::GetInstance();

    // get cell volume
    amrex::Geometry const & geom = warpx.Geom(0);
    const amrex::Real dV = AMREX_D_TERM(geom.CellSize(0), *geom.CellSize(1), *geom.CellSize(2));

    const auto get_idx = [&](const std::string& name){
        return m_headers_indices.at(name).idx;
    };

    std::array<std::unique_ptr<amrex::MultiFab>, 2> num_dens;

    // loop over species
    for (int i_s = 0; i_s < 2; ++i_s)
    {
        // get WarpXParticleContainer class object
        WarpXParticleContainer& myspc = mypc.GetParticleContainerFromName(m_beam_name[i_s]);
        // get charge
        amrex::ParticleReal const q = myspc.getCharge();

        using PType = typename WarpXParticleContainer::SuperParticleType;

        num_dens[i_s] = myspc.GetChargeDensity(0);
        num_dens[i_s]->mult(1._prt/q);

#if defined(WARPX_DIM_1D_Z)
        // w_tot
        amrex::Real w_tot = ReduceSum( myspc,
            [=] AMREX_GPU_HOST_DEVICE (const PType& p)
            {
                return p.rdata(PIdx::w);
            });
        amrex::ParallelDescriptor::ReduceRealSum(w_tot);
#elif defined(WARPX_DIM_XZ)
        // w_tot
        // x_ave,
        // thetax_min, thetax_ave, thetax_max
        amrex::ReduceOps<ReduceOpSum,
                         ReduceOpSum,
                         ReduceOpMin, ReduceOpSum, ReduceOpMax> reduce_ops;
        auto r = amrex::ParticleReduce<amrex::ReduceData<Real,
                                                         Real,
                                                         Real, Real, Real>>(
            myspc,
            [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<Real,
                                                                             Real,
                                                                             Real, Real, Real>
            {
                const amrex::Real w  = p.rdata(PIdx::w);
                const amrex::Real x = p.pos(0);
                const amrex::Real ux = p.rdata(PIdx::ux);
                const amrex::Real uz = p.rdata(PIdx::uz);
                const amrex::Real thetax = std::atan2(ux, uz);
                return {w, w*x, thetax, w*thetax, thetax};
            },
            reduce_ops);

        amrex::Real w_tot  = amrex::get<0>(r);
        amrex::Real x_ave  = amrex::get<1>(r);
        amrex::Real thetax_min  = amrex::get<2>(r);
        amrex::Real thetax_ave  = amrex::get<3>(r);
        amrex::Real thetax_max  = amrex::get<4>(r);

        amrex::ParallelDescriptor::ReduceRealSum({w_tot, x_ave, thetax_ave});
        amrex::ParallelDescriptor::ReduceRealMin({thetax_min});
        amrex::ParallelDescriptor::ReduceRealMax({thetax_max});

        // x_std, thetax_std
        amrex::Real x_std = 0.0_rt;
        amrex::Real thetax_std = 0.0_rt;

        if (w_tot > 0.0_rt)
        {
            x_ave = x_ave / w_tot;
            thetax_ave = thetax_ave / w_tot;

            amrex::ReduceOps<ReduceOpSum, ReduceOpSum> reduce_ops_std;
            auto r_std = amrex::ParticleReduce<amrex::ReduceData<Real, Real>>(
                myspc,
                [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<Real, Real>
                {
                    const amrex::Real w  = p.rdata(PIdx::w);
                    const amrex::Real x = p.pos(0);
                    const amrex::Real ux = p.rdata(PIdx::ux);
                    const amrex::Real uz = p.rdata(PIdx::uz);
                    const amrex::Real thetax = std::atan2(ux, uz);
                    const amrex::Real tmp1 = (x - x_ave)*(x - x_ave)*w;
                    const amrex::Real tmp2 = (thetax - thetax_ave)*(thetax - thetax_ave)*w;
                    return {tmp1, tmp2};
                },
                reduce_ops_std);

            x_std = amrex::get<0>(r_std);
            thetax_std = amrex::get<1>(r_std);

            amrex::ParallelDescriptor::ReduceRealSum({x_std, thetax_std});

            x_std = std::sqrt(x_std / w_tot);
            thetax_std = std::sqrt(thetax_std / w_tot);
        }

        m_data[get_idx("x_ave_"+m_beam_name[i_s])] = x_ave;
        m_data[get_idx("x_std_"+m_beam_name[i_s])] = x_std;
        m_data[get_idx("thetax_min_"+m_beam_name[i_s])] = thetax_min;
        m_data[get_idx("thetax_ave_"+m_beam_name[i_s])] = thetax_ave;
        m_data[get_idx("thetax_max_"+m_beam_name[i_s])] = thetax_max;
        m_data[get_idx("thetax_std_"+m_beam_name[i_s])] = thetax_std;
#elif defined(WARPX_DIM_3D)
        // w_tot
        // x_ave, y_ave,
        // thetax_min, thetax_ave, thetax_max
        // thetay_min, thetay_ave, thetay_max
        amrex::ReduceOps<ReduceOpSum,
                         ReduceOpSum, ReduceOpSum,
                         ReduceOpMin, ReduceOpSum, ReduceOpMax,
                         ReduceOpMin, ReduceOpSum, ReduceOpMax> reduce_ops;
        auto r = amrex::ParticleReduce<amrex::ReduceData<Real,
                                                         Real, Real,
                                                         Real, Real, Real,
                                                         Real, Real, Real>>(
            myspc,
            [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<Real,
                                                                             Real, Real,
                                                                             Real, Real, Real,
                                                                             Real, Real, Real>
            {
                const amrex::Real w  = p.rdata(PIdx::w);
                const amrex::Real x = p.pos(0);
                const amrex::Real y = p.pos(1);
                const amrex::Real ux = p.rdata(PIdx::ux);
                const amrex::Real uy = p.rdata(PIdx::uy);
                const amrex::Real uz = p.rdata(PIdx::uz);
                const amrex::Real thetax = std::atan2(ux, uz);
                const amrex::Real thetay = std::atan2(uy, uz);
                return {w, w*x, w*y,
                        thetax, w*thetax, thetax,
                        thetay, w*thetay, thetay};
            },
            reduce_ops);

        amrex::Real w_tot  = amrex::get<0>(r);
        amrex::Real x_ave  = amrex::get<1>(r);
        amrex::Real y_ave  = amrex::get<2>(r);
        amrex::Real thetax_min  = amrex::get<3>(r);
        amrex::Real thetax_ave  = amrex::get<4>(r);
        amrex::Real thetax_max  = amrex::get<5>(r);
        amrex::Real thetay_min  = amrex::get<6>(r);
        amrex::Real thetay_ave  = amrex::get<7>(r);
        amrex::Real thetay_max  = amrex::get<8>(r);

        amrex::ParallelDescriptor::ReduceRealSum({w_tot, x_ave, y_ave, thetax_ave, thetay_ave});
        amrex::ParallelDescriptor::ReduceRealMin({thetax_min, thetay_min});
        amrex::ParallelDescriptor::ReduceRealMax({thetax_max, thetay_max});

        // x_std, y_std, thetax_std, thetay_std
        amrex::Real x_std = 0.0_rt;
        amrex::Real y_std = 0.0_rt;
        amrex::Real thetax_std = 0.0_rt;
        amrex::Real thetay_std = 0.0_rt;

        if (w_tot > 0.0_rt)
        {
            x_ave = x_ave / w_tot;
            y_ave = y_ave / w_tot;
            thetax_ave = thetax_ave / w_tot;
            thetay_ave = thetay_ave / w_tot;

            amrex::ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_ops_std;
            auto r_std = amrex::ParticleReduce<amrex::ReduceData<Real, Real, Real, Real>>(
                myspc,
                [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<Real, Real, Real, Real>
                {
                    const amrex::Real w  = p.rdata(PIdx::w);
                    const amrex::Real x = p.pos(0);
                    const amrex::Real ux = p.rdata(PIdx::ux);
                    const amrex::Real y = p.pos(1);
                    const amrex::Real uy = p.rdata(PIdx::uy);
                    const amrex::Real uz = p.rdata(PIdx::uz);
                    const amrex::Real thetax = std::atan2(ux, uz);
                    const amrex::Real thetay = std::atan2(uy, uz);
                    const amrex::Real tmp1 = (x - x_ave)*(x - x_ave)*w;
                    const amrex::Real tmp2 = (y - y_ave)*(y - y_ave)*w;
                    const amrex::Real tmp3 = (thetax - thetax_ave)*(thetax - thetax_ave)*w;
                    const amrex::Real tmp4 = (thetay - thetay_ave)*(thetay - thetay_ave)*w;
                    return {tmp1, tmp2, tmp3, tmp4};
                },
                reduce_ops_std);

            x_std = amrex::get<0>(r_std);
            y_std = amrex::get<1>(r_std);
            thetax_std = amrex::get<2>(r_std);
            thetay_std = amrex::get<3>(r_std);

            amrex::ParallelDescriptor::ReduceRealSum({x_std, y_std, thetax_std, thetay_std});

            x_std = std::sqrt(x_std / w_tot);
            y_std = std::sqrt(y_std / w_tot);
            thetax_std = std::sqrt(thetax_std / w_tot);
            thetay_std = std::sqrt(thetay_std / w_tot);
        }

        m_data[get_idx("x_ave_"+m_beam_name[i_s])] = x_ave;
        m_data[get_idx("x_std_"+m_beam_name[i_s])] = x_std;
        m_data[get_idx("y_ave_"+m_beam_name[i_s])] = y_ave;
        m_data[get_idx("y_std_"+m_beam_name[i_s])] = y_std;
        m_data[get_idx("thetax_min_"+m_beam_name[i_s])] = thetax_min;
        m_data[get_idx("thetax_ave_"+m_beam_name[i_s])] = thetax_ave;
        m_data[get_idx("thetax_max_"+m_beam_name[i_s])] = thetax_max;
        m_data[get_idx("thetax_std_"+m_beam_name[i_s])] = thetax_std;
        m_data[get_idx("thetay_min_"+m_beam_name[i_s])] = thetay_min;
        m_data[get_idx("thetay_ave_"+m_beam_name[i_s])] = thetay_ave;
        m_data[get_idx("thetay_max_"+m_beam_name[i_s])] = thetay_max;
        m_data[get_idx("thetay_std_"+m_beam_name[i_s])] = thetay_std;
#endif

#if (defined WARPX_QED)
        // get mass
        amrex::ParticleReal m = myspc.getMass();
        const bool is_photon = myspc.AmIA<PhysicalSpecies::photon>();
        if (is_photon) {
            m = PhysConst::m_e;
        }

        // compute chimin, chiave and chimax
        amrex::Real chimin_f = 0.0_rt;
        amrex::Real chimax_f = 0.0_rt;
        amrex::Real chiave_f = 0.0_rt;

        using ablastr::fields::Direction;

        if (myspc.DoQED())
        {
            // define variables in preparation for field gatheeduce_data.value()ring
            const int n_rz_azimuthal_modes = WarpX::n_rz_azimuthal_modes;
            const int nox = WarpX::nox;
            const bool galerkin_interpolation = WarpX::galerkin_interpolation;
            const amrex::IntVect ngEB = warpx.getngEB();

            // TODO loop over refinement levels: for (int lev = 0; lev <= level_number; ++lev)
            const int lev = 0;

            // define variables in preparation for field gathering
            using warpx::fields::FieldType;
            const amrex::XDim3 dinv = WarpX::InvCellSize(std::max(lev, 0));
            const amrex::MultiFab & Ex = *warpx.m_fields.get(FieldType::Efield_aux, Direction{0}, lev);
            const amrex::MultiFab & Ey = *warpx.m_fields.get(FieldType::Efield_aux, Direction{1}, lev);
            const amrex::MultiFab & Ez = *warpx.m_fields.get(FieldType::Efield_aux, Direction{2}, lev);
            const amrex::MultiFab & Bx = *warpx.m_fields.get(FieldType::Bfield_aux, Direction{0}, lev);
            const amrex::MultiFab & By = *warpx.m_fields.get(FieldType::Bfield_aux, Direction{1}, lev);
            const amrex::MultiFab & Bz = *warpx.m_fields.get(FieldType::Bfield_aux, Direction{2}, lev);

            // declare reduce_op
            ReduceOps<ReduceOpMin, ReduceOpMax, ReduceOpSum> reduce_op;
            ReduceData<Real, Real, Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

            // Loop over boxes
            for (WarpXParIter pti(myspc, lev); pti.isValid(); ++pti)
            {
                const auto GetPosition = GetParticlePosition<PIdx>(pti);
                // get particle arrays
                amrex::ParticleReal* const AMREX_RESTRICT ux = pti.GetAttribs()[PIdx::ux].dataPtr();
                amrex::ParticleReal* const AMREX_RESTRICT uy = pti.GetAttribs()[PIdx::uy].dataPtr();
                amrex::ParticleReal* const AMREX_RESTRICT uz = pti.GetAttribs()[PIdx::uz].dataPtr();
                amrex::ParticleReal* const AMREX_RESTRICT w = pti.GetAttribs()[PIdx::w].dataPtr();
                // declare external fields
                const int offset = 0;
                const auto getExternalEB = GetExternalEBField(pti, offset);
                const amrex::ParticleReal Ex_external_particle = myspc.m_E_external_particle[0];
                const amrex::ParticleReal Ey_external_particle = myspc.m_E_external_particle[1];
                const amrex::ParticleReal Ez_external_particle = myspc.m_E_external_particle[2];
                const amrex::ParticleReal Bx_external_particle = myspc.m_B_external_particle[0];
                const amrex::ParticleReal By_external_particle = myspc.m_B_external_particle[1];
                const amrex::ParticleReal Bz_external_particle = myspc.m_B_external_particle[2];

                // define variables in preparation for field gathering
                amrex::Box box = pti.tilebox();
                box.grow(ngEB);
                const amrex::Dim3 lo = amrex::lbound(box);
                const amrex::XDim3 xyzmin = WarpX::LowerCorner(box, lev, 0._rt);
                const amrex::Array4<const amrex::Real> & ex_arr = Ex[pti].array();
                const amrex::Array4<const amrex::Real> & ey_arr = Ey[pti].array();
                const amrex::Array4<const amrex::Real> & ez_arr = Ez[pti].array();
                const amrex::Array4<const amrex::Real> & bx_arr = Bx[pti].array();
                const amrex::Array4<const amrex::Real> & by_arr = By[pti].array();
                const amrex::Array4<const amrex::Real> & bz_arr = Bz[pti].array();
                const amrex::IndexType ex_type = Ex[pti].box().ixType();
                const amrex::IndexType ey_type = Ey[pti].box().ixType();
                const amrex::IndexType ez_type = Ez[pti].box().ixType();
                const amrex::IndexType bx_type = Bx[pti].box().ixType();
                const amrex::IndexType by_type = By[pti].box().ixType();
                const amrex::IndexType bz_type = Bz[pti].box().ixType();

                // evaluate reduce_op
                reduce_op.eval(pti.numParticles(), reduce_data,
                [=] AMREX_GPU_DEVICE (int i) -> ReduceTuple
                {
                    // get external fields
                    amrex::ParticleReal xp, yp, zp;
                    GetPosition(i, xp, yp, zp);
                    amrex::ParticleReal ex = Ex_external_particle;
                    amrex::ParticleReal ey = Ey_external_particle;
                    amrex::ParticleReal ez = Ez_external_particle;
                    amrex::ParticleReal bx = Bx_external_particle;
                    amrex::ParticleReal by = By_external_particle;
                    amrex::ParticleReal bz = Bz_external_particle;

                    getExternalEB(i, ex, ey, ez, bx, by, bz);

                    // gather E and B
                    doGatherShapeN(xp, yp, zp,
                        ex, ey, ez, bx, by, bz,
                        ex_arr, ey_arr, ez_arr, bx_arr, by_arr, bz_arr,
                        ex_type, ey_type, ez_type,
                        bx_type, by_type, bz_type,
                        dinv, xyzmin, lo,
                        n_rz_azimuthal_modes, nox, galerkin_interpolation);
                    // compute chi
                    amrex::Real chi = 0.0_rt;
                    if (is_photon) {
                        chi = QedUtils::chi_photon(ux[i]*m, uy[i]*m, uz[i]*m,
                                            ex, ey, ez, bx, by, bz);
                    } else {
                        chi = QedUtils::chi_ele_pos(ux[i]*m, uy[i]*m, uz[i]*m,
                                            ex, ey, ez, bx, by, bz);
                    }
                    return {chi, chi, chi*w[i]};
                });
            }
            auto val = reduce_data.value();
            chimin_f = get<0>(val);
            chimax_f = get<1>(val);
            chiave_f = get<2>(val);
            amrex::ParallelDescriptor::ReduceRealMin(chimin_f);
            amrex::ParallelDescriptor::ReduceRealMax(chimax_f);
            amrex::ParallelDescriptor::ReduceRealSum(chiave_f);

            m_data[get_idx("chimin_"+m_beam_name[i_s])] = chimin_f;
            m_data[get_idx("chiave_"+m_beam_name[i_s])] = chiave_f/w_tot;
            m_data[get_idx("chimax_"+m_beam_name[i_s])] = chimax_f;
        }
#endif
    } // end loop over species

    // make density MultiFabs from nodal to cell centered
    amrex::BoxArray ba = warpx.boxArray(0);
    const amrex::DistributionMapping dmap = warpx.DistributionMap(0);
    constexpr int ncomp = 1;
    constexpr int ngrow = 0;
    amrex::MultiFab mf_dst1(ba.convert(amrex::IntVect::TheCellVector()), dmap, ncomp, ngrow);
    amrex::MultiFab mf_dst2(ba.convert(amrex::IntVect::TheCellVector()), dmap, ncomp, ngrow);
    ablastr::coarsen::sample::Coarsen(mf_dst1, *num_dens[0], 0, 0, ncomp, ngrow);
    ablastr::coarsen::sample::Coarsen(mf_dst2, *num_dens[1], 0, 0, ncomp, ngrow);

    // compute luminosity
    amrex::Real const n1_dot_n2 = amrex::MultiFab::Dot(mf_dst1, 0, mf_dst2, 0, 1, 0);
    amrex::Real const lumi = 2._rt * PhysConst::c * n1_dot_n2 * dV;
    m_data[get_idx("dL_dt")] = lumi;
#endif // not RZ
}
