/* Copyright 2019-2020 Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "BeamRelevant.H"

#include "Diagnostics/ReducedDiags/ReducedDiags.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParticleReduce.H>
#include <AMReX_Particles.H>
#include <AMReX_REAL.H>
#include <AMReX_Tuple.H>
#include <AMReX_TypeList.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <vector>

using namespace amrex;

// constructor
BeamRelevant::BeamRelevant (const std::string& rd_name)
: ReducedDiags{rd_name}
{
    // read beam name
    const ParmParse pp_rd_name(rd_name);
    pp_rd_name.get("species",m_beam_name);

    // resize data array
#if (defined WARPX_DIM_3D || defined WARPX_DIM_RZ)
    //  0, 1, 2: mean x,y,z
    //  3, 4, 5: mean px,py,pz
    //        6: gamma
    //  7, 8, 9: rms x,y,z
    // 10,11,12: rms px,py,pz
    //       13: rms gamma
    // 14,15,16: emittance x,y,z
    //    17,18: Twiss-alpha x,y
    //    19,20: beta-function x,y
    //       21: charge
    m_data.resize(22, 0.0_rt);
#elif (defined WARPX_DIM_XZ)
    //     0, 1: mean x,z
    //  2, 3, 4: mean px,py,pz
    //        5: gamma
    //     6, 7: rms x,z
    //  8, 9,10: rms px,py,pz
    //       11: rms gamma
    //    12,13: emittance x,z
    //       14: Twiss-alpha x
    //       15: beta-function x
    //       16: charge
    m_data.resize(17, 0.0_rt);
#elif (defined WARPX_DIM_1D_Z)
    //       0 : mean z
    //   1,2,3 : mean px,py,pz
    //       4 : gamma
    //       5 : rms z
    //   6,7,8 : rms px,py,pz
    //       9 : rms gamma
    //      10 : emittance z
    //      11 : charge
    m_data.resize(12, 0.0_rt);
#endif

    if (ParallelDescriptor::IOProcessor())
    {
        if ( m_write_header )
        {
            // open file
            std::ofstream ofs{m_path + m_rd_name + "." + m_extension, std::ofstream::out};
            // write header row
#if (defined WARPX_DIM_3D || defined WARPX_DIM_RZ)
            int c = 0;
            ofs << "#";
            ofs << "[" << c++ << "]step()";           ofs << m_sep;
            ofs << "[" << c++ << "]time(s)";          ofs << m_sep;
            ofs << "[" << c++ << "]x_mean(m)";        ofs << m_sep;
            ofs << "[" << c++ << "]y_mean(m)";        ofs << m_sep;
            ofs << "[" << c++ << "]z_mean(m)";        ofs << m_sep;
            ofs << "[" << c++ << "]px_mean(kg*m/s)";  ofs << m_sep;
            ofs << "[" << c++ << "]py_mean(kg*m/s)";  ofs << m_sep;
            ofs << "[" << c++ << "]pz_mean(kg*m/s)";  ofs << m_sep;
            ofs << "[" << c++ << "]gamma_mean()";     ofs << m_sep;
            ofs << "[" << c++ << "]x_rms(m)";         ofs << m_sep;
            ofs << "[" << c++ << "]y_rms(m)";         ofs << m_sep;
            ofs << "[" << c++ << "]z_rms(m)";         ofs << m_sep;
            ofs << "[" << c++ << "]px_rms(kg*m/s)";   ofs << m_sep;
            ofs << "[" << c++ << "]py_rms(kg*m/s)";   ofs << m_sep;
            ofs << "[" << c++ << "]pz_rms(kg*m/s)";   ofs << m_sep;
            ofs << "[" << c++ << "]gamma_rms()";      ofs << m_sep;
            ofs << "[" << c++ << "]emittance_x(m)";   ofs << m_sep;
            ofs << "[" << c++ << "]emittance_y(m)";   ofs << m_sep;
            ofs << "[" << c++ << "]emittance_z(m)";   ofs << m_sep;
            ofs << "[" << c++ << "]alpha_x()";        ofs << m_sep;
            ofs << "[" << c++ << "]alpha_y()";        ofs << m_sep;
            ofs << "[" << c++ << "]beta_x(m)";        ofs << m_sep;
            ofs << "[" << c++ << "]beta_y(m)";        ofs << m_sep;
            ofs << "[" << c++ << "]charge(C)";        ofs << "\n";
#elif (defined WARPX_DIM_XZ)
            int c = 0;
            ofs << "#";
            ofs << "[" << c++ << "]step()";           ofs << m_sep;
            ofs << "[" << c++ << "]time(s)";          ofs << m_sep;
            ofs << "[" << c++ << "]x_mean(m)";        ofs << m_sep;
            ofs << "[" << c++ << "]z_mean(m)";        ofs << m_sep;
            ofs << "[" << c++ << "]px_mean(kg*m/s)";  ofs << m_sep;
            ofs << "[" << c++ << "]py_mean(kg*m/s)";  ofs << m_sep;
            ofs << "[" << c++ << "]pz_mean(kg*m/s)";  ofs << m_sep;
            ofs << "[" << c++ << "]gamma_mean()";     ofs << m_sep;
            ofs << "[" << c++ << "]x_rms(m)";         ofs << m_sep;
            ofs << "[" << c++ << "]z_rms(m)";         ofs << m_sep;
            ofs << "[" << c++ << "]px_rms(kg*m/s)";   ofs << m_sep;
            ofs << "[" << c++ << "]py_rms(kg*m/s)";   ofs << m_sep;
            ofs << "[" << c++ << "]pz_rms(kg*m/s)";   ofs << m_sep;
            ofs << "[" << c++ << "]gamma_rms()";      ofs << m_sep;
            ofs << "[" << c++ << "]emittance_x(m)";   ofs << m_sep;
            ofs << "[" << c++ << "]emittance_z(m)";   ofs << m_sep;
            ofs << "[" << c++ << "]alpha_x()";        ofs << m_sep;
            ofs << "[" << c++ << "]beta_x(m)";        ofs << m_sep;
            ofs << "[" << c++ << "]charge(C)";        ofs << "\n";
#elif (defined WARPX_DIM_1D_Z)
            int c = 0;
            ofs << "#";
            ofs << "[" << c++ << "]step()";           ofs << m_sep;
            ofs << "[" << c++ << "]time(s)";          ofs << m_sep;
            ofs << "[" << c++ << "]z_mean(m)";        ofs << m_sep;
            ofs << "[" << c++ << "]px_mean(kg*m/s)";  ofs << m_sep;
            ofs << "[" << c++ << "]py_mean(kg*m/s)";  ofs << m_sep;
            ofs << "[" << c++ << "]pz_mean(kg*m/s)";  ofs << m_sep;
            ofs << "[" << c++ << "]gamma_mean()";     ofs << m_sep;
            ofs << "[" << c++ << "]z_rms(m)";         ofs << m_sep;
            ofs << "[" << c++ << "]px_rms(kg*m/s)";   ofs << m_sep;
            ofs << "[" << c++ << "]py_rms(kg*m/s)";   ofs << m_sep;
            ofs << "[" << c++ << "]pz_rms(kg*m/s)";   ofs << m_sep;
            ofs << "[" << c++ << "]gamma_rms()";      ofs << m_sep;
            ofs << "[" << c++ << "]emittance_z(m)";   ofs << m_sep;
            ofs << "[" << c++ << "]charge(C)";        ofs << "\n";
#endif
            // close file
            ofs.close();
        }
    }
}
// end constructor

// function that compute beam relevant quantities
void BeamRelevant::ComputeDiags (int step)
{
    // Judge if the diags should be done
    if (!m_intervals.contains(step+1)) { return; }

    // get MultiParticleContainer class object
    const auto & mypc = WarpX::GetInstance().GetPartContainer();

    // get number of species
    int const nSpecies = mypc.nSpecies();

    // get species names (std::vector<std::string>)
    auto const species_names = mypc.GetSpeciesNames();

    // inverse of speed of light squared
    Real constexpr inv_c2 = 1.0_rt / (PhysConst::c * PhysConst::c);

    // loop over species
    for (int i_s = 0; i_s < nSpecies; ++i_s)
    {
        // only beam species does
        if (species_names[i_s] != m_beam_name) { continue; }

        // get WarpXParticleContainer class object
        auto const & myspc = mypc.GetParticleContainer(i_s);

        // get mass and charge
        ParticleReal const m = myspc.getMass();
        ParticleReal const q = myspc.getCharge();

        using PType = typename WarpXParticleContainer::SuperParticleType;

        // number of reduction operations in first concurrent batch
        constexpr size_t num_red_ops_1 = 8;
        TypeMultiplier<amrex::ReduceOps, ReduceOpSum[num_red_ops_1]> reduce_ops_1;
        using ReducedDataT1 = TypeMultiplier<amrex::ReduceData, ParticleReal[num_red_ops_1]>;

        auto r1 = amrex::ParticleReduce<ReducedDataT1>(
            myspc,
            [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> ReducedDataT1::Type
            {
                const ParticleReal p_ux = p.rdata(PIdx::ux);
                const ParticleReal p_uy = p.rdata(PIdx::uy);
                const ParticleReal p_uz = p.rdata(PIdx::uz);
                const ParticleReal p_us = p_ux*p_ux + p_uy*p_uy + p_uz*p_uz;
                const ParticleReal p_w = p.rdata(PIdx::w);

                ParticleReal p_x, p_y, p_z;
                get_particle_position(p, p_x, p_y, p_z);

                const ParticleReal p_x_mean = p_x*p_w;
                const ParticleReal p_y_mean = p_y*p_w;
                const ParticleReal p_z_mean = p_z*p_w;

                const ParticleReal p_ux_mean = p_ux*p_w;
                const ParticleReal p_uy_mean = p_uy*p_w;
                const ParticleReal p_uz_mean = p_uz*p_w;
                const ParticleReal p_gm_mean = std::sqrt(1.0_rt+p_us*inv_c2)*p_w;

                return {p_w,
                        p_x_mean, p_y_mean, p_z_mean,
                        p_ux_mean, p_uy_mean, p_uz_mean,
                        p_gm_mean};
            },
            reduce_ops_1);

        std::vector<ParticleReal> values_per_rank_1st(num_red_ops_1);

        /* contains in this order:
         * w, x_mean, y_mean, z_mean
         * ux_mean, uy_mean, uz_mean, gm_mean
         */
        amrex::constexpr_for<0, num_red_ops_1> ([&](auto i) {
            values_per_rank_1st[i] = amrex::get<i>(r1);
        });

        // reduced sum over mpi ranks (allreduce)
        amrex::ParallelAllReduce::Sum
        ( values_per_rank_1st.data(), static_cast<int>(values_per_rank_1st.size()), ParallelDescriptor::Communicator());

        const ParticleReal w_sum   = values_per_rank_1st.at(0);
        const ParticleReal x_mean  = values_per_rank_1st.at(1) /= w_sum;
        const ParticleReal y_mean  = values_per_rank_1st.at(2) /= w_sum;
        const ParticleReal z_mean  = values_per_rank_1st.at(3) /= w_sum;
        const ParticleReal ux_mean = values_per_rank_1st.at(4) /= w_sum;
        const ParticleReal uy_mean = values_per_rank_1st.at(5) /= w_sum;
        const ParticleReal uz_mean = values_per_rank_1st.at(6) /= w_sum;
        const ParticleReal gm_mean = values_per_rank_1st.at(7) /= w_sum;

        if (w_sum < std::numeric_limits<Real>::min() )
        {
            for (auto& item: m_data) { item = 0.0_rt; }

            return;
        }

        // number of reduction operations in second concurrent batch
        constexpr size_t num_red_ops_2 = 11;

        TypeMultiplier<amrex::ReduceOps, ReduceOpSum[num_red_ops_2]> reduce_ops2;
        using ReducedDataT2 = TypeMultiplier<amrex::ReduceData, ParticleReal[num_red_ops_2]>;

        auto r2 = amrex::ParticleReduce<ReducedDataT2>(
            myspc,
            [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> ReducedDataT2::Type
            {
                const ParticleReal p_ux = p.rdata(PIdx::ux);
                const ParticleReal p_uy = p.rdata(PIdx::uy);
                const ParticleReal p_uz = p.rdata(PIdx::uz);
                const ParticleReal p_us = p_ux*p_ux + p_uy*p_uy + p_uz*p_uz;
                const ParticleReal p_gm = std::sqrt(1.0_rt+p_us*inv_c2);
                const ParticleReal p_w = p.rdata(PIdx::w);

                ParticleReal p_x, p_y, p_z;
                get_particle_position(p, p_x, p_y, p_z);

                const ParticleReal p_x_ms = (p_x-x_mean)*(p_x-x_mean)*p_w;
                const ParticleReal p_y_ms = (p_y-y_mean)*(p_y-y_mean)*p_w;
                const ParticleReal p_z_ms = (p_z-z_mean)*(p_z-z_mean)*p_w;

                const ParticleReal p_ux_ms = (p_ux-ux_mean)*(p_ux-ux_mean)*p_w;
                const ParticleReal p_uy_ms = (p_uy-uy_mean)*(p_uy-uy_mean)*p_w;
                const ParticleReal p_uz_ms = (p_uz-uz_mean)*(p_uz-uz_mean)*p_w;
                const ParticleReal p_gm_ms = (p_gm-gm_mean)*(p_gm-gm_mean)*p_w;

                const ParticleReal p_xux = (p_x-x_mean)*(p_ux-ux_mean)*p_w;
                const ParticleReal p_yuy = (p_y-y_mean)*(p_uy-uy_mean)*p_w;
                const ParticleReal p_zuz = (p_z-z_mean)*(p_uz-uz_mean)*p_w;

                const ParticleReal p_charge = q*p_w;

                return {p_x_ms, p_y_ms, p_z_ms,
                        p_ux_ms, p_uy_ms, p_uz_ms,
                        p_gm_ms,
                        p_xux, p_yuy, p_zuz,
                        p_charge};
            },
            reduce_ops2);

        std::vector<ParticleReal> values_per_rank_2nd(num_red_ops_2);

        /* contains in this order:
         * x_ms, y_ms, z_ms
         * ux_ms, uy_ms, uz_ms,
         * gm_ms
         * xux, yuy, zuz,
         * charge
         */
        amrex::constexpr_for<0, num_red_ops_2> ([&](auto i) {
            values_per_rank_2nd[i] = amrex::get<i>(r2);
        });

        // reduced sum over mpi ranks (reduce to IO rank)
        ParallelDescriptor::ReduceRealSum
        ( values_per_rank_2nd.data(), static_cast<int>(values_per_rank_2nd.size()), ParallelDescriptor::IOProcessorNumber());

        const ParticleReal x_ms   = values_per_rank_2nd.at(0) /= w_sum;
        const ParticleReal y_ms   = values_per_rank_2nd.at(1) /= w_sum;
        const ParticleReal z_ms   = values_per_rank_2nd.at(2) /= w_sum;
        const ParticleReal ux_ms  = values_per_rank_2nd.at(3) /= w_sum;
        const ParticleReal uy_ms  = values_per_rank_2nd.at(4) /= w_sum;
        const ParticleReal uz_ms  = values_per_rank_2nd.at(5) /= w_sum;
        const ParticleReal gm_ms  = values_per_rank_2nd.at(6) /= w_sum;
        const ParticleReal xux    = values_per_rank_2nd.at(7) /= w_sum;
        const ParticleReal yuy    = values_per_rank_2nd.at(8) /= w_sum;
        const ParticleReal zuz    = values_per_rank_2nd.at(9) /= w_sum;
        const ParticleReal charge = values_per_rank_2nd.at(10);

        // save data
#if (defined WARPX_DIM_3D || defined WARPX_DIM_RZ)
        m_data[0]  = x_mean;
        m_data[1]  = y_mean;
        m_data[2]  = z_mean;
        m_data[3]  = ux_mean * m;
        m_data[4]  = uy_mean * m;
        m_data[5]  = uz_mean * m;
        m_data[6]  = gm_mean;
        m_data[7]  = std::sqrt(x_ms);
        m_data[8]  = std::sqrt(y_ms);
        m_data[9]  = std::sqrt(z_ms);
        m_data[10] = std::sqrt(ux_ms) * m;
        m_data[11] = std::sqrt(uy_ms) * m;
        m_data[12] = std::sqrt(uz_ms) * m;
        m_data[13] = std::sqrt(gm_ms);
        m_data[14] = std::sqrt(x_ms*ux_ms-xux*xux) / PhysConst::c;
        m_data[15] = std::sqrt(y_ms*uy_ms-yuy*yuy) / PhysConst::c;
        m_data[16] = std::sqrt(z_ms*uz_ms-zuz*zuz) / PhysConst::c;
        m_data[17] = - (PhysConst::c * xux) / std::sqrt(x_ms*ux_ms-xux*xux);
        m_data[18] = - (PhysConst::c * yuy) / std::sqrt(y_ms*uy_ms-yuy*yuy);
        m_data[19] = (PhysConst::c * x_ms) / std::sqrt(x_ms*ux_ms-xux*xux);
        m_data[20] = (PhysConst::c * y_ms) / std::sqrt(y_ms*uy_ms-yuy*yuy);
        m_data[21] = charge;
#elif (defined WARPX_DIM_XZ)
        m_data[0]  = x_mean;
        m_data[1]  = z_mean;
        m_data[2]  = ux_mean * m;
        m_data[3]  = uy_mean * m;
        m_data[4]  = uz_mean * m;
        m_data[5]  = gm_mean;
        m_data[6]  = std::sqrt(x_ms);
        m_data[7]  = std::sqrt(z_ms);
        m_data[8]  = std::sqrt(ux_ms) * m;
        m_data[9]  = std::sqrt(uy_ms) * m;
        m_data[10] = std::sqrt(uz_ms) * m;
        m_data[11] = std::sqrt(gm_ms);
        m_data[12] = std::sqrt(x_ms*ux_ms-xux*xux) / PhysConst::c;
        m_data[13] = std::sqrt(z_ms*uz_ms-zuz*zuz) / PhysConst::c;
        m_data[14] = - (PhysConst::c * xux) / std::sqrt(x_ms*ux_ms-xux*xux);
        m_data[15] = (PhysConst::c * x_ms) / std::sqrt(x_ms*ux_ms-xux*xux);
        m_data[16] = charge;
        amrex::ignore_unused(y_mean, y_ms, yuy);
#elif (defined WARPX_DIM_1D_Z)
        m_data[0]  = z_mean;
        m_data[1]  = ux_mean * m;
        m_data[2]  = uy_mean * m;
        m_data[3]  = uz_mean * m;
        m_data[4]  = gm_mean;
        m_data[5]  = std::sqrt(z_ms);
        m_data[6]  = std::sqrt(ux_ms) * m;
        m_data[7]  = std::sqrt(uy_ms) * m;
        m_data[8]  = std::sqrt(uz_ms) * m;
        m_data[9]  = std::sqrt(gm_ms);
        m_data[10] = std::sqrt(z_ms*uz_ms-zuz*zuz) / PhysConst::c;
        m_data[11] = charge;
        amrex::ignore_unused(x_mean, x_ms, xux, y_mean, y_ms, yuy);
#endif
    }
    // end loop over species
}
// end void BeamRelevant::ComputeDiags
