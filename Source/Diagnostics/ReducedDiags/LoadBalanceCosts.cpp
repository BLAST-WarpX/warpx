/* Copyright 2019-2020 Michael Rowan, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "LoadBalanceCosts.H"

#include "Diagnostics/ReducedDiags/ReducedDiags.H"
#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_LayoutData.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>

#ifdef AMREX_USE_MPI
#   include <mpi.h>
#endif

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <istream>
#include <memory>
#include <string>
#include <utility>

using namespace amrex;
using warpx::fields::FieldType;

namespace
{
    amrex::Long
    countBoxMacroParticles (amrex::MFIter const & mfi, int const lev)
    {
        const int gid = mfi.index();
        const int tid = mfi.LocalTileIndex();
        auto box_index = std::make_pair(gid, tid);

        auto & warpx = WarpX::GetInstance();
        MultiParticleContainer const & mpc = warpx.GetPartContainer();
        int const nSpecies = mpc.nSpecies();

        amrex::Long num_macro_particles = 0;
        for (int i_s = 0; i_s < nSpecies; ++i_s)
        {
            WarpXParticleContainer const & pc = mpc.GetParticleContainer(i_s);
            auto const & plev  = pc.GetParticles(lev);

            auto const & ptile = plev.at(box_index);
            auto const np = ptile.numParticles();
            num_macro_particles += np;
        }

        return num_macro_particles;
    }
}

// constructor
LoadBalanceCosts::LoadBalanceCosts (const std::string& rd_name)
    : ReducedDiags{rd_name}
{
}

// function that gathers costs
void LoadBalanceCosts::ComputeDiags (int step)
{
    // get a reference to WarpX instance
    auto& warpx = WarpX::GetInstance();

    // judge if the diags should be done
    // costs is initialized only if we're doing load balance
    if (!m_intervals.contains(step+1) ||
        !warpx.get_load_balance_intervals().isActivated() ) { return; }

    // get number of boxes over all levels
    auto nLevels = warpx.finestLevel() + 1;
    int nBoxes = 0;
    for (int lev = 0; lev < nLevels; ++lev)
    {
        auto *const cost = WarpX::getCosts(lev);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            cost, "ERROR: costs are not initialized on level " + std::to_string(lev) + " !");
        nBoxes += cost->size();
    }

    // keep track of the max number of boxes, this is needed later on to fill
    // the jagged array (in case each step does not have the same number of boxes)
    m_nBoxesMax = std::max(m_nBoxesMax, nBoxes);

    // resize and clear data array
    const size_t dataSize =
        static_cast<size_t>(m_nDataFields)*
        static_cast<size_t>(nBoxes);
    m_data.resize(dataSize, 0.0_rt);
    m_data.assign(dataSize, 0.0_rt);

    // read in WarpX costs to local copy; compute if using `Heuristic` update
    amrex::Vector<std::unique_ptr<amrex::LayoutData<amrex::Real> > > costs;

    costs.resize(nLevels);
    for (int lev = 0; lev < nLevels; ++lev)
    {
        costs[lev] = std::make_unique<LayoutData<Real>>(*WarpX::getCosts(lev));
    }

    if (WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Heuristic)
    {
        warpx.ComputeCostsHeuristic(costs);
    }

    // keep track of correct index in array over all boxes on all levels
    // shift index for m_data
    int shift_m_data = 0;

    using ablastr::fields::Direction;

    // save data
    for (int lev = 0; lev < nLevels; ++lev)
    {
        const amrex::DistributionMapping& dm = warpx.DistributionMap(lev);
        const MultiFab & Ex = *warpx.m_fields.get(FieldType::Efield_aux, Direction{0}, lev);
        for (MFIter mfi(Ex, false); mfi.isValid(); ++mfi)
        {
            const Box& tbx = mfi.tilebox();
            m_data[shift_m_data + mfi.index()*m_nDataFields + 0] = (*costs[lev])[mfi.index()];
            m_data[shift_m_data + mfi.index()*m_nDataFields + 1] = dm[mfi.index()];
            m_data[shift_m_data + mfi.index()*m_nDataFields + 2] = lev;
            m_data[shift_m_data + mfi.index()*m_nDataFields + 3] = tbx.loVect()[0];
#if (AMREX_SPACEDIM >= 2)
            m_data[shift_m_data + mfi.index()*m_nDataFields + 4] = tbx.loVect()[1];
#else
            m_data[shift_m_data + mfi.index()*m_nDataFields + 4] = 0.;
#endif
#if defined(WARPX_DIM_3D)
            m_data[shift_m_data + mfi.index()*m_nDataFields + 5] = tbx.loVect()[2];
#else
            m_data[shift_m_data + mfi.index()*m_nDataFields + 5] = 0.;
#endif
            m_data[shift_m_data + mfi.index()*m_nDataFields + 6] = static_cast<amrex::Real>(tbx.d_numPts()); // note: difference to volume
            m_data[shift_m_data + mfi.index()*m_nDataFields + 7] = countBoxMacroParticles(mfi, lev);
#ifdef AMREX_USE_GPU
            m_data[shift_m_data + mfi.index()*m_nDataFields + 8] = amrex::Gpu::Device::deviceId();
#endif
            // ...
        }

        // we looped through all the boxes on level lev, update the shift index
        shift_m_data += m_nDataFields*(costs[lev]->size());
    }

    // parallel reduce to IO proc and get data over all procs
    ParallelDescriptor::ReduceRealSum(m_data.data(),
                                      static_cast<int>(m_data.size()),
                                      ParallelDescriptor::IOProcessorNumber());

#ifdef AMREX_USE_MPI
    // now parallel reduce to IO proc and get string data (host name) over all procs
    // MPI Gatherv preliminaries

    // get the MPI host name and number of characters
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int length;

    BL_MPI_REQUIRE( MPI_Get_processor_name( hostname, &length ) );

    // IO proc will collect messages from other procs;
    // receive counts and displacements needed only by IO proc
    m_data_string_recvcount.resize(ParallelDescriptor::NProcs(), 0);
    m_data_string_recvcount.assign(ParallelDescriptor::NProcs(), 0);
    m_data_string_disp.resize(ParallelDescriptor::NProcs(), 0);
    m_data_string_disp.assign(ParallelDescriptor::NProcs(), 0);

    // get the string lengths on IO proc
    ParallelDescriptor::Gather(&length, 1,                     // send
                               m_data_string_recvcount.data(), 1, // receive
                               ParallelDescriptor::IOProcessorNumber());

    // determine total length of collected strings for root, and set displacements;
    // + 1 is for chosen separation between words in the gathered string; this
    // chosen separator character is set further below when elements of
    // m_data_string_recvbuf are initialized
    m_data_string_recvbuf_length += m_data_string_recvcount[0] + 1;
    for (int i=1; i<m_data_string_disp.size(); i++)
    {
        m_data_string_recvbuf_length += (m_data_string_recvcount[i] + 1);

        // displacements is cumulative sum along recvcount, include the (+ 1) space separator or null terminator
        m_data_string_disp[i] = m_data_string_disp[i-1] + m_data_string_recvcount[i-1] + 1;
    }

    // knowing the total length of string on IOProc, we can allocate memory for the receive buffer
    // initialize spaces, null terminator is the last element
    m_data_string_recvbuf.resize(m_data_string_recvbuf_length, ' ');
    m_data_string_recvbuf.assign(m_data_string_recvbuf_length, ' ');
    m_data_string_recvbuf[m_data_string_recvbuf_length-1] = '\0';

    // now the root process knows cnts and locations to place messages from sending processes;
    // collect the hostnames; m_data_string_recvbuf will provide mapping from rank-->hostname
    ParallelDescriptor::Gatherv(&hostname[0],              /* hostname ID */
                                length,                    /* length of hostname */
                                m_data_string_recvbuf.data(), /* write data into string buffer */
                                m_data_string_recvcount,   /* how many messages to receive */
                                m_data_string_disp,        /* starting position in recv buffer to place received msg */
                                ParallelDescriptor::IOProcessorNumber());
#endif

    // cleanup
    if (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber())
    {
#ifdef AMREX_USE_MPI
        const std::string m_data_stdstring_recvbuf(m_data_string_recvbuf.begin(), m_data_string_recvbuf.end());
        m_data_string = amrex::Tokenize(m_data_stdstring_recvbuf, " ");
#endif
    }

    /* m_data now contains up-to-date values for:
     *  [[cost, proc, lev, i_low, j_low, k_low, num_cells, num_macro_particles(, gpu_ID [if GPU run]) ] of box 0 at level 0,
     *   [cost, proc, lev, i_low, j_low, k_low, num_cells, num_macro_particles(, gpu_ID [if GPU run]) ] of box 1 at level 0,
     *   [cost, proc, lev, i_low, j_low, k_low, num_cells, num_macro_particles(, gpu_ID [if GPU run]) ] of box 2 at level 0,
     *   ...
     *   [cost, proc, lev, i_low, j_low, k_low num_cells, num_macro_particles(, gpu_ID [if GPU run]) ] of box 0 at level 1,
     *   [cost, proc, lev, i_low, j_low, k_low, num_cells, num_macro_particles(, gpu_ID [if GPU run]) ] of box 1 at level 1,
     *   [cost, proc, lev, i_low, j_low, k_low, num_cells, num_macro_particles(, gpu_ID [if GPU run]) ] of box 2 at level 1,
     *   ...]
     * and m_data_string contains:
     *  [hostname of box 0 at level 0,
     *   hostname of box 1 at level 0,
     *   hostname of box 2 at level 0,
     *   ...
     *   hostname of box 0 at level 1,
     *   hostname of box 1 at level 1,
     *   hostname of box 2 at level 1,
     *   ...]
     */
}

// write to file function for cost
void LoadBalanceCosts::WriteToFile (int step) const
{
    // open file
    std::ofstream ofs{m_path + m_rd_name + "." + m_extension,
            std::ofstream::out | std::ofstream::app};

    // write step
    ofs << step+1 << m_sep;

    // set precision
    ofs << std::fixed << std::setprecision(14) << std::scientific;

    // write time
    ofs << WarpX::GetInstance().gett_new(0);

    // loop over data size and write
    for (int i = 0; i < static_cast<int>(m_data.size()); ++i)
    {
        ofs << m_sep << m_data[i];
        if ((i - m_nDataFields + 1)%m_nDataFields == 0)
        {
            // at the end of current group of m_nDatafields, output the string data (hostname)
            const int ind_rank = i - m_nDataFields + 2; // index for the rank corresponding to current box

            // m_data --> rank --> hostname
            ofs << m_sep << m_data_string[static_cast<long unsigned int>(m_data[ind_rank])];
        }
    }
    // end loop over data size

    // end line
    ofs << "\n";

    // close file
    ofs.close();

    // get a reference to WarpX instance
    auto& warpx = WarpX::GetInstance();

    if (!ParallelDescriptor::IOProcessor()) { return; }

    // final step is a special case, fill jagged array with NaN
    if (m_intervals.nextContains(step+1) > warpx.maxStep())
    {
        // open tmp file to copy data
        const std::string fileTmpName = m_path + m_rd_name + ".tmp." + m_extension;
        std::ofstream ofstmp(fileTmpName, std::ofstream::out);

        // write header row
        // for each box on each level we saved 9(10) data fields:
        //   [cost, proc, lev, i_low, j_low, k_low, num_cells, num_macro_particles(, gpu_ID_box), hostname]
        // nDataFieldsToWrite = below accounts for the Real data fields (m_nDataFields), then 1 string output to write
        const int nDataFieldsToWrite = m_nDataFields + 1;

        int c = 0;
        ofstmp << "#";
        ofstmp << "[" << c++ << "]step()";
        ofstmp << m_sep;
        ofstmp << "[" << c++ << "]time(s)";

        for (int boxNumber=0; boxNumber<m_nBoxesMax; ++boxNumber)
        {
            ofstmp << m_sep;
            ofstmp << "[" << c++ << "]cost_box_" + std::to_string(boxNumber) + "()";
            ofstmp << m_sep;
            ofstmp << "[" << c++ << "]proc_box_" + std::to_string(boxNumber) + "()";
            ofstmp << m_sep;
            ofstmp << "[" << c++ << "]lev_box_" + std::to_string(boxNumber) + "()";
            ofstmp << m_sep;
            ofstmp << "[" << c++ << "]i_low_box_" + std::to_string(boxNumber) + "()";
            ofstmp << m_sep;
            ofstmp << "[" << c++ << "]j_low_box_" + std::to_string(boxNumber) + "()";
            ofstmp << m_sep;
            ofstmp << "[" << c++ << "]k_low_box_" + std::to_string(boxNumber) + "()";
            ofstmp << m_sep;
            ofstmp << "[" << c++ << "]num_cells_" + std::to_string(boxNumber) + "()";
            ofstmp << m_sep;
            ofstmp << "[" << c++ << "]num_macro_particles_" + std::to_string(boxNumber) + "()";
#ifdef AMREX_USE_GPU
            ofstmp << m_sep;
            ofstmp << "[" << c++ << "]gpu_ID_box_" + std::to_string(boxNumber) + "()";
#endif
            ofstmp << m_sep;
            ofstmp << "[" << c++ << "]hostname_box_" + std::to_string(boxNumber) + "()";
        }
        ofstmp << "\n";

        // open the data-containing file
        const std::string fileDataName = m_path + m_rd_name + "." + m_extension;
        std::ifstream ifs(fileDataName, std::ifstream::in);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(ifs, "Failed to load balance file");
        ifs.exceptions(std::ios_base::badbit); // | std::ios_base::failbit

        // Fill in the tmp costs file with data, padded with NaNs
        for (std::string lineIn; std::getline(ifs, lineIn);)
        {
            // count the elements in the input line
            int cnt = 0;
            std::stringstream ss(lineIn);
            std::string token;

            while (std::getline(ss, token, m_sep[0]))
            {
                cnt += 1;
                if (ss.peek() == m_sep[0]) { ss.ignore(); }
            }

            // 2 columns for step, time; then nBoxes*nDatafields columns for data;
            // then nBoxes*1 columns for hostname;
            // then fill the remaining columns (i.e., up to 2 + m_nBoxesMax*nDataFieldsToWrite)
            // with NaN, so the array is not jagged
            ofstmp << lineIn;
            for (int i=0; i<(m_nBoxesMax*nDataFieldsToWrite - (cnt - 2)); ++i)
            {
                ofstmp << m_sep << "NaN";
            }
            ofstmp << "\n";
        }

        // close files
        ifs.close();
        ofstmp.close();

        // remove the original, rename tmp file
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::remove(fileDataName.c_str()) == EXIT_SUCCESS,
            "Failed to remove " + fileDataName);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::rename(fileTmpName.c_str(), fileDataName.c_str()) == EXIT_SUCCESS,
            "Failed to rename " + fileTmpName + " into " + fileDataName);
    }
}
