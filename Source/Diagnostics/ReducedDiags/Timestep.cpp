/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Thomas Marks
 *
 * License: BSD-3-Clause-LBNL
 */

#include "Timestep.H"

#include "WarpX.H"

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H> // TODO: remove this
#include <AMReX_REAL.H>

#include <ostream>

using namespace amrex::literals;

// constructor
Timestep::Timestep (const std::string& rd_name)
:ReducedDiags{rd_name}
{
    const auto& warpx = WarpX::GetInstance();
    const auto max_level = warpx.maxLevel();

    // data size should be equal to the number of refinement levels
    m_data.resize(max_level + 1, 0.0_rt);

    if (amrex::ParallelDescriptor::IOProcessor() && m_write_header) {
        // open file
        std::ofstream ofs{m_path + m_rd_name + "." + m_extension, std::ofstream::out};

        // write header row
        int c = 0;
        ofs << "#";
        ofs << "[" << c++ << "]step()";
        ofs << m_sep;
        ofs << "[" << c++ << "]time(s)";
        ofs << m_sep;

        for (int lev = 0; lev <= max_level; lev++) {
            ofs << "[" << c++ << "]timestep[" << lev << "](s)";
            if (lev < max_level) {
                ofs << m_sep;
            }
        }

        // close file
        ofs << std::endl;
        ofs.close();
    }
}
// end constructor

// function to get current simulation timestep at all refinement levels
void Timestep::ComputeDiags (int step) {
    // Check if diagnostic should be done
    if (!m_intervals.contains(step+1)) { return; }

    const auto& warpx = WarpX::GetInstance();
    const auto max_level = warpx.maxLevel();
    const auto dt = warpx.getdt();

    for (int lev = 0; lev <= max_level; lev++) {
        m_data[lev] = dt[lev];
    }
}
// end Timestep::ComputeDiags
