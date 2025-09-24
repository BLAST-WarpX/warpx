/* Copyright 2020-2021 Michael Rowan, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "LoadBalanceEfficiency.H"

#include "Diagnostics/ReducedDiags/ReducedDiags.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <ostream>
#include <vector>

using namespace amrex;

// constructor
LoadBalanceEfficiency::LoadBalanceEfficiency (
    const std::string& rd_name,
    std::shared_ptr<warpx::load_balancing::LoadBalancer> p_load_balancer)
    : ReducedDiags{rd_name}, m_load_balancer{p_load_balancer}
{
    // read number of levels
    int nLevel = 0;
    const ParmParse pp_amr("amr");
    pp_amr.query("max_level", nLevel);
    nLevel += 1;

    // resize data array
    m_data.resize(nLevel, 0.0_rt);

    if (ParallelDescriptor::IOProcessor())
    {
        if ( m_write_header )
        {
            // open file
            std::ofstream ofs{m_path + m_rd_name + "." + m_extension, std::ofstream::out};

            // write header row
            int c = 0;
            ofs << "#";
            ofs << "[" << c++ << "]step()";
            ofs << m_sep;
            ofs << "[" << c++ << "]time(s)";
            for (int lev = 0; lev < nLevel; ++lev)
            {
                ofs << m_sep;
                ofs << "[" << c++ << "]lev" + std::to_string(lev);
            }
            ofs << "\n";

            // close file
            ofs.close();
        }
    }
}

// Get the load balance efficiency
void LoadBalanceEfficiency::ComputeDiags (int step)
{
    // Judge if the diags should be done
    if (!m_intervals.contains(step+1)) { return; }

    // loop over refinement levels
    for (int lev = 0; lev < m_load_balancer->get_nlevs(); ++lev)
    {
        // save data
        m_data[lev] = m_load_balancer->get_load_balance_efficiency(lev);
    }
    // end loop over refinement levels

    /* m_data now contains up-to-date values for:
     *  [load balance efficiency at level 0,
     *   load balance efficiency at level 1,
     *   load balance efficiency at level 2,
     *   ......] */
}
