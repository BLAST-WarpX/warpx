/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef MATRIX_PRINT_CONTROL_H_
#define MATRIX_PRINT_CONTROL_H_

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

/**
 * \brief When and where to write the matrix of a preconditioner or linear solver.
 *
 *  Inputs, with the name of the PC or linear solver as prefix:
 *  - print_matrix (bool, default false)
 *  - print_matrix_steps (ints, zero-based time steps; default: only the
 *    first matrix update of the run). Every update in these steps is written.
 *  - print_matrix_dir (string, default "diags/pc_matrices")
 */
class MatrixPrintControl
{
public:

    void Define (const std::string& a_prefix)
    {
        m_prefix = a_prefix;
        const amrex::ParmParse pp(a_prefix);
        pp.query("print_matrix", m_print);
        m_steps_given = (pp.queryarr("print_matrix_steps", m_steps) != 0);
        pp.query("print_matrix_dir", m_dir);
    }

    /**
     * \brief Call at every matrix update. Returns true if this update is written,
     *        and creates the output directory if needed.
     */
    bool ShouldPrint (int a_step)
    {
        if (a_step != m_step) {
            m_step = a_step;
            m_update = 0;
        } else {
            ++m_update;
        }
        if (!m_print) { return false; }
        const bool print = m_steps_given
            ? (std::find(m_steps.begin(), m_steps.end(), a_step) != m_steps.end())
            : (m_num_printed == 0);
        if (!print) { return false; }

        if (m_num_printed == 0) {
            if (amrex::ParallelDescriptor::IOProcessor()) {
                if (!amrex::UtilCreateDirectory(m_dir, 0755)) {
                    amrex::CreateDirectoryFailed(m_dir);
                }
            }
            amrex::ParallelDescriptor::Barrier();
        }
        ++m_num_printed;
        std::ostringstream name;
        name << m_dir << "/" << m_prefix << "_step" << std::setw(5) << std::setfill('0')
             << a_step << "_update" << m_update << ".mtx";
        m_filename = name.str();
        amrex::Print() << m_prefix << ": writing matrix to " << m_filename << "\n";
        return true;
    }

    /** \brief Output file of the last update for which ShouldPrint() returned true. */
    [[nodiscard]] const std::string& FileName () const { return m_filename; }

    /** \brief Metadata lines identifying the matrix. */
    [[nodiscard]] std::vector<std::string> HeaderLines () const
    {
        return {"name: " + m_prefix,
                "step: " + std::to_string(m_step),
                "update_in_step: " + std::to_string(m_update)};
    }

    void printParameters () const
    {
        amrex::Print() << m_prefix << " print matrix:         "
                       << (m_print ? "true" : "false") << "\n";
    }

private:

    std::string m_prefix;
    bool m_print = false;
    bool m_steps_given = false;
    std::vector<int> m_steps;
    std::string m_dir = "diags/pc_matrices";

    int m_step = -1;
    int m_update = 0;
    int m_num_printed = 0;
    std::string m_filename;
};

#endif
