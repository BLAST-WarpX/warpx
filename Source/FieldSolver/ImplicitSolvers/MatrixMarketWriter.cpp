/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "MatrixMarketWriter.H"
#include "WarpXSolverDOF.H"

#include <AMReX_iMultiFab.H>
#include <AMReX_IndexType.H>
#include <AMReX_Loop.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <utility>

namespace
{
    // Sort the entries by (row, column)
    void SortTriplets (MatrixMarketWriter::Triplets& a_t)
    {
        std::vector<std::size_t> perm(a_t.row.size());
        std::iota(perm.begin(), perm.end(), std::size_t(0));
        std::sort(perm.begin(), perm.end(), [&a_t] (std::size_t a, std::size_t b) {
            return std::make_pair(a_t.row[a], a_t.col[a]) < std::make_pair(a_t.row[b], a_t.col[b]);
        });
        MatrixMarketWriter::Triplets sorted;
        sorted.row.reserve(perm.size());
        sorted.col.reserve(perm.size());
        sorted.val.reserve(perm.size());
        for (auto p : perm) { sorted.add(a_t.row[p], a_t.col[p], a_t.val[p]); }
        a_t = std::move(sorted);
    }

    // Call a_write on each rank in turn, in rank order
    template <class F>
    void InRankOrder (F&& a_write)
    {
        for (int r = 0; r < amrex::ParallelDescriptor::NProcs(); ++r) {
            if (r == amrex::ParallelDescriptor::MyProc()) { a_write(); }
            amrex::ParallelDescriptor::Barrier();
        }
    }

    std::string Join (const amrex::Vector<amrex::Long>& a_v)
    {
        std::ostringstream ss;
        for (amrex::Long i = 0; i < a_v.size(); ++i) { ss << (i > 0 ? " " : "") << a_v[i]; }
        return ss.str();
    }

    std::string Format (amrex::Real a_x)
    {
        std::ostringstream ss;
        ss.precision(10);
        ss << a_x;
        return ss.str();
    }

    // Row statistics of sorted entries, MPI-reduced
    std::vector<std::string> RowStatistics (const MatrixMarketWriter::Triplets& a_t)
    {
        constexpr auto big = std::numeric_limits<amrex::Real>::max();
        amrex::Real diag_min = big, diag_max = -big, ratio_max = 0, inf_norm = 0;
        amrex::Long zero_diag = 0;
        std::size_t k = 0;
        while (k < a_t.row.size()) {
            const amrex::Long r = a_t.row[k];
            amrex::Real diag = 0, offdiag = 0;
            for (; k < a_t.row.size() && a_t.row[k] == r; ++k) {
                if (a_t.col[k] == r) { diag += a_t.val[k]; }
                else { offdiag += std::abs(a_t.val[k]); }
            }
            diag_min = std::min(diag_min, diag);
            diag_max = std::max(diag_max, diag);
            inf_norm = std::max(inf_norm, std::abs(diag) + offdiag);
            if (diag == 0) { ++zero_diag; }
            else { ratio_max = std::max(ratio_max, offdiag / std::abs(diag)); }
        }
        amrex::ParallelDescriptor::ReduceRealMin(diag_min);
        amrex::ParallelDescriptor::ReduceRealMax(diag_max);
        amrex::ParallelDescriptor::ReduceRealMax(ratio_max);
        amrex::ParallelDescriptor::ReduceRealMax(inf_norm);
        amrex::ParallelDescriptor::ReduceLongSum(zero_diag);

        // Gershgorin: the eigenvalues of D^-1 A lie in [1 - r, 1 + r]
        return {
            "diagonal_min: " + Format(diag_min),
            "diagonal_max: " + Format(diag_max),
            "zero_diagonal_rows: " + std::to_string(zero_diag),
            "max_offdiag_ratio: " + Format(ratio_max),
            "inf_norm: " + Format(inf_norm),
            "gershgorin_bounds_DinvA: " + Format(1 - ratio_max) + " " + Format(1 + ratio_max),
            "gershgorin_omega: " + Format(2 / (2 + ratio_max)),
        };
    }

    std::string ParentDirectory (const std::string& a_filename)
    {
        const auto pos = a_filename.rfind('/');
        return (pos == std::string::npos) ? std::string(".") : a_filename.substr(0, pos);
    }
}

void MatrixMarketWriter::Write (const std::string& a_filename,
                                const std::vector<std::string>& a_header,
                                const WarpXSolverDOF& a_dofs,
                                Triplets& a_local)
{
    BL_PROFILE("MatrixMarketWriter::Write()");
    using namespace amrex;

    SortTriplets(a_local);
    const auto stats = RowStatistics(a_local);

    const auto nnz_l = static_cast<Long>(a_local.row.size());
    Long nnz_g = nnz_l;
    ParallelDescriptor::ReduceLongSum(nnz_g);
    Vector<Long> nnz_all(ParallelDescriptor::NProcs());
    ParallelDescriptor::Gather(&nnz_l, 1, nnz_all.data(), 1,
                               ParallelDescriptor::IOProcessorNumber());

    WriteDOFMap(ParentDirectory(a_filename), a_dofs);

    const Long n_g = a_dofs.m_nDoFs_g;
    if (ParallelDescriptor::IOProcessor()) {
        std::ofstream f(a_filename, std::ios::trunc);
        f << "%%MatrixMarket matrix coordinate real general\n";
        for (const auto& h : a_header) { f << "% " << h << "\n"; }
        for (const auto& h : stats) { f << "% " << h << "\n"; }
        f << "% global_size: " << n_g << " x " << n_g << "\n";
        f << "% nnz: " << nnz_g << "\n";
        f << "% ranks: " << ParallelDescriptor::NProcs() << "\n";
        f << "% row_offsets: " << Join(a_dofs.m_row_offsets) << "\n";
        f << "% local_nnz: " << Join(nnz_all) << "\n";
        f << "% dof_map: dofmap.txt\n";
        f << n_g << " " << n_g << " " << nnz_g << "\n";
    }
    ParallelDescriptor::Barrier();

    InRankOrder([&] () {
        std::ofstream f(a_filename, std::ios::app);
        f.precision(17);
        for (std::size_t k = 0; k < a_local.row.size(); ++k) {
            f << a_local.row[k] + 1 << " " << a_local.col[k] + 1 << " " << a_local.val[k] << "\n";
        }
    });
}

void MatrixMarketWriter::WriteDOFMap (const std::string& a_dir, const WarpXSolverDOF& a_dofs)
{
    BL_PROFILE("MatrixMarketWriter::WriteDOFMap()");
    using namespace amrex;

    static std::set<std::string> written;
    if (written.contains(a_dir)) { return; }
    written.insert(a_dir);

    // row, level, field component, i, j, k
    std::vector<std::array<Long,6>> entries;
    for (int lev = 0; lev < static_cast<int>(a_dofs.m_array.size()); ++lev) {
        for (int n = 0; n < 3; ++n) {
            const auto* dof = a_dofs.m_array[lev][n].get();
            if (dof == nullptr) { continue; }
            for (MFIter mfi(*dof); mfi.isValid(); ++mfi) {
                const auto h = HostCopy((*dof)[mfi]);
                const auto a = h.const_array();
                LoopOnCpu(mfi.validbox(), [&] (int i, int j, int k) {
                    if (a(i,j,k,0) < 0) { return; }
                    entries.push_back({Long(a(i,j,k,1)), lev, n, i, j, k});
                });
            }
        }
    }
    std::sort(entries.begin(), entries.end());

    const std::string filename = a_dir + "/dofmap.txt";
    if (ParallelDescriptor::IOProcessor()) {
        std::ofstream f(filename, std::ios::trunc);
        f << "# WarpX DOF map of the preconditioner matrices\n";
        f << "# row: global row (1-based, as in the .mtx files); comp: field component (0: x, 1: y, 2: z)\n";
        f << "# i j k: cell indices; spatial dimensions: " << AMREX_SPACEDIM
          << " (indices beyond them are 0)\n";
        for (int n = 0; n < 3; ++n) {
            if (a_dofs.m_array.empty() || !a_dofs.m_array[0][n]) { continue; }
            const IndexType t = a_dofs.m_array[0][n]->ixType();
            f << "# index type of comp " << n << " (1: nodal):";
            for (int d = 0; d < AMREX_SPACEDIM; ++d) { f << " " << (t.nodeCentered(d) ? 1 : 0); }
            f << "\n";
        }
        f << "# row lev comp i j k\n";
    }
    ParallelDescriptor::Barrier();

    InRankOrder([&] () {
        std::ofstream f(filename, std::ios::app);
        for (const auto& e : entries) {
            f << e[0] + 1 << " " << e[1] << " " << e[2] << " "
              << e[3] << " " << e[4] << " " << e[5] << "\n";
        }
    });
}
