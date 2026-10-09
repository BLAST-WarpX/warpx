/* Copyright 2026 Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "MatrixPCAlgMG.H"

#include "Utils/TextMsg.H"

#include <AMReX_AlgMG.H>
#include <AMReX_AlgPartition.H>
#include <AMReX_AlgVector.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_SpMatrix.H>

#include <utility>

struct MatrixPCAlgMG::Impl
{
    using RT = amrex::Real;

    std::string m_name;

    int m_max_iter = 1;
    int m_verbose = 0;
    amrex::AlgMGSmoother m_smoother = amrex::AlgMGSmoother::chebyshev;
    int m_cheby_degree = 4;
    RT m_strong_threshold = RT(0.25);
    int m_aggressive_levels = 0;
    int m_max_levels = 25;

    amrex::AlgPartition m_partition;
    std::unique_ptr<amrex::SpMatrix<RT>> m_mat;
    std::unique_ptr<amrex::AlgMG<RT>> m_algmg;
    amrex::AlgVector<RT> m_x;
    amrex::AlgVector<RT> m_b;
};

MatrixPCAlgMG::MatrixPCAlgMG (const std::string& a_name,
                              const amrex::Vector<amrex::Long>& a_row_offsets)
    : m_impl(std::make_unique<Impl>())
{
    auto& d = *m_impl;
    d.m_name = a_name;

    const amrex::ParmParse pp(a_name);
    pp.query("max_iter", d.m_max_iter);
    pp.query("algmg_verbose", d.m_verbose);
    pp.query_enum_case_insensitive("smoother", d.m_smoother);
    pp.query("chebyshev_degree", d.m_cheby_degree);
    pp.query("strong_threshold", d.m_strong_threshold);
    pp.query("aggressive_levels", d.m_aggressive_levels);
    pp.query("max_levels", d.m_max_levels);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(d.m_max_iter >= 1,
        a_name + ".max_iter must be at least 1");

    d.m_partition.define(a_row_offsets);
    d.m_x.define(d.m_partition);
    d.m_b.define(d.m_partition);
}

MatrixPCAlgMG::~MatrixPCAlgMG () = default;

void MatrixPCAlgMG::printParameters () const
{
    auto const& d = *m_impl;
    amrex::Print() << d.m_name << " max_iter:               " << d.m_max_iter << "\n";
    amrex::Print() << d.m_name << " smoother:               "
                   << amrex::getEnumNameString(d.m_smoother) << "\n";
    amrex::Print() << d.m_name << " chebyshev_degree:       " << d.m_cheby_degree << "\n";
    amrex::Print() << d.m_name << " strong_threshold:       " << d.m_strong_threshold << "\n";
    amrex::Print() << d.m_name << " aggressive_levels:      " << d.m_aggressive_levels << "\n";
    amrex::Print() << d.m_name << " max_levels:             " << d.m_max_levels << "\n";
}

void MatrixPCAlgMG::Setup (int a_nrows, int a_ncols_max, const int* a_num_nz,
                           const int* a_cols, const amrex::Real* a_vals)
{
    BL_PROFILE("MatrixPCAlgMG::Setup()");
    using RT = Impl::RT;
    auto& d = *m_impl;

    // Row-major arrays padded to a_ncols_max -> CSR; padding is dropped as
    // invalid entries.
    const auto n = amrex::Long(a_nrows);
    const int nc = a_ncols_max;
    amrex::Gpu::DeviceVector<amrex::Long> cols(n*nc);
    amrex::Gpu::DeviceVector<amrex::Long> offsets(n+1);
    amrex::Gpu::DeviceVector<RT> vals(n*nc);
    auto* pc = cols.data();
    auto* po = offsets.data();
    auto* pv = vals.data();
    amrex::ParallelFor(n+1, [=] AMREX_GPU_DEVICE (amrex::Long i)
    {
        po[i] = i*nc;
        if (i < n) {
            for (int k = 0; k < nc; ++k) {
                const bool valid = k < a_num_nz[i];
                pc[i*nc+k] = valid ? amrex::Long(a_cols[i*nc+k]) : amrex::Long(-1);
                pv[i*nc+k] = valid ? RT(a_vals[i*nc+k]) : RT(0);
            }
        }
    });
    amrex::Gpu::streamSynchronize();

    d.m_algmg.reset(); // it refers to m_mat
    d.m_mat = std::make_unique<amrex::SpMatrix<RT>>();
    d.m_mat->define(d.m_partition, pv, pc, n*nc, po,
                    amrex::CsrSorted{false}, amrex::CsrValid{false});

    d.m_algmg = std::make_unique<amrex::AlgMG<RT>>(*d.m_mat);
    d.m_algmg->setVerbose(d.m_verbose);
    d.m_algmg->setSmoother(d.m_smoother);
    d.m_algmg->setChebyshevDegree(d.m_cheby_degree);
    d.m_algmg->setStrongThreshold(d.m_strong_threshold);
    d.m_algmg->setAggressiveNumLevels(d.m_aggressive_levels);
    d.m_algmg->setMaxLevels(d.m_max_levels);
    d.m_algmg->setFixedIter(d.m_max_iter);
    d.m_algmg->setup();
}

amrex::Real* MatrixPCAlgMG::rhs ()
{
    return m_impl->m_b.data();
}

const amrex::Real* MatrixPCAlgMG::sol () const
{
    return m_impl->m_x.data();
}

void MatrixPCAlgMG::Apply ()
{
    auto& d = *m_impl;
    if (d.m_max_iter == 1) {
        d.m_algmg->precond(d.m_x, d.m_b);
    } else {
        // A fixed number of V-cycles keeps the preconditioner linear.
        d.m_x.setVal(Impl::RT(0));
        d.m_algmg->solve(d.m_x, d.m_b);
    }
}
