/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "MomentBalance.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>

namespace warpx::radiation::detail
{
void
UpdateMomentGlobalBalance (amrex::MultiFab const& state, amrex::MultiFab const& old,
                           amrex::MultiFab const& transfer, ImplicitMomentTransportResult& result,
                           FourVector const& boundary, FourVector const& geometric)
{
#if !defined(WARPX_DIM_RZ)
    amrex::ignore_unused(geometric);
#endif
    result.momentum_residual = 0;
    for (int d = 0; d < 4; ++d)
    {
        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
                         amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum>
            ops;
        amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real,
                          amrex::Real>
            data(ops);
        using Tuple = typename decltype(data)::Type;
        for (amrex::MFIter iterator(state); iterator.isValid(); ++iterator)
        {
            auto const current = state.const_array(iterator);
            auto const initial = old.const_array(iterator);
            auto const source = transfer.const_array(iterator);
            ops.eval(iterator.validbox(), data,
                     [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple
                     {
                         auto const a = current(i, j, k, d);
                         auto const b = initial(i, j, k, d);
                         auto const c = source(i, j, k, d);
                         return {a, b, c, std::abs(a), std::abs(b), std::abs(c)};
                     });
        }
        auto const values = data.value();
        amrex::Real totals[6] = {amrex::get<0>(values), amrex::get<1>(values),
                                 amrex::get<2>(values), amrex::get<3>(values),
                                 amrex::get<4>(values), amrex::get<5>(values)};
        amrex::ParallelDescriptor::ReduceRealSum(totals, 6);
#if defined(WARPX_DIM_RZ)
        auto const imbalance = totals[0] - totals[1] + totals[2] + boundary[d] - geometric[d];
        auto const scale =
            totals[3] + totals[4] + totals[5] + std::abs(boundary[d]) + std::abs(geometric[d]);
#else
        auto const imbalance = totals[0] - totals[1] + totals[2] + boundary[d];
        auto const scale = totals[3] + totals[4] + totals[5] + std::abs(boundary[d]);
#endif
        auto const error = std::isfinite(imbalance) && std::isfinite(scale)
                               ? (scale > 0 ? std::abs(imbalance) / scale : std::abs(imbalance))
                               : std::numeric_limits<amrex::Real>::infinity();
        if (d == 0)
        {
            result.energy_residual = error;
        }
        else
        {
            result.momentum_residual = amrex::max(result.momentum_residual, error);
        }
    }
}
} // namespace warpx::radiation::detail
