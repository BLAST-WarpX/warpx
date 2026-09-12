/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "MomentBoundaryExchange.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <cmath>

namespace warpx::radiation
{
    bool
    ComputeMomentBoundaryExchange (std::array<amrex::MultiFab const*, AMREX_SPACEDIM> const& fluxes,
                                   amrex::Geometry const& geometry, amrex::Real dt,
                                   FourVector& exchange)
    {
        if (geometry.Coord() != 0 || !(dt >= 0) || !std::isfinite(dt)) {
            return false;
        }
        // Validate metadata before entering collectives. Every rank must supply
        // the same globally defined face layouts, as for a MultiFab operation.
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            auto const* flux = fluxes[direction];
            if (!flux || flux->nComp() < 4 ||
                flux->ixType().toIntVect() != amrex::IntVect::TheDimensionVector(direction) ||
                !(geometry.CellSize(direction) > 0) ||
                !std::isfinite(dt / geometry.CellSize(direction))) {
                return false;
            }
            auto cells = flux->boxArray();
            cells.enclosedCells();
            // Disjoint cell boxes ensure each external face is counted once,
            // even though neighboring internal nodal faces have two copies.
            if (!cells.isDisjoint() || !cells.contains(geometry.Domain()) ||
                cells.numPts() != geometry.Domain().numPts()) {
                return false;
            }
        }
        FourVector candidate{};
        auto const domain = geometry.Domain();
        for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
            if (geometry.isPeriodic(direction)) {
                continue;
            }
            auto const& flux = *fluxes[direction];
            auto const factor = dt / geometry.CellSize(direction);
            for (int component = 0; component < 4; ++component) {
                amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpMax> ops;
                amrex::ReduceData<amrex::Real, int> data(ops);
                using Tuple = typename decltype(data)::Type;
                for (amrex::MFIter iterator(flux); iterator.isValid(); ++iterator) {
                    auto const values = flux.const_array(iterator);
                    for (int const side : {-1, 1}) {
                        auto face = iterator.validbox();
                        int const index =
                            side < 0 ? domain.smallEnd(direction) : domain.bigEnd(direction) + 1;
                        if (index < face.smallEnd(direction) || index > face.bigEnd(direction)) {
                            continue;
                        }
                        face.setSmall(direction, index);
                        face.setBig(direction, index);
                        ops.eval(face, data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Tuple {
                            auto const value = values(i, j, k, component);
                            auto const increment = side * factor * value;
                            bool const valid =
                                amrex::Math::isfinite(value) && amrex::Math::isfinite(increment);
                            return {valid ? increment : 0, valid ? 0 : 1};
                        });
                    }
                }
                auto const values = data.value();
                auto sum = amrex::get<0>(values);
                int invalid = amrex::get<1>(values);
                amrex::ParallelDescriptor::ReduceRealSum(sum);
                amrex::ParallelDescriptor::ReduceIntMax(invalid);
                if (invalid || !std::isfinite(sum)) {
                    return false;
                }
                candidate[component] += sum;
            }
        }
        for (auto value : candidate) {
            if (!std::isfinite(value)) {
                return false;
            }
        }
        exchange = candidate;
        return true;
    }
} // namespace warpx::radiation
