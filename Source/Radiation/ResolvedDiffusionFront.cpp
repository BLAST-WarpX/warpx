/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "ResolvedDiffusionFront.H"

#include "ConstitutiveDiffusionMomentum.H"
#include "MovingAsymptoticDiffusionField.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <memory>
#include <utility>

namespace warpx::radiation
{
ResolvedDiffusionFront BuildResolvedDiffusionFront (
    amrex::MultiFab const& energy, amrex::MultiFab const& momentum,
    amrex::MultiFab const& opacity, amrex::iMultiFab const& eligible,
    amrex::Geometry const& geometry)
{
    ResolvedDiffusionFront result;
    auto const& boxes = energy.boxArray();
    auto const& distribution = energy.DistributionMap();
    result.mask.define(boxes, distribution, 1, 1);
    result.opacity.define(boxes, distribution, 1, 1);
    result.mask.setVal(1); // Reflecting radial exterior in RZ.
    result.opacity.setVal(0);
    amrex::ReduceOps<amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Long> data(ops);
    for (amrex::MFIter it(result.mask); it.isValid(); ++it) {
        auto const desired = eligible.const_array(it);
        auto const mask = result.mask.array(it);
        auto const e = energy.const_array(it);
        auto const q = momentum.const_array(it);
        auto const chi = opacity.const_array(it);
        auto const selected_chi = result.opacity.array(it);
        ops.eval(it.validbox(), data, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                 -> amrex::GpuTuple<amrex::Long> {
            bool const empty = desired(i,j,k) == 1 && e(i,j,k) == 0 &&
                q(i,j,k,0) == 0 && q(i,j,k,1) == 0 && q(i,j,k,2) == 0;
            mask(i,j,k) = empty ? 0 : desired(i,j,k);
            selected_chi(i,j,k) = empty ? 0 : chi(i,j,k);
            return {empty ? amrex::Long(1) : amrex::Long(0)};
        });
    }
    result.kinetic_cells = amrex::get<0>(data.value());
    if (energy.local_size() == 0) { result.kinetic_cells = 0; }
    amrex::ParallelDescriptor::ReduceLongSum(result.kinetic_cells);
    result.mask.FillBoundary(geometry.periodicity());
    result.opacity.FillBoundary(geometry.periodicity());
    return result;
}

bool AdmissibleResolvedDiffusionCensus (
    amrex::MultiFab const& energy, amrex::MultiFab const& momentum,
    amrex::MultiFab const& opacity, amrex::iMultiFab const& eligible,
    amrex::MultiFab const& velocity, amrex::Geometry const& geometry, amrex::Real dt,
    AsymptoticDiffusionFieldOptions const& options, bool moving)
{
    auto front = BuildResolvedDiffusionFront(energy, momentum, opacity, eligible, geometry);
    auto const& boxes = energy.boxArray();
    auto const& distribution = energy.DistributionMap();
    amrex::MultiFab zero_work(boxes, distribution, 1, 0);
    amrex::MultiFab zero_stress(boxes, distribution, 3, 0);
    amrex::MultiFab unused_impulse(boxes, distribution, 3, 0);
    zero_work.setVal(0);
    zero_stress.setVal(0);
    AsymptoticDiffusionFieldResult prediction;
    amrex::MultiFab lab_q;
    auto controls = options;
    controls.verbosity = 0;
    if (moving) {
        auto trial = TryAdvanceMovingAsymptoticDiffusionField(energy, front.opacity,
            front.mask, velocity, geometry, dt, zero_work, controls, true);
        if (!trial.valid) { return false; }
        lab_q = std::move(trial.lab_radiation_inertia);
        prediction = std::move(trial.diffusion);
    } else {
        amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> zero_faces;
        amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> incoming{};
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            auto const faces = amrex::convert(boxes, amrex::IntVect::TheDimensionVector(d));
            zero_faces[d] = std::make_unique<amrex::MultiFab>(faces, distribution, 1, 0);
            zero_faces[d]->setVal(0);
            incoming[d] = zero_faces[d].get();
        }
        prediction = TryAdvanceAsymptoticDiffusionField(energy, front.opacity, front.mask,
            geometry, dt, zero_work, incoming, nullptr, controls);
        if (!prediction.valid) { return false; }
    }
    ConstitutiveDiffusionMomentumReport report;
    return TryComputeConstitutiveDiffusionMomentum(energy, momentum, prediction.energy,
        moving ? lab_q : prediction.radiation_inertia, zero_stress, front.mask,
        geometry, unused_impulse, report);
}
}
