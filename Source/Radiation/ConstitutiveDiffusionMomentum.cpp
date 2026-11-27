/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "ConstitutiveDiffusionMomentum.H"

#include "Utils/WarpXConst.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace warpx::radiation
{
using namespace amrex::literals;
namespace
{
    enum class Failure : int
    {
        none = 0,
        unsupported_geometry = 1,
        aliased_fields = 2,
        invalid_layout = 3,
        invalid_state = 4
    };

    constexpr char const* failure_reason (Failure failure) noexcept
    {
        switch (failure) {
            case Failure::none:
                return "";
            case Failure::unsupported_geometry:
                return "constitutive diffusion momentum requires Cartesian 1D, 2D, or 3D";
            case Failure::aliased_fields:
                return "input/output MultiFabs must not alias storage";
            case Failure::invalid_layout:
                return "inconsistent geometry, BoxArray, DistributionMapping, or components";
            case Failure::invalid_state:
                return "nonfinite, negative-energy, mask, or moment-realizability failure";
        }
        return "unknown failure";
    }

    bool
    collective_valid (bool local)
    {
        int invalid = local ? 0 : 1;
        amrex::ParallelDescriptor::ReduceIntMax(invalid);
        return invalid == 0;
    }

    Failure
    collective_failure (Failure local) noexcept
    {
        int code = static_cast<int>(local);
        amrex::ParallelDescriptor::ReduceIntMax(code);
        return static_cast<Failure>(code);
    }

    void
    set_report (ConstitutiveDiffusionMomentumReport& report, Failure failure)
    {
        report.valid = failure == Failure::none;
        report.failure = failure != Failure::none;
        report.reason = failure_reason(failure);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    vector_norm (amrex::Real const* values) noexcept
    {
        amrex::Real scale = 0.0_rt;
        for (int n = 0; n < 3; ++n) {
            scale = amrex::max(scale, std::abs(values[n]));
        }
        if (scale == 0.0_rt) { return 0.0_rt; }
        amrex::Real sum = 0.0_rt;
        for (int n = 0; n < 3; ++n) {
            amrex::Real const ratio = values[n] / scale;
            sum += ratio * ratio;
        }
        return scale * std::sqrt(sum);
    }

    bool
    storage_overlaps (amrex::MultiFab const& first, amrex::MultiFab const& second)
    {
        for (amrex::MFIter iterator(first); iterator.isValid(); ++iterator) {
            auto const& a = first[iterator];
            auto const& b = second[iterator];
            auto const first_begin = reinterpret_cast<std::uintptr_t>(a.dataPtr());
            auto const second_begin = reinterpret_cast<std::uintptr_t>(b.dataPtr());
            auto const first_bytes = static_cast<std::uintptr_t>(a.box().numPts()) *
                static_cast<std::uintptr_t>(a.nComp()) * sizeof(amrex::Real);
            auto const second_bytes = static_cast<std::uintptr_t>(b.box().numPts()) *
                static_cast<std::uintptr_t>(b.nComp()) * sizeof(amrex::Real);
            auto const maximum = std::numeric_limits<std::uintptr_t>::max();
            if (!a.dataPtr() || !b.dataPtr() || first_bytes > maximum-first_begin ||
                second_bytes > maximum-second_begin) { return true; }
            auto const first_end = first_begin + first_bytes;
            auto const second_end = second_begin + second_bytes;
            if (first_begin < second_end && second_begin < first_end) { return true; }
        }
        return false;
    }

    bool
    any_real_storage_alias (
        std::vector<amrex::MultiFab const*> const& fields,
        amrex::MultiFab const& output)
    {
        for (auto const* input : fields) {
            if (input == &output || storage_overlaps(*input, output)) {
                return true;
            }
        }
        return false;
    }
}

bool
TryComputeConstitutiveDiffusionMomentum (
    amrex::MultiFab const& old_energy,
    amrex::MultiFab const& old_q,
    amrex::MultiFab const& new_energy,
    amrex::MultiFab const& new_q,
    amrex::MultiFab const& outward_face_stress_resultant,
    amrex::iMultiFab const& diffusion_mask,
    amrex::Geometry const& geometry,
    amrex::MultiFab& output_impulse,
    ConstitutiveDiffusionMomentumReport& report)
{
    std::vector<amrex::MultiFab const*> const real_inputs {
        &old_energy, &old_q, &new_energy, &new_q, &outward_face_stress_resultant};

    bool const supported_cartesian =
#if defined(WARPX_DIM_1D_Z)
        AMREX_SPACEDIM == 1
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        AMREX_SPACEDIM == 2
#elif defined(WARPX_DIM_3D)
        AMREX_SPACEDIM == 3
#else
        false
#endif
        ;
    bool const coordinates_valid =
#if defined(WARPX_DIM_RZ)
        (geometry.Coord() == 0 || geometry.Coord() == 1) && geometry.ProbLo(0) >= 0 &&
        !geometry.isPeriodic(0);
#else
        geometry.Coord() == 0;
#endif
    bool geometry_valid = supported_cartesian && coordinates_valid &&
        geometry.Domain().ok() && std::isfinite(PhysConst::c) && PhysConst::c > 0.0_rt;
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
        geometry_valid = geometry_valid && geometry.Domain().length(axis) > 0 &&
            std::isfinite(geometry.ProbLo(axis)) && std::isfinite(geometry.ProbHi(axis)) &&
            geometry.ProbHi(axis) > geometry.ProbLo(axis) &&
            std::isfinite(geometry.CellSize(axis)) && geometry.CellSize(axis) > 0.0_rt;
    }
    if (!collective_valid(geometry_valid)) {
        set_report(report, collective_failure(supported_cartesian ?
            Failure::invalid_layout : Failure::unsupported_geometry));
        return false;
    }

    auto const& boxes = old_energy.boxArray();
    auto const& distribution = old_energy.DistributionMap();
    bool const layout_valid = boxes.ok() && boxes.isDisjoint() &&
                              boxes.contains(geometry.Domain()) &&
                              boxes.numPts() == geometry.Domain().numPts() &&
                              old_energy.nComp() == 1 &&
                              old_energy.ixType().cellCentered() &&
                              old_q.nComp() == 3 && old_q.ixType().cellCentered() &&
                              old_q.boxArray() == boxes &&
                              old_q.DistributionMap() == distribution &&
                              new_energy.nComp() == 1 &&
                              new_energy.ixType().cellCentered() &&
                              new_energy.boxArray() == boxes &&
                              new_energy.DistributionMap() == distribution &&
                              new_q.nComp() == 3 && new_q.ixType().cellCentered() &&
                              new_q.boxArray() == boxes &&
                              new_q.DistributionMap() == distribution &&
                              outward_face_stress_resultant.nComp() == 3 &&
                              outward_face_stress_resultant.ixType().cellCentered() &&
                              outward_face_stress_resultant.boxArray() == boxes &&
                              outward_face_stress_resultant.DistributionMap() == distribution &&
                              diffusion_mask.nComp() == 1 &&
                              diffusion_mask.ixType().cellCentered() &&
                              diffusion_mask.boxArray() == boxes &&
                              diffusion_mask.DistributionMap() == distribution &&
                              output_impulse.nComp() == 3 &&
                              output_impulse.ixType().cellCentered() &&
                              output_impulse.boxArray() == boxes &&
                              output_impulse.DistributionMap() == distribution;
    if (!collective_valid(layout_valid)) {
        set_report(report, collective_failure(Failure::invalid_layout));
        return false;
    }

    if (!collective_valid(!any_real_storage_alias(real_inputs, output_impulse))) {
        set_report(report, collective_failure(Failure::aliased_fields));
        return false;
    }

    amrex::MultiFab candidate(boxes, distribution, 3, 0);
    amrex::ReduceOps<amrex::ReduceOpMax> ops;
    amrex::ReduceData<int> data(ops);
    for (amrex::MFIter iterator(old_energy); iterator.isValid(); ++iterator) {
        auto const e_old = old_energy.const_array(iterator);
        auto const q_old = old_q.const_array(iterator);
        auto const e_new = new_energy.const_array(iterator);
        auto const q_new = new_q.const_array(iterator);
        auto const stress = outward_face_stress_resultant.const_array(iterator);
        auto const mask = diffusion_mask.const_array(iterator);
        auto const impulse = candidate.array(iterator);
        ops.eval(
            iterator.validbox(), data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> amrex::GpuTuple<int>
            {
                int const flag = mask(i, j, k);
                bool bad = flag != 0 && flag != 1;

                amrex::Real const old_momentum[3] = {q_old(i, j, k, 0), q_old(i, j, k, 1),
                                                     q_old(i, j, k, 2)};
                amrex::Real const new_momentum[3] = {q_new(i, j, k, 0), q_new(i, j, k, 1),
                                                     q_new(i, j, k, 2)};
                amrex::Real const old_norm = vector_norm(old_momentum);
                amrex::Real const new_norm = vector_norm(new_momentum);
                amrex::Real const energy_old = e_old(i, j, k);
                amrex::Real const energy_new = e_new(i, j, k);

                bad = bad || !amrex::Math::isfinite(energy_old) || energy_old < 0.0_rt ||
                      !amrex::Math::isfinite(energy_new) || energy_new < 0.0_rt ||
                      !amrex::Math::isfinite(old_norm) || !amrex::Math::isfinite(new_norm);

                for (int n = 0; n < 3; ++n) {
                    bad = bad || !amrex::Math::isfinite(old_momentum[n]) ||
                          !amrex::Math::isfinite(new_momentum[n]) ||
                          !amrex::Math::isfinite(stress(i, j, k, n));
                }

                // These are roundoff tolerances scaled by the operands under test. They
                // are not physical floors; the vectors are never clipped or projected.
                constexpr amrex::Real roundoff = 32.0_rt *
                    std::numeric_limits<amrex::Real>::epsilon();
                bad = bad || old_norm - energy_old > roundoff *
                    amrex::max(energy_old, old_norm);
                bad = bad || new_norm - energy_new / std::sqrt(3.0_rt) > roundoff *
                    amrex::max(energy_new, new_norm);

                if (bad) { return {1}; }
                for (int n = 0; n < 3; ++n) {
                    amrex::Real const component = old_momentum[n]/PhysConst::c -
                        new_momentum[n]/PhysConst::c - stress(i, j, k, n);
                    if (!amrex::Math::isfinite(component)) { return {1}; }
                    impulse(i, j, k, n) = flag == 1 ? component : 0.0_rt;
                }
                return {0};
            });
    }

    int invalid = amrex::get<0>(data.value());
    if (old_energy.local_size() == 0) { invalid = 0; }
    amrex::ParallelDescriptor::ReduceIntMax(invalid);
    if (invalid != 0) {
        set_report(report, Failure::invalid_state);
        return false;
    }

    amrex::MultiFab::Copy(output_impulse, candidate, 0, 0, 3, 0);
    set_report(report, Failure::none);
    return true;
}

} // namespace warpx::radiation
