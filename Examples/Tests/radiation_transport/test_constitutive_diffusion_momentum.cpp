/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/ConstitutiveDiffusionMomentum.H"
#include "Utils/WarpXConst.H"

#include <AMReX.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>

#include <array>
#include <cmath>
#include <limits>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        using namespace warpx::radiation;
        AMREX_ALWAYS_ASSERT(AMREX_SPACEDIM == 1);
        amrex::Box const domain(amrex::IntVect(0), amrex::IntVect(3));
        amrex::RealBox const physical({AMREX_D_DECL(0.,0.,0.)},
                                       {AMREX_D_DECL(2.,2.,2.)});
        int periodic[AMREX_SPACEDIM] = {AMREX_D_DECL(1,1,1)};
        amrex::Geometry geometry(domain, &physical, 0, periodic);
        amrex::BoxArray boxes(domain);
        boxes.maxSize(2);
        amrex::DistributionMapping distribution(boxes);
        amrex::MultiFab old_e(boxes, distribution, 1, 0), new_e(boxes, distribution, 1, 0);
        amrex::MultiFab old_q(boxes, distribution, 3, 1), new_q(boxes, distribution, 3, 0);
        amrex::MultiFab stress(boxes, distribution, 3, 0), output(boxes, distribution, 3, 1);
        amrex::iMultiFab mask(boxes, distribution, 1, 0);
        old_e.setVal(10); new_e.setVal(12); old_q.setVal(0); new_q.setVal(0);
        stress.setVal(0); mask.setVal(1); output.setVal(27);
        for (amrex::MFIter mfi(mask); mfi.isValid(); ++mfi) {
            auto const flags = mask.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                flags(i,j,k) = i == 1 ? 0 : 1;
            });
        }
        ConstitutiveDiffusionMomentumReport report;
        auto run = [&] {
            return TryComputeConstitutiveDiffusionMomentum(old_e, old_q, new_e, new_q,
                stress, mask, geometry, output, report);
        };
        auto check = [&] (std::array<amrex::Real,3> const& expected) {
            amrex::MultiFab error(boxes, distribution, 3, 0);
            amrex::MultiFab::Copy(error, output, 0, 0, 3, 0);
            for (int d = 0; d < 3; ++d) {
                auto const value = expected[d];
                for (amrex::MFIter mfi(error); mfi.isValid(); ++mfi) {
                    auto const data = error.array(mfi);
                    auto const flags = mask.const_array(mfi);
                    amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                        data(i,j,k,d) -= flags(i,j,k) == 1 ? value : 0;
                    });
                }
                AMREX_ALWAYS_ASSERT(error.norm0(d) <=
                    64*std::numeric_limits<amrex::Real>::epsilon()*std::abs(value));
            }
        };
        auto rejected = [&] {
            output.setVal(27);
            AMREX_ALWAYS_ASSERT(!run() && !report.valid && report.failure);
            amrex::MultiFab difference(boxes, distribution, 3, 1);
            amrex::MultiFab::Copy(difference, output, 0, 0, 3, 1);
            difference.plus(-27, 0, 3, 1);
            for (int d = 0; d < 3; ++d) { AMREX_ALWAYS_ASSERT(difference.norm0(d,1) == 0); }
        };
        AMREX_ALWAYS_ASSERT(run() && report.valid && !report.failure);
        check({0,0,0});
        old_q.setVal(3,0,1); old_q.setVal(4,1,1);
        new_q.setVal(1,0,1); new_q.setVal(2,1,1); new_q.setVal(3,2,1);
        stress.setVal(1/PhysConst::c,0,1);
        stress.setVal(-3/PhysConst::c,1,1);
        stress.setVal(4/PhysConst::c,2,1);
        AMREX_ALWAYS_ASSERT(run()); check({1/PhysConst::c,5/PhysConst::c,-7/PhysConst::c});

        // Independent pressure resultant balances the change of stored q.
        old_q.setVal(0);
        for (int d = 0; d < 3; ++d) { stress.setVal(-(d+1)/PhysConst::c,d,1); }
        AMREX_ALWAYS_ASSERT(run()); check({0,0,0});
        // A beam inventory is allowed before the explicit stiff projection.
        new_q.setVal(0); stress.setVal(0); old_q.setVal(10,2,1);
        AMREX_ALWAYS_ASSERT(run()); check({0,0,10/PhysConst::c});
        old_q.setVal(11,2,1); rejected(); old_q.setVal(0);
        new_q.setVal(8,2,1); rejected(); new_q.setVal(0);
        stress.setVal(std::numeric_limits<amrex::Real>::quiet_NaN()); rejected(); stress.setVal(0);
        mask.setVal(2); rejected(); mask.setVal(1);
        old_e.setVal(-1); rejected(); old_e.setVal(10);

        // Only the owning rank carries this error; collective rejection must
        // still leave every rank's output unchanged (including an empty rank).
        for (amrex::MFIter mfi(old_e); mfi.isValid(); ++mfi) {
            auto const e = old_e.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) {
                if (i == 0) { e(i,j,k) = -1; }
            });
        }
        rejected(); old_e.setVal(10);
        AMREX_ALWAYS_ASSERT(TryComputeConstitutiveDiffusionMomentum(old_e, old_q, old_e,
            new_q, stress, mask, geometry, output, report)); // read-only input alias is safe
        AMREX_ALWAYS_ASSERT(!TryComputeConstitutiveDiffusionMomentum(old_e, old_q, new_e,
            new_q, stress, mask, geometry, old_q, report));
        // Component-offset alias: pointer equality alone does not detect it.
        amrex::MultiFab backing(boxes, distribution, 4, 1); backing.setVal(0);
        amrex::MultiFab overlapping_old(backing, amrex::make_alias, 0, 3);
        amrex::MultiFab overlapping_output(backing, amrex::make_alias, 1, 3);
        AMREX_ALWAYS_ASSERT(!TryComputeConstitutiveDiffusionMomentum(old_e, overlapping_old,
            new_e, new_q, stress, mask, geometry, overlapping_output, report));
        AMREX_ALWAYS_ASSERT(backing.norm0(0,1) == 0);
        amrex::MultiFab wrong_components(boxes, distribution, 2, 0);
        AMREX_ALWAYS_ASSERT(!TryComputeConstitutiveDiffusionMomentum(old_e, wrong_components,
            new_e, new_q, stress, mask, geometry, output, report));
    }
    amrex::Finalize();
}
