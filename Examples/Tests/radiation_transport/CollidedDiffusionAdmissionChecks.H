/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_COLLIDED_DIFFUSION_ADMISSION_CHECKS_H_
#define WARPX_COLLIDED_DIFFUSION_ADMISSION_CHECKS_H_

#include "Radiation/CollidedDiffusionAdmission.H"
#include <AMReX_Arena.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MFIter.H>
#include <cmath>
#include <limits>

namespace warpx::radiation
{
// Analytic cell tests; native eligibility and the physical path are tested separately.
inline void CheckCollidedDiffusionCohortCriteria ()
{
    CollidedDiffusionAdmissionControls controls;
    CollidedDiffusionCohortMoments m;
    m.cohort_energy=6; m.cohort_pressure={2,2,2,0,0,0};
    AMREX_ALWAYS_ASSERT(ShouldAdmitCollidedDiffusionCohort(m,controls));
    m.cohort_energy=2; m.cohort_pressure={0,0,2,0,0,0};
    AMREX_ALWAYS_ASSERT(!ShouldAdmitCollidedDiffusionCohort(m,controls)); // opposite beams, q=0
    m.cohort_energy=1; m.cohort_momentum={0,0,1}; m.cohort_pressure={0,0,1,0,0,0};
    AMREX_ALWAYS_ASSERT(!ShouldAdmitCollidedDiffusionCohort(m,controls)); // single beam
    m.background_energy=100;
    AMREX_ALWAYS_ASSERT(ShouldAdmitCollidedDiffusionCohort(m,controls)); // isotropic dilution
    m={}; m.cohort_energy=3; m.cohort_pressure={1,1,1,0,0,0};
    m.cohort_momentum={0.75,0.75,0.75};
    AMREX_ALWAYS_ASSERT(!ShouldAdmitCollidedDiffusionCohort(m,controls)); // vector, not component cut
    m.cohort_momentum={}; m.cohort_pressure[3]=0.24;
    AMREX_ALWAYS_ASSERT(!ShouldAdmitCollidedDiffusionCohort(m,controls)); // sqrt(2)*0.08 > 0.1
    for (amrex::Real const e : {amrex::Real(1.e-300),amrex::Real(1.e300)}) {
        m={}; m.cohort_energy=e; m.cohort_pressure={e/3,e/3,e/3,0,0,0};
        AMREX_ALWAYS_ASSERT(ShouldAdmitCollidedDiffusionCohort(m,controls));
    }
    m={}; m.background_energy=10;
    AMREX_ALWAYS_ASSERT(!ShouldAdmitCollidedDiffusionCohort(m,controls)); // no eligible cohort
    m.cohort_energy=3; m.cohort_pressure={1,1,1,0,0,0};
    m.background_energy=-1;
    AMREX_ALWAYS_ASSERT(!ShouldAdmitCollidedDiffusionCohort(m,controls));
    m.background_energy=0; m.cohort_pressure[1]=std::numeric_limits<amrex::Real>::quiet_NaN();
    AMREX_ALWAYS_ASSERT(!ShouldAdmitCollidedDiffusionCohort(m,controls));
    m.cohort_pressure[1]=1; controls.max_pressure_anisotropy=0.11;
    AMREX_ALWAYS_ASSERT(!ShouldAdmitCollidedDiffusionCohort(m,controls));
}

// Call immediately around an admission operation, before any other field update
// or leakage birth. This checks independently measured fields and native photons.
inline void CheckCollidedDiffusionAdmissionTransfer (
    amrex::MultiFab const& before_e, amrex::MultiFab const& before_q,
    amrex::MultiFab const& after_e, amrex::MultiFab const& after_q,
    CollidedDiffusionAdmissionReport const& report,
    amrex::GpuArray<amrex::Real,4> const& photons_before,
    amrex::GpuArray<amrex::Real,4> const& photons_after)
{
    AMREX_ALWAYS_ASSERT(report.valid);
    auto const pinned=amrex::MFInfo().SetArena(amrex::The_Pinned_Arena());
    auto copy=[&](amrex::MultiFab const& source) {
        amrex::MultiFab host(source.boxArray(),source.DistributionMap(),source.nComp(),
                             source.nGrowVect(),pinned);
        amrex::dtoh_memcpy(host,source);
        return host;
    };
    auto e0=copy(before_e), e1=copy(after_e), q0=copy(before_q), q1=copy(after_q);
    auto moved=copy(report.admitted_four_inventory);
    amrex::Gpu::streamSynchronize();
    auto check=[](amrex::Real before,amrex::Real after,amrex::Real transfer) {
        auto const scale=std::abs(before)+std::abs(after)+std::abs(transfer);
        AMREX_ALWAYS_ASSERT(std::isfinite(scale));
        AMREX_ALWAYS_ASSERT(std::abs((after-before)-transfer)<=1.e-11*scale);
    };
    for (amrex::MFIter it(e0); it.isValid(); ++it) {
        auto const a=e0.const_array(it), b=e1.const_array(it);
        auto const u=q0.const_array(it), v=q1.const_array(it), t=moved.const_array(it);
        for (int i=it.validbox().smallEnd(0); i<=it.validbox().bigEnd(0); ++i) {
            check(a(i,0,0),b(i,0,0),t(i,0,0,0));
            for (int d=0; d<3; ++d) { check(u(i,0,0,d),v(i,0,0,d),t(i,0,0,d+1)); }
        }
    }
    // A signed global first moment can cancel; native reduction ordering may
    // change during Redistribute. Photon energies bound its uncancelled scale.
    auto const photon_scale=std::abs(photons_before[0])+std::abs(photons_after[0])+
        std::abs(report.actual_admitted_four_inventory[0]);
    AMREX_ALWAYS_ASSERT(std::isfinite(photon_scale));
    for (int d=0; d<4; ++d) {
        AMREX_ALWAYS_ASSERT(std::abs(photons_after[d]-photons_before[d]+
            report.actual_admitted_four_inventory[d])<=1.e-11*photon_scale);
    }
}
}
#endif
