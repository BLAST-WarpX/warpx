/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "PML_RZ_FDTD.H"

#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Utils/WarpXConst.H"

#include <ablastr/utils/Communication.H>

#include <AMReX_BoxList.H>
#include <AMReX_MFIter.H>
#include <AMReX_VisMF.H>

using namespace amrex::literals;

ablastr::fields::VectorField
PML_RZ_FDTD::View (Storage const& fields)
{
    return {fields[0].get(), fields[1].get(), fields[2].get()};
}

PML_RZ_FDTD::PML_RZ_FDTD (amrex::BoxArray const& grids,
                          amrex::DistributionMapping const& dm,
                          amrex::Geometry const& geom,
                          ablastr::fields::VectorField E,
                          ablastr::fields::VectorField B, int ncell, int delta)
    : m_geom(geom), m_delta(delta)
{
    // Extend only grids touching the outer radial face, retaining their owners.
    amrex::BoxList boxes;
    amrex::Vector<int> ranks;
    for (int i = 0; i < grids.size(); ++i) {
        if (grids[i].bigEnd(0) == geom.Domain().bigEnd(0)) {
            boxes.push_back(amrex::adjCellHi(grids[i], 0, ncell));
            ranks.push_back(dm[i]);
        }
    }
    const amrex::BoxArray ba(boxes);
    const amrex::DistributionMapping pml_dm(ranks);
    for (int n = 0; n < 3; ++n) {
        const auto eba = amrex::convert(ba, E[n]->ixType());
        const auto bba = amrex::convert(ba, B[n]->ixType());
        const int nc = E[n]->nComp();
        m_E[n] = std::make_unique<amrex::MultiFab>(eba, pml_dm, nc,
                                                   E[n]->nGrowVect());
        m_B[n] = std::make_unique<amrex::MultiFab>(bba, pml_dm, nc,
                                                   B[n]->nGrowVect());
        m_aux_E[n] = std::make_unique<amrex::MultiFab>(eba, pml_dm, nc, 0);
        m_aux_B[n] = std::make_unique<amrex::MultiFab>(bba, pml_dm, nc, 0);
        m_increment_E[n] =
            std::make_unique<amrex::MultiFab>(eba, pml_dm, nc, 0);
        m_increment_B[n] =
            std::make_unique<amrex::MultiFab>(bba, pml_dm, nc, 0);
        m_zero_current[n] =
            std::make_unique<amrex::MultiFab>(eba, pml_dm, nc, 0);
        m_E[n]->setVal(0.0_rt);
        m_B[n]->setVal(0.0_rt);
        m_aux_E[n]->setVal(0.0_rt);
        m_aux_B[n]->setVal(0.0_rt);
        m_zero_current[n]->setVal(0.0_rt);
    }
}

void
PML_RZ_FDTD::EvolveE (FiniteDifferenceSolver& solver, amrex::Real dt)
{
    for (auto& field : m_increment_E) {
        field->setVal(0.0_rt);
    }
    solver.EvolveEPMLRZ(View(m_increment_E), View(m_B), View(m_zero_current), dt);
    UpdateConstitutive(m_E, m_aux_E, m_increment_E, dt);
}

void
PML_RZ_FDTD::EvolveB (FiniteDifferenceSolver& solver, amrex::Real dt)
{
    for (auto& field : m_increment_B) {
        field->setVal(0.0_rt);
    }
    solver.EvolveBPMLRZ(View(m_increment_B), View(m_E), dt);
    UpdateConstitutive(m_B, m_aux_B, m_increment_B, dt);
}

void
PML_RZ_FDTD::UpdateConstitutive (Storage& field, Storage& auxiliary,
                                 Storage const& increment, amrex::Real dt)
{
    const amrex::Real dr = m_geom.CellSize(0);
    const amrex::Real rlo = m_geom.ProbLo(0);
    const amrex::Real r_interface = m_geom.ProbHi(0);
    const int ilo = m_geom.Domain().smallEnd(0);
    const amrex::Real factor = 4.0_rt * PhysConst::c / (dr * m_delta * m_delta);
    for (int n = 0; n < 3; ++n) {
        const amrex::Real shift =
            field[n]->ixType().nodeCentered(0) ? 0.0_rt : 0.5_rt;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(*field[n], amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            const auto f = field[n]->array(mfi);
            const auto aux = auxiliary[n]->array(mfi);
            const auto curl = increment[n]->const_array(mfi);
            amrex::ParallelFor(
                mfi.tilebox(), field[n]->nComp(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int m) {
                    const amrex::Real r = rlo + (i - ilo + shift) * dr;
                    const amrex::Real depth =
                        amrex::max(r - r_interface, 0.0_rt);
                    const amrex::Real sigma =
                        factor * (depth / dr) * (depth / dr);
                    // Account for the stretched cylindrical radius.
                    const amrex::Real a = 0.5_rt * dt * sigma;
                    const amrex::Real b = a * depth / (3.0_rt * r);
                    const amrex::Real old_aux = aux(i, j, k, m);
                    const amrex::Real delta = curl(i, j, k, m);
                    if (n == 0) {
                        f(i, j, k, m) =
                            ((1.0_rt - b) * f(i, j, k, m) +
                             (1.0_rt + a) * delta + 2.0_rt * a * old_aux) /
                            (1.0_rt + b);
                        aux(i, j, k, m) += delta;
                    } else if (n == 1) {
                        f(i, j, k, m) =
                            ((1.0_rt - a) * f(i, j, k, m) +
                             (1.0_rt + b) * delta + 2.0_rt * b * old_aux) /
                            (1.0_rt + a);
                        aux(i, j, k, m) += delta;
                    } else {
                        const amrex::Real next_aux =
                            ((1.0_rt - a) * old_aux + delta) / (1.0_rt + a);
                        f(i, j, k, m) = ((1.0_rt - b) * f(i, j, k, m) +
                                         next_aux - old_aux) /
                                        (1.0_rt + b);
                        aux(i, j, k, m) = next_aux;
                    }
                });
        }
    }
}

void
PML_RZ_FDTD::FillBoundary (ablastr::fields::VectorField regular, bool electric,
                           bool single_precision,
                           std::optional<bool> nodal_sync)
{
    auto pml = View(electric ? m_E : m_B);
    const auto& period = m_geom.periodicity();
    for (int n = 0; n < 3; ++n) {
        const int nc = regular[n]->nComp();
        const auto ng = regular[n]->nGrowVect();
        amrex::MultiFab tmp(regular[n]->boxArray(),
                            regular[n]->DistributionMap(), nc, ng);
        amrex::MultiFab::Copy(tmp, *regular[n], 0, 0, nc, ng);
        ablastr::utils::communication::ParallelCopy(tmp, *pml[n], 0, 0, nc,
                                                    amrex::IntVect(0), ng,
                                                    single_precision, period);
        // Regular nodes own the interface. Only update their ghost cells.
        for (amrex::MFIter mfi(*regular[n]); mfi.isValid(); ++mfi) {
            const auto src = tmp.const_array(mfi);
            const auto dst = regular[n]->array(mfi);
            const auto ghosts =
                amrex::boxDiff((*regular[n])[mfi].box(), mfi.validbox());
            for (const auto& box : ghosts) {
                amrex::ParallelFor(
                    box, nc, [=] AMREX_GPU_DEVICE(int i, int j, int k, int m) {
                        dst(i, j, k, m) = src(i, j, k, m);
                    });
            }
        }
        ablastr::utils::communication::ParallelCopy(
            *pml[n], *regular[n], 0, 0, nc, amrex::IntVect(0),
            pml[n]->nGrowVect(), single_precision, period);
        ablastr::utils::communication::FillBoundary(*pml[n], single_precision,
                                                    period, nodal_sync);
    }
}

void
PML_RZ_FDTD::CheckPoint (std::string const& prefix) const
{
    for (int n = 0; n < 3; ++n) {
        const auto suffix = std::to_string(n);
        amrex::VisMF::Write(*m_E[n], std::string(prefix).append("_E").append(suffix));
        amrex::VisMF::Write(*m_B[n], std::string(prefix).append("_B").append(suffix));
        amrex::VisMF::Write(*m_aux_E[n], std::string(prefix).append("_aux_E").append(suffix));
        amrex::VisMF::Write(*m_aux_B[n], std::string(prefix).append("_aux_B").append(suffix));
    }
}

void
PML_RZ_FDTD::Restart (std::string const& prefix)
{
    for (int n = 0; n < 3; ++n) {
        const auto suffix = std::to_string(n);
        amrex::VisMF::Read(*m_E[n], std::string(prefix).append("_E").append(suffix));
        amrex::VisMF::Read(*m_B[n], std::string(prefix).append("_B").append(suffix));
        amrex::VisMF::Read(*m_aux_E[n], std::string(prefix).append("_aux_E").append(suffix));
        amrex::VisMF::Read(*m_aux_B[n], std::string(prefix).append("_aux_B").append(suffix));
    }
}
