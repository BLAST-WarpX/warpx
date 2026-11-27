/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "EmbeddedBoundary/WarpXFaceInfoBox.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H"
#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/RZPressureInductionStage.H"
#include "Fields.H"
#include "Initialization/WarpXInit.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>

using namespace amrex::literals;
using warpx::fields::FieldType;

namespace
{
void
zero (ablastr::fields::VectorField const& fields)
{
    for (auto* f : fields)
    {
        f->setVal(0);
    }
}
} // namespace

int
main (int argc, char* argv[])
{
    warpx::initialization::initialize_external_libraries(argc, argv);
    {
        auto& sim = WarpX::GetInstance();
        sim.InitData();
        auto const& geom = sim.Geom(0);
        auto* hybrid = sim.get_pointer_HybridPICModel();
        AMREX_ALWAYS_ASSERT(hybrid != nullptr && AMREX_SPACEDIM == 2);
        // Native RZ kernels select cylindrical physics at compile time even
        // when AMReX's geometry metadata is Cartesian. The standalone stage
        // explicitly requires cylindrical metadata; adapt the same physical
        // mesh without changing the native simulation geometry or operators.
        amrex::RealBox const extent(geom.ProbLo(), geom.ProbHi());
        int const periodic[2] = {geom.isPeriodic(0), geom.isPeriodic(1)};
        amrex::Geometry const radial_geometry(geom.Domain(), &extent, 1, periodic);
        auto B = sim.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
        auto E = sim.m_fields.get_alldirs(FieldType::Efield_fp, 0);
        auto J = sim.m_fields.get_alldirs(FieldType::current_fp, 0);
        auto const& ba = B[1]->boxArray();
        auto const& dm = B[1]->DistributionMap();
        amrex::MultiFab charge(ba, dm, 1, 1), temperature(ba, dm, 1, 1);
        amrex::MultiFab pressure(ba, dm, 1, 1), internal(ba, dm, 1, 1);
        amrex::MultiFab volume(ba, dm, 1, 1), stage_b(ba, dm, 1, 1);
        amrex::MultiFab initial_b(ba, dm, 1, 1), difference(ba, dm, 1, 0);
        amrex::MultiFab stage_u(ba, dm, 1, 1), stage_t(ba, dm, 1, 1);
        amrex::MultiFab total_current_storage[3], pressure_e_storage[3];
        ablastr::fields::VectorField total_current{}, pressure_e{};
        for (int d = 0; d < 3; ++d)
        {
            total_current_storage[d].define(ba, dm, 1, B[d]->nGrowVect());
            pressure_e_storage[d].define(ba, dm, 1, E[d]->nGrowVect());
            total_current[d] = &total_current_storage[d];
            pressure_e[d] = &pressure_e_storage[d];
        }
        // max_step=0 does not populate the live rho scratch field. Use the
        // native particle deposition, including real boundary/volume scaling.
        amrex::MultiFab deposited_charge(ba, dm, 1,
                                         sim.m_fields.get(FieldType::rho_fp, 0)->nGrowVect());
        sim.GetPartContainer().GetParticleContainerFromName("ions").DepositChargeOnLevel(
            &deposited_charge, 0, false, true);
        sim.FinalizeRhoOnLevel(0, &deposited_charge, PatchType::fine);
        amrex::MultiFab::Copy(charge, deposited_charge, 0, 0, 1, 0);
        charge.OverrideSync(geom.periodicity());
        charge.FillBoundary(geom.periodicity());
        AMREX_ALWAYS_ASSERT(charge.min(0) > 0);
        AMREX_ALWAYS_ASSERT(charge.max(0) > 1.1_rt * charge.min(0));
        auto const eos = hybrid->electronThermodynamicsExecutor();
        auto const dr = geom.CellSize(0), dz = geom.CellSize(1);
        int const outer = geom.Domain().bigEnd(0) + 1;
        FiniteDifferenceSolver solver(ElectromagneticSolverAlgo::HybridPIC, {dr, 1._rt, dz},
                                      ablastr::utils::enums::GridType::Collocated);
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> no_eb;
        std::array<std::unique_ptr<amrex::LayoutData<FaceInfoBox>>, 3> no_borrowing;
        constexpr amrex::Real dt = 1.e-7_rt;
        for (int control = 0; control < 2; ++control)
        {
            for (amrex::MFIter mfi(temperature); mfi.isValid(); ++mfi)
            {
                auto const t = temperature.array(mfi), p = pressure.array(mfi);
                auto const u = internal.array(mfi), v = volume.array(mfi);
                auto const rho = charge.const_array(mfi);
                auto const b = initial_b.array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    {
                        auto const r = i * dr, z = j * dz;
                        auto const kelvin =
                            1.e5_rt *
                            (1 + (control == 0 ? .1_rt * std::sin(2 * MathConst::pi * z) : 0));
                        auto const thermal =
                            eos.stateFromChargeDensityTemperature(rho(i, j, k), kelvin);
                        t(i, j, k) = kelvin;
                        p(i, j, k) = thermal.pressure;
                        u(i, j, k) = thermal.internal_energy_density;
                        b(i, j, k) = 1.e-4_rt * r * (1 - r * r);
                        // Current native caloric metric, not a new geometric remap.
                        v(i, j, k) = i == 0 ? MathConst::pi * dr * dr * dz / 3
                                            : 2 * MathConst::pi * i * dr * dr * dz *
                                                  (i == outer ? .5_rt : 1._rt);
                    });
            }
            pressure.OverrideSync(geom.periodicity());
            pressure.FillBoundary(geom.periodicity());
            // Only interior pressure neighbors are read by the masked native
            // gradient; physical wall/axis derivatives are explicitly zero.
            zero(B);
            zero(E);
            zero(J);
            zero(total_current);
            zero(pressure_e);
            solver.HybridPICSolveE(E, pressure_e, total_current, J, B, charge, pressure, no_eb, 0,
                                   hybrid, false);
            // B=J=Ji=0 isolates the pressure term in the actual native Ohm call.
            // Use real native boundary routines before the Faraday witness.
            sim.ApplyEfieldBoundary(0, PatchType::fine, 0);
            sim.ApplyFieldBoundaryOnAxis(E[0], E[1], E[2], 0);
            for (auto* f : E)
            {
                f->OverrideSync(geom.periodicity());
                f->FillBoundary(geom.periodicity());
            }
            amrex::MultiFab::Copy(*B[1], initial_b, 0, 0, 1, 0);
            amrex::MultiFab::Copy(stage_b, initial_b, 0, 0, 1, 0);
            amrex::MultiFab::Copy(stage_u, internal, 0, 0, 1, 0);
            amrex::MultiFab::Copy(stage_t, temperature, 0, 0, 1, 0);
            AMREX_ALWAYS_ASSERT(warpx::hybrid::tryAdvanceRZPressureInductionStage(
                stage_b, stage_u, stage_t, charge, volume, radial_geometry, dt, eos));
            solver.EvolveB(sim.m_fields, 0, PatchType::fine, no_eb, no_borrowing, dt);
            amrex::MultiFab::Copy(difference, *B[1], 0, 0, 1, 0);
            amrex::MultiFab::Subtract(difference, stage_b, 0, 0, 1, 0);
            auto const error = difference.norm0(0, 0);
            amrex::MultiFab::Copy(difference, *B[1], 0, 0, 1, 0);
            amrex::MultiFab::Subtract(difference, initial_b, 0, 0, 1, 0);
            auto const increment = difference.norm0(0, 0);
            auto const scale = dt * pressure.norm0(0, 0) / charge.min(0) / (dr * dz);
            amrex::Print() << "native_pressure_control=" << control
                           << " B_increment_T=" << increment << " operator_error_T=" << error
                           << " normalized_error=" << error / scale << '\n';
            AMREX_ALWAYS_ASSERT(error < 1.e-11_rt * scale);
            if (control == 0)
            {
                AMREX_ALWAYS_ASSERT(increment > 1.e-8_rt);
            }
            else
            {
                AMREX_ALWAYS_ASSERT(increment < 1.e-12_rt * scale);
            }
            // L is measured through native Ohm/BC/Faraday basis responses.
            // -dt*P_k*(L^T W_B Bmid/mu0)_k is local material energy [J],
            // not an E.J density and not only a global conservation check.
            if (control == 0)
            {
                int const nr = outer;
                int const nz = geom.Domain().length(1);
                amrex::MultiFab bmid(ba, dm, 1, 0);
                amrex::MultiFab::Copy(bmid, initial_b, 0, 0, 1, 0);
                amrex::MultiFab::Add(bmid, stage_b, 0, 0, 1, 0);
                bmid.mult(0.5_rt, 0, 1, 0);
                auto const owner_mask = stage_b.OwnerMask(geom.periodicity());
                amrex::MultiFab basis_p(ba, dm, 1, 1);
                amrex::Real max_error = 0._rt, max_work = 0._rt, max_old_thermal = 0._rt;
                constexpr auto eps = std::numeric_limits<amrex::Real>::epsilon();

                for (int pi = 0; pi <= nr; ++pi)
                {
                    for (int pj = 0; pj < nz; ++pj)
                    {
                        basis_p.setVal(0._rt);
                        for (amrex::MFIter mfi(basis_p); mfi.isValid(); ++mfi)
                        {
                            auto const bp = basis_p.array(mfi);
                            auto const bx = mfi.validbox();
                            amrex::ParallelFor(bx,
                                               [=] AMREX_GPU_DEVICE(int i, int j, int k)
                                               {
                                                   if (i == pi &&
                                                       (j % nz == pj || (pj == 0 && j == nz)))
                                                   {
                                                       bp(i, j, k) = 1._rt;
                                                   }
                                               });
                        }
                        basis_p.OverrideSync(geom.periodicity());
                        basis_p.FillBoundary(geom.periodicity());

                        zero(B);
                        zero(E);
                        zero(J);
                        zero(total_current);
                        zero(pressure_e);
                        solver.HybridPICSolveE(E, pressure_e, total_current, J, B, charge, basis_p,
                                               no_eb, 0, hybrid, false);
                        sim.ApplyEfieldBoundary(0, PatchType::fine, 0);
                        sim.ApplyFieldBoundaryOnAxis(E[0], E[1], E[2], 0);
                        for (auto* f : E)
                        {
                            f->OverrideSync(geom.periodicity());
                            f->FillBoundary(geom.periodicity());
                        }
                        solver.EvolveB(sim.m_fields, 0, PatchType::fine, no_eb, no_borrowing,
                                       1._rt);

                        amrex::Real loc_dot = 0._rt, loc_p = 0._rt, loc_du = 0._rt, loc_u0 = 0._rt;
                        for (amrex::MFIter mfi(*B[1]); mfi.isValid(); ++mfi)
                        {
                            auto const b_resp = B[1]->const_array(mfi);
                            auto const b_m = bmid.const_array(mfi);
                            auto const msk = owner_mask->const_array(mfi);
                            auto const p_arr = pressure.const_array(mfi);
                            auto const su_arr = stage_u.const_array(mfi);
                            auto const in_arr = internal.const_array(mfi);
                            auto const v_arr = volume.const_array(mfi);
                            auto const bx = mfi.validbox();

                            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum,
                                             amrex::ReduceOpSum, amrex::ReduceOpSum>
                                rop;
                            amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real>
                                rdata(rop);
                            using RTuple = typename decltype(rdata)::Type;

                            rop.eval(bx, rdata,
                                     [=] AMREX_GPU_DEVICE(int i, int j, int k) -> RTuple
                                     {
                                         amrex::Real d_val = 0._rt, p_val = 0._rt, du_val = 0._rt,
                                                     u0_val = 0._rt;
                                         if (msk(i, j, k) != 0)
                                         {
                                             if (i > 0)
                                             {
                                                 auto const wb = 2 * MathConst::pi * i * dr * dr *
                                                                 dz * (i == outer ? .5_rt : 1._rt);
                                                 d_val = b_resp(i, j, k) * b_m(i, j, k) *
                                                         (wb / PhysConst::mu0);
                                             }
                                             if (i == pi && (j % nz == pj || (pj == 0 && j == nz)))
                                             {
                                                 p_val = p_arr(i, j, k);
                                                 du_val = (su_arr(i, j, k) - in_arr(i, j, k)) *
                                                          v_arr(i, j, k);
                                                 u0_val = in_arr(i, j, k) * v_arr(i, j, k);
                                             }
                                         }
                                         return {d_val, p_val, du_val, u0_val};
                                     });

                            auto const vals = rdata.value();
                            loc_dot += amrex::get<0>(vals);
                            loc_p += amrex::get<1>(vals);
                            loc_du += amrex::get<2>(vals);
                            loc_u0 += amrex::get<3>(vals);
                        }

                        amrex::Real dot = loc_dot, p_k = loc_p, du_k = loc_du, u0_k = loc_u0;
                        amrex::ParallelDescriptor::ReduceRealSum(dot);
                        amrex::ParallelDescriptor::ReduceRealSum(p_k);
                        amrex::ParallelDescriptor::ReduceRealSum(du_k);
                        amrex::ParallelDescriptor::ReduceRealSum(u0_k);

                        auto const expected_work = -dt * p_k * dot;
                        auto const discrepancy = std::abs(expected_work - du_k);
                        max_error = std::max(max_error, discrepancy);
                        max_work = std::max(max_work, std::abs(expected_work));
                        max_old_thermal = std::max(max_old_thermal, u0_k);
                    }
                }
                amrex::Print() << "material_work_oracle: max_work_J=" << max_work
                               << " max_error_J=" << max_error
                               << " max_old_thermal_J=" << max_old_thermal << '\n';
                AMREX_ALWAYS_ASSERT(max_work > 1.e-9_rt * max_old_thermal);
                AMREX_ALWAYS_ASSERT(max_error <=
                                    1.e-9_rt * max_work + 128._rt * eps * max_old_thermal);
            }
        }
        amrex::Print() << "Native Ohm pressure/BC/Faraday witness passed; "
                          "not full native magnetic/material evolution qualification.\n";
        WarpX::Finalize();
    }
    warpx::initialization::finalize_external_libraries();
}
