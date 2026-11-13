/* Copyright 2026 WarpX contributors
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FiniteDifferenceSolver.H"

#ifdef WARPX_DIM_RZ
#include "BoundaryConditions/PML.H"
#include "Fields.H"
#include "FiniteDifferenceAlgorithms/CylindricalYeeAlgorithm.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_Array4.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

#include <cmath>

using namespace amrex::literals;
using warpx::fields::FieldType;

namespace {
// Integrate the convolution memory for a constant staggered curl over this
// field substep. The physical field uses the time-integrated memory, so two
// magnetic half-steps compose exactly when their electric field is fixed.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
convolved_increment (amrex::Real driving, amrex::Real& memory, amrex::Real sigma, amrex::Real dt)
{
    amrex::Real const integral = sigma == 0 ? dt : -std::expm1(-sigma * dt) / sigma;
    amrex::Real const change = (driving + memory) * integral;
    memory -= sigma * change;
    return change;
}
} // namespace

void
FiniteDifferenceSolver::EvolvePMLCylindrical (ablastr::fields::MultiFabRegister& fields,
                                              PatchType patch_type, int level,
                                              MultiSigmaBox const& sigma_boxes, amrex::Real dt,
                                              bool magnetic)
{
    bool const fine = patch_type == PatchType::fine;
    auto const electric =
        fields.get_alldirs(fine ? FieldType::pml_E_fp : FieldType::pml_E_cp, level);
    auto const magnetic_field =
        fields.get_alldirs(fine ? FieldType::pml_B_fp : FieldType::pml_B_cp, level);
    auto const destination = magnetic ? magnetic_field : electric;
    auto const source = magnetic ? electric : magnetic_field;
    int const modes = m_nmodes;
    int const components = 2 * modes - 1;
    amrex::Real const dr = m_dr, rmin = m_rmin;
    auto const* const radial_coefficients = m_stencil_coefs_r.dataPtr();
    auto const* const axial_coefficients = m_stencil_coefs_z.dataPtr();
    int const radial_order = static_cast<int>(m_stencil_coefs_r.size());
    int const axial_order = static_cast<int>(m_stencil_coefs_z.size());
    using Curl = CylindricalYeeAlgorithm;
    auto const& electric_masks = WarpX::GetInstance().GetPML(level)->GetEBUpdateEFlag(patch_type);

    for (int direction = 0; direction < 3; ++direction) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            destination[direction]->nComp() == 3 * components,
            "Cylindrical QCPML needs physical fields and radial/axial convolution memories.");
        auto const staggering = destination[direction]->ixType();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(*destination[direction], amrex::TilingIfNotGPU()); mfi.isValid();
             ++mfi) {
            auto const output = destination[direction]->array(mfi);
            auto const active = !magnetic && electric_masks[direction]
                                    ? electric_masks[direction]->const_array(mfi)
                                    : amrex::Array4<int const>{};
            auto const radial = source[0]->const_array(mfi);
            auto const azimuthal = source[1]->const_array(mfi);
            auto const axial = source[2]->const_array(mfi);
            auto const electric_radial = electric[0]->const_array(mfi);
            auto const& radial_profile = staggering.nodeCentered(0)
                                             ? sigma_boxes[mfi].sigma[0]
                                             : sigma_boxes[mfi].sigma_star[0];
            auto const& axial_profile = staggering.nodeCentered(1) ? sigma_boxes[mfi].sigma[1]
                                                                   : sigma_boxes[mfi].sigma_star[1];
            auto const* const sigma_r = radial_profile.data();
            auto const* const sigma_z = axial_profile.data();
            int const first_r = radial_profile.lo(), first_z = axial_profile.lo();
            amrex::ParallelFor(
                mfi.tilebox(), components, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                    if (active && active(i, j, k) == 0) {
                        output(i, j, k, n) = 0;
                        output(i, j, k, n + components) = 0;
                        output(i, j, k, n + 2 * components) = 0;
                        return;
                    }
                    int const mode = (n + 1) / 2;
                    bool const imaginary = n > 0 && n % 2 == 0;
                    int const partner = n == 0 ? 0 : imaginary ? n - 1 : n + 1;
                    amrex::Real const angular_sign = imaginary ? -1.0_rt : 1.0_rt;
                    amrex::Real const r =
                        rmin + (i + (staggering.nodeCentered(0) ? 0.0_rt : 0.5_rt)) * dr;
                    amrex::Real radial_drive = 0, axial_drive = 0, angular_drive = 0;
                    if (magnetic) {
                        if (direction == 0) {
                            if (r == 0 && mode != 1) {
                                output(i, j, k, n) = 0;
                                output(i, j, k, n + components) = 0;
                                output(i, j, k, n + 2 * components) = 0;
                                return;
                            }
                            axial_drive = Curl::UpwardDz(azimuthal, axial_coefficients, axial_order,
                                                         i, j, k, n);
                            angular_drive = mode == 0
                                                ? 0
                                                : -angular_sign * mode *
                                                      (r == 0 ? axial(i + 1, j, k, partner) / dr
                                                              : axial(i, j, k, partner) / r);
                        } else if (direction == 1) {
                            radial_drive = Curl::UpwardDr(axial, radial_coefficients, radial_order,
                                                          i, j, k, n);
                            axial_drive = -Curl::UpwardDz(radial, axial_coefficients, axial_order,
                                                          i, j, k, n);
                        } else {
                            radial_drive = -Curl::UpwardDrr_over_r(
                                azimuthal, r, dr, radial_coefficients, radial_order, i, j, k, n);
                            angular_drive =
                                mode == 0 ? 0 : angular_sign * mode * radial(i, j, k, partner) / r;
                        }
                    } else {
                        if (direction == 0) {
                            axial_drive =
                                -PhysConst::c2 * Curl::DownwardDz(azimuthal, axial_coefficients,
                                                                  axial_order, i, j, k, n);
                            angular_drive = mode == 0 ? 0
                                                      : PhysConst::c2 * angular_sign * mode *
                                                            axial(i, j, k, partner) / r;
                        } else if (direction == 1) {
                            if (r == 0) {
                                output(i, j, k, n) =
                                    mode == 1 ? (imaginary ? -electric_radial(i, j, k, partner)
                                                           : electric_radial(i, j, k, partner))
                                              : 0;
                                output(i, j, k, n + components) = 0;
                                output(i, j, k, n + 2 * components) = 0;
                                return;
                            }
                            radial_drive =
                                -PhysConst::c2 * Curl::DownwardDr(axial, radial_coefficients,
                                                                  radial_order, i, j, k, n);
                            axial_drive =
                                PhysConst::c2 * Curl::DownwardDz(radial, axial_coefficients,
                                                                 axial_order, i, j, k, n);
                        } else {
                            if (r == 0) {
                                if (mode != 0) {
                                    output(i, j, k, n) = 0;
                                    output(i, j, k, n + components) = 0;
                                    output(i, j, k, n + 2 * components) = 0;
                                    return;
                                }
                                radial_drive = PhysConst::c2 * 4 * azimuthal(i, j, k, n) / dr;
                            } else {
                                radial_drive =
                                    PhysConst::c2 *
                                    Curl::DownwardDrr_over_r(azimuthal, r, dr, radial_coefficients,
                                                             radial_order, i, j, k, n);
                                angular_drive = mode == 0 ? 0
                                                          : -PhysConst::c2 * angular_sign * mode *
                                                                radial(i, j, k, partner) / r;
                            }
                        }
                    }
                    output(i, j, k, n) +=
                        dt * angular_drive +
                        convolved_increment(radial_drive, output(i, j, k, n + components),
                                            sigma_r[i - first_r], dt) +
                        convolved_increment(axial_drive, output(i, j, k, n + 2 * components),
                                            sigma_z[j - first_z], dt);
                });
        }
    }
}
#endif
