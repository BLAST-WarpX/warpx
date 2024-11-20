/* Copyright 2019
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "PsatdAlgorithmFirstOrder.H"

#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpX_Complex.H"

#include <AMReX_Array4.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_BaseFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_GpuComplex.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_PODVector.H>

#include <cmath>

#if WARPX_USE_FFT

using namespace amrex::literals;

PsatdAlgorithmFirstOrder::PsatdAlgorithmFirstOrder (
    const SpectralKSpace& spectral_kspace,
    const amrex::DistributionMapping& dm,
    const SpectralFieldIndex& spectral_index,
    const int norder_x,
    const int norder_y,
    const int norder_z,
    ablastr::utils::enums::GridType grid_type,
    const amrex::Real dt,
    const bool div_cleaning,
    const JInTime J_in_time,
    const RhoInTime rho_in_time
)
    // Initializer list
    : SpectralBaseAlgorithm(spectral_kspace, dm, spectral_index, norder_x, norder_y, norder_z, grid_type),
    m_dt(dt),
    m_div_cleaning(div_cleaning),
    m_J_in_time(J_in_time),
    m_rho_in_time(rho_in_time)
{}

void
PsatdAlgorithmFirstOrder::pushSpectralFields (SpectralFieldData& f) const
{
    const bool div_cleaning = m_div_cleaning;

    const bool J_constant = (m_J_in_time == JInTime::Constant);
    const bool J_linear   = (m_J_in_time == JInTime::Linear);
    const bool rho_constant = (m_rho_in_time == RhoInTime::Constant);
    const bool rho_linear   = (m_rho_in_time == RhoInTime::Linear);

    const amrex::Real dt = m_dt;
    const amrex::Real dt2 = dt*dt;

    const SpectralFieldIndex& Idx = m_spectral_index;

    // Loop over boxes
    for (amrex::MFIter mfi(f.fields); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = f.fields[mfi].box();

        // Extract arrays for the fields to be updated
        const amrex::Array4<Complex> fields = f.fields[mfi].array();

        // Extract pointers for the k vectors
        const amrex::Real* modified_kx_arr = modified_kx_vec[mfi].dataPtr();
#if defined(WARPX_DIM_3D)
        const amrex::Real* modified_ky_arr = modified_ky_vec[mfi].dataPtr();
#endif
        const amrex::Real* modified_kz_arr = modified_kz_vec[mfi].dataPtr();

        // Loop over indices within one box
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            // Record old values of the fields to be updated
            const Complex Ex_old = fields(i,j,k,Idx.Ex);
            const Complex Ey_old = fields(i,j,k,Idx.Ey);
            const Complex Ez_old = fields(i,j,k,Idx.Ez);
            const Complex Bx_old = fields(i,j,k,Idx.Bx);
            const Complex By_old = fields(i,j,k,Idx.By);
            const Complex Bz_old = fields(i,j,k,Idx.Bz);

            // Shortcuts for the values of J and rho
            const Complex Jx_mid = (J_constant) ? fields(i,j,k,Idx.Jx_mid) : 0._rt;
            const Complex Jy_mid = (J_constant) ? fields(i,j,k,Idx.Jy_mid) : 0._rt;
            const Complex Jz_mid = (J_constant) ? fields(i,j,k,Idx.Jz_mid) : 0._rt;
            const Complex Jx_old = (J_linear  ) ? fields(i,j,k,Idx.Jx_old) : 0._rt;
            const Complex Jy_old = (J_linear  ) ? fields(i,j,k,Idx.Jy_old) : 0._rt;
            const Complex Jz_old = (J_linear  ) ? fields(i,j,k,Idx.Jz_old) : 0._rt;
            const Complex Jx_new = (J_linear  ) ? fields(i,j,k,Idx.Jx_new) : 0._rt;
            const Complex Jy_new = (J_linear  ) ? fields(i,j,k,Idx.Jy_new) : 0._rt;
            const Complex Jz_new = (J_linear  ) ? fields(i,j,k,Idx.Jz_new) : 0._rt;

            const Complex Jx_c0 = (J_constant) ? Jx_mid : Jx_old;
            const Complex Jy_c0 = (J_constant) ? Jy_mid : Jy_old;
            const Complex Jz_c0 = (J_constant) ? Jz_mid : Jz_old;
            const Complex Jx_c1 = (J_linear  ) ? (Jx_new-Jx_old)/dt : 0._rt;
            const Complex Jy_c1 = (J_linear  ) ? (Jy_new-Jy_old)/dt : 0._rt;
            const Complex Jz_c1 = (J_linear  ) ? (Jz_new-Jz_old)/dt : 0._rt;

            Complex rho_mid, rho_old, rho_new, F_old, G_old;
            Complex rho_c0, rho_c1;
            if (div_cleaning)
            {
                rho_mid = (rho_constant) ? fields(i,j,k,Idx.rho_mid) : 0._rt;
                rho_old = (rho_linear  ) ? fields(i,j,k,Idx.rho_old) : 0._rt;
                rho_new = (rho_linear  ) ? fields(i,j,k,Idx.rho_new) : 0._rt;

                F_old = fields(i,j,k,Idx.F);
                G_old = fields(i,j,k,Idx.G);

                rho_c0 = (rho_constant) ? rho_mid : rho_old;
                rho_c1 = (rho_linear  ) ? (rho_new-rho_old)/dt : 0._rt;
            }

            // k vector values
            const amrex::Real kx = modified_kx_arr[i];
#if defined(WARPX_DIM_3D)
            const amrex::Real ky = modified_ky_arr[j];
            const amrex::Real kz = modified_kz_arr[k];
#else
            constexpr amrex::Real ky = 0._rt;
            const     amrex::Real kz = modified_kz_arr[j];
#endif
            // Physical constants and imaginary unit
            constexpr amrex::Real c = PhysConst::c;
            constexpr amrex::Real c2 = c*c;
            constexpr amrex::Real inv_c = 1._rt/c;
            constexpr amrex::Real mu0 = PhysConst::mu0;
            constexpr Complex I = Complex{0._rt, 1._rt};

            const amrex::Real kx2 = kx*kx;
            const amrex::Real ky2 = ky*ky;
            const amrex::Real kz2 = kz*kz;

            const amrex::Real knorm = std::sqrt(kx2 + ky2 + kz2);
            const amrex::Real knorm2 = knorm*knorm;
            const amrex::Real knorm4 = knorm2*knorm2;

            // Auxiliary variables
            const amrex::Real inv_knorm  = 1._rt/knorm;
            const amrex::Real inv_knorm2 = 1._rt/knorm2;
            const amrex::Real inv_knorm4 = 1._rt/knorm4;

            const amrex::Real C = std::cos(c*knorm*dt);
            const amrex::Real S = std::sin(c*knorm*dt);

            // Update equations

            if (knorm == 0._rt)
            {
                fields(i,j,k,Idx.Ex) = Ex_old - mu0*c2*dt*Jx_c0 - 0.5_rt*mu0*c2*dt2*Jx_c1;
                fields(i,j,k,Idx.Ey) = Ey_old - mu0*c2*dt*Jy_c0 - 0.5_rt*mu0*c2*dt2*Jy_c1;
                fields(i,j,k,Idx.Ez) = Ez_old - mu0*c2*dt*Jz_c0 - 0.5_rt*mu0*c2*dt2*Jz_c1;

                if (div_cleaning)
                {
                    fields(i,j,k,Idx.F) = F_old - mu0*c2*dt*rho_c0 - 0.5_rt*mu0*c2*dt2*rho_c1;
                }
            }
            else // knorm != 0
            {
                Complex C01, C02, C03, C04, C05, C06, C07, C08,
                        C09, C10, C11, C12, C13, C14, C15, C16;

                // Ex
                C01 = (div_cleaning) ? C : (kx2+ky2*C+kz2*C)*inv_knorm2;
                C02 = (div_cleaning) ? 0._rt : kx*ky*(1._rt-C)*inv_knorm2;
                C03 = (div_cleaning) ? 0._rt : kx*kz*(1._rt-C)*inv_knorm2;
                C04 = 0._rt;
                C05 = -I*c*kz*S*inv_knorm;
                C06 =  I*c*ky*S*inv_knorm;
                C07 = (div_cleaning) ? I*c*kx*S*inv_knorm : 0._rt;
                C09 = (div_cleaning) ? -mu0*c*S*inv_knorm : -mu0*c*(dt*c*kx2*knorm2+ky2*knorm*S+kz2*knorm*S)*inv_knorm4;
                C10 = (div_cleaning) ? 0._rt : mu0*c*kx*ky*(knorm*S-dt*c*knorm2)*inv_knorm4;
                C11 = (div_cleaning) ? 0._rt : mu0*c*kx*kz*(knorm*S-dt*c*knorm2)*inv_knorm4;
                C12 = 0._rt; // This is not redundant, do not remove this
                if (J_linear) { C12 = (div_cleaning) ? mu0*(C-1._rt)*inv_knorm2 : mu0*(2._rt*ky2*(C-1._rt)+2._rt*kz2*(C-1._rt)-dt2*c2*kx2*knorm2)*inv_knorm4*0.5_rt; }
                C13 = (J_linear && !div_cleaning) ? mu0*kx*ky*(2._rt*(1._rt-C)-dt2*c2*knorm2)*inv_knorm4*0.5_rt : 0._rt;
                C14 = (J_linear && !div_cleaning) ? mu0*kx*kz*(2._rt*(1._rt-C)-dt2*c2*knorm2)*inv_knorm4*0.5_rt : 0._rt;
                C15 = (div_cleaning) ? I*mu0*c2*kx*(C-1._rt)*inv_knorm2 : 0._rt;
                C16 = (div_cleaning && rho_linear) ? I*mu0*c*kx*(knorm*S-dt*c*knorm2)*inv_knorm4 : 0._rt;

                fields(i,j,k,Idx.Ex) = C01*Ex_old + C02*Ey_old + C03*Ez_old
                                     + C04*Bx_old + C05*By_old + C06*Bz_old
                                     + C07*F_old // only with div cleaning
                                     + C09*Jx_c0 + C10*Jy_c0 + C11*Jz_c0
                                     + C12*Jx_c1 + C13*Jy_c1 + C14*Jz_c1 // only with J linear in time
                                     + C15*rho_c0  // only with div cleaning
                                     + C16*rho_c1; // only with div cleaning and rho linear in time

                // Ey
                C01 = (div_cleaning) ? 0._rt : kx*ky*(1._rt-C)*inv_knorm2;
                C02 = (div_cleaning) ? C : (kx2*C+ky2+kz2*C)*inv_knorm2;
                C03 = (div_cleaning) ? 0._rt : ky*kz*(1._rt-C)*inv_knorm2;
                C04 = I*c*kz*S*inv_knorm;
                C05 = 0._rt;
                C06 = -I*c*kx*S*inv_knorm;
                C07 = (div_cleaning) ? I*c*ky*S*inv_knorm : 0._rt;
                C09 = (div_cleaning) ? 0._rt : mu0*c*kx*ky*(knorm*S-dt*c*knorm2)*inv_knorm4;
                C10 = (div_cleaning) ? -mu0*c*S*inv_knorm : -mu0*c*(dt*c*ky2*knorm2+kx2*knorm*S+kz2*knorm*S)*inv_knorm4;
                C11 = (div_cleaning) ? 0._rt : mu0*c*ky*kz*(knorm*S-dt*c*knorm2)*inv_knorm4;
                C12 = (J_linear && !div_cleaning) ? mu0*kx*ky*(2._rt*(1._rt-C)-dt2*c2*knorm2)*inv_knorm4*0.5_rt : 0._rt;
                C13 = 0._rt; // This is not redundant, do not remove this
                if (J_linear) { C13 = (div_cleaning) ? mu0*(C-1._rt)*inv_knorm2 : mu0*(2._rt*kx2*(C-1._rt)+2._rt*kz2*(C-1._rt)-dt2*c2*ky2*knorm2)*inv_knorm4*0.5_rt; }
                C14 = (J_linear && !div_cleaning) ? mu0*ky*kz*(2._rt*(1._rt-C)-dt2*c2*knorm2)*inv_knorm4*0.5_rt : 0._rt;
                C15 = (div_cleaning) ? I*mu0*c2*ky*(C-1._rt)*inv_knorm2 : 0._rt;
                C16 = (div_cleaning && rho_linear) ? I*mu0*c*ky*(knorm*S-dt*c*knorm2)*inv_knorm4 : 0._rt;

                fields(i,j,k,Idx.Ey) = C01*Ex_old + C02*Ey_old + C03*Ez_old
                                     + C04*Bx_old + C05*By_old + C06*Bz_old
                                     + C07*F_old // only with div cleaning
                                     + C09*Jx_c0 + C10*Jy_c0 + C11*Jz_c0
                                     + C12*Jx_c1 + C13*Jy_c1 + C14*Jz_c1 // only with J linear in time
                                     + C15*rho_c0  // only with div cleaning
                                     + C16*rho_c1; // only with div cleaning and rho linear in time

                // Ez
                C01 = (div_cleaning) ? 0._rt : kx*kz*(1._rt-C)*inv_knorm2;
                C02 = (div_cleaning) ? 0._rt : ky*kz*(1._rt-C)*inv_knorm2;
                C03 = (div_cleaning) ? C : (kx2*C+ky2*C+kz2)*inv_knorm2;
                C04 = -I*c*ky*S*inv_knorm;
                C05 = I*c*kx*S*inv_knorm;
                C06 = 0._rt;
                C07 = (div_cleaning) ? I*c*kz*S*inv_knorm : 0._rt;
                C09 = (div_cleaning) ? 0._rt : mu0*c*kx*kz*(knorm*S-dt*c*knorm2)*inv_knorm4;
                C10 = (div_cleaning) ? 0._rt : mu0*c*ky*kz*(knorm*S-dt*c*knorm2)*inv_knorm4;
                C11 = (div_cleaning) ? -mu0*c*S*inv_knorm : -mu0*c*(dt*c*kz2*knorm2+kx2*knorm*S+ky2*knorm*S)*inv_knorm4;
                C12 = (J_linear && !div_cleaning) ? mu0*kx*kz*(2._rt*(1._rt-C)-dt2*c2*knorm2)*inv_knorm4*0.5_rt : 0._rt;
                C13 = (J_linear && !div_cleaning) ? mu0*ky*kz*(2._rt*(1._rt-C)-dt2*c2*knorm2)*inv_knorm4*0.5_rt : 0._rt;
                C14 = 0._rt; // This is not redundant, do not remove this
                if (J_linear) { C14 = (div_cleaning) ? mu0*(C-1._rt)*inv_knorm2 : mu0*(2._rt*kx2*(C-1._rt)+2._rt*ky2*(C-1._rt)-dt2*c2*kz2*knorm2)*inv_knorm4*0.5_rt; }
                C15 = (div_cleaning) ? I*mu0*c2*kz*(C-1._rt)*inv_knorm2 : 0._rt;
                C16 = (div_cleaning && rho_linear) ? I*mu0*c*kz*(knorm*S-dt*c*knorm2)*inv_knorm4 : 0._rt;

                fields(i,j,k,Idx.Ez) = C01*Ex_old + C02*Ey_old + C03*Ez_old
                                     + C04*Bx_old + C05*By_old + C06*Bz_old
                                     + C07*F_old // only with div cleaning
                                     + C09*Jx_c0 + C10*Jy_c0 + C11*Jz_c0
                                     + C12*Jx_c1 + C13*Jy_c1 + C14*Jz_c1 // only with J linear in time
                                     + C15*rho_c0  // only with div cleaning
                                     + C16*rho_c1; // only with div cleaning and rho linear in time

                // Bx
                C01 = 0._rt;
                C02 = I*kz*S*inv_knorm*inv_c;
                C03 = -I*ky*S*inv_knorm*inv_c;
                C04 = (div_cleaning) ? C : (kx2+ky2*C+kz2*C)*inv_knorm2;
                C05 = (div_cleaning) ? 0._rt : kx*ky*(1._rt-C)*inv_knorm2;
                C06 = (div_cleaning) ? 0._rt : kx*kz*(1._rt-C)*inv_knorm2;
                C08 = (div_cleaning) ? I*kx*S*inv_knorm*inv_c : 0._rt;
                C09 = 0._rt;
                C10 = I*mu0*kz*(C-1._rt)*inv_knorm2;
                C11 = -I*mu0*ky*(C-1._rt)*inv_knorm2;
                C12 = 0._rt;
                C13 = (J_linear) ? I*mu0*kz*(knorm*S-dt*c*knorm2)*inv_knorm4*inv_c : 0._rt;
                C14 = (J_linear) ? -I*mu0*ky*(knorm*S-dt*c*knorm2)*inv_knorm4*inv_c : 0._rt;

                fields(i,j,k,Idx.Bx) = C01*Ex_old + C02*Ey_old + C03*Ez_old
                                     + C04*Bx_old + C05*By_old + C06*Bz_old
                                     + C08*G_old // only with div cleaning
                                     + C09*Jx_c0 + C10*Jy_c0 + C11*Jz_c0
                                     + C12*Jx_c1 + C13*Jy_c1 + C14*Jz_c1; // only with J linear in time

                // By
                C01 = -I*kz*S*inv_knorm*inv_c;
                C02 = 0._rt;
                C03 = I*kx*S*inv_knorm*inv_c;
                C04 = (div_cleaning) ? 0._rt : kx*ky*(1._rt-C)*inv_knorm2;
                C05 = (div_cleaning) ? C : (kx2*C+ky2+kz2*C)*inv_knorm2;
                C06 = (div_cleaning) ? 0._rt : ky*kz*(1._rt-C)*inv_knorm2;
                C08 = (div_cleaning) ? I*ky*S*inv_knorm*inv_c : 0._rt;
                C09 = -I*mu0*kz*(C-1._rt)*inv_knorm2;
                C10 = 0._rt;
                C11 = I*mu0*kx*(C-1._rt)*inv_knorm2;
                C12 = (J_linear) ? -I*mu0*kz*(knorm*S-dt*c*knorm2)*inv_knorm4*inv_c : 0._rt;
                C13 = 0._rt;
                C14 = (J_linear) ? I*mu0*kx*(knorm*S-dt*c*knorm2)*inv_knorm4*inv_c : 0._rt;

                fields(i,j,k,Idx.By) = C01*Ex_old + C02*Ey_old + C03*Ez_old
                                     + C04*Bx_old + C05*By_old + C06*Bz_old
                                     + C08*G_old // only with div cleaning
                                     + C09*Jx_c0 + C10*Jy_c0 + C11*Jz_c0
                                     + C12*Jx_c1 + C13*Jy_c1 + C14*Jz_c1; // only with J linear in time

                // Bz
                C01 = I*ky*S*inv_knorm*inv_c;
                C02 = -I*kx*S*inv_knorm*inv_c;
                C03 = 0._rt;
                C04 = (div_cleaning) ? 0._rt : kx*kz*(1._rt-C)*inv_knorm2;
                C05 = (div_cleaning) ? 0._rt : ky*kz*(1._rt-C)*inv_knorm2;
                C06 = (div_cleaning) ? C : (kx2*C+ky2*C+kz2)*inv_knorm2;
                C08 = (div_cleaning) ? I*kz*S*inv_knorm*inv_c : 0._rt;
                C09 = I*mu0*ky*(C-1._rt)*inv_knorm2;
                C10 = -I*mu0*kx*(C-1._rt)*inv_knorm2;
                C11 = 0._rt;
                C12 = (J_linear) ? I*mu0*ky*(knorm*S-dt*c*knorm2)*inv_knorm4*inv_c : 0._rt;
                C13 = (J_linear) ? -I*mu0*kx*(knorm*S-dt*c*knorm2)*inv_knorm4*inv_c : 0._rt;
                C14 = 0._rt;

                fields(i,j,k,Idx.Bz) = C01*Ex_old + C02*Ey_old + C03*Ez_old
                                     + C04*Bx_old + C05*By_old + C06*Bz_old
                                     + C08*G_old // only with div cleaning
                                     + C09*Jx_c0 + C10*Jy_c0 + C11*Jz_c0
                                     + C12*Jx_c1 + C13*Jy_c1 + C14*Jz_c1; // only with J linear in time

                if (div_cleaning)
                {
                    // F
                    C01 = I*kx*S*inv_knorm*inv_c;
                    C02 = I*ky*S*inv_knorm*inv_c;
                    C03 = I*kz*S*inv_knorm*inv_c;
                    C07 = C;
                    C09 = I*mu0*kx*(C-1._rt)*inv_knorm2;
                    C10 = I*mu0*ky*(C-1._rt)*inv_knorm2;
                    C11 = I*mu0*kz*(C-1._rt)*inv_knorm2;
                    C12 = (J_linear) ? I*mu0*kx*(knorm*S-dt*c*knorm2)*inv_knorm4*inv_c : 0._rt;
                    C13 = (J_linear) ? I*mu0*ky*(knorm*S-dt*c*knorm2)*inv_knorm4*inv_c : 0._rt;
                    C14 = (J_linear) ? I*mu0*kz*(knorm*S-dt*c*knorm2)*inv_knorm4*inv_c : 0._rt;
                    C15 = -mu0*c*S*inv_knorm;
                    C16 = (rho_linear) ? mu0*(C-1._rt)*inv_knorm2 : 0._rt;

                    fields(i,j,k,Idx.F) = C01*Ex_old + C02*Ey_old + C03*Ez_old
                                        + C07*F_old
                                        + C09*Jx_c0 + C10*Jy_c0 + C11*Jz_c0
                                        + C12*Jx_c1 + C13*Jy_c1 + C14*Jz_c1 // only with J linear in time
                                        + C15*rho_c0
                                        + C16*rho_c1; // only with rho linear in time

                    // G
                    C04 = I*c*kx*S*inv_knorm;
                    C05 = I*c*ky*S*inv_knorm;
                    C06 = I*c*kz*S*inv_knorm;
                    C08 = C;

                    fields(i,j,k,Idx.G) = C04*Bx_old + C05*By_old + C06*Bz_old
                                        + C08*G_old;
                }
            }
        });
    }
}

void PsatdAlgorithmFirstOrder::CurrentCorrection (SpectralFieldData& field_data)
{
    // Profiling
    BL_PROFILE("PsatdAlgorithmFirstOrder::CurrentCorrection");

    amrex::ignore_unused(field_data);
    WARPX_ABORT_WITH_MESSAGE(
        "Current correction not implemented for first-order PSATD equations");
}

void
PsatdAlgorithmFirstOrder::VayDeposition (SpectralFieldData& field_data)
{
    // Profiling
    BL_PROFILE("PsatdAlgorithmFirstOrder::VayDeposition()");

    amrex::ignore_unused(field_data);
    WARPX_ABORT_WITH_MESSAGE(
        "Vay deposition not implemented for first-order PSATD equations");
}

#endif // WARPX_USE_FFT
