/* Copyright 2026 WarpX contributors
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "PrescribedCurrentParticleContainer.H"

#include "Fields.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/CurrentWaveform.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/profiler/ProfilerWrapper.H>
#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_ParIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace amrex;
using warpx::fields::FieldType;

bool
PrescribedCurrentParticleContainer::is_enabled () {
    bool enabled = false;
    const ParmParse pp_warpx("warpx");
    pp_warpx.query("current_injection", enabled);
    if (!enabled) {
        return false;
    }

    std::string source_type = "antenna";
    pp_warpx.query("current_injection.type", source_type);
    std::transform(
        source_type.begin(), source_type.end(), source_type.begin(),
        [] (unsigned char const value) { return std::tolower(value); });
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        source_type == "antenna",
        "warpx.current_injection.type must be 'antenna'. Paired terminals use "
        "the "
        "separate warpx.current_controlled_port namespace.");
    return enabled;
}

PrescribedCurrentParticleContainer::PrescribedCurrentParticleContainer (
    AmrCore* amr_core, int ispecies)
    : WarpXParticleContainer(amr_core, ispecies, "prescribed_current") {
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::electromagnetic_solver_id !=
                                         ElectromagneticSolverAlgo::HybridPIC,
                                     "The box-based warpx.current_injection "
                                     "source is an impressed-current antenna, "
                                     "not an ion species, and is not supported "
                                     "by the Hybrid-PIC solver. Use "
                                     "warpx.current_controlled_port for a "
                                     "current carried by the hybrid plasma.");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::current_deposition_algo == CurrentDepositionAlgo::Esirkepov ||
            WarpX::current_deposition_algo == CurrentDepositionAlgo::Villasenor,
        "The impressed-current antenna requires a charge-conserving current "
        "deposition "
        "scheme: algo.current_deposition = esirkepov or villasenor.");

    m_charge = 1.0;
    m_mass = std::numeric_limits<Real>::max();
    AddIntComp("face_id");

    const ParmParse pp_warpx("warpx");

    // --- Optional global waveform (used by any pair without its own file) ----
    warpx::utils::CurrentWaveform global_waveform;
    std::string global_file;
    const bool has_global_file =
        pp_warpx.query("current_injection.file", global_file);
    if (has_global_file) {
        global_waveform.load(global_file);
    }

    // --- Drive faces (one per pair) ------------------------------------------
    int n_pairs = 1;
    pp_warpx.query("current_injection.n_pairs", n_pairs);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        n_pairs >= 1, "warpx.current_injection.n_pairs must be >= 1");

    for (int p = 0; p < n_pairs; ++p) {
        const std::string base = "current_injection.pair_" + std::to_string(p);

        Face f;
        utils::parser::getWithParser(pp_warpx, (base + ".drive.xlo").c_str(),
                                     f.lo[0]);
        utils::parser::getWithParser(pp_warpx, (base + ".drive.xhi").c_str(),
                                     f.hi[0]);
        utils::parser::getWithParser(pp_warpx, (base + ".drive.ylo").c_str(),
                                     f.lo[1]);
        utils::parser::getWithParser(pp_warpx, (base + ".drive.yhi").c_str(),
                                     f.hi[1]);
        utils::parser::getWithParser(pp_warpx, (base + ".drive.zlo").c_str(),
                                     f.lo[2]);
        utils::parser::getWithParser(pp_warpx, (base + ".drive.zhi").c_str(),
                                     f.hi[2]);
        pp_warpx.query(base + ".drive.dir", f.dir);
        pp_warpx.query(base + ".drive.sign", f.sign);

        for (int d = 0; d < 3; ++d) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::isfinite(f.lo[d]) && std::isfinite(f.hi[d]) &&
                    f.lo[d] < f.hi[d],
                "current_injection drive bounds must be finite and satisfy lo "
                "< hi");
        }

        // Area A is the physical cross-section perpendicular to the current.
        // In RZ with dir = 0 (Jr) the imposed profile is the conserved-total-I
        // 1/r coaxial profile Jr = I/(2*pi*r*dz), which has no single A, so A
        // is unused and may be omitted. In every other case A is required.
#if defined(WARPX_DIM_RZ)
        const bool a_required = (f.dir != 0);
#else
        const bool a_required = true;
#endif
        if (a_required) {
            utils::parser::getWithParser(pp_warpx, (base + ".drive.A").c_str(),
                                         f.A);
        } else {
            pp_warpx.query(base + ".drive.A", f.A);
        }

        // Per-pair waveform file overrides the global one.
        std::string pair_file;
        if (pp_warpx.query(base + ".file", pair_file)) {
            f.waveform.load(pair_file);
        } else {
            std::string msg = "warpx.current_injection: ";
            msg += base;
            msg += " needs ";
            msg += base;
            msg += ".file (or set the global warpx.current_injection.file).";
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(has_global_file, msg);
            f.waveform = global_waveform;
        }

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            f.dir >= 0 && f.dir < 3,
            "current_injection drive.dir must be 0, 1 or 2");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            f.sign == -1 || f.sign == 1,
            "current_injection drive.sign must be -1 or 1");
#if defined(WARPX_DIM_RZ)
        if (f.dir == 0) {
            // Jr coaxial drive: box must lie off-axis (r > 0); Jr ~ 1/r
            // diverges at r = 0 and the inverse-volume scaling forces Jr = 0 on
            // axis.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                f.lo[0] > 0., "current_injection RZ drive (dir=0, Jr) needs "
                              "xlo > 0 (off-axis)");
        } else {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                f.A > 0., "current_injection drive.A must be > 0");
        }
#else
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            f.A > 0., "current_injection drive.A must be > 0");
#endif
        m_faces.push_back(std::move(f));
    }

    m_enabled = !m_faces.empty();
}

void
PrescribedCurrentParticleContainer::compute_velocity_scale () {
    Real I_peak = 0._rt;
    Real charge_bound = 0._rt;
    for (const Face& f : m_faces) {
        I_peak = std::max(I_peak, f.waveform.max_abs_current());
        charge_bound =
            std::max(charge_bound, f.waveform.absolute_charge_bound());
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        I_peak > 0._rt,
        "warpx.current_injection: all waveforms are identically zero.");

    Real min_cell_size = std::numeric_limits<Real>::max();
    auto const cell_size = Geom(0).CellSizeArray();
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
        min_cell_size = std::min(min_cell_size, cell_size[direction]);
    }
    Real const speed_limited = 0.05_rt * PhysConst::c / I_peak;
    Real const displacement_limited =
        charge_bound > 0.0_rt ? 0.05_rt * min_cell_size / charge_bound
                              : speed_limited;
    m_vel_coeff = std::min(speed_limited, displacement_limited);
}

void
PrescribedCurrentParticleContainer::InitData () {
    if (!m_enabled) {
        return;
    }

#if defined(WARPX_DIM_1D_Z)
    WARPX_ABORT_WITH_MESSAGE(
        "Prescribed current injection is not implemented in 1D_Z geometry.");
#else
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(maxLevel() == 0,
                                     "Prescribed current injection currently "
                                     "supports only a single mesh level.");

    const int lev = 0;
    const auto& geom = Geom(lev);
    const auto problo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    const Box& domain = geom.Domain();

#if defined(WARPX_DIM_3D)
    const Real dV = dx[0] * dx[1] * dx[2];
#elif defined(WARPX_DIM_XZ)
    // 2D XZ: per-unit-length in the invariant y-direction. AMReX index 0 -> x,
    // index 1 -> z. A drive face area f.A is likewise per-unit-length in y.
    const Real dV = dx[0] * dx[1];
#elif defined(WARPX_DIM_RZ)
    // RZ: AMReX index 0 -> r, index 1 -> z; theta is invariant. The physical
    // cell volume is 2*pi*r*dr*dz, but the RZ current-deposition path deposits
    // a *raw* J (without the 2*pi*r factor) and
    // ApplyInverseVolumeScalingToCurrent Density divides by 2*pi*r afterward,
    // so the physical current density is
    //   J = w * q * v / (2*pi*r*dr*dz).
    // See the per-face weight derivation in the seeding loop below.
    const Real dr = dx[0];
    const Real dz = dx[1];
    const Real two_pi = MathConst::pi * 2._rt;
#else
    WARPX_ABORT_WITH_MESSAGE("PrescribedCurrentParticleContainer is only "
                             "implemented in 3D, 2D (XZ) and RZ.");
#endif

    // Keep both the peak speed below 0.05c and the total charge-separation
    // excursion below 0.05 of the smallest cell.
    compute_velocity_scale();

    // Each cell gets a coincident +/- weight pair. The two members move in
    // opposite directions, so their currents add while their charge separation
    // supplies the rho required by div(J) at a finite antenna region's ends.
    Vector<ParticleReal> xs, ys, zs, ws;
    Vector<int> face_ids;
    Real const initial_time = WarpX::GetInstance().gett_new(lev);
    if (ParallelDescriptor::MyProc() == 0) {
        const auto lo = domain.smallEnd();
        const auto hi = domain.bigEnd();
        for (int face_id = 0; face_id < static_cast<int>(m_faces.size());
             ++face_id) {
            const Face& f = m_faces[face_id];
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
            const Real W = 0.5_rt * dV / (m_vel_coeff * f.A);
#endif
#if defined(WARPX_DIM_3D)
            for (int k = lo[2]; k <= hi[2]; ++k) {
                for (int j = lo[1]; j <= hi[1]; ++j) {
                    for (int i = lo[0]; i <= hi[0]; ++i) {
                        const Real xc = problo[0] + (i + 0.5_rt) * dx[0];
                        const Real yc = problo[1] + (j + 0.5_rt) * dx[1];
                        const Real zc = problo[2] + (k + 0.5_rt) * dx[2];
                        if (xc >= f.lo[0] && xc < f.hi[0] && yc >= f.lo[1] &&
                            yc < f.hi[1] && zc >= f.lo[2] && zc < f.hi[2]) {
                            Real const displacement =
                                m_vel_coeff * f.sign *
                                f.waveform.integral(initial_time);
                            for (int const polarity : {-1, 1}) {
                                xs.push_back(xc + (f.dir == 0
                                                       ? polarity * displacement
                                                       : 0.0_rt));
                                ys.push_back(yc + (f.dir == 1
                                                       ? polarity * displacement
                                                       : 0.0_rt));
                                zs.push_back(zc + (f.dir == 2
                                                       ? polarity * displacement
                                                       : 0.0_rt));
                                ws.push_back(polarity * W);
                                face_ids.push_back(face_id);
                            }
                        }
                    }
                }
            }
#elif defined(WARPX_DIM_XZ)
            // AMReX index 0 -> x, index 1 -> z; y is invariant (face y-bounds
            // f.lo[1]/f.hi[1] are ignored). Seed at y = 0 (AddNParticles drops
            // y).
            for (int j = lo[1]; j <= hi[1]; ++j) {
                for (int i = lo[0]; i <= hi[0]; ++i) {
                    const Real xc = problo[0] + (i + 0.5_rt) * dx[0];
                    const Real zc = problo[1] + (j + 0.5_rt) * dx[1];
                    if (xc >= f.lo[0] && xc < f.hi[0] && zc >= f.lo[2] &&
                        zc < f.hi[2]) {
                        Real const displacement =
                            m_vel_coeff * f.sign *
                            f.waveform.integral(initial_time);
                        for (int const polarity : {-1, 1}) {
                            xs.push_back(xc + (f.dir == 0
                                                   ? polarity * displacement
                                                   : 0.0_rt));
                            ys.push_back(0.0_rt);
                            zs.push_back(zc + (f.dir == 2
                                                   ? polarity * displacement
                                                   : 0.0_rt));
                            ws.push_back(polarity * W);
                            face_ids.push_back(face_id);
                        }
                    }
                }
            }
#elif defined(WARPX_DIM_RZ)
            // AMReX index 0 -> r, index 1 -> z; theta is invariant (face
            // y-bounds f.lo[1]/f.hi[1] are ignored). Seed at theta = 0, i.e.
            // (x = r, y = 0): AddNParticles stores (r, theta=0, z), and the
            // Cartesian position push in Evolve is then exactly a radial
            // (Jr), azimuthal (Jtheta) or axial (Jz) step at theta = 0.
            //
            // Per-cell weight so that, after ApplyInverseVolumeScalingToCurrent
            // Density divides the raw deposit by 2*pi*r, the physical current
            // density is the prescribed profile:
            //   dir = 0 (Jr, conserved total I): uniform pair weight
            //       |W| = dr / (2*Nz*vel_coeff)
            //     -> Jr = I(t) / (2*pi*r*Nz*dz) (the 1/r coaxial profile; total
            //        radial current I is distributed across the box in z)
            //   dir = 1,2 (Jtheta/Jz, uniform J): weight ~ r
            //       |W| = (2*pi*r*dr*dz) / (2*vel_coeff*A)
            //     -> J = sign * I(t) / A        (uniform over the r-z box)
            int n_axial_cells = 0;
            for (int j = lo[1]; j <= hi[1]; ++j) {
                const Real zc = problo[1] + (j + 0.5_rt) * dx[1];
                if (zc >= f.lo[2] && zc < f.hi[2]) {
                    ++n_axial_cells;
                }
            }
            for (int j = lo[1]; j <= hi[1]; ++j) {
                for (int i = lo[0]; i <= hi[0]; ++i) {
                    const Real rc = problo[0] + (i + 0.5_rt) * dx[0];
                    const Real zc = problo[1] + (j + 0.5_rt) * dx[1];
                    if (rc >= f.lo[0] && rc < f.hi[0] && zc >= f.lo[2] &&
                        zc < f.hi[2]) {
                        const int n_spokes =
                            std::max(1, WarpX::n_rz_azimuthal_modes);
                        const Real W =
                            (f.dir == 0)
                                ? 0.5_rt * dr /
                                      (n_spokes * n_axial_cells * m_vel_coeff)
                                : 0.5_rt * (two_pi * rc * dr * dz) /
                                      (n_spokes * m_vel_coeff * f.A);
                        Real const displacement =
                            m_vel_coeff * f.sign *
                            f.waveform.integral(initial_time);

                        for (int spoke = 0; spoke < n_spokes; spoke++) {
                            const Real phase = two_pi * spoke / n_spokes;
                            for (int const polarity : {-1, 1}) {
                                Real radius = rc;
                                Real angle = phase;
                                Real z_position = zc;
                                if (f.dir == 0) {
                                    radius += polarity * displacement;
                                } else if (f.dir == 1) {
                                    angle += polarity * displacement / rc;
                                } else {
                                    z_position += polarity * displacement;
                                }
                                xs.push_back(radius * std::cos(angle));
                                ys.push_back(radius * std::sin(angle));
                                zs.push_back(z_position);
                                ws.push_back(polarity * W);
                                face_ids.push_back(face_id);
                            }
                        }
                    }
                }
            }
#endif
        }
    }

    const auto np = static_cast<long>(xs.size());
    const Vector<ParticleReal> ux(np, 0.0), uy(np, 0.0), uz(np, 0.0);
    Vector<Vector<ParticleReal>> attr;
    attr.push_back(ws);
    const Vector<Vector<int>> attr_int{face_ids};
    AddNParticles(lev, np, xs, ys, zs, ux, uy, uz, 1, attr, 1, attr_int, 1);

    if (Verbose()) {
        amrex::Print() << Utils::TextMsg::Info(
            "PrescribedCurrentParticleContainer: " +
            std::to_string(TotalNumberOfParticles()) +
            " antenna particles over " + std::to_string(m_faces.size()) +
            " antenna region(s)");
    }

    if (TotalNumberOfParticles() == 0) {
        ablastr::warn_manager::WMRecordWarning(
            "CurrentInjection",
            "warpx.current_injection: no cells found inside any drive box.",
            ablastr::warn_manager::WarnPriority::high);
        m_enabled = false;
    }
#endif
}

void
PrescribedCurrentParticleContainer::PostRestart () {
    // This deterministic antenna is not checkpointed with physical species.
    // Recreate it from the input boxes and waveform after loading a checkpoint.
    InitData();
}

void
PrescribedCurrentParticleContainer::Evolve (
    ablastr::fields::MultiFabRegister& fields, int lev,
    const std::string& current_fp_string, Real t, Real dt,
    SubcyclingHalf /*subcycling_half*/, bool skip_deposition,
    PositionPushType /*position_push_type*/,
    MomentumPushType /*momentum_push_type*/,
    ImplicitOptions const* implicit_options) {
    using ablastr::fields::Direction;

    ABLASTR_PROFILE("PrescribedCurrentParticleContainer::Evolve()");

    if (!m_enabled) {
        return;
    }

    const PushType push_type =
        (implicit_options == nullptr) ? PushType::Explicit : PushType::Implicit;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        dt > 0.0_rt, "Current antenna timestep must be positive.");

    // Use the exact step-average current of the piecewise-linear waveform.
    // Then the pair displacement, deposited J, and deposited endpoint charge
    // obey the same finite-step continuity equation.
    const int nfaces = static_cast<int>(m_faces.size());
    Gpu::DeviceVector<Real> d_velocity(nfaces);
    Gpu::DeviceVector<int> d_dir(nfaces);
    {
        Vector<Real> h_velocity(nfaces);
        Vector<int> hdir(nfaces);
        for (int f = 0; f < nfaces; ++f) {
            h_velocity[f] = m_faces[f].sign * m_vel_coeff *
                            (m_faces[f].waveform.integral(t + dt) -
                             m_faces[f].waveform.integral(t)) /
                            dt;
            hdir[f] = m_faces[f].dir;
        }
        Gpu::copyAsync(Gpu::hostToDevice, h_velocity.begin(), h_velocity.end(),
                       d_velocity.begin());
        Gpu::copyAsync(Gpu::hostToDevice, hdir.begin(), hdir.end(),
                       d_dir.begin());
        Gpu::streamSynchronize();
    }
    const Real* const AMREX_RESTRICT velocity_f = d_velocity.dataPtr();
    const int* const AMREX_RESTRICT dir_f = d_dir.dataPtr();
    const Real c2 = PhysConst::c * PhysConst::c;

    const int thread_num = 0;

    for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti) {
        auto& attribs = pti.GetAttribs();
        auto& wp = attribs[PIdx::w];
        auto& uxp = attribs[PIdx::ux];
        auto& uyp = attribs[PIdx::uy];
        auto& uzp = attribs[PIdx::uz];

        const long np = pti.numParticles();

        const auto GetPosition = GetParticlePosition<PIdx>(pti);
        auto SetPosition = SetParticlePosition<PIdx>(pti);

        ParticleReal* const AMREX_RESTRICT ux_ptr = uxp.dataPtr();
        ParticleReal* const AMREX_RESTRICT uy_ptr = uyp.dataPtr();
        ParticleReal* const AMREX_RESTRICT uz_ptr = uzp.dataPtr();
        ParticleReal const* const AMREX_RESTRICT weight_ptr = wp.dataPtr();
        const int* const AMREX_RESTRICT face_id_ptr =
            pti.GetStructOfArrays().GetIntData("face_id").dataPtr();

        if (!skip_deposition && fields.has(FieldType::rho_fp, lev)) {
            int* ion_lev = nullptr;
            DepositCharge(pti, wp, ion_lev, fields.get(FieldType::rho_fp, lev),
                          0, 0, np, thread_num, lev, lev);
        }

#if !defined(WARPX_DIM_1D_Z)
        ParticleReal* x_n = nullptr;
        if (push_type == PushType::Implicit) {
            x_n = pti.GetAttribs("x_n").dataPtr();
        }
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
        ParticleReal* y_n = nullptr;
        if (push_type == PushType::Implicit) {
            y_n = pti.GetAttribs("y_n").dataPtr();
        }
#endif
        ParticleReal* z_n = nullptr;
        if (push_type == PushType::Implicit) {
            z_n = pti.GetAttribs("z_n").dataPtr();
        }

        // Opposite-weight members receive opposite velocities. Their current
        // contributions therefore add, while their separated charge supplies
        // the polarization rho dictated by continuity.
        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long ip) noexcept {
            ParticleReal x, y, z;
            if (push_type == PushType::Implicit) {
#if !defined(WARPX_DIM_1D_Z)
                x = x_n[ip];
#else
                x = ParticleReal(0.0);
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_RZ)
                y = y_n[ip];
#else
                y = ParticleReal(0.0);
#endif
                z = z_n[ip];
            } else {
                GetPosition(ip, x, y, z);
            }
            const int fc = face_id_ptr[ip];
            const int fdir = dir_f[fc];
            const Real polarity = weight_ptr[ip] >= 0.0_prt ? 1.0_rt : -1.0_rt;
            const Real v = polarity * velocity_f[fc];
            const Real u = v / std::sqrt(1._rt - v * v / c2);
#if defined(WARPX_DIM_RZ)
            if (fdir == 0) {
                const Real r = std::sqrt(x * x + y * y);
                const Real cos_theta = (r > 0._rt) ? x / r : 1._rt;
                const Real sin_theta = (r > 0._rt) ? y / r : 0._rt;
                ux_ptr[ip] = u * cos_theta;
                uy_ptr[ip] = u * sin_theta;
                uz_ptr[ip] = 0._rt;
            } else if (fdir == 1) {
                const Real r = std::sqrt(x * x + y * y);
                const Real cos_theta = (r > 0._rt) ? x / r : 1._rt;
                const Real sin_theta = (r > 0._rt) ? y / r : 0._rt;
                ux_ptr[ip] = -u * sin_theta;
                uy_ptr[ip] = u * cos_theta;
                uz_ptr[ip] = 0._rt;
            } else {
                ux_ptr[ip] = 0._rt;
                uy_ptr[ip] = 0._rt;
                uz_ptr[ip] = u;
            }
#else
            ux_ptr[ip] = (fdir == 0) ? u : 0._rt;
            uy_ptr[ip] = (fdir == 1) ? u : 0._rt;
            uz_ptr[ip] = (fdir == 2) ? u : 0._rt;
#endif

            // Explicit particles advance by a full step; implicit particles
            // are placed at n+1/2 and completed by
            // FinishImplicitParticleUpdate.
            const Real ginv = 1._rt / std::sqrt(1._rt + (u * u) / c2);
            const Real push_dt =
                push_type == PushType::Implicit ? 0.5_rt * dt : dt;
            x += ux_ptr[ip] * ginv * push_dt;
            y += uy_ptr[ip] * ginv * push_dt;
            z += uz_ptr[ip] * ginv * push_dt;
            SetPosition(ip, x, y, z);
        });

        if (!skip_deposition) {
            const Real relative_time = -0.5_rt * dt;
            int* ion_lev = nullptr;
            amrex::MultiFab* jx =
                fields.get(current_fp_string, Direction{0}, lev);
            amrex::MultiFab* jy =
                fields.get(current_fp_string, Direction{1}, lev);
            amrex::MultiFab* jz =
                fields.get(current_fp_string, Direction{2}, lev);
            DepositCurrent(pti, wp, uxp, uyp, uzp, ion_lev, jx, jy, jz, 0, np,
                           thread_num, lev, lev, dt, relative_time, push_type);
        }

        if (!skip_deposition && fields.has(FieldType::rho_fp, lev)) {
            int* ion_lev = nullptr;
            DepositCharge(pti, wp, ion_lev, fields.get(FieldType::rho_fp, lev),
                          1, 0, np, thread_num, lev, lev);
        }

        amrex::Gpu::synchronize();
    }
}
