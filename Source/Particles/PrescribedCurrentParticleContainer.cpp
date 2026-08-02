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
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace amrex;

namespace
{
    //! Load a two-column ASCII waveform file ("t [s]   I [A]") into vectors.
    void load_waveform (const std::string& path,
                        std::vector<Real>& tvec,
                        std::vector<Real>& ivec)
    {
        Vector<char> file_chars;
        ParallelDescriptor::ReadAndBcastFile(path, file_chars);
        const std::string file_contents(file_chars.dataPtr());
        std::istringstream wf_stream(file_contents);
        std::string line;
        int line_number = 0;
        while (std::getline(wf_stream, line)) {
            ++line_number;
            const auto first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos || line[first] == '#') { continue; }
            std::istringstream iss(line);
            Real t_val, I_val;
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                static_cast<bool>(iss >> t_val >> I_val),
                "warpx.current_injection: malformed waveform row " +
                    std::to_string(line_number) + " in '" + path + "'");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::isfinite(t_val) && std::isfinite(I_val),
                "warpx.current_injection: non-finite value on waveform row " +
                    std::to_string(line_number) + " in '" + path + "'");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                tvec.empty() || t_val > tvec.back(),
                "warpx.current_injection: waveform times must be strictly increasing in '" +
                    path + "'");
            tvec.push_back(t_val);
            ivec.push_back(I_val);
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            tvec.size() >= 2,
            "warpx.current_injection: waveform file '" + path +
            "' must contain at least 2 data rows");
    }
}

bool
PrescribedCurrentParticleContainer::is_enabled ()
{
    bool enabled = false;
    const ParmParse pp_warpx("warpx");
    pp_warpx.query("current_injection", enabled);
    return enabled;
}

PrescribedCurrentParticleContainer::PrescribedCurrentParticleContainer (
    AmrCore* amr_core, int ispecies)
    : WarpXParticleContainer(amr_core, ispecies, "prescribed_current")
{
    m_charge = 1.0;
    m_mass = std::numeric_limits<Real>::max();
    AddIntComp("face_id");

    const ParmParse pp_warpx("warpx");

    // --- Optional global waveform (used by any pair without its own file) ----
    std::vector<Real> global_t, global_I;
    std::string global_file;
    const bool has_global_file = pp_warpx.query("current_injection.file", global_file);
    if (has_global_file) { load_waveform(global_file, global_t, global_I); }

    // --- Drive faces (one per pair) ------------------------------------------
    // A "return" is just a drive face with sign = -1, so no dedicated return
    // block is needed: use another pair_N.drive.* with sign = -1 if ever wanted.
    int n_pairs = 1;
    pp_warpx.query("current_injection.n_pairs", n_pairs);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        n_pairs >= 1, "warpx.current_injection.n_pairs must be >= 1");

    for (int p = 0; p < n_pairs; ++p) {
        const std::string base = "current_injection.pair_" + std::to_string(p);

        Face f;
        utils::parser::getWithParser(pp_warpx, (base + ".drive.xlo").c_str(), f.lo[0]);
        utils::parser::getWithParser(pp_warpx, (base + ".drive.xhi").c_str(), f.hi[0]);
        utils::parser::getWithParser(pp_warpx, (base + ".drive.ylo").c_str(), f.lo[1]);
        utils::parser::getWithParser(pp_warpx, (base + ".drive.yhi").c_str(), f.hi[1]);
        utils::parser::getWithParser(pp_warpx, (base + ".drive.zlo").c_str(), f.lo[2]);
        utils::parser::getWithParser(pp_warpx, (base + ".drive.zhi").c_str(), f.hi[2]);
        pp_warpx.query(base + ".drive.dir",  f.dir);
        pp_warpx.query(base + ".drive.sign", f.sign);

        for (int d = 0; d < 3; ++d) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::isfinite(f.lo[d]) && std::isfinite(f.hi[d]) && f.lo[d] < f.hi[d],
                "current_injection drive bounds must be finite and satisfy lo < hi");
        }

        // Area A is the physical cross-section perpendicular to the current.
        // In RZ with dir = 0 (Jr) the imposed profile is the conserved-total-I
        // 1/r coaxial profile Jr = I/(2*pi*r*dz), which has no single A, so A is
        // unused and may be omitted. In every other case A is required.
#if defined(WARPX_DIM_RZ)
        const bool a_required = (f.dir != 0);
#else
        const bool a_required = true;
#endif
        if (a_required) {
            utils::parser::getWithParser(pp_warpx, (base + ".drive.A").c_str(), f.A);
        } else {
            pp_warpx.query(base + ".drive.A", f.A);
        }

        // Per-pair waveform file overrides the global one.
        std::string pair_file;
        if (pp_warpx.query(base + ".file", pair_file)) {
            load_waveform(pair_file, f.t, f.I);
        } else {
            std::string msg = "warpx.current_injection: ";
            msg += base;
            msg += " needs ";
            msg += base;
            msg += ".file (or set the global warpx.current_injection.file).";
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(has_global_file, msg);
            f.t = global_t; f.I = global_I;
        }

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            f.dir >= 0 && f.dir < 3, "current_injection drive.dir must be 0, 1 or 2");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            f.sign == -1 || f.sign == 1, "current_injection drive.sign must be -1 or 1");
#if defined(WARPX_DIM_RZ)
        if (f.dir == 0) {
            // Jr coaxial drive: box must lie off-axis (r > 0); Jr ~ 1/r diverges
            // at r = 0 and the inverse-volume scaling forces Jr = 0 on axis.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                f.lo[0] > 0., "current_injection RZ drive (dir=0, Jr) needs xlo > 0 (off-axis)");
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

Real
PrescribedCurrentParticleContainer::interpolate (
    const std::vector<Real>& tv, const std::vector<Real>& Iv, Real tt)
{
    if (tv.size() < 2) { return 0._rt; }
    if (tt < tv.front() || tt > tv.back()) { return 0._rt; }
    const auto it  = std::lower_bound(tv.begin(), tv.end(), tt);
    if (it == tv.begin()) { return Iv.front(); }
    const auto idx = static_cast<std::size_t>(std::distance(tv.begin(), it));
    const Real t0 = tv[idx-1], t1 = tv[idx];
    const Real I0 = Iv[idx-1], I1 = Iv[idx];
    return I0 + (I1 - I0) * (tt - t0) / (t1 - t0);
}

void
PrescribedCurrentParticleContainer::compute_velocity_scale ()
{
    Real I_peak = 0._rt;
    for (const Face& f : m_faces) {
        for (const Real I : f.I) { I_peak = std::max(I_peak, std::abs(I)); }
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        I_peak > 0._rt, "warpx.current_injection: all waveforms are identically zero.");
    m_vel_coeff = 0.05_rt * PhysConst::c / I_peak;
}

void
PrescribedCurrentParticleContainer::InitData ()
{
    if (!m_enabled) { return; }

#if defined(WARPX_DIM_1D_Z)
    WARPX_ABORT_WITH_MESSAGE(
        "Prescribed current injection is not implemented in 1D_Z geometry.");
#else
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        maxLevel() == 0,
        "Prescribed current injection currently supports only a single mesh level.");

    const int lev = 0;
    const auto& geom    = Geom(lev);
    const auto problo   = geom.ProbLoArray();
    const auto dx       = geom.CellSizeArray();
    const Box& domain   = geom.Domain();

#if defined(WARPX_DIM_3D)
    const Real dV = dx[0]*dx[1]*dx[2];
#elif defined(WARPX_DIM_XZ)
    // 2D XZ: per-unit-length in the invariant y-direction. AMReX index 0 -> x,
    // index 1 -> z. A drive face area f.A is likewise per-unit-length in y.
    const Real dV = dx[0]*dx[1];
#elif defined(WARPX_DIM_RZ)
    // RZ: AMReX index 0 -> r, index 1 -> z; theta is invariant. The physical
    // cell volume is 2*pi*r*dr*dz, but the RZ current-deposition path deposits a
    // *raw* J (without the 2*pi*r factor) and ApplyInverseVolumeScalingToCurrent
    // Density divides by 2*pi*r afterward, so the physical current density is
    //   J = w * q * v / (2*pi*r*dr*dz).
    // See the per-face weight derivation in the seeding loop below.
    const Real dr = dx[0];
    const Real dz = dx[1];
    const Real two_pi = MathConst::pi * 2._rt;
#else
    WARPX_ABORT_WITH_MESSAGE(
        "PrescribedCurrentParticleContainer is only implemented in 3D, 2D (XZ) and RZ.");
#endif

    // Peak |I| over all faces sets a target velocity scale with max |v|=0.05c.
    compute_velocity_scale();

    // One signed-weight particle per filled cell (only on rank 0; Redistribute
    // scatters them). Charge=1, and with DIRECT current deposition the antenna
    // deposits only J (no charge), so a single particle reproduces the
    // current_fp box injection: J = charge * w * v / dV = sign * I_f(t)/A with
    //   v_f = m_vel_coeff * I_f(t)               (per-face)
    //   w_f = sign_f * dV / (m_vel_coeff * A_f)  (carries the per-face sign/A).
    // A persistent face_id attribute selects the waveform and direction even
    // when multiple drive boxes overlap.
    Vector<ParticleReal> xs, ys, zs, ws;
    Vector<int> face_ids;
    if (ParallelDescriptor::MyProc() == 0)
    {
        const auto lo = domain.smallEnd();
        const auto hi = domain.bigEnd();
        for (int face_id = 0; face_id < static_cast<int>(m_faces.size()); ++face_id)
        {
            const Face& f = m_faces[face_id];
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ)
            const Real W = f.sign * dV / (m_vel_coeff * f.A);
#endif
#if defined(WARPX_DIM_3D)
            for (int k = lo[2]; k <= hi[2]; ++k) {
            for (int j = lo[1]; j <= hi[1]; ++j) {
            for (int i = lo[0]; i <= hi[0]; ++i) {
                const Real xc = problo[0] + (i + 0.5_rt) * dx[0];
                const Real yc = problo[1] + (j + 0.5_rt) * dx[1];
                const Real zc = problo[2] + (k + 0.5_rt) * dx[2];
                if (xc >= f.lo[0] && xc < f.hi[0] &&
                    yc >= f.lo[1] && yc < f.hi[1] &&
                    zc >= f.lo[2] && zc < f.hi[2])
                {
                    xs.push_back(xc); ys.push_back(yc); zs.push_back(zc);
                    ws.push_back(W);
                    face_ids.push_back(face_id);
                }
            }}}
#elif defined(WARPX_DIM_XZ)
            // AMReX index 0 -> x, index 1 -> z; y is invariant (face y-bounds
            // f.lo[1]/f.hi[1] are ignored). Seed at y = 0 (AddNParticles drops y).
            for (int j = lo[1]; j <= hi[1]; ++j) {
            for (int i = lo[0]; i <= hi[0]; ++i) {
                const Real xc = problo[0] + (i + 0.5_rt) * dx[0];
                const Real zc = problo[1] + (j + 0.5_rt) * dx[1];
                if (xc >= f.lo[0] && xc < f.hi[0] &&
                    zc >= f.lo[2] && zc < f.hi[2])
                {
                    xs.push_back(xc); ys.push_back(0._rt); zs.push_back(zc);
                    ws.push_back(W);
                    face_ids.push_back(face_id);
                }
            }}
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
            //   dir = 0 (Jr, conserved total I): uniform weight
            //       W = sign * dr / (Nz * vel_coeff)
            //     -> Jr = I(t) / (2*pi*r*Nz*dz) (the 1/r coaxial profile; total
            //        radial current I is distributed across the box in z)
            //   dir = 1,2 (Jtheta/Jz, uniform J): weight ~ r
            //       W = sign * (2*pi*r*dr*dz) / (vel_coeff * A)
            //     -> J = sign * I(t) / A        (uniform over the r-z box)
            int n_axial_cells = 0;
            for (int j = lo[1]; j <= hi[1]; ++j) {
                const Real zc = problo[1] + (j + 0.5_rt) * dx[1];
                if (zc >= f.lo[2] && zc < f.hi[2]) { ++n_axial_cells; }
            }
            for (int j = lo[1]; j <= hi[1]; ++j) {
            for (int i = lo[0]; i <= hi[0]; ++i) {
                const Real rc = problo[0] + (i + 0.5_rt) * dx[0];
                const Real zc = problo[1] + (j + 0.5_rt) * dx[1];
                if (rc >= f.lo[0] && rc < f.hi[0] &&
                    zc >= f.lo[2] && zc < f.hi[2])
                {
                    const int n_spokes = std::max(1, WarpX::n_rz_azimuthal_modes);
                    const Real W = (f.dir == 0)
                         ? f.sign * dr / (n_spokes * n_axial_cells * m_vel_coeff)
                         : f.sign * (two_pi * rc * dr * dz) / (n_spokes * m_vel_coeff * f.A);

                    for (int spoke = 0; spoke < n_spokes; spoke++) {
                        const Real phase = two_pi * spoke / n_spokes;
                        xs.push_back(rc * std::cos(phase));
                        ys.push_back(rc * std::sin(phase));
                        zs.push_back(zc);
                        ws.push_back(W);
                        face_ids.push_back(face_id);
                    }
                }
            }}
#endif
        }
    }

    const auto np = static_cast<long>(xs.size());
    const Vector<ParticleReal> ux(np, 0.0), uy(np, 0.0), uz(np, 0.0);
    Vector<Vector<ParticleReal>> attr;     attr.push_back(ws);
    const Vector<Vector<int>>    attr_int{face_ids};
    AddNParticles(lev, np, xs, ys, zs, ux, uy, uz,
                  1, attr, 1, attr_int, 1);

    if (Verbose()) {
        amrex::Print() << Utils::TextMsg::Info(
            "PrescribedCurrentParticleContainer: "
            + std::to_string(TotalNumberOfParticles()) + " antenna particles over "
            + std::to_string(m_faces.size()) + " drive face(s)");
    }

    if (TotalNumberOfParticles() == 0) {
        ablastr::warn_manager::WMRecordWarning("CurrentInjection",
            "warpx.current_injection: no cells found inside any drive box.",
            ablastr::warn_manager::WarnPriority::high);
        m_enabled = false;
    }
#endif
}

void
PrescribedCurrentParticleContainer::PostRestart ()
{
    // This deterministic antenna is not checkpointed with physical species.
    // Recreate it from the input boxes and waveform after loading a checkpoint.
    InitData();
}

void
PrescribedCurrentParticleContainer::Evolve (
    ablastr::fields::MultiFabRegister& fields,
    int lev,
    const std::string& current_fp_string,
    Real t, Real dt, SubcyclingHalf /*subcycling_half*/, bool skip_deposition,
    PositionPushType /*position_push_type*/,
    MomentumPushType /*momentum_push_type*/,
    ImplicitOptions const * implicit_options)
{
    using ablastr::fields::Direction;

    ABLASTR_PROFILE("PrescribedCurrentParticleContainer::Evolve()");

    if (!m_enabled) { return; }

    const PushType push_type = (implicit_options == nullptr) ? PushType::Explicit : PushType::Implicit;

    // Per-face target velocity this step, copied to device. Each particle keeps
    // the ID of the face that created it, so overlapping source boxes are valid.
    const int nfaces = static_cast<int>(m_faces.size());
    Gpu::DeviceVector<Real> d_velocity(nfaces);
    Gpu::DeviceVector<int>  d_dir(nfaces);
    {
        Vector<Real> h_velocity(nfaces);
        Vector<int>  hdir(nfaces);
        for (int f = 0; f < nfaces; ++f) {
            h_velocity[f] = m_vel_coeff * interpolate(m_faces[f].t, m_faces[f].I, t);
            hdir[f] = m_faces[f].dir;
        }
        Gpu::copyAsync(
            Gpu::hostToDevice, h_velocity.begin(), h_velocity.end(), d_velocity.begin());
        Gpu::copyAsync(Gpu::hostToDevice, hdir.begin(), hdir.end(), d_dir.begin());
        Gpu::streamSynchronize();
    }
    const Real* const AMREX_RESTRICT velocity_f = d_velocity.dataPtr();
    const int* const AMREX_RESTRICT dir_f = d_dir.dataPtr();
    const Real c2    = PhysConst::c * PhysConst::c;

    const int thread_num = 0;

    for (WarpXParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        auto& attribs = pti.GetAttribs();
        auto& wp  = attribs[PIdx::w ];
        auto& uxp = attribs[PIdx::ux];
        auto& uyp = attribs[PIdx::uy];
        auto& uzp = attribs[PIdx::uz];

        const long np = pti.numParticles();

        const auto GetPosition = GetParticlePosition<PIdx>(pti);
        auto       SetPosition = SetParticlePosition<PIdx>(pti);

        ParticleReal* const AMREX_RESTRICT ux_ptr = uxp.dataPtr();
        ParticleReal* const AMREX_RESTRICT uy_ptr = uyp.dataPtr();
        ParticleReal* const AMREX_RESTRICT uz_ptr = uzp.dataPtr();
        const int* const AMREX_RESTRICT face_id_ptr =
            pti.GetStructOfArrays().GetIntData("face_id").dataPtr();

        // Set momenta from this particle's source face: its waveform sets the
        // velocity and its dir sets the component. Push
        // positions by v*dt so the (possibly charge-conserving) deposition sees
        // the right displacement; the per-face sign of J is in the signed weight.
        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip) noexcept
        {
            ParticleReal x, y, z;
            GetPosition(ip, x, y, z);
            const int fc = face_id_ptr[ip];
            const int  fdir = dir_f[fc];
            const Real v = velocity_f[fc];
            const Real u = v / std::sqrt(1._rt - v*v/c2);
#if defined(WARPX_DIM_RZ)
            if (fdir == 0) {
                const Real r = std::sqrt(x*x + y*y);
                const Real cos_theta = (r > 0._rt) ? x/r : 1._rt;
                const Real sin_theta = (r > 0._rt) ? y/r : 0._rt;
                ux_ptr[ip] = u * cos_theta;
                uy_ptr[ip] = u * sin_theta;
                uz_ptr[ip] = 0._rt;
            } else if (fdir == 1) {
                const Real r = std::sqrt(x*x + y*y);
                const Real cos_theta = (r > 0._rt) ? x/r : 1._rt;
                const Real sin_theta = (r > 0._rt) ? y/r : 0._rt;
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

            // Push along whichever component is non-zero (dir-agnostic).
            const Real ginv = 1._rt / std::sqrt(1._rt + (u*u)/c2);
            x += ux_ptr[ip]*ginv*dt;
            y += uy_ptr[ip]*ginv*dt;
            z += uz_ptr[ip]*ginv*dt;
            SetPosition(ip, x, y, z);
        });

        if (!skip_deposition)
        {
            const Real relative_time = -0.5_rt * dt;
            int* ion_lev = nullptr;
            amrex::MultiFab* jx = fields.get(current_fp_string, Direction{0}, lev);
            amrex::MultiFab* jy = fields.get(current_fp_string, Direction{1}, lev);
            amrex::MultiFab* jz = fields.get(current_fp_string, Direction{2}, lev);
            DepositCurrent(pti, wp, uxp, uyp, uzp, ion_lev, jx, jy, jz,
                           0, np, thread_num, lev, lev, dt, relative_time, push_type);
        }

        // Reset positions: the antenna must not drift for a sustained waveform.
        // Reuse the momentum just set (exact undo, no re-box-test).
        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip) noexcept
        {
            const Real ux = ux_ptr[ip], uy = uy_ptr[ip], uz = uz_ptr[ip];
            const Real ginv = 1._rt / std::sqrt(1._rt + (ux*ux+uy*uy+uz*uz)/c2);
            ParticleReal x, y, z;
            GetPosition(ip, x, y, z);
            x -= ux*ginv*dt;
            y -= uy*ginv*dt;
            z -= uz*ginv*dt;
            SetPosition(ip, x, y, z);
        });

        amrex::Gpu::synchronize();
    }
}
