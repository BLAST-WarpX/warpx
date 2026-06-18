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
#include <fstream>
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
        std::ifstream wf_stream(path);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            wf_stream.is_open(),
            "warpx.current_injection: cannot open waveform file '" + path + "'");
        std::string line;
        while (std::getline(wf_stream, line)) {
            const auto first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos || line[first] == '#') { continue; }
            std::istringstream iss(line);
            Real t_val, I_val;
            if (iss >> t_val >> I_val) {
                tvec.push_back(t_val);
                ivec.push_back(I_val);
            }
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

    const ParmParse pp_warpx("warpx");

    // --- Optional global waveform (used by any pair without its own file) ----
    std::vector<Real> global_t, global_I;
    std::string global_file;
    const bool has_global_file = pp_warpx.query("current_injection.file", global_file);
    if (has_global_file) { load_waveform(global_file, global_t, global_I); }

    // --- Drive faces (one per pair) ------------------------------------------
    // A "return" is just a drive face with sign = -1, so no dedicated return
    // block is needed: use another pair_N.drive.* with sign = -1 if ever wanted.
    int n_pairs = 0;
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
        utils::parser::getWithParser(pp_warpx, (base + ".drive.A").c_str(),   f.A);
        pp_warpx.query((base + ".drive.dir").c_str(),  f.dir);
        pp_warpx.query((base + ".drive.sign").c_str(), f.sign);

        // Per-pair waveform file overrides the global one.
        std::string pair_file;
        if (pp_warpx.query((base + ".file").c_str(), pair_file)) {
            load_waveform(pair_file, f.t, f.I);
        } else {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                has_global_file,
                "warpx.current_injection: " + base + " needs " + base +
                ".file (or set the global warpx.current_injection.file).");
            f.t = global_t; f.I = global_I;
        }

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            f.dir >= 0 && f.dir < 3, "current_injection drive.dir must be 0, 1 or 2");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            f.A > 0., "current_injection drive.A must be > 0");
        m_faces.push_back(std::move(f));
    }

    m_enabled = !m_faces.empty();
}

Real
PrescribedCurrentParticleContainer::interpolate (
    const std::vector<Real>& tv, const std::vector<Real>& Iv, Real tt)
{
    if (tv.size() < 2) { return 0._rt; }
    if (tt <= tv.front() || tt >= tv.back()) { return 0._rt; }
    const auto it  = std::lower_bound(tv.begin(), tv.end(), tt);
    const auto idx = static_cast<std::size_t>(std::distance(tv.begin(), it));
    const Real t0 = tv[idx-1], t1 = tv[idx];
    const Real I0 = Iv[idx-1], I1 = Iv[idx];
    return I0 + (I1 - I0) * (tt - t0) / (t1 - t0);
}

void
PrescribedCurrentParticleContainer::InitData ()
{
    if (!m_enabled) { return; }

    const int lev = 0;
    const auto& geom    = Geom(lev);
    const auto problo   = geom.ProbLoArray();
    const auto dx       = geom.CellSizeArray();
    const Box& domain   = geom.Domain();

#if defined(WARPX_DIM_3D)
    const Real dV = dx[0]*dx[1]*dx[2];
#else
    WARPX_ABORT_WITH_MESSAGE(
        "PrescribedCurrentParticleContainer is only implemented in 3D for now.");
    const Real dV = 1._rt;
#endif

    // Peak |I| over ALL faces' waveforms -> velocity scale so the fastest
    // particle stays well below c: speed = m_vel_coeff * I(t), peak = 0.05 c.
    Real I_peak = 0._rt;
    for (const Face& f : m_faces) {
        for (const Real I : f.I) { I_peak = std::max(I_peak, std::abs(I)); }
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        I_peak > 0._rt, "warpx.current_injection: all waveforms are identically zero.");
    const Real u_max = 0.05_rt * PhysConst::c;
    m_vel_coeff = u_max / I_peak;

    // One signed-weight particle per filled cell (only on rank 0; Redistribute
    // scatters them). Charge=1, and with DIRECT current deposition the antenna
    // deposits only J (no charge), so a single particle reproduces the
    // current_fp box injection: J = charge * w * v / dV = sign * I_f(t)/A with
    //   v_f = m_vel_coeff * I_f(t)               (per-face; Evolve box-tests pos)
    //   w_f = sign_f * dV / (m_vel_coeff * A_f)  (carries the per-face sign/A).
    // No per-particle waveform tag is needed: the particle never leaves its
    // drive box (positions are reset each step), so Evolve recovers the face
    // by box-testing the position.
    Vector<ParticleReal> xs, ys, zs, ws;
    if (ParallelDescriptor::MyProc() == 0)
    {
        const auto lo = domain.smallEnd();
        const auto hi = domain.bigEnd();
        for (const Face& f : m_faces)
        {
            const Real W = f.sign * dV / (m_vel_coeff * f.A);
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
                }
            }}}
        }
    }

    const auto np = static_cast<long>(xs.size());
    const Vector<ParticleReal> ux(np, 0.0), uy(np, 0.0), uz(np, 0.0);
    Vector<Vector<ParticleReal>> attr;     attr.push_back(ws);
    const Vector<Vector<int>>    attr_int;
    AddNParticles(lev, np, xs, ys, zs, ux, uy, uz,
                  1, attr, 0, attr_int, 1);

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

    // Per-face speed this step (speed_f = m_vel_coeff * I_f(t)) and the face
    // bounding boxes, copied to device. Each particle picks its face by
    // box-testing its (fixed) position -- no per-particle tag needed.
    const int nfaces = static_cast<int>(m_faces.size());
    Gpu::DeviceVector<Real> d_speed(nfaces), d_lo(3*nfaces), d_hi(3*nfaces);
    Gpu::DeviceVector<int>  d_dir(nfaces);
    {
        Vector<Real> h_speed(nfaces), hlo(3*nfaces), hhi(3*nfaces);
        Vector<int>  hdir(nfaces);
        for (int f = 0; f < nfaces; ++f) {
            h_speed[f] = m_vel_coeff * interpolate(m_faces[f].t, m_faces[f].I, t);
            hdir[f]    = m_faces[f].dir;
            for (int d = 0; d < 3; ++d) { hlo[3*f+d] = m_faces[f].lo[d]; hhi[3*f+d] = m_faces[f].hi[d]; }
        }
        Gpu::copyAsync(Gpu::hostToDevice, h_speed.begin(), h_speed.end(), d_speed.begin());
        Gpu::copyAsync(Gpu::hostToDevice, hdir.begin(), hdir.end(), d_dir.begin());
        Gpu::copyAsync(Gpu::hostToDevice, hlo.begin(), hlo.end(), d_lo.begin());
        Gpu::copyAsync(Gpu::hostToDevice, hhi.begin(), hhi.end(), d_hi.begin());
        Gpu::streamSynchronize();
    }
    const Real* const AMREX_RESTRICT speed_f = d_speed.dataPtr();
    const int*  const AMREX_RESTRICT dir_f   = d_dir.dataPtr();
    const Real* const AMREX_RESTRICT lo_p    = d_lo.dataPtr();
    const Real* const AMREX_RESTRICT hi_p    = d_hi.dataPtr();
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

        // Set momenta from this cell's face (found by box-testing the position):
        // its waveform sets the speed and its dir sets the component. Push
        // positions by v*dt so the (possibly charge-conserving) deposition sees
        // the right displacement; the per-face sign of J is in the signed weight.
        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip) noexcept
        {
            ParticleReal x, y, z;
            GetPosition(ip, x, y, z);
            int fc = 0;
            for (int f = 0; f < nfaces; ++f) {
                if (x >= lo_p[3*f] && x < hi_p[3*f] &&
                    y >= lo_p[3*f+1] && y < hi_p[3*f+1] &&
                    z >= lo_p[3*f+2] && z < hi_p[3*f+2]) { fc = f; break; }
            }
            const int  fdir = dir_f[fc];
            const Real u    = speed_f[fc];
            ux_ptr[ip] = (fdir == 0) ? u : 0._rt;
            uy_ptr[ip] = (fdir == 1) ? u : 0._rt;
            uz_ptr[ip] = (fdir == 2) ? u : 0._rt;

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
