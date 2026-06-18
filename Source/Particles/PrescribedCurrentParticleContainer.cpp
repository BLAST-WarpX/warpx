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
    : WarpXParticleContainer(amr_core, ispecies)
{
    charge = 1.0;
    m_mass = std::numeric_limits<Real>::max();

    const ParmParse pp_warpx("warpx");

    // --- Global waveform -----------------------------------------------------
    std::string ci_file;
    const bool has_global_file = pp_warpx.query("current_injection.file", ci_file);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        has_global_file,
        "warpx.current_injection.file is required (per-pair files are a follow-up).");
    load_waveform(ci_file, m_time, m_current);

    // --- Drive faces ---------------------------------------------------------
    int n_pairs = 0;
    pp_warpx.query("current_injection.n_pairs", n_pairs);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        n_pairs >= 1, "warpx.current_injection.n_pairs must be >= 1");

    bool dir_set = false;
    for (int p = 0; p < n_pairs; ++p) {
        const std::string base = "current_injection.pair_" + std::to_string(p);

        // Reject the parts of the public interface this first port does not cover.
        Real dummy = 0.;
        const bool has_return = utils::parser::queryWithParser(
            pp_warpx, (base + ".return.xlo").c_str(), dummy);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !has_return,
            "PrescribedCurrentParticleContainer: explicit return faces are not "
            "supported yet -- the +/- particle pair already injects net current "
            "with zero net charge, so the return face is usually unnecessary.");

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

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            f.sign == 1,
            "PrescribedCurrentParticleContainer: only sign = +1 drive faces are "
            "supported in this first port.");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            f.dir >= 0 && f.dir < 3, "current_injection drive.dir must be 0, 1 or 2");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            f.A > 0., "current_injection drive.A must be > 0");

        if (!dir_set) { m_dir = f.dir; dir_set = true; }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            f.dir == m_dir,
            "PrescribedCurrentParticleContainer: all drive faces must share the "
            "same current direction in this first port.");

        m_faces.push_back(f);
    }

    m_enabled = !m_faces.empty();
}

Real
PrescribedCurrentParticleContainer::interpolate_current (Real t) const
{
    if (m_time.size() < 2) { return 0._rt; }
    if (t <= m_time.front() || t >= m_time.back()) { return 0._rt; }
    const auto it  = std::lower_bound(m_time.begin(), m_time.end(), t);
    const auto idx = static_cast<std::size_t>(std::distance(m_time.begin(), it));
    const Real t0 = m_time[idx-1], t1 = m_time[idx];
    const Real I0 = m_current[idx-1], I1 = m_current[idx];
    return I0 + (I1 - I0) * (t - t0) / (t1 - t0);
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

    // Peak |I| -> set the velocity scale so the fastest particle stays well
    // below c: speed = m_vel_coeff * I(t), with peak speed = 0.05 c.
    Real I_peak = 0._rt;
    for (const Real I : m_current) { I_peak = std::max(I_peak, std::abs(I)); }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        I_peak > 0._rt, "warpx.current_injection: waveform is identically zero.");
    const Real u_max = 0.05_rt * PhysConst::c;
    m_vel_coeff = u_max / I_peak;

    // Build the +/- particle pairs (only on rank 0; Redistribute scatters them).
    Vector<ParticleReal> xs, ys, zs, ws;
    if (ParallelDescriptor::MyProc() == 0)
    {
        const auto lo = domain.smallEnd();
        const auto hi = domain.bigEnd();
        for (const Face& f : m_faces)
        {
            // weight magnitude so that the pair deposits J = I(t)/A:
            //   J = 2 * W * (C * I) / dV  ==>  W = dV / (2 * C * A)
            const Real W = dV / (2._rt * m_vel_coeff * f.A);
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
                    // + partner
                    xs.push_back(xc); ys.push_back(yc); zs.push_back(zc);
                    ws.push_back( W);
                    // - partner (co-located)
                    xs.push_back(xc); ys.push_back(yc); zs.push_back(zc);
                    ws.push_back(-W);
                }
            }}}
        }
    }

    const auto np = static_cast<long>(xs.size());
    const Vector<ParticleReal> ux(np, 0.0), uy(np, 0.0), uz(np, 0.0);
    Vector<Vector<ParticleReal>> attr;
    attr.push_back(ws);
    const Vector<Vector<int>> attr_int;
    AddNParticles(lev, np, xs, ys, zs, ux, uy, uz,
                  1, attr, 0, attr_int, 1);

    amrex::Print() << "[CurrentInjection] PrescribedCurrentParticleContainer: "
                   << TotalNumberOfParticles() << " antenna particles ("
                   << TotalNumberOfParticles()/2 << " +/- pairs), I_peak=" << I_peak
                   << " A, peak speed=" << 0.05_rt*PhysConst::c << " m/s.\n";

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

    // Speed magnitude carried by every particle this step (sign set per partner).
    const Real speed = m_vel_coeff * interpolate_current(t);
    const int  dir   = m_dir;
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

        ParticleReal* const AMREX_RESTRICT w_ptr  = wp.dataPtr();
        ParticleReal* const AMREX_RESTRICT ux_ptr = uxp.dataPtr();
        ParticleReal* const AMREX_RESTRICT uy_ptr = uyp.dataPtr();
        ParticleReal* const AMREX_RESTRICT uz_ptr = uzp.dataPtr();

        // Set momenta from the waveform and push positions by v*dt so that the
        // (possibly charge-conserving) deposition sees the right displacement.
        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip) noexcept
        {
            const Real s = (w_ptr[ip] > 0) ? 1._rt : -1._rt;
            const Real u = s * speed;
            ux_ptr[ip] = (dir == 0) ? u : 0._rt;
            uy_ptr[ip] = (dir == 1) ? u : 0._rt;
            uz_ptr[ip] = (dir == 2) ? u : 0._rt;

            const Real v = u / std::sqrt(1._rt + (u*u)/c2);
            ParticleReal x, y, z;
            GetPosition(ip, x, y, z);
            if      (dir == 0) { x += v*dt; }
            else if (dir == 1) { y += v*dt; }
            else               { z += v*dt; }
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
        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (long ip) noexcept
        {
            const Real s = (w_ptr[ip] > 0) ? 1._rt : -1._rt;
            const Real u = s * speed;
            const Real v = u / std::sqrt(1._rt + (u*u)/c2);
            ParticleReal x, y, z;
            GetPosition(ip, x, y, z);
            if      (dir == 0) { x -= v*dt; }
            else if (dir == 1) { y -= v*dt; }
            else               { z -= v*dt; }
            SetPosition(ip, x, y, z);
        });

        amrex::Gpu::synchronize();
    }
}
