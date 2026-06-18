/* Copyright 2026 Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ParticleThermalizer.H"

#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Particles/ParticleCreation/AddPlasmaUtilities.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cctype>
#include <string>

using namespace amrex::literals;

ParticleThermalizer::ParticleThermalizer():
    m_start(0._rt), m_end(-1._rt),
    m_momentum_threshold{-1._rt, -1._rt, -1._rt}, m_theta(-1._rt)
{
  const amrex::ParmParse pp("particle_thermalizer");

  // Read normal as a string (x, y, or z)
  std::string normal_str;
  const bool thermalizer_present = pp.query("normal", normal_str);
  if (!thermalizer_present) {
    // If no normal is specified, the thermalizer is not defined
    return;
  }

  // normalize to lowercase
  std::transform(normal_str.begin(), normal_str.end(), normal_str.begin(), [](unsigned char c){ return std::tolower(c); });
#if defined(WARPX_DIM_1D_Z)
  if (normal_str == "z") {
    m_normal = 0;
  } else {
    amrex::Abort("particle_thermalizer: normal must be 'z' in 1D simulations");
  }
#elif defined(WARPX_DIM_XZ)
  if (normal_str == "x") {
    m_normal = 0;
  } else if (normal_str == "z") {
    m_normal = 1;
  } else {
    amrex::Abort("particle_thermalizer: normal must be 'x' or 'z' in 2D simulations");
  }
#elif defined(WARPX_DIM_RZ)
  amrex::Abort("particle_thermalizer: thermalizer not supported in RZ geometry");
#elif defined(WARPX_DIM_RCYLINDER)
  amrex::Abort("particle_thermalizer: thermalizer not supported in RCYLINDER geometry");
#elif defined(WARPX_DIM_RSPHERE)
  amrex::Abort("particle_thermalizer: thermalizer not supported in RSPHERE geometry");
#elif defined(WARPX_DIM_3D)
  if (normal_str == "x") {
    m_normal = 0;
  } else if (normal_str == "y") {
    m_normal = 1;
  } else if (normal_str == "z") {
    m_normal = 2;
  } else {
    amrex::Abort("particle_thermalizer: normal must be 'x', 'y', or 'z'");
  }
#endif

  // Read numeric parameters with defaults
  pp.get("start", m_start);
  pp.get("end", m_end);
  WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
      m_end > m_start,
      "particle_thermalizer: 'end' must be greater than 'start'");

  auto tvec = std::vector<amrex::Real>{};
  utils::parser::getArrWithParser(pp,"momentum_threshold", tvec );
  if (tvec.size() == 1){
      m_momentum_threshold = amrex::Array<amrex::Real, 3>{tvec[0], tvec[0], tvec[0]};
  }
  else if (tvec.size() == 3){
      m_momentum_threshold = amrex::Array<amrex::Real, 3>{tvec[0], tvec[1], tvec[2]};
  }
  else{
    WARPX_ABORT_WITH_MESSAGE("particle_thermalizer: one or three values must be specified for 'momentum_threshold'");
  }
  WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
      std::all_of(
          m_momentum_threshold.begin(),
          m_momentum_threshold.end(),
          [](const auto& el){return (el >= 0._rt);}),
        "particle_thermalizer: 'momentum_threshold' must be non-negative");

  pp.get("theta", m_theta);
  WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
      m_theta >= 0._rt,
      "particle_thermalizer: 'theta' must be non-negative");

  pp.queryarr("species", m_species_names);

  m_defined = true;
}

bool ParticleThermalizer::defined() const {
  return m_defined;
}

void ParticleThermalizer::applyThermalizer(MultiParticleContainer &mpc) const
{
  if (m_species_names.empty()) {
    // No species filter: apply to all species.
    for (auto &pc_uptr : mpc) {
      if (!pc_uptr) { continue; }
      applyThermalizer(*pc_uptr);
    }
  } else {
    // Apply only to the named species.
    for (const auto &name : m_species_names) {
      applyThermalizer(mpc.GetParticleContainerFromName(name));
    }
  }
}

void ParticleThermalizer::applyThermalizer(WarpXParticleContainer &pc) const
{
    for (int lev = 0; lev < pc.numLevels(); ++lev) {
        const auto& geom = pc.Geom(lev);
        const auto& dx = geom.CellSizeArray();
        const auto& problo = geom.ProbLoArray();
        const auto dir = static_cast<int>(m_normal);

        amrex::RealBox thermalizer_region = geom.ProbDomain();
        thermalizer_region.setLo(dir, m_start);
        thermalizer_region.setHi(dir, m_end);
        for (WarpXParIter pti(pc, lev); pti.isValid(); ++pti) {
            const long np = pti.numParticles();

            // early exit for tiles that do not overlap the thermalizer region
            const amrex::Box& tile_box = pti.tilebox();
            const amrex::RealBox tile_realbox = WarpX::getRealBox(tile_box, lev);

            amrex::RealBox overlap_realbox;
            amrex::Box overlap_box;
            amrex::IntVect shifted;
            const bool no_overlap = find_overlap(tile_realbox, thermalizer_region, dx, problo, overlap_realbox, overlap_box, shifted);
            if (no_overlap) {
                continue; // Go to the next tile
            }

            const auto getPosition = GetParticlePosition(pti);

            // Acquire pointers to particle attribute arrays as needed.
            amrex::ParticleReal* ux = pti.GetAttribs(PIdx::ux).data();
            amrex::ParticleReal* uy = pti.GetAttribs(PIdx::uy).data();
            amrex::ParticleReal* uz = pti.GetAttribs(PIdx::uz).data();

            const amrex::Real loend = thermalizer_region.lo(dir);
            const amrex::Real hiend = thermalizer_region.hi(dir);

            const amrex::Real ux_threshold = m_momentum_threshold[0];
            const amrex::Real uy_threshold = m_momentum_threshold[1];
            const amrex::Real uz_threshold = m_momentum_threshold[2];
            const amrex::Real theta = m_theta;

            // Parallel loop over particles in the tile.
            amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (long ip, amrex::RandomEngine const& engine) noexcept {
                amrex::ParticleReal x, y, z;
                amrex::ParticleReal norm_pos = 0.0_prt; //NOLINT (misc-const-correctness)

                getPosition(ip, x, y, z);
#if defined(WARPX_DIM_1D_Z)
                norm_pos = z;  // only one possibility
#elif defined(WARPX_DIM_XZ)
                norm_pos = dir ? z : x;  // if dir = 1, z; if dir = 0, x
#elif defined(WARPX_DIM_3D)
                if (dir == 0) {
                    norm_pos = x;
                } else if (dir == 1) {
                    norm_pos = y;
                } else if (dir == 2) {
                    norm_pos = z;
                }
#endif  // other geometries have already been ruled out.

                amrex::Real prob; // stopping probability
                if (norm_pos < loend) {
                  prob = 0._rt;
                } else if (norm_pos > hiend - dx[dir]) {
                  prob = 1._rt;
                } else {
                  prob = 1.0_rt - std::pow((hiend - dx[dir] - norm_pos) /
                                           (hiend - dx[dir] - loend),
                                            0.25_rt);
                }

                if (amrex::Random(engine) > prob) {
                    return; // do not thermalize this particle
                } else {
                    // assign new momentum from thermal distribution
                    const amrex::Real vave = std::sqrt(theta);
                    if (amrex::Math::abs(ux[ip]) > ux_threshold*PhysConst::c) {
                        ux[ip] = std::copysign(amrex::RandomNormal(0._rt, vave, engine)*PhysConst::c, ux[ip]);
                    }
                    if (amrex::Math::abs(uy[ip]) > uy_threshold*PhysConst::c) {
                        uy[ip] = std::copysign(amrex::RandomNormal(0._rt, vave, engine)*PhysConst::c, uy[ip]);
                    }
                    if (amrex::Math::abs(uz[ip]) > uz_threshold*PhysConst::c) {
                        uz[ip] = std::copysign(amrex::RandomNormal(0._rt, vave, engine)*PhysConst::c, uz[ip]);
                    }
                }
            });
        }
    }
}
