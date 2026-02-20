/* Copyright 2026 Andrew Myers
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ParticleThermalizer.H"
#include <AMReX_REAL.H>
#include <AMReX_ParmParse.H>
#include <algorithm>
#include <cctype>
#include <string>

#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Particles/ParticleCreation/AddPlasmaUtilities.H"
#include "WarpX.H"

using namespace amrex::literals;

ParticleThermalizer::ParticleThermalizer()
  : m_defined(false),
    m_normal(-1), m_start(0._rt), m_end(-1._rt),
    m_momentum_threshold(-1._rt), m_theta(-1._rt)
{
  const amrex::ParmParse pp("particle_thermalizer");

  // Read normal as a string (x, y, or z)
  m_normal_str = "";
  bool thermalizer_present = pp.query("normal", m_normal_str);
  if (!thermalizer_present) {
    // If no normal is specified, the thermalizer is not defined
    return;
  }

  // normalize to lowercase
  std::transform(m_normal_str.begin(), m_normal_str.end(), m_normal_str.begin(), [](unsigned char c){ return std::tolower(c); });
#if defined(WARPX_DIM_1D_Z)
  if (m_normal_str == "z") {
    m_normal = 0;
  } else {
    amrex::Abort("particle_thermalizer: normal must be 'z' in 1D simulations");
  }
#elif defined(WARPX_DIM_XZ)
  if (m_normal_str == "x") {
    m_normal = 0;
  } else if (m_normal_str == "z") {
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
  if (m_normal_str == "x") {
    m_normal = 0;
  } else if (m_normal_str == "y") {
    m_normal = 1;
  } else if (m_normal_str == "z") {
    m_normal = 2;
  } else {
    amrex::Abort("particle_thermalizer: normal must be 'x', 'y', or 'z'");
  }
#endif

  // Read numeric parameters with defaults
  pp.get("start", m_start);
  pp.get("end", m_end);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      m_end > m_start,
      "particle_thermalizer: 'end' must be greater than 'start'");
  pp.get("momentum_threshold", m_momentum_threshold);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      m_momentum_threshold >= 0._rt,
      "particle_thermalizer: 'momentum_threshold' must be non-negative");
  pp.get("theta", m_theta);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      m_theta >= 0._rt,
      "particle_thermalizer: 'theta' must be non-negative");

  pp.queryarr("species", m_species_names);

  m_defined = true;
}

bool ParticleThermalizer::defined() const {
  return m_defined;
}

void ParticleThermalizer::applyThermalizer(MultiParticleContainer &mpc)
{
  if (m_species_names.empty()) {
    // No species filter: apply to all species.
    for (auto &pc_uptr : mpc) {
      if (!pc_uptr) continue;
      applyThermalizer(*pc_uptr);
    }
  } else {
    // Apply only to the named species.
    for (const auto &name : m_species_names) {
      applyThermalizer(mpc.GetParticleContainerFromName(name));
    }
  }
}

void ParticleThermalizer::applyThermalizer(WarpXParticleContainer &pc)
{
    for (int lev = 0; lev < pc.numLevels(); ++lev) {
        const auto& geom = pc.Geom(lev);
        const auto& dx = geom.CellSizeArray();
        const auto& problo = geom.ProbLoArray();
        int dir = static_cast<int>(m_normal);

        // TODO - Handle non-3D
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
            amrex::ParticleReal* unorm = nullptr;
            if (m_normal_str == "x") {
                unorm = ux;
            } else if (m_normal_str == "y") {
                unorm = uy;
            } else if (m_normal_str == "z") {
                unorm = uz;
            }

            amrex::Real loend = thermalizer_region.lo(dir);
            amrex::Real hiend = thermalizer_region.hi(dir);

            amrex::Real u_threshold = m_momentum_threshold;
            amrex::Real theta = m_theta;

            // Parallel loop over particles in the tile.
            amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (long ip, amrex::RandomEngine const& engine) noexcept {
                amrex::ParticleReal x, y, z;
                getPosition(ip, x, y, z);

                amrex::Real prob; // stopping probability
                if (z < loend) {
                  prob = 0._rt;
                } else if (z > hiend - dx[dir]) {
                  prob = 1._rt;
                } else {
                  prob = 1.0 - std::pow((hiend - dx[dir] - z) /
                                        (hiend - dx[dir] - loend),
                                        0.25_rt);
                }

                if (amrex::Random(engine) > prob) {
                    return; // do not thermalize this particle
                } else {
                  if (amrex::Math::abs(unorm[ip]) > u_threshold*PhysConst::c) {
                      // assign new momentum from thermal distribution
                      amrex::Real vave = std::sqrt(theta);
                      unorm[ip] = std::copysign(amrex::RandomNormal(0._rt, vave, engine)*PhysConst::c, unorm[ip]);
                  }
                }
            });
        }
    }
}
