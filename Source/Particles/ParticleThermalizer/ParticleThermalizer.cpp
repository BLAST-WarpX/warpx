#include "ParticleThermalizer.H"
#include <AMReX_REAL.H>
#include <AMReX_ParmParse.H>
#include <algorithm>
#include <cctype>
#include <string>

#include "Particles/MultiParticleContainer.H"

using namespace amrex::literals;

ParticleThermalizer::ParticleThermalizer()
  : m_defined(false),
    m_normal(Normal::X), m_start(0._rt), m_end(-1._rt),
    m_momentum_threshold(-1._rt), m_temperature(-1._rt)
{
  const amrex::ParmParse pp("particle_thermalizer");

  // Read normal as a string (x, y, or z)
  std::string normal_str = "";
  bool thermalizer_present = pp.query("normal", normal_str);
  if (!thermalizer_present) {
    // If no normal is specified, the thermalizer is not defined
    return;
  }
  // normalize to lowercase
  std::transform(normal_str.begin(), normal_str.end(), normal_str.begin(), [](unsigned char c){ return std::tolower(c); });
  if (normal_str == "x") {
    m_normal = Normal::X;
  } else if (normal_str == "y") {
    m_normal = Normal::Y;
  } else if (normal_str == "z") {
    m_normal = Normal::Z;
  } else {
    amrex::Abort("particle_thermalizer: normal must be 'x', 'y', or 'z'");
  }

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
  pp.get("temperature", m_temperature);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      m_temperature >= 0._rt,
      "particle_thermalizer: 'temperature' must be non-negative");
  
  m_defined = true;
}

bool ParticleThermalizer::defined() const {
  return m_defined;
}

void ParticleThermalizer::applyThermalizer(MultiParticleContainer &mpc)
{
    // Iterate over all species/particle containers. Keep this a no-op
    // for now but structure the loop so the thermalization implementation
    // can be added per-species.
    for (auto &pc_uptr : mpc) {
        if (!pc_uptr) continue;
        auto &pc = *pc_uptr;
        // Placeholder per-species work: currently no-op.
        // Example: we could inspect species name via pc.GetSpeciesName() or
        // call a method to modify particle momenta.
        (void)pc; // silence unused variable warnings until implemented
    }
}
