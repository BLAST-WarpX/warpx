#include "ParticleThermalizer.H"
#include <AMReX_REAL.H>
#include <AMReX_ParmParse.H>
#include <algorithm>
#include <cctype>
#include <string>

ParticleThermalizer::ParticleThermalizer()
  : m_normal(Normal::X), m_start(0._rt), m_end(0._rt),
    m_momentum_threshold(0._rt), m_temperature(0._rt)
{
  const amrex::ParmParse pp("particle_thermalizer");

  // Read normal as a string (x, y, or z)
  std::string normal_str = "x";
  pp.query("normal", normal_str);
  // normalize to lowercase
  std::transform(normal_str.begin(), normal_str.end(), normal_str.begin(), [](unsigned char c){ return std::tolower(c); });
  if (normal_str == "x") {
    m_normal = Normal::X;
  } else if (normal_str == "y") {
    m_normal = Normal::Y;
  } else if (normal_str == "z") {
    m_normal = Normal::Z;
  } else {
    // unknown value -> keep default X
  }

  // Read numeric parameters with defaults
  pp.query("start", m_start);
  pp.query("end", m_end);
  pp.query("momentum_threshold", m_momentum_threshold);
  pp.query("temperature", m_temperature);
}
