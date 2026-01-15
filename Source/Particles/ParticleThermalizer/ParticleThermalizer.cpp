#include "ParticleThermalizer.H"
#include <AMReX_REAL.H>

ParticleThermalizer::ParticleThermalizer(Normal normal, amrex::Real start, amrex::Real end,
                                         amrex::Real momentum_threshold, amrex::Real temperature)
    : m_normal(normal), m_start(start), m_end(end),
      m_momentum_threshold(momentum_threshold), m_temperature(temperature)
{}

ParticleThermalizer::Normal ParticleThermalizer::normal() const { return m_normal; }
amrex::Real ParticleThermalizer::start() const { return m_start; }
amrex::Real ParticleThermalizer::end() const { return m_end; }
amrex::Real ParticleThermalizer::momentum_threshold() const { return m_momentum_threshold; }
amrex::Real ParticleThermalizer::temperature() const { return m_temperature; }

void ParticleThermalizer::set_normal(Normal n) { m_normal = n; }
void ParticleThermalizer::set_start(amrex::Real s) { m_start = s; }
void ParticleThermalizer::set_end(amrex::Real e) { m_end = e; }
void ParticleThermalizer::set_momentum_threshold(amrex::Real mt) { m_momentum_threshold = mt; }
void ParticleThermalizer::set_temperature(amrex::Real t) { m_temperature = t; }
