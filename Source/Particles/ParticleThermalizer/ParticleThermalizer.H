#ifndef WARPX_PARTICLETHERMALIZER_H
#define WARPX_PARTICLETHERMALIZER_H

#include <string>
#include <AMReX_REAL.H>

class ParticleThermalizer {
public:
    enum class Normal { X, Y, Z };

    ParticleThermalizer(Normal normal, amrex::Real start, amrex::Real end,
                        amrex::Real momentum_threshold, amrex::Real temperature);

    // Accessors
    Normal normal() const;
    amrex::Real start() const;
    amrex::Real end() const;
    amrex::Real momentum_threshold() const;
    amrex::Real temperature() const;

    // Mutators
    void set_normal(Normal n);
    void set_start(amrex::Real s);
    void set_end(amrex::Real e);
    void set_momentum_threshold(amrex::Real mt);
    void set_temperature(amrex::Real t);

private:
    Normal m_normal;
    amrex::Real m_start;
    amrex::Real m_end;
    amrex::Real m_momentum_threshold;
    amrex::Real m_temperature;
};

#endif // WARPX_PARTICLETHERMALIZER_H
