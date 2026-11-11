/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIAL_PARTICLE_COUPLING_H_
#define WARPX_RADIAL_PARTICLE_COUPLING_H_

#include <ablastr/fields/MultiFabRegister.H>
#include <string>

class WarpX;

namespace warpx::radial_particles {
bool enabled (ablastr::fields::MultiFabRegister const& fields);
void initialize (WarpX& w, std::string const& restart);
void write_identity (WarpX& w, std::string const& directory);
void update_gather (WarpX& w);
ablastr::fields::MultiLevelVectorField deposit (WarpX& w, amrex::Real dt);
void add_current (WarpX& w, ablastr::fields::VectorField const& current);
} // namespace warpx::radial_particles
#endif
