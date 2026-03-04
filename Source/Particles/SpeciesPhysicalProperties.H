/* Copyright 2020 Maxence Thevenet
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_SPECIESPHYSICALPROPERTIES_H_
#define WARPX_SPECIESPHYSICALPROPERTIES_H_

#include <AMReX_REAL.H>

#include <optional>
#include <string>

enum struct PhysicalSpecies{
    unspecified=0, electron, positron, muon, antimuon, photon, neutron, hydrogen, proton, hydrogen2, hydrogen3,
    helium, helium3, helium4, lithium, lithium6, lithium7, beryllium, beryllium9, boron, boron10, boron11, carbon,
    carbon12, carbon13, carbon14, nitrogen, nitrogen14, nitrogen15, oxygen, oxygen16, oxygen17, oxygen18, fluorine,
    fluorine19, neon, neon20, neon21, neon22, aluminium, argon, copper, xenon, gold};

namespace species
{
    /**
     * \brief Returns the PhysicalSpecies associated to a given name
     *
     * \param[in] species_name the name of a species
     * \return the PhysicalSpecies corresponding to species_name (if it exists)
     */
    std::optional<PhysicalSpecies> from_string (const std::string& species_name);

    /**
     * \brief Returns the charge associated to a PhysicalSpecies
     *
     * \param[in] ps the PhysicalSpecies
     * \return the charge associated to the PhysicalSpecies
     */
    amrex::Real get_charge (const PhysicalSpecies& ps);

    /**
     * \brief Returns the mass associated to a PhysicalSpecies
     *
     * \param[in] ps the PhysicalSpecies
     * \return the mass associated to the PhysicalSpecies
     */
    amrex::Real get_mass (const PhysicalSpecies& ps);

    /**
     * \brief Returns the name associated to a PhysicalSpecies
     *
     * \param[in] ps the PhysicalSpecies
     * \return the name associated to the PhysicalSpecies
     */
    std::string get_name (const PhysicalSpecies& ps);
}

#endif // WARPX_SPECIESPHYSICALPROPERTIES_H_
