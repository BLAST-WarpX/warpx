/* Copyright 2022 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ACCELERATORLATTICE_ACCELERATORLATTICE_H_
#define WARPX_ACCELERATORLATTICE_ACCELERATORLATTICE_H_

#include "LatticeElementFinder.H"
#include "LatticeElements/Drift.H"
#include "LatticeElements/HardEdgedQuadrupole.H"
#include "LatticeElements/HardEdgedPlasmaLens.H"

#include <memory>
#include <string>

/* \brief AcceleratorLattice contains the lattice elements
 */
class AcceleratorLattice
{
public:

    AcceleratorLattice ();

    /**
     * \brief Read in the MAD like lattice description
     *
     * @param[in] root_name base name for the input variables
     * @param[in] z_location starting z location for the lattice
     */
    void
    ReadLattice (std::string const & root_name, amrex::ParticleReal & z_location);

    /* Refinement level at which this accelerator lattice is defined */
    int m_level;

    /* Flags where any lattice elements have been specified by the user */
    bool m_lattice_defined = false;

    /**
     * \brief Initialize the element finder instance at the given level of refinement
     *
     * @param[in] lev the level of refinement
     * @param[in] gamma_boost the Lorentz factor of the boosted frame
     * @param[in] ba the box array at the level of refinement
     * @param[in] dm the distribution map at the level of refinement
     */
    void InitElementFinder (
        int lev,
        amrex::Real gamma_boost,
        amrex::BoxArray const & ba,
        amrex::DistributionMapping const & dm);

    /**
     * \brief Update the element finder, needed when the simulation frame has moved relative to the lab frame
     *
     * @param[in] lev the level of refinement
     */
    void UpdateElementFinder (int lev);

    /* The lattice element finder handles the lookup that finds the elements at the particle locations.
     * It should follow the same grid layout as the main grids.
     */
    std::unique_ptr<amrex::LayoutData<LatticeElementFinder>> m_element_finder;

    /**
     * \brief Return an instance of a lattice finder associated with the grid and that can be used
     * on the device (such as a GPU)
     *
     * @param[in] a_pti the grid where the finder is needed
     * @param[in] a_offset the particle offset since the finded needs information about the particles as well
     */
    [[nodiscard]] LatticeElementFinderDevice
    GetFinderDeviceInstance (WarpXParIter const& a_pti, int a_offset) const;

    /* All of the available lattice element types */
    Drift h_drift;
    HardEdgedQuadrupole h_quad;
    HardEdgedPlasmaLens h_plasmalens;

};

#endif // WARPX_ACCELERATORLATTICE_ACCELERATORLATTICE_H_
