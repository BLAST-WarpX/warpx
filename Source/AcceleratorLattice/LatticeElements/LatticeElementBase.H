/* Copyright 2022 David Grote
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ACCELERATORLATTICE_LATTICEELEMENTS_LATTICEELEMENTBASE_H_
#define WARPX_ACCELERATORLATTICE_LATTICEELEMENTS_LATTICEELEMENTBASE_H_

#include <AMReX_REAL.H>
#include <AMReX_ParmParse.H>
#include <AMReX_GpuContainers.H>

#include <string>
#include <vector>

struct LatticeElementBase
{

    /**
     * \brief LatticeElementBase constructor
     * This gets the input z extent of the element and check for its correctness
     *
     * @param[in] element_name the name of the element type
     */
    LatticeElementBase (std::string const& element_name);

    /**
     * \brief Read in an element base data and add it to the lists
     *
     * @param[in] pp_element The ParmParse instance to read in the data
     * @param[inout] z_location The current z location in the lattice
     */
    void
    AddElementBase(amrex::ParmParse & pp_element, amrex::ParticleReal & z_location);

    /**
     * \brief Write the base element information to the device
     */
    void
    WriteToDeviceBase ();

    std::string m_element_name;

    int nelements = 0;

    // The host level copy of the data.
    std::vector<amrex::ParticleReal> h_zs;
    std::vector<amrex::ParticleReal> h_ze;

    // The device level copy of the data.
    amrex::Gpu::DeviceVector<amrex::ParticleReal> d_zs;
    amrex::Gpu::DeviceVector<amrex::ParticleReal> d_ze;

};

#endif // WARPX_ACCELERATORLATTICE_LATTICEELEMENTS_LATTICEELEMENTBASE_H_
