/* Copyright 2021 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ParticleCreationFunc.H"

#include "BinaryCollisionUtils.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/TextMsg.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

namespace {

    void readFusionCrossSectionFile (
        const std::string& cross_section_file,
        amrex::Gpu::HostVector<amrex::ParticleReal>& energies,
        amrex::Gpu::HostVector<amrex::ParticleReal>& coefficients,
        int& num_coefficients)
    {
        std::ifstream infile(cross_section_file);
        if (!infile.is_open()) {
            WARPX_ABORT_WITH_MESSAGE(
                "Failed to open fusion cross-section data file: " + cross_section_file);
        }

        // Will be set from the first non-empty row; all subsequent rows must match.
        num_coefficients = 0;
        std::string line;
        int line_number = 0;
        while (std::getline(infile, line)) {
            ++line_number;

            // Parse all whitespace-separated values on this line.
            std::istringstream line_stream(line);
            amrex::Vector<amrex::ParticleReal> values;
            amrex::ParticleReal value;
            while (line_stream >> value) {
                values.push_back(value);
            }
            if (values.empty()) {
                continue; // skip blank lines (no parseable floats)
            }

            // Each row must have: one energy column + at least one coefficient column.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                values.size() > 1u,
                "Fusion cross-section data must contain one energy column and at least one "
                "coefficient column.");

            int const row_num_coefficients = static_cast<int>(values.size()) - 1;
            if (num_coefficients == 0) {
                // Lock in the column count from the first data row.
                num_coefficients = row_num_coefficients;
            } else {
                // Subsequent rows must have the same number of coefficient columns.
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    row_num_coefficients == num_coefficients,
                    "Inconsistent number of columns in fusion cross-section data file " +
                    cross_section_file + " at line " + std::to_string(line_number));
            }

            // Energy grid must be strictly increasing for interpolation.
            if (!energies.empty()) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    values[0] > energies.back(),
                    "Fusion cross-section energy values must be strictly increasing.");
            }

            // Store the energy (column 0) and the coefficient(s) (columns 1..N).
            energies.push_back(values[0]);
            for (int i = 0; i < num_coefficients; ++i) {
                coefficients.push_back(values[i + 1]);
            }
        }

        // Distinguish a clean EOF from a low-level I/O error.
        if (infile.bad()) {
            WARPX_ABORT_WITH_MESSAGE(
                "Failed to read fusion cross-section data from file: " + cross_section_file);
        }

        // At least two energy points are required to perform linear interpolation.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            energies.size() > 1u,
            "Fusion cross-section data file must contain at least two energy rows for "
            "interpolation: " +
            cross_section_file);
    }

}

ParticleCreationFunc::ParticleCreationFunc (const std::string& collision_name,
                                            MultiParticleContainer const * const mypc):
    m_collision_type{BinaryCollisionUtils::get_collision_type(collision_name, mypc)}
{
    const amrex::ParmParse pp_collision_name(collision_name);

    if (m_collision_type == CollisionType::ProtonBoronToAlphasFusion)
    {
        // Proton-Boron fusion only produces alpha particles
        m_num_product_species = 1;
        // Proton-Boron fusion produces 3 alpha particles per fusion reaction
        m_num_products_host.push_back(3);
#ifndef AMREX_USE_GPU
        // On CPU, the device vector can be filled immediately
        m_num_products_device.push_back(3);
#endif
    }
    else if ((BinaryCollisionUtils::is_two_product_fusion_type(m_collision_type))
        || (m_collision_type == CollisionType::LinearBreitWheeler)
        || (m_collision_type == CollisionType::LinearCompton))
    {
        m_num_product_species = 2;
        m_num_products_host.push_back(1);
        m_num_products_host.push_back(1);
#ifndef AMREX_USE_GPU
        // On CPU, the device vector can be filled immediately
        m_num_products_device.push_back(1);
        m_num_products_device.push_back(1);
#endif
    }
    else
    {
        WARPX_ABORT_WITH_MESSAGE("Unknown collision type in ParticleCreationFunc");
    }

    if (m_collision_type == CollisionType::ProtonBoronToAlphasFusion
        || BinaryCollisionUtils::is_two_product_fusion_type(m_collision_type))
    {
        pp_collision_name.query_enum_case_insensitive("scattering_angle_model", m_scattering_angle_model);
    }

    // Optionally load an energy-dependent table of coefficients that
    // describes the anisotropic angular distribution of fusion products.
    std::string fusion_angular_distribution_coefficients_file;
    if (pp_collision_name.query("fusion_angular_distribution_coefficients", fusion_angular_distribution_coefficients_file)) {
        // Temporary host-side storage for the table data.
        amrex::Gpu::HostVector<amrex::ParticleReal> h_energies;
        amrex::Gpu::HostVector<amrex::ParticleReal> h_coefficients;
        // Parse the file; sets m_fusion_angular_distribution_num_coefficients
        // to the number of coefficients per energy point.
        readFusionCrossSectionFile(
            fusion_angular_distribution_coefficients_file, h_energies, h_coefficients,
            m_fusion_angular_distribution_num_coefficients);

        // Record the number of energy points and size the device vectors.
        m_fusion_angular_distribution_num_energies = static_cast<int>(h_energies.size());
        m_fusion_angular_distribution_energies.resize(h_energies.size());
        m_fusion_angular_distribution_coefficients.resize(h_coefficients.size());
#ifdef AMREX_USE_GPU
        // Upload energy grid and coefficient table to device memory.
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_energies.begin(), h_energies.end(),
                         m_fusion_angular_distribution_energies.begin());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_coefficients.begin(), h_coefficients.end(),
                         m_fusion_angular_distribution_coefficients.begin());
#else
        // CPU path: DeviceVector uses host memory, so std::copy suffices.
        std::copy(h_energies.begin(), h_energies.end(), m_fusion_angular_distribution_energies.begin());
        std::copy(h_coefficients.begin(), h_coefficients.end(),
                  m_fusion_angular_distribution_coefficients.begin());
#endif
    }

    // If anisotropic scattering is requested, the coefficient table must have
    // been loaded above; enforce this requirement before the simulation continues.
    if (m_collision_type == CollisionType::ProtonBoronToAlphasFusion
        || BinaryCollisionUtils::is_two_product_fusion_type(m_collision_type))
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_scattering_angle_model != ScatteringAngleModel::Anisotropic
            || m_fusion_angular_distribution_num_energies > 0,
            "<collision_name>.scattering_angle_model = anisotropic requires "
            "<collision_name>.fusion_angular_distribution_coefficients to be set "
            "to a valid table file.");
    }

#ifdef AMREX_USE_GPU
     m_num_products_device.resize(m_num_product_species);
     amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_num_products_host.begin(),
                           m_num_products_host.end(),
                           m_num_products_device.begin());
     amrex::Gpu::streamSynchronize();
#endif
}
