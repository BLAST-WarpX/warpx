/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "FieldSolver/FiniteDifferenceSolver/HybridPICModel/SingularityEOS.H"
#include "Utils/MaterialTableIdentity.H"

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>

#include <hdf5.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using warpx::materials::MaterialTableFingerprint;
    using warpx::materials::MaterialTableIdentity;
    using warpx::materials::MaterialTableIdentityContract;

    [[noreturn]] void fail (std::string const& message)
    {
        std::cerr << message << '\n';
        std::abort();
    }

    [[nodiscard]] std::string case_name ()
    {
        char const* const value =
            std::getenv("MATERIAL_RESTART_IDENTITY_CASE");
        return value == nullptr ? "unchanged" : value;
    }

    [[nodiscard]] std::filesystem::path make_directory ()
    {
        auto const stamp = std::chrono::steady_clock::now()
            .time_since_epoch().count();
        auto const rank =
            amrex::ParallelDescriptor::MyProc();
        auto directory = std::filesystem::temp_directory_path()
            / ("warpx-material-restart-identity-" + std::to_string(stamp)
               + "-" + std::to_string(rank));
        std::filesystem::create_directories(directory);
        return directory;
    }

    void write_hdf5_fixture (
        std::filesystem::path const& path, int const material_id,
        std::string const& marker)
    {
        hid_t const file = H5Fcreate(
            path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file < 0) { fail("could not create HDF5 fixture " + path.string()); }
        std::string const group_name = std::to_string(material_id);
        hid_t const group = H5Gcreate(
            file, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (group < 0) {
            H5Fclose(file);
            fail("could not create HDF5 material group");
        }
        hid_t const space = H5Screate(H5S_SCALAR);
        hid_t const type = H5Tcopy(H5T_C_S1);
        if (space < 0 || type < 0
            || H5Tset_size(type, marker.size()) < 0)
        {
            fail("could not create HDF5 fixture metadata");
        }
        hid_t const attribute = H5Acreate(
            group, "identity_marker", type, space, H5P_DEFAULT, H5P_DEFAULT);
        if (attribute < 0
            || H5Awrite(attribute, type, marker.data()) < 0)
        {
            fail("could not write HDF5 fixture metadata");
        }
        H5Aclose(attribute);
        H5Tclose(type);
        H5Sclose(space);
        H5Gclose(group);
        if (H5Fclose(file) < 0) {
            fail("could not close HDF5 fixture");
        }
    }

    /** Change one byte without changing file size. */
    void mutate_one_byte (std::filesystem::path const& path)
    {
        std::fstream file{path, std::ios::binary | std::ios::in | std::ios::out};
        if (!file) { fail("could not mutate fixture " + path.string()); }
        file.seekg(64);
        char byte = 0;
        file.read(&byte, 1);
        byte = static_cast<char>(byte ^ 0x01);
        file.seekp(64);
        file.write(&byte, 1);
        if (!file) { fail("could not mutate fixture " + path.string()); }
    }

    [[nodiscard]] MaterialTableFingerprint fingerprint (
        std::filesystem::path const& path,
        std::string role,
        std::string binding,
        std::int64_t material_id,
        std::string material_key,
        std::string interpretation)
    {
        std::uint64_t size = 0;
        auto const digest = MaterialTableIdentity::HashFile(path.string(), size);
        return {
            std::move(role), std::move(binding), material_id,
            std::move(material_key), digest, size,
            std::move(interpretation)};
    }

    [[nodiscard]] MaterialTableIdentityContract contract (
        std::filesystem::path const& eos,
        std::filesystem::path const& opacity,
        std::string eos_binding = "material:\"W\" species:\"ions\"",
        std::string opacity_binding = "material:\"W\" species:\"ions\"")
    {
        auto const split =
            static_cast<int>(singularity::TableSplit::ElectronOnly);
        std::string const eos_interpretation =
            "singularity_spiner_rho_temp_electron_only_v1"
            "|units=cgs_to_si|energy_reference=table_absolute"
            "|density_tail=evaluate_rho_min_scale_pressure"
            "|temperature_bounds=table_intersection|inverse=bracket_64"
            "|table_split=" + std::to_string(split);
        std::string const opacity_interpretation =
            "warpx_opacity_multigroup_hdf5_v1|units=si_mass_opacity"
            "|array_order=rho_temperature_group|interpolation=log_log"
            "|density_conversion=linear|groups=table_order|range=reject";
        return {
            "fixed_charge=true;selection=resolved_single_material"
            ";electron_model=singularity_spiner;opacity_backend=hdf5",
            {
                fingerprint(
                    eos, "electron_eos", std::move(eos_binding), 7400,
                    "W", eos_interpretation),
                fingerprint(
                    opacity, "opacity", std::move(opacity_binding), 7400,
                    "W", opacity_interpretation)}};
    }

    void install (MaterialTableIdentityContract value)
    {
        MaterialTableIdentity::ResetForTesting();
        MaterialTableIdentity::Install(std::move(value));
        MaterialTableIdentity::ValidateCurrentAcrossRanks();
    }

    void create_fixtures (
        std::filesystem::path const& directory,
        MaterialTableIdentityContract& value)
    {
        auto const eos = directory / "electron.sp5";
        auto const opacity = directory / "opacity.h5";
        write_hdf5_fixture(eos, 7400, "eos-manufactured");
        write_hdf5_fixture(opacity, 7400, "opacity-manufactured");
        value = contract(eos, opacity);
    }

    void write_manifest (
        std::filesystem::path const& directory,
        MaterialTableIdentityContract const& value)
    {
        install(value);
        MaterialTableIdentity::WriteCheckpoint(directory.string());
    }

    void run_unchanged ()
    {
        auto const directory = make_directory();
        MaterialTableIdentityContract value;
        create_fixtures(directory, value);
        write_manifest(directory, value);
        install(value);
        MaterialTableIdentity::ValidateRestart(directory.string());
    }

    void run_relocated ()
    {
        auto const directory = make_directory();
        MaterialTableIdentityContract original;
        create_fixtures(directory, original);
        write_manifest(directory, original);

        auto const relocated = directory / "relocated";
        std::filesystem::create_directories(relocated);
        auto const eos = relocated / "relocated.sp5";
        auto const opacity = relocated / "relocated.h5";
        std::filesystem::copy_file(
            directory / "electron.sp5", eos);
        std::filesystem::copy_file(
            directory / "opacity.h5", opacity);
        auto moved = contract(eos, opacity);
        install(std::move(moved));
        MaterialTableIdentity::ValidateRestart(directory.string());
    }

    void run_changed_content ()
    {
        auto const directory = make_directory();
        MaterialTableIdentityContract original;
        create_fixtures(directory, original);
        write_manifest(directory, original);

        auto const eos = directory / "electron.sp5";
        mutate_one_byte(eos);
        auto changed = contract(eos, directory / "opacity.h5");
        install(std::move(changed));
        MaterialTableIdentity::ValidateRestart(directory.string());
    }

    void run_changed_binding ()
    {
        auto const directory = make_directory();
        MaterialTableIdentityContract original;
        create_fixtures(directory, original);
        write_manifest(directory, original);
        auto changed = contract(
            directory / "electron.sp5", directory / "opacity.h5",
            "material:\"W\" species:\"other\"",
            "material:\"W\" species:\"other\"");
        install(std::move(changed));
        MaterialTableIdentity::ValidateRestart(directory.string());
    }

    void run_changed_material_id ()
    {
        auto const directory = make_directory();
        MaterialTableIdentityContract original;
        create_fixtures(directory, original);
        write_manifest(directory, original);
        auto changed = original;
        changed.tables.front().material_id += 1;
        install(std::move(changed));
        MaterialTableIdentity::ValidateRestart(directory.string());
    }

    void run_missing ()
    {
        auto const directory = make_directory();
        MaterialTableIdentityContract original;
        create_fixtures(directory, original);
        write_manifest(directory, original);
        std::filesystem::remove(directory / "electron.sp5");
        try {
            auto missing = contract(
                directory / "electron.sp5", directory / "opacity.h5");
            install(std::move(missing));
        } catch (std::exception const& exception) {
            std::cerr << exception.what() << '\n';
            fail("could not read externally supplied table");
        }
        fail("missing table was accepted");
    }

    void run_invalid_manifest ()
    {
        auto const directory = make_directory();
        MaterialTableIdentityContract original;
        create_fixtures(directory, original);
        write_manifest(directory, original);
        auto const manifest =
            directory / "MaterialTableIdentity.txt";
        std::filesystem::resize_file(manifest, 12);
        install(original);
        MaterialTableIdentity::ValidateRestart(directory.string());
    }

    void run_legacy ()
    {
        auto const directory = make_directory();
        MaterialTableIdentityContract original;
        create_fixtures(directory, original);
        install(original);
        MaterialTableIdentity::ValidateRestart(directory.string());
    }

    void run_legacy_rejected ()
    {
        auto const directory = make_directory();
        MaterialTableIdentityContract original;
        create_fixtures(directory, original);
        install(original);
        MaterialTableIdentity::ValidateRestart(directory.string());
    }

    void run_mpi_agreement ()
    {
        auto const directory = make_directory();
        MaterialTableIdentityContract original;
        create_fixtures(directory, original);
        install(original);
        MaterialTableIdentity::WriteCheckpoint(directory.string());
        MaterialTableIdentity::ValidateRestart(directory.string());
    }

    void run_mpi_failure ()
    {
        auto const directory = make_directory();
        MaterialTableIdentityContract original;
        create_fixtures(directory, original);
        if (amrex::ParallelDescriptor::MyProc() == 1) {
            original.tables.front().sha256 =
                std::string(64, '0');
        }
        install(std::move(original));
        fail("MPI rank mismatch was accepted");
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        auto const test_case = case_name();
        if (test_case == "unchanged") { run_unchanged(); }
        else if (test_case == "relocated") { run_relocated(); }
        else if (test_case == "changed_content") { run_changed_content(); }
        else if (test_case == "changed_binding") { run_changed_binding(); }
        else if (test_case == "changed_material_id") { run_changed_material_id(); }
        else if (test_case == "missing") { run_missing(); }
        else if (test_case == "invalid_manifest") { run_invalid_manifest(); }
        else if (test_case == "legacy") { run_legacy(); }
        else if (test_case == "legacy_rejected") { run_legacy_rejected(); }
        else if (test_case == "mpi_agree") { run_mpi_agreement(); }
        else if (test_case == "mpi_failure") { run_mpi_failure(); }
        else { fail("unknown material restart identity test case"); }
    }
    amrex::Finalize();
    return 0;
}
