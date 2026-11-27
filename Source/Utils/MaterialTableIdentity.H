/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_UTILS_MATERIALTABLEIDENTITY_H_
#define WARPX_UTILS_MATERIALTABLEIDENTITY_H_

#include <cstdint>
#include <string>
#include <vector>

namespace warpx::materials
{
    /** Immutable identity of one externally supplied material table.
     *
     * The hash covers the complete file, including HDF5 metadata and
     * bookkeeping.  File paths are deliberately not part of equality: a
     * byte-identical relocation with the same logical binding is equivalent.
     */
    struct MaterialTableFingerprint
    {
        std::string role;
        std::string binding;
        std::int64_t material_id = -1;
        std::string material_key;
        std::string sha256;
        std::uint64_t size = 0;
        std::string interpretation;
    };

    /** Versioned, canonical contract stored beside a checkpoint. */
    struct MaterialTableIdentityContract
    {
        std::string selection_contract;
        std::vector<MaterialTableFingerprint> tables;
    };

    /**
     * Shared checkpoint identity state.
     *
     * MaterialRegistry installs the contract after reading and staging the
     * exact bytes supplied to table loaders.  Checkpoint writers and readers
     * use the same canonical serialization and perform a collective agreement
     * check before publishing or consuming checkpoint state.
     */
    class MaterialTableIdentity
    {
    public:
        static void Install (MaterialTableIdentityContract contract);
        [[nodiscard]] static bool HasTables () noexcept;
        [[nodiscard]] static MaterialTableIdentityContract Current ();

        /** Write the versioned manifest.  A no-table contract writes nothing. */
        static void WriteCheckpoint (std::string const& directory);

        /** Validate the current contract against a restart manifest.
         *
         * The call is idempotent for one process and never returns after a
         * local or collective failure.  A missing manifest for a table-driven
         * checkpoint is rejected unless
         * `materials.restart_allow_legacy_tables=1` is explicitly set.
         */
        static void ValidateRestart (std::string const& directory);

        /** Compare the installed contract across MPI ranks. */
        static void ValidateCurrentAcrossRanks ();

        /** Complete-file SHA-256.  Throws std::runtime_error on I/O failure. */
        [[nodiscard]] static std::string HashFile (
            std::string const& filename, std::uint64_t& size);

        /** Resolve a staged load path for a logical table input. */
        [[nodiscard]] static std::string ResolveLoadPath (
            std::string const& logical_file,
            std::string const& role,
            std::int64_t material_id,
            std::string const& material_key);

        /** Register the exact-byte staging path for a logical table input. */
        static void RegisterLoadAlias (
            std::string const& logical_file,
            std::string const& role,
            std::int64_t material_id,
            std::string const& material_key,
            std::string const& load_file);

        /** Reset process-local aliases and validation state, for tests. */
        static void ResetForTesting ();
    };
}

#endif // WARPX_UTILS_MATERIALTABLEIDENTITY_H_
