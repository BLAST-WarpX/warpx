/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "MaterialTableIdentity.H"

#include "Utils/TextMsg.H"

#include <ablastr/warn_manager/WarnManager.H>

#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace warpx::materials
{
    namespace
    {
        constexpr char manifest_name[] = "MaterialTableIdentity.txt";
        constexpr char manifest_version[] = "warpx_material_table_identity_v1";
        constexpr char hash_mode[] = "complete_file_sha256";

        struct Sha256
        {
            std::array<std::uint32_t, 8> state{
                0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
            std::array<std::uint8_t, 64> block{};
            std::size_t block_size = 0;
            std::uint64_t total_size = 0;
        };

        constexpr std::array<std::uint32_t, 64> rounds{
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

        [[nodiscard]] std::uint32_t rotate_right (
            std::uint32_t const value, int const count) noexcept
        {
            return (value >> count) | (value << (32 - count));
        }

        void sha256_compress (
            Sha256& hash, std::uint8_t const* const block) noexcept
        {
            std::array<std::uint32_t, 64> words{};
            for (std::size_t i = 0; i < 16; ++i) {
                std::size_t const offset = 4U * i;
                words[i] = (static_cast<std::uint32_t>(block[offset]) << 24U)
                    | (static_cast<std::uint32_t>(block[offset + 1U]) << 16U)
                    | (static_cast<std::uint32_t>(block[offset + 2U]) << 8U)
                    | static_cast<std::uint32_t>(block[offset + 3U]);
            }
            for (std::size_t i = 16; i < words.size(); ++i) {
                std::uint32_t const s0 =
                    rotate_right(words[i - 15U], 7)
                    ^ rotate_right(words[i - 15U], 18)
                    ^ (words[i - 15U] >> 3U);
                std::uint32_t const s1 =
                    rotate_right(words[i - 2U], 17)
                    ^ rotate_right(words[i - 2U], 19)
                    ^ (words[i - 2U] >> 10U);
                words[i] = words[i - 16U] + s0 + words[i - 7U] + s1;
            }

            auto a = hash.state[0];
            auto b = hash.state[1];
            auto c = hash.state[2];
            auto d = hash.state[3];
            auto e = hash.state[4];
            auto f = hash.state[5];
            auto g = hash.state[6];
            auto h = hash.state[7];
            for (std::size_t i = 0; i < words.size(); ++i) {
                std::uint32_t const s1 = rotate_right(e, 6)
                    ^ rotate_right(e, 11) ^ rotate_right(e, 25);
                std::uint32_t const ch = (e & f) ^ (~e & g);
                std::uint32_t const temporary1 =
                    h + s1 + ch + rounds[i] + words[i];
                std::uint32_t const s0 = rotate_right(a, 2)
                    ^ rotate_right(a, 13) ^ rotate_right(a, 22);
                std::uint32_t const maj = (a & b) ^ (a & c) ^ (b & c);
                std::uint32_t const temporary2 = s0 + maj;
                h = g;
                g = f;
                f = e;
                e = d + temporary1;
                d = c;
                c = b;
                b = a;
                a = temporary1 + temporary2;
            }
            hash.state[0] += a;
            hash.state[1] += b;
            hash.state[2] += c;
            hash.state[3] += d;
            hash.state[4] += e;
            hash.state[5] += f;
            hash.state[6] += g;
            hash.state[7] += h;
        }

        void sha256_update (
            Sha256& hash, std::uint8_t const* const data,
            std::size_t const size) noexcept
        {
            hash.total_size += size;
            std::size_t offset = 0;
            while (offset < size) {
                std::size_t const count = std::min(
                    hash.block.size() - hash.block_size, size - offset);
                std::copy(
                    data + offset, data + offset + count,
                    hash.block.begin()
                        + static_cast<std::ptrdiff_t>(hash.block_size));
                hash.block_size += count;
                offset += count;
                if (hash.block_size == hash.block.size()) {
                    sha256_compress(hash, hash.block.data());
                    hash.block_size = 0;
                }
            }
        }

        [[nodiscard]] std::string sha256_finish (Sha256 hash)
        {
            std::uint64_t const bit_size = hash.total_size * 8U;
            std::uint8_t const marker = 0x80U;
            sha256_update(hash, &marker, 1U);
            std::uint8_t const zero = 0U;
            while (hash.block_size != 56U) {
                sha256_update(hash, &zero, 1U);
            }
            std::array<std::uint8_t, 8> size_bytes{};
            for (std::size_t i = 0; i < size_bytes.size(); ++i) {
                size_bytes[size_bytes.size() - 1U - i] =
                    static_cast<std::uint8_t>(bit_size >> (8U * i));
            }
            sha256_update(hash, size_bytes.data(), size_bytes.size());
            if (hash.block_size != 0U) {
                throw std::runtime_error("internal SHA-256 finalization error");
            }

            std::ostringstream output;
            output << std::hex << std::setfill('0');
            for (std::uint32_t const word : hash.state) {
                output << std::setw(8) << word;
            }
            return output.str();
        }

        [[nodiscard]] std::string sha256_bytes (
            std::uint8_t const* const data, std::size_t const size)
        {
            Sha256 hash;
            sha256_update(hash, data, size);
            return sha256_finish(hash);
        }

        [[nodiscard]] std::string sha256_string (std::string const& value)
        {
            return sha256_bytes(
                reinterpret_cast<std::uint8_t const*>(value.data()),
                value.size());
        }

        [[nodiscard]] bool same_record (
            MaterialTableFingerprint const& lhs,
            MaterialTableFingerprint const& rhs) noexcept
        {
            return lhs.role == rhs.role && lhs.binding == rhs.binding
                && lhs.material_id == rhs.material_id
                && lhs.material_key == rhs.material_key
                && lhs.sha256 == rhs.sha256 && lhs.size == rhs.size
                && lhs.interpretation == rhs.interpretation;
        }

        [[nodiscard]] bool record_less (
            MaterialTableFingerprint const& lhs,
            MaterialTableFingerprint const& rhs) noexcept
        {
            return std::tie(
                lhs.role, lhs.binding, lhs.material_id, lhs.material_key,
                lhs.sha256, lhs.size, lhs.interpretation)
                < std::tie(
                    rhs.role, rhs.binding, rhs.material_id, rhs.material_key,
                    rhs.sha256, rhs.size, rhs.interpretation);
        }

        void canonicalize (MaterialTableIdentityContract& contract)
        {
            std::sort(
                contract.tables.begin(), contract.tables.end(), record_less);
        }

        [[nodiscard]] std::string serialize (
            MaterialTableIdentityContract const& contract)
        {
            std::ostringstream output;
            output << manifest_version << '\n'
                   << hash_mode << '\n'
                   << std::quoted(contract.selection_contract) << '\n'
                   << "count " << contract.tables.size() << '\n';
            for (auto const& table : contract.tables) {
                output << "table " << std::quoted(table.role) << ' '
                       << std::quoted(table.binding) << ' '
                       << table.material_id << ' '
                       << std::quoted(table.material_key) << ' '
                       << table.sha256 << ' ' << table.size << ' '
                       << std::quoted(table.interpretation) << '\n';
            }
            return output.str();
        }

        struct State
        {
            MaterialTableIdentityContract contract;
            std::map<std::string, std::string> load_aliases;
            std::string validated_directory;
            bool installed = false;
        };

        State& state ()
        {
            static State value;
            return value;
        }

        [[nodiscard]] std::string alias_key (
            std::string const& logical_file,
            std::string const& role,
            std::int64_t const material_id,
            std::string const& material_key)
        {
            std::ostringstream output;
            output << std::quoted(logical_file) << ' ' << std::quoted(role)
                   << ' ' << material_id << ' ' << std::quoted(material_key);
            return output.str();
        }

        [[nodiscard]] bool manifest_exists (std::string const& directory)
        {
            int exists = 0;
            if (amrex::ParallelDescriptor::IOProcessor()) {
                exists = std::filesystem::exists(
                    std::filesystem::path(directory) / manifest_name);
            }
            amrex::ParallelDescriptor::Bcast(
                &exists, 1,
                amrex::ParallelDescriptor::IOProcessorNumber());
            return exists != 0;
        }

        [[noreturn]] void fail (std::string const& message)
        {
            amrex::Abort(
                Utils::TextMsg::Err(
                    "Material table restart identity: " + message));
            std::abort();
        }

        [[nodiscard]] MaterialTableIdentityContract parse_manifest (
            std::string const& text)
        {
            std::istringstream input{text};
            std::string version;
            std::string mode;
            std::string selection;
            std::string count_label;
            std::size_t count = 0;
            if (!(input >> version >> mode >> std::quoted(selection)
                    >> count_label >> count)
                || version != manifest_version || mode != hash_mode
                || count_label != "count")
            {
                fail("manifest header is invalid or truncated.");
            }
            MaterialTableIdentityContract contract;
            contract.selection_contract = selection;
            contract.tables.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                std::string label;
                MaterialTableFingerprint table;
                if (!(input >> label >> std::quoted(table.role)
                        >> std::quoted(table.binding) >> table.material_id
                        >> std::quoted(table.material_key) >> table.sha256
                        >> table.size >> std::quoted(table.interpretation))
                    || label != "table" || table.material_id < 0
                    || table.sha256.size() != 64U
                    || table.sha256.find_first_not_of("0123456789abcdef")
                        != std::string::npos)
                {
                    fail("manifest table record is invalid or truncated.");
                }
                contract.tables.push_back(std::move(table));
            }
            input >> std::ws;
            if (!input.eof()) {
                fail("manifest has trailing or inconsistent data.");
            }
            canonicalize(contract);
            return contract;
        }

        [[nodiscard]] bool allow_legacy_override ()
        {
            bool allow = false;
            amrex::ParmParse const pp("materials");
            pp.query("restart_allow_legacy_tables", allow);
            return allow;
        }

        void compare_contracts (
            MaterialTableIdentityContract const& expected,
            MaterialTableIdentityContract const& actual)
        {
            if (expected.selection_contract != actual.selection_contract) {
                fail(
                    "material selection or interpretation settings changed; "
                    "expected '" + expected.selection_contract + "', got '"
                    + actual.selection_contract + "'.");
            }
            if (expected.tables.size() != actual.tables.size()) {
                fail(
                    "selected material table count changed; expected "
                    + std::to_string(expected.tables.size()) + ", got "
                    + std::to_string(actual.tables.size()) + ".");
            }
            for (std::size_t i = 0; i < expected.tables.size(); ++i) {
                auto const& stored = expected.tables[i];
                auto const& current = actual.tables[i];
                if (same_record(stored, current)) { continue; }
                if (stored.role != current.role
                    || stored.binding != current.binding
                    || stored.material_id != current.material_id
                    || stored.material_key != current.material_key)
                {
                    fail(
                        "material table selection or logical binding changed "
                        "for " + stored.role + "/" + stored.binding + ".");
                }
                if (stored.interpretation != current.interpretation) {
                    fail(
                        "material table interpretation changed for "
                        + stored.role + "/" + stored.binding + ".");
                }
                if (stored.size != current.size) {
                    fail(
                        "material table size changed for " + stored.role + "/"
                        + stored.binding + "; expected "
                        + std::to_string(stored.size) + ", got "
                        + std::to_string(current.size) + ".");
                }
                fail(
                    "material table content changed for " + stored.role + "/"
                    + stored.binding + "; expected SHA-256 " + stored.sha256
                    + ", got " + current.sha256 + ".");
            }
        }
    }

    void MaterialTableIdentity::Install (MaterialTableIdentityContract contract)
    {
        canonicalize(contract);
        state().contract = std::move(contract);
        state().installed = true;
        state().validated_directory.clear();
    }

    bool MaterialTableIdentity::HasTables () noexcept
    {
        return state().installed && !state().contract.tables.empty();
    }

    MaterialTableIdentityContract MaterialTableIdentity::Current ()
    {
        return state().contract;
    }

    std::string MaterialTableIdentity::HashFile (
        std::string const& filename, std::uint64_t& size)
    {
        std::ifstream input{filename, std::ios::binary};
        if (!input) {
            throw std::runtime_error("could not read table file '" + filename + "'");
        }
        Sha256 hash;
        std::array<char, 64U * 1024U> buffer{};
        while (input) {
            input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            auto const count = input.gcount();
            if (count > 0) {
                sha256_update(
                    hash,
                    reinterpret_cast<std::uint8_t const*>(buffer.data()),
                    static_cast<std::size_t>(count));
            }
        }
        if (input.bad()) {
            throw std::runtime_error("could not read table file '" + filename + "'");
        }
        size = hash.total_size;
        return sha256_finish(hash);
    }

    void MaterialTableIdentity::RegisterLoadAlias (
        std::string const& logical_file,
        std::string const& role,
        std::int64_t const material_id,
        std::string const& material_key,
        std::string const& load_file)
    {
        state().load_aliases.emplace(
            alias_key(logical_file, role, material_id, material_key),
            load_file);
    }

    std::string MaterialTableIdentity::ResolveLoadPath (
        std::string const& logical_file,
        std::string const& role,
        std::int64_t const material_id,
        std::string const& material_key)
    {
        auto const it = state().load_aliases.find(
            alias_key(logical_file, role, material_id, material_key));
        return it == state().load_aliases.end() ? logical_file : it->second;
    }

    void MaterialTableIdentity::ValidateCurrentAcrossRanks ()
    {
        auto const digest_text = serialize(state().contract);
        auto const digest = sha256_string(digest_text);
        std::array<unsigned char, 32> local{};
        for (std::size_t i = 0; i < local.size(); ++i) {
            std::istringstream input{digest.substr(2U * i, 2U)};
            unsigned int byte = 0;
            input >> std::hex >> byte;
            local[i] = static_cast<unsigned char>(byte);
        }
        std::array<unsigned char, 32> root = local;
        amrex::ParallelDescriptor::Bcast(
            root.data(), static_cast<int>(root.size()),
            amrex::ParallelDescriptor::IOProcessorNumber());
        if (root != local) {
            fail(
                "inconsistent material table identity inputs across MPI "
                "ranks.");
        }
    }

    void MaterialTableIdentity::WriteCheckpoint (std::string const& directory)
    {
        ValidateCurrentAcrossRanks();
        if (!HasTables()) { return; }

        int write_failed = 0;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            try {
                auto const path =
                    std::filesystem::path(directory) / manifest_name;
                auto const temporary = path.string() + ".tmp";
                std::ofstream output{
                    temporary, std::ios::binary | std::ios::trunc};
                if (!output) {
                    throw std::runtime_error("could not open manifest");
                }
                output << serialize(state().contract);
                output.close();
                if (!output) {
                    throw std::runtime_error("could not write manifest");
                }
                std::error_code error;
                std::filesystem::remove(path, error);
                error.clear();
                std::filesystem::rename(temporary, path, error);
                if (error) {
                    throw std::runtime_error(
                        "could not publish manifest: " + error.message());
                }
            } catch (std::exception const& exception) {
                amrex::Print()
                    << Utils::TextMsg::Err(
                           std::string("Material table identity write failed: ")
                           + exception.what())
                    << '\n';
                write_failed = 1;
            }
        }
        amrex::ParallelDescriptor::ReduceIntMax(&write_failed, 1);
        if (write_failed != 0) {
            fail("checkpoint manifest could not be published atomically.");
        }
    }

    void MaterialTableIdentity::ValidateRestart (std::string const& directory)
    {
        if (state().validated_directory == directory) { return; }
        ValidateCurrentAcrossRanks();
        bool const has_manifest = manifest_exists(directory);
        if (!HasTables()) {
            if (has_manifest) {
                fail(
                    "a non-table restart configuration encountered a "
                    "table-identity manifest; table selection changed.");
            }
            state().validated_directory = directory;
            return;
        }
        if (!has_manifest) {
            if (!allow_legacy_override()) {
                fail(
                    "checkpoint has external material tables but no identity "
                    "manifest. This legacy table checkpoint is unverified and "
                    "must not be silently accepted; set "
                    "materials.restart_allow_legacy_tables=1 only for an "
                    "explicit, audited legacy recovery.");
            }
            ablastr::warn_manager::WMRecordWarning(
                "Material table restart identity",
                "UNVERIFIED LEGACY TABLE CHECKPOINT OVERRIDE: external "
                "material-table fingerprints are absent. Restart proceeds "
                "without content identity enforcement.",
                ablastr::warn_manager::WarnPriority::high);
            state().validated_directory = directory;
            return;
        }

        auto const path =
            std::filesystem::path(directory) / manifest_name;
        amrex::Vector<char> buffer;
        amrex::ParallelDescriptor::ReadAndBcastFile(
            path.string(), buffer);
        std::string text(buffer.data());
        auto const stored = parse_manifest(text);
        compare_contracts(stored, state().contract);
        state().validated_directory = directory;
    }

    void MaterialTableIdentity::ResetForTesting ()
    {
        state() = {};
    }
}
