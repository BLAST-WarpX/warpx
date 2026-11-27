/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include <Radiation/ImplicitMomentTransport.H>
#include <Radiation/MomentTransportLedger.H>

#include <AMReX.H>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

using namespace warpx::radiation;

namespace
{
std::string
Record (MomentTransportLedger const& ledger)
{
    std::ostringstream output;
    AMREX_ALWAYS_ASSERT(ledger.Write(output));
    return output.str();
}
} // namespace

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        MomentTransportLedger ledger;
        MomentTransportAccounting increment;
        // A weak physical imbalance must survive large cancelling wall/stress
        // accounts, including checkpoint and subsequent continuation.
        // Construct the exact power of two without ldexp: NVHPC 25.1's
        // x86 optimizer can lower ldexp to an unsupported SCALEFS instruction.
        static_assert(std::numeric_limits<amrex::Real>::digits + 2 < 64);
        amrex::Real const large = static_cast<amrex::Real>(
            std::uint64_t{1} << (std::numeric_limits<amrex::Real>::digits + 2));
        increment.boundary = {large, large, large, large};
        increment.geometric = increment.boundary;
        AMREX_ALWAYS_ASSERT(ledger.TryAccumulate(increment));
        increment = {};
        increment.boundary = {1, 2, -1, -2};
        AMREX_ALWAYS_ASSERT(ledger.TryAccumulate(increment));
        for (int d = 0; d < 4; ++d)
        {
            AMREX_ALWAYS_ASSERT(ledger.Balance()[d] == increment.boundary[d]);
        }
        auto const saved = Record(ledger);
        MomentTransportLedger restored;
        std::istringstream input(saved);
        AMREX_ALWAYS_ASSERT(restored.Read(input));
        AMREX_ALWAYS_ASSERT(Record(restored) == saved);
        for (int d = 0; d < 4; ++d)
        {
            AMREX_ALWAYS_ASSERT(restored.Balance()[d] == increment.boundary[d]);
        }
        increment.geometric = {3, -3, 5, -5};
        AMREX_ALWAYS_ASSERT(ledger.TryAccumulate(increment));
        AMREX_ALWAYS_ASSERT(restored.TryAccumulate(increment));
        AMREX_ALWAYS_ASSERT(Record(restored) == Record(ledger));
        FourVector const expected{-1, 7, -7, 1};
        for (int d = 0; d < 4; ++d)
        {
            AMREX_ALWAYS_ASSERT(ledger.Balance()[d] == expected[d]);
        }

        auto const before = Record(ledger);
        for (auto const invalid : {std::numeric_limits<amrex::Real>::quiet_NaN(),
                                   std::numeric_limits<amrex::Real>::infinity()})
        {
            increment = {};
            increment.boundary[0] = 7;
            increment.geometric[3] = invalid;
            AMREX_ALWAYS_ASSERT(!ledger.TryAccumulate(increment));
            AMREX_ALWAYS_ASSERT(Record(ledger) == before);
        }
        for (auto const& malformed :
             {std::string{}, std::string("unknown_schema\n"), saved.substr(0, saved.size() / 2),
              saved + "extra\n", std::string("moment_transport_ledger_v1\nnan\n")})
        {
            std::istringstream bad(malformed);
            AMREX_ALWAYS_ASSERT(!ledger.Read(bad));
            AMREX_ALWAYS_ASSERT(Record(ledger) == before);
        }
        MomentTransportLedger overflowing;
        increment = {};
        increment.boundary[0] = std::numeric_limits<amrex::Real>::max();
        AMREX_ALWAYS_ASSERT(overflowing.TryAccumulate(increment));
        auto const max_record = Record(overflowing);
        AMREX_ALWAYS_ASSERT(!overflowing.TryAccumulate(increment));
        AMREX_ALWAYS_ASSERT(Record(overflowing) == max_record);
        // Individually finite accounts can still have an unrepresentable net.
        increment = {};
        increment.geometric[0] = -std::numeric_limits<amrex::Real>::max();
        AMREX_ALWAYS_ASSERT(!overflowing.TryAccumulate(increment));
        AMREX_ALWAYS_ASSERT(Record(overflowing) == max_record);
    }
    amrex::Finalize();
}
