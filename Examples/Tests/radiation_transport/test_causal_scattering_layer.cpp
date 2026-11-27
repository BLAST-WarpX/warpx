/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "Radiation/CausalScatteringLayer.H"

#include <AMReX.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>
#include <vector>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        using namespace warpx::radiation;

        // 1. Transparent right/left flight and census
        {
            CausalLayerSlab slab;
            slab.lo = 0.0;
            slab.hi = 1.0;
            slab.chi = 0.0;

            amrex::Real const dt = 0.4 / PhysConst::c;
            FourVector const p_right{1.0, 0.0, 0.0, 1.0};
            FourVector const p_left{1.0, 0.0, 0.0, -1.0};

            std::vector<CausalLayerPacket> pkts{
                {0, 0.2, p_right, -1, 0, LayerDisposition::Resident},
                {1, 0.8, p_right, -1, 0, LayerDisposition::Resident},
                {2, 0.8, p_left, -1, 0, LayerDisposition::Resident},
                {3, 0.2, p_left, -1, 0, LayerDisposition::Resident}
            };

            auto res = AdvanceCausalScatteringLayer(pkts, slab, dt, 12345ULL);
            AMREX_ALWAYS_ASSERT(res.valid);
            AMREX_ALWAYS_ASSERT(res.candidate.size() == 4);
            AMREX_ALWAYS_ASSERT(res.collisions == 0);
            AMREX_ALWAYS_ASSERT(res.escape_times[0] == -1 && res.escape_times[2] == -1);
            AMREX_ALWAYS_ASSERT(std::abs(res.escape_times[1]*PhysConst::c-0.2) < 1.e-14);
            AMREX_ALWAYS_ASSERT(std::abs(res.escape_times[3]*PhysConst::c-0.2) < 1.e-14);

            // Packet 0: Resident at z = 0.6
            AMREX_ALWAYS_ASSERT(res.candidate[0].disposition == LayerDisposition::Resident);
            AMREX_ALWAYS_ASSERT(std::abs(res.candidate[0].z - 0.6) < 1e-12);

            // Packet 1: EscapedRight at z = 1.0
            AMREX_ALWAYS_ASSERT(res.candidate[1].disposition == LayerDisposition::EscapedRight);
            AMREX_ALWAYS_ASSERT(std::abs(res.candidate[1].z - 1.0) < 1e-12);

            // Packet 2: Resident at z = 0.4
            AMREX_ALWAYS_ASSERT(res.candidate[2].disposition == LayerDisposition::Resident);
            AMREX_ALWAYS_ASSERT(std::abs(res.candidate[2].z - 0.4) < 1e-12);

            // Packet 3: EscapedLeft at z = 0.0
            AMREX_ALWAYS_ASSERT(res.candidate[3].disposition == LayerDisposition::EscapedLeft);
            AMREX_ALWAYS_ASSERT(std::abs(res.candidate[3].z - 0.0) < 1e-12);
        }

        // 2. Split vs whole and identical retry
        {
            CausalLayerSlab slab;
            slab.lo = 0.0;
            slab.hi = 1.0;
            slab.chi = 5.0;
            slab.initial_rate_beta = {0.1, 0.0, 0.0};
            slab.work_beta = {0.1, 0.0, 0.0};

            FourVector const p_in{2.0, 0.0, 0.0, 2.0};
            std::vector<CausalLayerPacket> pkts{
                {10, 0.5, p_in, -1, 0, LayerDisposition::Resident},
                {20, 0.3, p_in, -1, 0, LayerDisposition::Resident}
            };

            amrex::Real const dt = 2.0e-9;
            uint64_t const seed = 98765ULL;

            auto res_whole = AdvanceCausalScatteringLayer(pkts, slab, dt, seed);
            AMREX_ALWAYS_ASSERT(res_whole.valid);
            for (std::size_t i = 0; i < pkts.size(); ++i) {
                AMREX_ALWAYS_ASSERT(res_whole.candidate[i].collision_count ==
                                    res_whole.collision_counts[i]);
                AMREX_ALWAYS_ASSERT(res_whole.candidate[i].id == pkts[i].id);
            }

            // Identical retry produces bit-for-bit identical results
            auto res_retry = AdvanceCausalScatteringLayer(pkts, slab, dt, seed);
            AMREX_ALWAYS_ASSERT(res_retry.valid);
            for (std::size_t i = 0; i < pkts.size(); ++i)
            {
                AMREX_ALWAYS_ASSERT(
                    res_whole.candidate[i].rng_counter == res_retry.candidate[i].rng_counter);
                AMREX_ALWAYS_ASSERT(res_whole.candidate[i].z == res_retry.candidate[i].z);
                AMREX_ALWAYS_ASSERT(
                    res_whole.candidate[i].remaining_tau == res_retry.candidate[i].remaining_tau);
                AMREX_ALWAYS_ASSERT(
                    res_whole.candidate[i].disposition == res_retry.candidate[i].disposition);
                for (int c = 0; c < 4; ++c)
                {
                    AMREX_ALWAYS_ASSERT(
                        res_whole.candidate[i].photon[c] == res_retry.candidate[i].photon[c]);
                }
            }

            // Split timestep dt = dt1 + dt2 matches whole dt
            amrex::Real const dt1 = 0.4 * dt;
            amrex::Real const dt2 = 0.6 * dt;
            auto res_step1 = AdvanceCausalScatteringLayer(pkts, slab, dt1, seed);
            AMREX_ALWAYS_ASSERT(res_step1.valid);
            auto res_step2 = AdvanceCausalScatteringLayer(res_step1.candidate, slab, dt2, seed);
            AMREX_ALWAYS_ASSERT(res_step2.valid);

            for (std::size_t i = 0; i < pkts.size(); ++i)
            {
                AMREX_ALWAYS_ASSERT(
                    res_whole.candidate[i].rng_counter == res_step2.candidate[i].rng_counter);
                AMREX_ALWAYS_ASSERT(
                    std::abs(res_whole.candidate[i].z - res_step2.candidate[i].z) < 1e-12);
                AMREX_ALWAYS_ASSERT(
                    std::abs(res_whole.candidate[i].remaining_tau -
                             res_step2.candidate[i].remaining_tau) < 1e-12);
                AMREX_ALWAYS_ASSERT(
                    res_whole.candidate[i].disposition == res_step2.candidate[i].disposition);
                for (int c = 0; c < 4; ++c)
                {
                    AMREX_ALWAYS_ASSERT(
                        std::abs(res_whole.candidate[i].photon[c] -
                                 res_step2.candidate[i].photon[c]) < 1e-12);
                }
            }
        }

        // 3. Work beta energy identity including differing rate beta & collision telescope
        {
            CausalLayerSlab slab;
            slab.lo = 0.0;
            slab.hi = 1.0;
            slab.chi = 20.0;
            slab.initial_rate_beta = {0.2, -0.1, 0.05};
            slab.work_beta = {-0.1, 0.25, -0.15};

            FourVector const p_in{5.0, 3.0, 0.0, 4.0};
            std::vector<CausalLayerPacket> pkts{
                {100, 0.5, p_in, -1, 0, LayerDisposition::Resident},
                {101, 0.4, p_in, -1, 0, LayerDisposition::Resident}
            };

            amrex::Real const dt = 3.0e-9;
            auto res = AdvanceCausalScatteringLayer(pkts, slab, dt, 42ULL);
            AMREX_ALWAYS_ASSERT(res.valid);
            AMREX_ALWAYS_ASSERT(res.collisions > 0);

            // Conservation and telescoping identity:
            // resident_inventory + escaped_inventory + collision_transfer == initial_inventory
            FourVector initial_inv{};
            for (auto const& p : pkts)
            {
                for (int c = 0; c < 4; ++c) { initial_inv[c] += p.photon[c]; }
            }
            for (int c = 0; c < 4; ++c)
            {
                amrex::Real const final_sum = res.resident_inventory[c] +
                                              res.escaped_inventory[c] +
                                              res.collision_transfer[c];
                AMREX_ALWAYS_ASSERT(
                    std::abs(final_sum - initial_inv[c]) < 1e-12 * initial_inv[0]);
            }
            amrex::Real frame_work = 0;
            for (int d = 0; d < 3; ++d) {
                frame_work += slab.work_beta[d]*res.collision_transfer[d+1];
            }
            AMREX_ALWAYS_ASSERT(std::abs(res.collision_transfer[0]-frame_work) <
                                1.e-12*initial_inv[0]);
            for (std::size_t i = 0; i < pkts.size(); ++i) {
                auto old_rest = pkts[i].photon[0];
                auto new_rest = res.candidate[i].photon[0];
                for (int d = 0; d < 3; ++d) {
                    old_rest -= slab.work_beta[d]*pkts[i].photon[d+1];
                    new_rest -= slab.work_beta[d]*res.candidate[i].photon[d+1];
                }
                AMREX_ALWAYS_ASSERT(std::abs(old_rest-new_rest) < 1.e-12*pkts[i].photon[0]);
            }
        }

        // 4. Late budget reject leaves input unchanged and produces empty candidate
        {
            CausalLayerSlab slab;
            slab.lo = 0.0;
            slab.hi = 1.0;
            slab.chi = 100.0;

            FourVector const p_in{1.0, 0.0, 0.0, 1.0};
            std::vector<CausalLayerPacket> const pkts_orig{
                {200, 0.5, p_in, -1, 0, LayerDisposition::Resident}
            };
            std::vector<CausalLayerPacket> pkts = pkts_orig;

            auto res = AdvanceCausalScatteringLayer(
                pkts, slab, 1.0e-8, 111ULL, /*max_events_per_packet=*/1);
            AMREX_ALWAYS_ASSERT(!res.valid);
            AMREX_ALWAYS_ASSERT(res.candidate.empty());
            AMREX_ALWAYS_ASSERT(res.escape_times.empty() && res.collision_counts.empty());
            AMREX_ALWAYS_ASSERT(res.collisions == 0);
            for (int c = 0; c < 4; ++c)
            {
                AMREX_ALWAYS_ASSERT(res.collision_transfer[c] == 0);
                AMREX_ALWAYS_ASSERT(res.escaped_inventory[c] == 0);
                AMREX_ALWAYS_ASSERT(res.resident_inventory[c] == 0);
            }
            AMREX_ALWAYS_ASSERT(pkts.size() == pkts_orig.size());
            AMREX_ALWAYS_ASSERT(pkts[0].remaining_tau == pkts_orig[0].remaining_tau);
            AMREX_ALWAYS_ASSERT(pkts[0].rng_counter == pkts_orig[0].rng_counter);
            AMREX_ALWAYS_ASSERT(pkts[0].z == pkts_orig[0].z);
        }

        // 5. Malformed state, NaN, duplicate IDs, counter overflow, bad beta, bad tau
        {
            CausalLayerSlab slab;
            slab.lo = 0.0;
            slab.hi = 1.0;
            slab.chi = 1.0;

            FourVector const valid_p{1.0, 0.0, 0.0, 1.0};

            // Duplicate IDs
            {
                std::vector<CausalLayerPacket> bad{
                    {42, 0.5, valid_p, -1, 0, LayerDisposition::Resident},
                    {42, 0.5, valid_p, -1, 0, LayerDisposition::Resident}
                };
                AMREX_ALWAYS_ASSERT(!AdvanceCausalScatteringLayer(bad, slab, 1e-9, 1).valid);
            }
            // Unique id 0 is allowed
            {
                std::vector<CausalLayerPacket> ok_zero_id{
                    {0, 0.5, valid_p, -1, 0, LayerDisposition::Resident}
                };
                AMREX_ALWAYS_ASSERT(AdvanceCausalScatteringLayer(ok_zero_id, slab, 1e-9, 1).valid);
            }
            // NaN in photon
            {
                FourVector nan_p{std::numeric_limits<amrex::Real>::quiet_NaN(), 0, 0, 1};
                std::vector<CausalLayerPacket> bad{
                    {1, 0.5, nan_p, -1, 0, LayerDisposition::Resident}
                };
                AMREX_ALWAYS_ASSERT(!AdvanceCausalScatteringLayer(bad, slab, 1e-9, 1).valid);
            }
            // NaN in z
            {
                std::vector<CausalLayerPacket> bad{
                    {1, std::numeric_limits<amrex::Real>::quiet_NaN(), valid_p, -1, 0,
                     LayerDisposition::Resident}
                };
                AMREX_ALWAYS_ASSERT(!AdvanceCausalScatteringLayer(bad, slab, 1e-9, 1).valid);
            }
            // z out of slab
            {
                std::vector<CausalLayerPacket> bad{
                    {1, 1.5, valid_p, -1, 0, LayerDisposition::Resident}
                };
                AMREX_ALWAYS_ASSERT(!AdvanceCausalScatteringLayer(bad, slab, 1e-9, 1).valid);
            }
            // Negative tau other than exact -1 sentinel
            {
                std::vector<CausalLayerPacket> bad{
                    {1, 0.5, valid_p, -0.5, 0, LayerDisposition::Resident}
                };
                AMREX_ALWAYS_ASSERT(!AdvanceCausalScatteringLayer(bad, slab, 1e-9, 1).valid);
            }
            // Sentinel -1 with rng_counter != 0
            {
                std::vector<CausalLayerPacket> bad{
                    {1, 0.5, valid_p, -1, 5, LayerDisposition::Resident}
                };
                AMREX_ALWAYS_ASSERT(!AdvanceCausalScatteringLayer(bad, slab, 1e-9, 1).valid);
            }
            // Counter overflow reject
            {
                std::vector<CausalLayerPacket> bad{
                    {1, 0.5, valid_p, 0.5, std::numeric_limits<uint64_t>::max(),
                     LayerDisposition::Resident}
                };
                CausalLayerSlab thick_slab = slab;
                thick_slab.chi = 100.0;
                AMREX_ALWAYS_ASSERT(!AdvanceCausalScatteringLayer(bad, thick_slab, 1e-9, 1).valid);
            }
            // Bad beta (>= 1)
            {
                CausalLayerSlab bad_slab = slab;
                bad_slab.initial_rate_beta = {1.0, 0, 0};
                std::vector<CausalLayerPacket> pkts{
                    {1, 0.5, valid_p, -1, 0, LayerDisposition::Resident}
                };
                AMREX_ALWAYS_ASSERT(!AdvanceCausalScatteringLayer(pkts, bad_slab, 1e-9, 1).valid);
            }
            // max_events_per_packet == 0
            {
                std::vector<CausalLayerPacket> pkts{
                    {1, 0.5, valid_p, -1, 0, LayerDisposition::Resident}
                };
                AMREX_ALWAYS_ASSERT(!AdvanceCausalScatteringLayer(pkts, slab, 1e-9, 1, 0).valid);
            }
            // dt < 0
            {
                std::vector<CausalLayerPacket> pkts{
                    {1, 0.5, valid_p, -1, 0, LayerDisposition::Resident}
                };
                AMREX_ALWAYS_ASSERT(!AdvanceCausalScatteringLayer(pkts, slab, -1e-9, 1).valid);
            }
        }

        // 6. Zero optical depth valid clock
        {
            CausalLayerSlab slab;
            slab.lo = 0.0;
            slab.hi = 1.0;
            slab.chi = 0.0;

            FourVector const valid_p{1.0, 0.0, 0.0, 1.0};
            std::vector<CausalLayerPacket> pkts{
                {300, 0.5, valid_p, -1, 0, LayerDisposition::Resident}
            };
            auto res = AdvanceCausalScatteringLayer(pkts, slab, 1.0e-10, 777ULL);
            AMREX_ALWAYS_ASSERT(res.valid);
            AMREX_ALWAYS_ASSERT(res.candidate[0].remaining_tau > 0);
            AMREX_ALWAYS_ASSERT(res.candidate[0].rng_counter == 1);
            AMREX_ALWAYS_ASSERT(res.collisions == 0);
        }

        // 7. dt = 0 no RNG initialization
        {
            CausalLayerSlab slab;
            slab.lo = 0.0;
            slab.hi = 1.0;
            slab.chi = 1.0;

            FourVector const valid_p{1.0, 0.0, 0.0, 1.0};
            std::vector<CausalLayerPacket> pkts{
                {400, 0.5, valid_p, -1, 0, LayerDisposition::Resident}
            };
            auto res = AdvanceCausalScatteringLayer(pkts, slab, 0.0, 777ULL);
            AMREX_ALWAYS_ASSERT(res.valid);
            AMREX_ALWAYS_ASSERT(res.candidate[0].remaining_tau == -1);
            AMREX_ALWAYS_ASSERT(res.candidate[0].rng_counter == 0);
            AMREX_ALWAYS_ASSERT(res.collisions == 0);
        }

        // Independent Python analog-MC reference: normal incidence, seed 7,
        // 20000 histories. Compare physical angular law, not a fitted cosine.
        for (int thick = 0; thick < 2; ++thick) {
            constexpr int count = 20000;
            std::vector<CausalLayerPacket> ensemble(count);
            for (int i = 0; i < count; ++i) {
                ensemble[i].id = static_cast<std::uint64_t>(i);
                ensemble[i].photon = {1, 0, 0, 1};
            }
            CausalLayerSlab slab;
            slab.chi = thick ? 8 : 2;
            auto response = AdvanceCausalScatteringLayer(ensemble, slab, 100/PhysConst::c, 7);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(response.valid, response.reason.c_str());
            int transmitted = 0;
            long double mu = 0, mu2 = 0;
            for (auto const& packet : response.candidate) {
                AMREX_ALWAYS_ASSERT(packet.disposition != LayerDisposition::Resident);
                if (packet.disposition == LayerDisposition::EscapedRight) {
                    auto const cosine = packet.photon[3]/packet.photon[0];
                    ++transmitted;
                    mu += cosine;
                    mu2 += cosine*cosine;
                }
            }
            auto const expected_count = thick ? 3549 : 9468;
            auto const expected_mu = thick ? 0.7151305099632086L : 0.7677803513982281L;
            auto const reference_se = thick ? 0.0036379786329532033L : 0.0024917966090721205L;
            auto const probability = static_cast<long double>(expected_count)/count;
            AMREX_ALWAYS_ASSERT(std::abs(transmitted-expected_count) <=
                                6*std::sqrt(2*count*probability*(1-probability)));
            auto const mean = mu/transmitted;
            auto const variance = (mu2-transmitted*mean*mean)/(transmitted-1);
            AMREX_ALWAYS_ASSERT(std::abs(mean-expected_mu) <=
                                6*std::sqrt(variance/transmitted+reference_se*reference_se));
        }

        // A zero clock is an immediate physical event, not an initialization sentinel.
        CausalLayerPacket zero_clock{901, 0.5, {1, 0, 0, 1}, 0, 0, LayerDisposition::Resident};
        CausalLayerSlab finite;
        finite.chi = 1;
        auto immediate = AdvanceCausalScatteringLayer({zero_clock}, finite, 1.e-20, 41);
        AMREX_ALWAYS_ASSERT(immediate.valid && immediate.collisions == 1);
        AMREX_ALWAYS_ASSERT(immediate.candidate[0].rng_counter == 3);
        auto overflow = AdvanceCausalScatteringLayer({zero_clock}, finite,
            std::numeric_limits<amrex::Real>::max(), 41);
        AMREX_ALWAYS_ASSERT(!overflow.valid && overflow.candidate.empty() &&
                            overflow.escape_times.empty() && overflow.collision_counts.empty());
        amrex::Print() << "Causal scattering layer unit tests PASSED successfully!\n";
    }
    amrex::Finalize();
    return 0;
}
