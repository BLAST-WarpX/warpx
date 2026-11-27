/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "PacketDiffusionInterface.H"

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace warpx::radiation
{
    namespace
    {
        PacketDiffusionInterfaceResult
        rejectPacket (std::string reason)
        {
            PacketDiffusionInterfaceResult result;
            result.accepted = false;
            result.rejection_reason = std::move(reason);
            return result;
        }

        bool
        finitePositive (double value)
        {
            return std::isfinite(value) && value > 0.0;
        }

        bool
        finiteNonnegative (double value)
        {
            return std::isfinite(value) && value >= 0.0;
        }
    } // namespace

    double
    PacketDiffusionInterfaceProbability (double diffusion_optical_depth,
                                         double inward_cosine,
                                         double marshak_extrapolation_lambda)
    {
        if (!finitePositive(diffusion_optical_depth) || !finiteNonnegative(inward_cosine) ||
            inward_cosine > 1.0 || !finitePositive(marshak_extrapolation_lambda))
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        long double const probability =
            4.0L * (1.0L + 1.5L * static_cast<long double>(inward_cosine)) /
            (3.0L * static_cast<long double>(diffusion_optical_depth) +
             6.0L * static_cast<long double>(marshak_extrapolation_lambda));
        if (!std::isfinite(probability) || probability < 0.0L || probability > 1.0L)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return static_cast<double>(probability);
    }

    PacketDiffusionInterfaceResult
    AdvancePacketDiffusionInterface (PacketDiffusionInterfaceInput const& input,
                                     AsymptoticDiffusionControls const& controls)
    {
        if (!input.diffusion.boundary_admissions.empty())
        {
            return rejectPacket(
                "Packet wrapper requires empty scalar boundary admissions.");
        }

        AsymptoticDiffusionInput scalar_input = input.diffusion;
        scalar_input.boundary_admissions.assign(scalar_input.boundary_faces.size(), 0.0);
        std::vector<CrossedPacketDiffusionSplit> splits;
        splits.reserve(input.packets.size());

        for (std::size_t packet_index = 0; packet_index < input.packets.size(); ++packet_index)
        {
            CrossedPacketAtDiffusionInterface const& packet = input.packets[packet_index];
            if (packet.boundary_face >= scalar_input.boundary_faces.size() ||
                !finiteNonnegative(packet.energy) || !finiteNonnegative(packet.inward_cosine) ||
                packet.inward_cosine > 1.0)
            {
                return rejectPacket("Invalid crossed packet record.");
            }

            AsymptoticDiffusionBoundaryFace const& face =
                scalar_input.boundary_faces[packet.boundary_face];
            long double const optical_depth = static_cast<long double>(face.opacity) *
                                              static_cast<long double>(face.width);
            double const probability = PacketDiffusionInterfaceProbability(
                static_cast<double>(optical_depth), packet.inward_cosine,
                controls.marshak_extrapolation_lambda);
            if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0)
            {
                return rejectPacket("Crossed packet interface probability is invalid.");
            }

            long double const energy = static_cast<long double>(packet.energy);
            long double const admitted = static_cast<long double>(probability) * energy;
            long double const reflected = (1.0L - static_cast<long double>(probability)) * energy;
            if (!std::isfinite(optical_depth) || !std::isfinite(admitted) ||
                !std::isfinite(reflected) || admitted < 0.0L || reflected < 0.0L)
            {
                return rejectPacket("Non-finite deterministic packet split.");
            }

            CrossedPacketDiffusionSplit split;
            split.packet = packet_index;
            split.boundary_face = packet.boundary_face;
            split.crossed_energy = packet.energy;
            split.inward_cosine = packet.inward_cosine;
            split.diffusion_optical_depth = static_cast<double>(optical_depth);
            split.probability = probability;
            split.admitted_energy = static_cast<double>(admitted);
            split.reflected_energy = static_cast<double>(reflected);
            splits.push_back(split);

            long double total_admission = static_cast<long double>(
                scalar_input.boundary_admissions[packet.boundary_face]);
            total_admission += admitted;
            if (!std::isfinite(total_admission))
            {
                return rejectPacket("Boundary admission ledger overflow.");
            }
            scalar_input.boundary_admissions[packet.boundary_face] =
                static_cast<double>(total_admission);
            if (!finiteNonnegative(
                    scalar_input.boundary_admissions[packet.boundary_face]))
            {
                return rejectPacket("Boundary admission ledger is invalid.");
            }
        }

        AsymptoticDiffusionResult scalar_result =
            AdvanceAsymptoticDiffusion(scalar_input, controls);
        if (!scalar_result.accepted || !scalar_result.candidate.has_value())
        {
            std::string reason = scalar_result.rejection_reason.empty()
                                     ? "Scalar diffusion candidate was rejected."
                                     : scalar_result.rejection_reason;
            return rejectPacket(std::move(reason));
        }

        PacketDiffusionInterfaceResult result;
        result.accepted = true;
        result.candidate = PacketDiffusionInterfaceCandidate{
            std::move(*scalar_result.candidate), std::move(splits)};
        return result;
    }
} // namespace warpx::radiation
