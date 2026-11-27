/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_RADIATION_PACKET_DIFFUSION_INTERFACE_H_
#define WARPX_RADIATION_PACKET_DIFFUSION_INTERFACE_H_

#include "AsymptoticDiffusion.H"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace warpx::radiation
{

    /** A packet that has already crossed a D/K interface face.
     *
     * `energy` is the already-crossed packet energy [J].  It must not be
     * multiplied by `mu` again.  `inward_cosine` is the nonnegative cosine
     * between the packet direction and the inward D-side face normal.
     */
    struct CrossedPacketAtDiffusionInterface
    {
        std::size_t boundary_face = 0;
        double energy = 0.0;
        double inward_cosine = 0.0;
    };

    /** Deterministic virtual-interface split of one crossed packet.
     *
     * The explicit virtual redirection impulse is exactly zero.  The mean D-side
     * effect is counted once by the all-D estimator in the scalar candidate.
     */
    struct CrossedPacketDiffusionSplit
    {
        std::size_t packet = 0;
        std::size_t boundary_face = 0;
        double crossed_energy = 0.0;
        double inward_cosine = 0.0;
        double diffusion_optical_depth = 0.0;
        double probability = 0.0;
        double admitted_energy = 0.0;
        double reflected_energy = 0.0;
    };

    struct PacketDiffusionInterfaceInput
    {
        /** Scalar private region.  `boundary_admissions` must be empty because
         * this wrapper owns packet-to-D admission bookkeeping. */
        AsymptoticDiffusionInput diffusion;
        std::vector<CrossedPacketAtDiffusionInterface> packets;
    };

    struct PacketDiffusionInterfaceCandidate
    {
        AsymptoticDiffusionCandidate diffusion;
        std::vector<CrossedPacketDiffusionSplit> packets;
    };

    struct PacketDiffusionInterfaceResult
    {
        bool accepted = false;
        std::string rejection_reason;
        std::optional<PacketDiffusionInterfaceCandidate> candidate;

        explicit operator bool () const noexcept { return accepted; }
    };

    /** Return P(mu)=4(1+3 mu/2)/(3 tau+6 lambda).
     *
     * Invalid optical depth, cosine, extrapolation constant, or probability
     * returns a NaN; the caller must reject rather than clip it.
     */
    double
    PacketDiffusionInterfaceProbability (double diffusion_optical_depth,
                                         double inward_cosine,
                                         double marshak_extrapolation_lambda = 0.7104);

    /** Split crossed packets deterministically, then solve the resulting cold,
     * frozen-material backward-Euler scalar candidate.
     *
     * All inputs are const.  Rejection returns no candidate and therefore no
     * accepted leakage, material impulse, or packet split.
     */
    PacketDiffusionInterfaceResult
    AdvancePacketDiffusionInterface (PacketDiffusionInterfaceInput const& input,
                                     AsymptoticDiffusionControls const& controls);

} // namespace warpx::radiation

#endif // WARPX_RADIATION_PACKET_DIFFUSION_INTERFACE_H_
