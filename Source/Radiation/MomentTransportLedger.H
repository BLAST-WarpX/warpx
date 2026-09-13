/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_RADIATION_MOMENTTRANSPORTLEDGER_H_
#define WARPX_RADIATION_MOMENTTRANSPORTLEDGER_H_

#include "FrameTransform.H"

#include <iosfwd>

namespace warpx::radiation
{
struct MomentTransportAccounting;

/** Cumulative outward wall exchange and cylindrical stress, in (E,c*p)
 * units. Geometric stress is NOT material recoil. Updates and reads are
 * transactional; checkpointing preserves the compensated summation state.
 * The caller supplies globally reduced, accepted interval accounts once.
 */
class MomentTransportLedger
{
  public:
    bool TryAccumulate (MomentTransportAccounting const& increment);
    [[nodiscard]] FourVector Boundary () const;
    [[nodiscard]] FourVector Geometric () const;
    /** Outward minus geometric contribution to the radiation inventory.
     * Subtract before rounding the individual compensated totals. */
    [[nodiscard]] FourVector Balance () const;
    bool Write (std::ostream& stream) const;
    bool Read (std::istream& stream);

  private:
    FourVector m_boundary_sum{};
    FourVector m_boundary_correction{};
    FourVector m_geometric_sum{};
    FourVector m_geometric_correction{};
};
} // namespace warpx::radiation
#endif
