/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "MomentTransportLedger.H"

#include "ImplicitMomentTransport.H"

#include <cmath>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>

namespace warpx::radiation
{
namespace
{
bool
Add (amrex::Real value, amrex::Real& sum, amrex::Real& correction)
{
    if (!std::isfinite(value))
    {
        return false;
    }
    amrex::Real const next = sum + value;
    if (!std::isfinite(next))
    {
        return false;
    }
    correction += std::abs(sum) >= std::abs(value) ? (sum - next) + value : (value - next) + sum;
    sum = next;
    return std::isfinite(correction) && std::isfinite(sum + correction);
}

FourVector
Total (FourVector const& sum, FourVector const& correction)
{
    FourVector result{};
    for (int d = 0; d < 4; ++d)
    {
        result[d] = sum[d] + correction[d];
    }
    return result;
}
} // namespace

bool
MomentTransportLedger::TryAccumulate (MomentTransportAccounting const& increment)
{
    auto candidate = *this;
    for (int d = 0; d < 4; ++d)
    {
        if (!Add(increment.boundary[d], candidate.m_boundary_sum[d],
                 candidate.m_boundary_correction[d]) ||
            !Add(increment.geometric[d], candidate.m_geometric_sum[d],
                 candidate.m_geometric_correction[d]))
        {
            return false;
        }
    }
    for (auto const value : candidate.Balance())
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    *this = candidate;
    return true;
}

FourVector
MomentTransportLedger::Boundary () const
{
    return Total(m_boundary_sum, m_boundary_correction);
}

FourVector
MomentTransportLedger::Geometric () const
{
    return Total(m_geometric_sum, m_geometric_correction);
}

FourVector
MomentTransportLedger::Balance () const
{
    FourVector result{};
    for (int d = 0; d < 4; ++d)
    {
        amrex::Real sum = m_boundary_sum[d];
        amrex::Real correction = 0;
        bool const valid = Add(-m_geometric_sum[d], sum, correction) &&
                           Add(m_boundary_correction[d], sum, correction) &&
                           Add(-m_geometric_correction[d], sum, correction);
        result[d] = valid ? sum + correction : std::numeric_limits<amrex::Real>::quiet_NaN();
    }
    return result;
}

bool
MomentTransportLedger::Write (std::ostream& stream) const
{
    // Do not change formatting on the caller's stream.
    std::ostringstream record;
    record << "moment_transport_ledger_v1\n"
           << std::scientific << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
    for (int d = 0; d < 4; ++d)
    {
        record << m_boundary_sum[d] << ' ' << m_boundary_correction[d] << ' ' << m_geometric_sum[d]
               << ' ' << m_geometric_correction[d] << '\n';
    }
    stream << record.str();
    return static_cast<bool>(stream);
}

bool
MomentTransportLedger::Read (std::istream& stream)
{
    MomentTransportLedger candidate;
    std::string version;
    if (!(stream >> version) || version != "moment_transport_ledger_v1")
    {
        return false;
    }
    for (int d = 0; d < 4; ++d)
    {
        if (!(stream >> candidate.m_boundary_sum[d] >> candidate.m_boundary_correction[d] >>
              candidate.m_geometric_sum[d] >> candidate.m_geometric_correction[d]))
        {
            return false;
        }
        for (auto const value : {candidate.m_boundary_sum[d], candidate.m_boundary_correction[d],
                                 candidate.m_geometric_sum[d], candidate.m_geometric_correction[d]})
        {
            if (!std::isfinite(value))
            {
                return false;
            }
        }
    }
    stream >> std::ws;
    if (stream.bad() || !stream.eof())
    {
        return false;
    }
    // Validate both totals and their independently compensated difference.
    if (!candidate.TryAccumulate({}))
    {
        return false;
    }
    *this = candidate;
    return true;
}
} // namespace warpx::radiation
