/* Copyright 2026 WarpX contributors
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "CurrentWaveform.H"

#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

using namespace amrex::literals;

namespace warpx::utils {
void
CurrentWaveform::load (std::string const& path) {
    m_time.clear();
    m_current.clear();

    amrex::Vector<char> file_chars;
    amrex::ParallelDescriptor::ReadAndBcastFile(path, file_chars);
    std::string const file_contents(file_chars.dataPtr());
    std::istringstream waveform_stream(file_contents);
    std::string line;
    int line_number = 0;
    while (std::getline(waveform_stream, line)) {
        ++line_number;
        auto const first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }

        std::istringstream row(line);
        amrex::Real sample_time = 0.0_rt;
        amrex::Real sample_current = 0.0_rt;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<bool>(row >> sample_time >> sample_current),
            "warpx current waveform: malformed row " +
                std::to_string(line_number) + " in '" + path + "'");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(sample_time) && std::isfinite(sample_current),
            "warpx current waveform: non-finite value on row " +
                std::to_string(line_number) + " in '" + path + "'");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_time.empty() ||
                                             sample_time > m_time.back(),
                                         "warpx current waveform: sample times "
                                         "must be strictly increasing in '" +
                                             path + "'");
        m_time.push_back(sample_time);
        m_current.push_back(sample_current);
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_time.size() >= 2, "warpx current waveform: file '" + path +
                                "' must contain at least two data rows");
}

amrex::Real
CurrentWaveform::value (amrex::Real const time) const {
    if (m_time.size() < 2 || time < m_time.front() || time > m_time.back()) {
        return 0.0_rt;
    }
    auto const upper = std::lower_bound(m_time.begin(), m_time.end(), time);
    if (upper == m_time.begin()) {
        return m_current.front();
    }
    auto const index =
        static_cast<std::size_t>(std::distance(m_time.begin(), upper));
    amrex::Real const t0 = m_time[index - 1];
    amrex::Real const t1 = m_time[index];
    amrex::Real const i0 = m_current[index - 1];
    amrex::Real const i1 = m_current[index];
    return i0 + (i1 - i0) * (time - t0) / (t1 - t0);
}

amrex::Real
CurrentWaveform::integral (amrex::Real const time) const {
    if (m_time.size() < 2 || time <= m_time.front()) {
        return 0.0_rt;
    }

    amrex::Real charge = 0.0_rt;
    for (std::size_t index = 1; index < m_time.size(); ++index) {
        amrex::Real const t0 = m_time[index - 1];
        amrex::Real const t1 = m_time[index];
        if (time >= t1) {
            charge +=
                0.5_rt * (m_current[index - 1] + m_current[index]) * (t1 - t0);
            continue;
        }

        amrex::Real const duration = time - t0;
        amrex::Real const slope =
            (m_current[index] - m_current[index - 1]) / (t1 - t0);
        charge += m_current[index - 1] * duration +
                  0.5_rt * slope * duration * duration;
        break;
    }
    return charge;
}

amrex::Real
CurrentWaveform::max_abs_current () const {
    amrex::Real result = 0.0_rt;
    for (amrex::Real const current : m_current) {
        result = std::max(result, std::abs(current));
    }
    return result;
}

amrex::Real
CurrentWaveform::absolute_charge_bound () const {
    amrex::Real result = 0.0_rt;
    for (std::size_t index = 1; index < m_time.size(); ++index) {
        amrex::Real const duration = m_time[index] - m_time[index - 1];
        amrex::Real const i0 = m_current[index - 1];
        amrex::Real const i1 = m_current[index];
        if (i0 * i1 >= 0.0_rt) {
            result += 0.5_rt * (std::abs(i0) + std::abs(i1)) * duration;
        } else {
            amrex::Real const first_duration =
                duration * std::abs(i0) / (std::abs(i0) + std::abs(i1));
            result += 0.5_rt * std::abs(i0) * first_duration;
            result += 0.5_rt * std::abs(i1) * (duration - first_duration);
        }
    }
    return result;
}
} // namespace warpx::utils
