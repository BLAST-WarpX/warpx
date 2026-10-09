/* Copyright 2026 WarpX contributors
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CURRENTWAVEFORM_H_
#define WARPX_CURRENTWAVEFORM_H_

#include <AMReX_REAL.H>

#include <string>
#include <vector>

namespace warpx::utils {
/** A linearly interpolated, two-column current waveform.
 *
 * The current is zero before the first and after the last sample.  The
 * time integral is nevertheless retained after the last sample, which is
 * required for charge-conserving impressed-current sources.
 */
class CurrentWaveform {
  public:
    CurrentWaveform () = default;
    explicit CurrentWaveform (std::string const& path) { load(path); }

    void load (std::string const& path);

    [[nodiscard]] amrex::Real value (amrex::Real time) const;
    [[nodiscard]] amrex::Real integral (amrex::Real time) const;
    [[nodiscard]] amrex::Real max_abs_current () const;
    [[nodiscard]] amrex::Real absolute_charge_bound () const;
    [[nodiscard]] bool
    empty () const {
        return m_time.empty();
    }

  private:
    std::vector<amrex::Real> m_time;
    std::vector<amrex::Real> m_current;
};
} // namespace warpx::utils

#endif // WARPX_CURRENTWAVEFORM_H_
