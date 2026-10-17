/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Axel Huebl
 *
 * License: BSD-3-Clause-LBNL
 */
#include "GIL.H"

#include <ablastr/profiler/ProfilerWrapper.H>


namespace
{
    // Set once, when our Python module is imported: before any simulation can run.
    warpx::python::AcquireGIL acquire_gil = nullptr;
    warpx::python::ReleaseGIL release_gil = nullptr;
}

namespace warpx::python
{
    void SetGILHooks ( AcquireGIL acquire, ReleaseGIL release )
    {
        acquire_gil = acquire;
        release_gil = release;
    }

    ScopedGIL::ScopedGIL ()
    {
        if (acquire_gil != nullptr && release_gil != nullptr) {
            // this waits until other Python threads hand over the GIL
            ABLASTR_PROFILE("warpx_py_acquire_gil");
            m_state = acquire_gil();
            m_release = release_gil;
        }
    }

    ScopedGIL::~ScopedGIL ()
    {
        if (m_release != nullptr) {
            m_release(m_state);
        }
    }

} // namespace warpx::python
