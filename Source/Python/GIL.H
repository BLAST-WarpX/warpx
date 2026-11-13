/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Axel Huebl
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PY_GIL_H_
#define WARPX_PY_GIL_H_


namespace warpx::python
{
    /**
     * \brief Function that acquires the Python global interpreter lock (GIL)
     *
     * \return opaque state, to be passed to the matching ReleaseGIL call
     */
    using AcquireGIL = void* (*) ();

    /**
     * \brief Function that undoes the matching AcquireGIL call
     */
    using ReleaseGIL = void (*) (void* state);

    /**
     * \brief Install the functions that ScopedGIL uses
     *
     * The WarpX library does not depend on Python. If WarpX is driven from Python, then our
     * Python module installs these functions when it is imported.
     */
    void SetGILHooks ( AcquireGIL acquire, ReleaseGIL release );

    /**
     * \brief Hold the Python global interpreter lock (GIL) for the lifetime of this object
     *
     * Our Python bindings release the GIL while WarpX initializes and evolves a simulation. Use
     * this class around all code that is reached from there and uses the Python interpreter:
     * Python callbacks and in situ libraries that run Python scripts in the interpreter that
     * drives WarpX.
     *
     * This can be nested and is safe to use if the current thread already holds the GIL.
     * It does nothing if WarpX is not driven from Python.
     *
     * The time to acquire the GIL, which includes waiting for other Python threads, is reported
     * as warpx_py_acquire_gil by the TinyProfiler.
     */
    class ScopedGIL
    {
    public:
        ScopedGIL ();
        ~ScopedGIL ();

        ScopedGIL ( ScopedGIL const & )             = delete;
        ScopedGIL& operator= ( ScopedGIL const & )  = delete;
        ScopedGIL ( ScopedGIL&& )                   = delete;
        ScopedGIL& operator= ( ScopedGIL&& )        = delete;

    private:
        //! function that releases m_state, if the GIL was acquired
        ReleaseGIL m_release = nullptr;
        //! state returned by the AcquireGIL function
        void* m_state = nullptr;
    };

} // namespace warpx::python

#endif // WARPX_PY_GIL_H_
