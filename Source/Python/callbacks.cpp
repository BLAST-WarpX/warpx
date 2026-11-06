/* Copyright 2019-2022 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: David Grote, Maxence Thevenet, Weiqun Zhang, Roelof Groenewald, Axel Huebl
 *
 * License: BSD-3-Clause-LBNL
 */
#include "callbacks.H"

#include "GIL.H"

#include <cstdlib>
#include <exception>
#include <iostream>


std::map< std::string, std::function<void()> > warpx_callback_py_map;

void InstallPythonCallback ( const std::string& name, std::function<void()> callback )
{
    warpx_callback_py_map[name] = std::move(callback);
}

bool IsPythonCallbackInstalled ( const std::string& name )
{
    return (warpx_callback_py_map.count(name) == 1u);
}

// Execute Python callbacks of the type given by the input string
void ExecutePythonCallback ( const std::string& name )
{
    if ( IsPythonCallbackInstalled(name) ) {
        // Our Python bindings release the GIL while the simulation runs. Acquire it before the
        // callback timer starts: waiting for other Python threads is not part of the callback.
        const warpx::python::ScopedGIL gil;

        // removing callbacks from another Python thread is not supported, but if it happened
        // while we waited for the GIL, then return instead of calling an empty function
        const auto it = warpx_callback_py_map.find(name);
        if (it == warpx_callback_py_map.end()) { return; }

        ABLASTR_PROFILE("warpx_py_" + name);
        try {
            it->second();
        } catch (std::exception &e) {
            std::cerr << "Python callback '" << name << "' failed!" << "\n";
            std::cerr << e.what() << "\n";
            std::exit(3);  // note: NOT amrex::Abort(), to avoid hangs with MPI

            // future note:
            // if we want to rethrow/raise exceptions from Python callbacks through here (C++) and
            // back the managing Python interpreter, we first need to discard and clear
            // out the Python error in py::error_already_set. Otherwise, MPI-runs will hang
            // (and Python will be in continued error state).
            // https://pybind11.readthedocs.io/en/stable/advanced/exceptions.html#handling-unraisable-exceptions
        }
    }
}

void ClearPythonCallback ( const std::string& name )
{
    warpx_callback_py_map.erase(name);
}
