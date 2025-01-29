/* Copyright 2019-2022 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: David Grote, Maxence Thevenet, Weiqun Zhang, Roelof Groenewald, Axel Huebl
 *
 * License: BSD-3-Clause-LBNL
 */
#include "callbacks.H"

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
        WARPX_PROFILE("warpx_py_" + name);
        try {
            warpx_callback_py_map[name]();
        } catch (std::exception &e) {
            std::cerr << "Python callback '" << name << "' failed!" << std::endl;
            std::cerr << e.what() << std::endl;
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
