/* Copyright 2019-2022 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: David Grote, Maxence Thevenet, Weiqun Zhang, Roelof Groenewald, Axel Huebl
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PY_CALLBACKS_H_
#define WARPX_PY_CALLBACKS_H_

#include "Utils/export.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <functional>
#include <map>
#include <string>


/**
 * Declare global map to hold python callback functions.
 *
 * The keys of the map describe at what point in the simulation the python
 * functions will be called. See ``WarpX/Python/pywarpx/callbacks.py`` for a
 * list of currently supported callback names.
*/
extern WARPX_EXPORT std::map< std::string, std::function<void()> > warpx_callback_py_map;

/**
 * \brief Function to install the given name and function in warpx_callback_py_map
 */
void InstallPythonCallback ( const std::string& name, std::function<void()> callback );

/**
 * \brief Function to check if the given name is a key in warpx_callback_py_map
 */
bool IsPythonCallbackInstalled ( const std::string& name );

/**
 * \brief Function to look for and execute Python callbacks
 */
void ExecutePythonCallback ( const std::string& name );

/**
 * \brief Function to clear the given callback name from warpx_callback_py_map
 */
void ClearPythonCallback ( const std::string& name );

#endif // WARPX_PY_CALLBACKS_H_
