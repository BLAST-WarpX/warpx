/* Copyright 2021-2022 The WarpX Community
 *
 * This header is used to centrally define classes that shall not violate the
 * C++ one-definition-rule (ODR) for various Python translation units.
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PYWARPX_H_
#define WARPX_PYWARPX_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <list>

namespace py = pybind11;
//using namespace warpx;

// PYBIND11_MAKE_OPAQUE(std::list<...>)

#endif  // WARPX_PYWARPX_H_
