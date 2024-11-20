/* Copyright 2019-2020 Andrew Myers, Axel Huebl, Maxence Thevenet,
 * Revathi Jambunathan, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "InjectorMomentum.H"

using namespace amrex;

void InjectorMomentum::clear () // NOLINT(readability-make-member-function-const)
{
    switch (type)
    {
    case Type::parser:
    case Type::gaussian:
    case Type::gaussianparser:
    case Type::gaussianflux:
    case Type::uniform:
    case Type::boltzmann:
    case Type::juttner:
    case Type::constant:
    case Type::radial_expansion:
    {
        break;
    }
    }
}
