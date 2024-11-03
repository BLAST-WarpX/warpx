/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Remi Lehe, Roelof Groenewald, Arianna Formenti, Revathi Jambunathan
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.H"

#include "Fields.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"


void WarpX::ComputeSpaceChargeField (bool const reset_fields)
{
    WARPX_PROFILE("WarpX::ComputeSpaceChargeField");
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    if (reset_fields) {
        // Reset all E and B fields to 0, before calculating space-charge fields
        WARPX_PROFILE("WarpX::ComputeSpaceChargeField::reset_fields");
        for (int lev = 0; lev <= max_level; lev++) {
            for (int comp=0; comp<3; comp++) {
                m_fields.get(FieldType::Efield_fp, Direction{comp}, lev)->setVal(0);
                m_fields.get(FieldType::Bfield_fp, Direction{comp}, lev)->setVal(0);
            }
        }
    }

    m_electrostatic_solver->ComputeSpaceChargeField(
        m_fields, *mypc, myfl.get(), max_level );
}
