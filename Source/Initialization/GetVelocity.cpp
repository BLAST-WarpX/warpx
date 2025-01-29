/* Copyright 2021 Hannah Klion
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "GetVelocity.H"

GetVelocity::GetVelocity (VelocityProperties const& vel) noexcept:
    m_type{vel.m_type}, m_dir{vel.m_dir}, m_sign_dir{vel.m_sign_dir}
{
    if (m_type == VelConstantValue) {
        m_velocity = vel.m_velocity;
    }
    else if (m_type == VelParserFunction) {
        m_velocity_parser = vel.m_ptr_velocity_parser->compile<3>();
    }
}
