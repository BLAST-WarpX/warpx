/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 * Authors: Luca Fedeli, Justin Angus, Remi Lehe, Axel Huebl
 */
#ifndef WARPX_FIELDS_H_
#define WARPX_FIELDS_H_

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Enum.H>

#include <algorithm>
#include <iterator>


namespace warpx::fields
{
    /** Unique identifiers for WarpX scalar and vector fields.
     *
     * These are implemented as amrex::MultiFab (one or one per component "direction",
     * respectively) and stored in the ablastr::fields::MultiFabRegister .
     */
    AMREX_ENUM(FieldType,
        None,
        Efield_aux, /**< Field that the particles gather from. Obtained from Efield_fp (and Efield_cp when using MR); see UpdateAuxilaryData */
        Bfield_aux, /**< Field that the particles gather from. Obtained from Bfield_fp (and Bfield_cp when using MR); see UpdateAuxilaryData */
        Efield_fp,  /**< The field that is updated by the field solver at each timestep */
        Bfield_fp,  /**< The field that is updated by the field solver at each timestep */
        Efield_fp_external, /**< Stores grid particle fields provided by the user as  through an openPMD file */
        Bfield_fp_external, /**< Stores grid particle fields provided by the user as  through an openPMD file */
        current_fp, /**< The current that is used as a source for the field solver */
        current_fp_nodal, /**< Only used when using nodal current deposition */
        current_fp_vay,   /**< Only used when using Vay current deposition */
        current_buf, /**< Particles that are close to the edge of the MR patch (i.e. in the deposition buffer) deposit to this field. */
        current_store, /**< Only used when doing subcycling with mesh refinement, for book-keeping of currents */
        rho_buf, /**< Particles that are close to the edge of the MR patch (i.e. in the deposition buffer) deposit to this field. */
        rho_fp,  /**< The charge density that is used as a source for the field solver (mostly for labframe electrostatic and PSATD) */
        F_fp, /**< Used for divE cleaning */
        G_fp, /**< Used for divB cleaning */
        phi_fp, /**< Obtained by the Poisson solver, for labframe electrostatic */
        vector_potential_fp, /**< Obtained by the magnetostatic solver */
        vector_potential_fp_nodal,
        vector_potential_grad_buf_e_stag,
        vector_potential_grad_buf_b_stag,
        hybrid_electron_pressure_fp, /**< Used with Ohm's law solver. Stores the electron pressure */
        hybrid_rho_fp_temp,          /**< Used with Ohm's law solver. Stores the time interpolated/extrapolated charge density */
        hybrid_current_fp_temp,      /**< Used with Ohm's law solver. Stores the time interpolated/extrapolated current density */
        hybrid_current_fp_plasma,    /**< Used with Ohm's law solver. Stores plasma current calculated as J_plasma = curl x B / mu0 - J_ext */
        hybrid_current_fp_external,  /**< Used with Ohm's law solver. Stores external current */
        Efield_cp,  /**< Only used with MR. The field that is updated by the field solver at each timestep, on the coarse patch of each level */
        Bfield_cp,  /**< Only used with MR. The field that is updated by the field solver at each timestep, on the coarse patch of each level */
        current_cp, /**< Only used with MR. The current that is used as a source for the field solver, on the coarse patch of each level */
        rho_cp, /**< Only used with MR. The charge density that is used as a source for the field solver, on the coarse patch of each level */
        F_cp,   /**< Only used with MR. Used for divE cleaning, on the coarse patch of each level */
        G_cp,   /**< Only used with MR. Used for divB cleaning, on the coarse patch of each level */
        Efield_cax, /**< Only used with MR. Particles that are close to the edge of the MR patch (i.e. in the gather buffer) gather from this field */
        Bfield_cax, /**< Only used with MR. Particles that are close to the edge of the MR patch (i.e. in the gather buffer) gather from this field */
        E_external_particle_field, /**< Stores external particle fields provided by the user as  through an openPMD file */
        B_external_particle_field, /**< Stores external particle fields provided by the user as  through an openPMD file */
        distance_to_eb, /**< Only used with embedded boundaries (EB). Stores the distance to the nearest EB */
        edge_lengths,   /**< Only used with the ECT solver. Indicates the length of the cell edge that is covered by the EB, in SI units */
        face_areas,     /**< Only used with the ECT solver. Indicates the area of the cell face that is covered by the EB, in SI units */
        area_mod,
        pml_E_fp,
        pml_B_fp,
        pml_j_fp,
        pml_F_fp,
        pml_G_fp,
        pml_E_cp,
        pml_B_cp,
        pml_j_cp,
        pml_F_cp,
        pml_G_cp,
        pml_edge_lengths,
        Efield_avg_fp,
        Bfield_avg_fp,
        Efield_avg_cp,
        Bfield_avg_cp,
        B_old, /**< Stores the value of B at the beginning of the timestep, for the implicit solver */
        ECTRhofield,
        Venl
    );

    /** these are vector fields */
    constexpr FieldType ArrayFieldTypes[] = {
        FieldType::Efield_aux,
        FieldType::Bfield_aux,
        FieldType::Efield_fp,
        FieldType::Bfield_fp,
        FieldType::current_fp,
        FieldType::current_fp_nodal,
        FieldType::current_fp_vay,
        FieldType::current_buf,
        FieldType::current_store,
        FieldType::vector_potential_fp,
        FieldType::vector_potential_fp_nodal,
        FieldType::vector_potential_grad_buf_e_stag,
        FieldType::vector_potential_grad_buf_b_stag,
        FieldType::hybrid_current_fp_temp,
        FieldType::hybrid_current_fp_plasma,
        FieldType::hybrid_current_fp_external,
        FieldType::Efield_cp,
        FieldType::Bfield_cp,
        FieldType::current_cp,
        FieldType::Efield_cax,
        FieldType::Bfield_cax,
        FieldType::E_external_particle_field,
        FieldType::B_external_particle_field,
        FieldType::pml_E_fp,
        FieldType::pml_B_fp,
        FieldType::pml_j_fp,
        FieldType::pml_E_cp,
        FieldType::pml_B_cp,
        FieldType::pml_j_cp,
        FieldType::Efield_avg_fp,
        FieldType::Bfield_avg_fp,
        FieldType::Efield_avg_cp,
        FieldType::Bfield_avg_cp,
        FieldType::B_old,
        FieldType::ECTRhofield,
        FieldType::Venl
    };

    /** Returns true if a FieldType represents a vector field */
    inline bool
    isFieldArray (const FieldType field_type)
    {
        return std::any_of( std::begin(ArrayFieldTypes), std::end(ArrayFieldTypes),
            [field_type](const FieldType& f) { return f == field_type; });
    }

}

#endif //WARPX_FIELDS_H_
