# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
# Manufactured material interface, not tungsten-data calibration.
# The exact analytic control uses declared mass/charge in double precision.
# Float particle metadata rounds the carrier ratio differently from that parser.
if(NOT WarpX_PRECISION STREQUAL DOUBLE OR
   NOT WarpX_PARTICLE_PRECISION STREQUAL DOUBLE OR NOT WarpX_LIB)
    return()
endif()

set(table_fixture_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/material_opacity_runtime_fixtures")
set(native_material_input inputs_base_1d_radiation_transport_coupled_material)
set(native_material_base
    "materials.names=tungsten materials.tungsten.species=ions materials.tungsten.opacity_material_id=1001 radiation_transport.require_material_metadata_consistency=1")
foreach(model IN ITEMS constant nonlinear)
    if(model STREQUAL constant)
        set(table_file material_w_lte.h5)
        set(table_key manufactured-w-lte)
        set(coefficient_factor "1")
    else()
        set(table_file material_w_nonlinear.h5)
        set(table_key manufactured-w-nonlinear)
        set(coefficient_factor "sqrt(ne*183.84*amu/200)*(Te/10000)^0.25")
    endif()
    set(table_args "${native_material_base} materials.tungsten.opacity_table_file='${table_fixture_dir}/${table_file}' materials.tungsten.opacity_material_key=${table_key}")
    set(analytic_args "radiation_transport.planck_absorption_coefficient(x,y,z,t,photon_energy,ne,Te)='3.162277660168379*ne*183.84*amu/20*if(photon_energy<2.e-19,1,10)*(${coefficient_factor})' radiation_transport.rosseland_transport_coefficient(x,y,z,t,photon_energy,ne,Te)='6.324555320336758*ne*183.84*amu/20*if(photon_energy<2.e-19,1,10)*(${coefficient_factor})'")
    set(control test_1d_radiation_transport_coupled_material_${model}_analytic)
    set(native test_1d_radiation_transport_coupled_material_${model})
    add_warpx_test(${control} 1 1 "${native_material_input} radiation_transport.absorption_coefficient=0 ${analytic_args}"
        "analysis_coupled_material.py" OFF OFF)
    add_warpx_test(${native} 1 1 "${native_material_input} ${table_args}"
        "analysis_coupled_material.py --reference ../${control}" OFF ${control})
    set_tests_properties(${native}.run PROPERTIES FIXTURES_REQUIRED material_opacity_runtime_fixtures)
    if(model STREQUAL nonlinear)
        add_warpx_test(${native}_mpi 1 2 "${native_material_input} ${table_args}"
            "analysis_coupled_material.py --reference ../${native}" OFF ${native})
        if(TEST ${native}_mpi.run)
            set_tests_properties(${native}_mpi.run PROPERTIES
                FIXTURES_REQUIRED material_opacity_runtime_fixtures)
        endif()
        add_warpx_test(${native}_restart 1 1
            "${native_material_input} ${table_args} amr.restart='../${native}/diags/chk000010'"
            "analysis_coupled_material.py --restart --reference ../${native}" OFF ${native})
        set_tests_properties(${native}_restart.run PROPERTIES
            FIXTURES_REQUIRED material_opacity_runtime_fixtures)
        foreach(rejected IN ITEMS no_audit moving out_of_table legacy_handle)
            if(rejected STREQUAL no_audit)
                set(overrides "radiation_transport.require_material_metadata_consistency=0")
                set(reason "coupled_implicit material opacity requires")
            elseif(rejected STREQUAL moving)
                set(overrides "ions.do_not_push=0")
                set(reason "coupled_implicit currently requires stationary")
            elseif(rejected STREQUAL out_of_table)
                set(overrides "hybrid_pic_model.initial_elec_temp(x,y,z)='1.e6*kb/q_e'")
                set(reason "Coupled implicit radiation/material solve failed")
            else()
                set(overrides "radiation_transport.material_opacity_table_file='${table_fixture_dir}/${table_file}' radiation_transport.material_opacity_table_id=1001")
                set(reason "coupled_implicit material opacity requires")
            endif()
            set(reject_name ${native}_reject_${rejected})
            add_radiation_failure_test(${reject_name} 1
                "${native_material_input} ${table_args} ${overrides}" "${reason}")
            set_tests_properties(${reject_name}.run PROPERTIES
                FIXTURES_REQUIRED material_opacity_runtime_fixtures)
        endforeach()
    endif()
endforeach()
