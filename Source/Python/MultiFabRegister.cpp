/* Copyright 2024 The WarpX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "Python/pyWarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_IntVect.H>

void init_MultiFabRegister (py::module & m)
{
    using namespace ablastr::fields;

    py::class_<ablastr::fields::Direction> pyDirection(m, "Direction");
    pyDirection
        .def(py::init<int>())
        .def(py::init<std::string>())
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        .def_property_readonly_static("r", [](py::object /* self */) {
            return ablastr::fields::Direction::r;
        })
#endif
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        .def_property_readonly_static("theta", [](py::object /* self */) {
            return ablastr::fields::Direction::theta;
        })
#endif

#if defined(WARPX_DIM_RSPHERE)
        .def_property_readonly_static("phi", [](py::object /* self */) {
            return ablastr::fields::Direction::phi;
        })
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_1D_Z)
        .def_property_readonly_static("x", [](py::object /* self */) {
            return ablastr::fields::Direction::x;
        })
#endif
#if defined(WARPX_DIM_3D) || defined(WARPX_DIM_XZ) || defined(WARPX_DIM_1D_Z)
        .def_property_readonly_static("y", [](py::object /* self */) {
            return ablastr::fields::Direction::y;
        })
#endif
#if !defined(WARPX_DIM_RSPHERE)
        .def_property_readonly_static("z", [](py::object /* self */) {
            return ablastr::fields::Direction::z;
        })
#endif
    ;
    py::implicitly_convertible<std::string, ablastr::fields::Direction>();

    py::class_<ablastr::fields::MultiFabRegister>(m, "MultiFabRegister")

        .def("alloc_init",
             py::overload_cast<
                 std::string,
                 int,
                 amrex::BoxArray const &,
                 amrex::DistributionMapping const &,
                 int,
                 amrex::IntVect const &,
                 std::optional<const amrex::Real>,
                 bool,
                 bool
             >(&MultiFabRegister::alloc_init<std::string>),
             py::return_value_policy::reference_internal,
             py::arg("name"),
             py::arg("level"),
             py::arg("ba"),
             py::arg("dm"),
             py::arg("ncomp"),
             py::arg("ngrow"),
             py::arg("initial_value"),
             py::arg("redistribute"),
             py::arg("redistribute_on_remake")
        )

        .def("alloc_init",
             py::overload_cast<
                 std::string,
                 ablastr::fields::Direction,
                 int,
                 amrex::BoxArray const &,
                 amrex::DistributionMapping const &,
                 int,
                 amrex::IntVect const &,
                 std::optional<const amrex::Real>,
                 bool,
                 bool
             >(&MultiFabRegister::alloc_init<std::string>),
             py::return_value_policy::reference_internal,
             py::arg("name"),
             py::arg("dir"),
             py::arg("level"),
             py::arg("ba"),
             py::arg("dm"),
             py::arg("ncomp"),
             py::arg("ngrow"),
             py::arg("initial_value"),
             py::arg("redistribute"),
             py::arg("redistribute_on_remake")
        )

        .def("alias_init",
             py::overload_cast<
                 std::string,
                 std::string,
                 int,
                 std::optional<const amrex::Real>
             >(&MultiFabRegister::alias_init<std::string, std::string>),
             py::return_value_policy::reference_internal,
             py::arg("new_name"),
             py::arg("alias_name"),
             py::arg("level"),
             py::arg("initial_value")
        )

        .def("alias_init",
             py::overload_cast<
                 std::string,
                 std::string,
                 ablastr::fields::Direction,
                 int,
                 std::optional<const amrex::Real>
             >(&MultiFabRegister::alias_init<std::string, std::string>),
             py::return_value_policy::reference_internal,
             py::arg("new_name"),
             py::arg("alias_name"),
             py::arg("dir"),
             py::arg("level"),
             py::arg("initial_value")
        )

        .def("has",
             py::overload_cast<
                 std::string,
                 int
             >(&MultiFabRegister::has<std::string>, py::const_),
             py::arg("name"),
             py::arg("level")
        )

        .def("has",
             py::overload_cast<
                 std::string,
                 ablastr::fields::Direction,
                 int
             >(&MultiFabRegister::has<std::string>, py::const_),
             py::arg("name"),
             py::arg("dir"),
             py::arg("level")
        )

        .def("get",
             py::overload_cast<
                 std::string,
                 int
             >(&MultiFabRegister::get<std::string>),
             py::return_value_policy::reference_internal,
             py::arg("name"),
             py::arg("level")
        )

        .def("get",
             py::overload_cast<
                 std::string,
                 ablastr::fields::Direction,
                 int
             >(&MultiFabRegister::get<std::string>),
             py::return_value_policy::reference_internal,
             py::arg("name"),
             py::arg("dir"),
             py::arg("level")
        )

        .def("list",
             &MultiFabRegister::list,
             "List the internal names of all registered fields"
        )

        .def("erase",
             py::overload_cast<
                 std::string,
                 int
             >(&MultiFabRegister::erase<std::string>),
             py::arg("name"),
             py::arg("level")
        )

        .def("erase",
             py::overload_cast<
                 std::string,
                 ablastr::fields::Direction,
                 int
             >(&MultiFabRegister::erase<std::string>),
             py::arg("name"),
             py::arg("dir"),
             py::arg("level")
        )
    ;
}
