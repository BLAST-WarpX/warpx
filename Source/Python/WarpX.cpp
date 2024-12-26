/* Copyright 2021-2022 The WarpX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyWarpX.H"

#include <WarpX.H>
// see WarpX.cpp - full includes for _fwd.H headers
#include <BoundaryConditions/PEC_Insulator.H>
#include <BoundaryConditions/PML.H>
#include <Diagnostics/MultiDiagnostics.H>
#include <Diagnostics/ReducedDiags/MultiReducedDiags.H>
#include <EmbeddedBoundary/WarpXFaceInfoBox.H>
#include <FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H>
#include <FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H>
#include <FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H>
#ifdef WARPX_USE_FFT
#   include <FieldSolver/SpectralSolver/SpectralKSpace.H>
#   ifdef WARPX_DIM_RZ
#       include <FieldSolver/SpectralSolver/SpectralSolverRZ.H>
#       include <BoundaryConditions/PML_RZ.H>
#   else
#       include <FieldSolver/SpectralSolver/SpectralSolver.H>
#   endif // RZ ifdef
#endif // use PSATD ifdef
#include <FieldSolver/WarpX_FDTD.H>
#include <Filter/NCIGodfreyFilter.H>
#include <Initialization/ExternalField.H>
#include <Particles/MultiParticleContainer.H>
#include <Fluids/MultiFluidContainer.H>
#include <Fluids/WarpXFluidContainer.H>
#include <Particles/ParticleBoundaryBuffer.H>
#include <AcceleratorLattice/AcceleratorLattice.H>
#include <Utils/TextMsg.H>
#include <Utils/WarpXAlgorithmSelection.H>
#include <Utils/WarpXConst.H>
#include <Utils/WarpXProfilerWrapper.H>
#include <Utils/WarpXUtil.H>
#include "FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.H"
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>

#if defined(AMREX_DEBUG) || defined(DEBUG)
#   include <cstdio>
#endif
#include <string>


//using namespace warpx;

namespace warpx {
    struct Config {};
}

void init_WarpX (py::module& m)
{
    using ablastr::fields::Direction;

    // Expose the WarpX instance
    m.def("get_instance",
        [] () { return &WarpX::GetInstance(); },
        "Return a reference to the WarpX object.");

    m.def("finalize", &WarpX::Finalize,
        "Close out the WarpX related data");

    py::class_<WarpX> warpx(m, "WarpX");
    warpx
        // WarpX is a Singleton Class with a private constructor
        //   https://github.com/ECP-WarpX/WarpX/pull/4104
        //   https://pybind11.readthedocs.io/en/stable/advanced/classes.html?highlight=singleton#custom-constructors
        .def(py::init([]() {
            return &WarpX::GetInstance();
        }))
        .def_static("get_instance",
            [] () { return &WarpX::GetInstance(); },
            "Return a reference to the WarpX object."
        )
        .def_static("finalize", &WarpX::Finalize,
            "Close out the WarpX related data"
        )

        .def("initialize_data", &WarpX::InitData,
            "Initializes the WarpX simulation"
        )
        .def("evolve", &WarpX::Evolve,
            "Evolve the simulation the specified number of steps"
        )

        // from amrex::AmrCore / amrex::AmrMesh
        .def_property_readonly("max_level",
            [](WarpX const & wx){ return wx.maxLevel(); },
            "The maximum mesh-refinement level for the simulation."
        )
        .def_property_readonly("finest_level",
            [](WarpX const & wx){ return wx.finestLevel(); },
            "The currently finest level of mesh-refinement used. This is always less or equal to max_level."
        )
        .def("Geom",
            //[](WarpX const & wx, int const lev) { return wx.Geom(lev); },
            py::overload_cast< int >(&WarpX::Geom, py::const_),
            py::arg("lev")
        )
        .def("DistributionMap",
            [](WarpX const & wx, int const lev) { return wx.DistributionMap(lev); },
            //py::overload_cast< int >(&WarpX::DistributionMap, py::const_),
            py::arg("lev")
        )
        .def("boxArray",
            [](WarpX const & wx, int const lev) { return wx.boxArray(lev); },
            //py::overload_cast< int >(&WarpX::boxArray, py::const_),
            py::arg("lev")
        )
        .def("multifab",
             [](WarpX & wx, std::string internal_name) {
                 if (wx.m_fields.internal_has(internal_name)) {
                     return wx.m_fields.internal_get(internal_name);
                 } else {
                     throw std::runtime_error("MultiFab '" + internal_name + "' is unknown or is not allocated!");
                 }
             },
             py::arg("internal_name"),
             py::return_value_policy::reference_internal,
             R"doc(Return a MultiFab by its internal name (deprecated).

The multifab('internal_name') signature is deprecated.
Please use:
- multifab('prefix', level=...) for scalar fields
- multifab('prefix', dir=..., level=...) for vector field components
where 'prefix' is the part of 'internal_name';'  before the [])doc"
        )
        .def("multifab",
            [](WarpX & wx, std::string scalar_name, int level) {
                if (wx.m_fields.has(scalar_name, level)) {
                    return wx.m_fields.get(scalar_name, level);
                } else {
                    throw std::runtime_error("The scalar field '" + scalar_name + "' is unknown or is not allocated!");
                }
            },
            py::arg("scalar_name"),
            py::arg("level"),
            py::return_value_policy::reference_internal,
            R"doc(Return scalar fields (MultiFabs) by name and level. The name is in the form like``\"rho_fp\"``, ``\"phi_fp"``. The level is an integer with 0 being the lowest level.

The physical fields in WarpX have the following naming:

- ``_fp`` are the "fine" patches, the regular resolution of a current mesh-refinement level
- ``_aux`` are temporary (auxiliar) patches at the same resolution as ``_fp``.
  They usually include contributions from other levels and can be interpolated for gather routines of particles.
- ``_cp`` are "coarse" patches, at the same resolution (but not necessary values) as the ``_fp`` of ``level - 1``
  (only for level 1 and higher).)doc"
        )
        .def("multifab",
            [](WarpX & wx, std::string vector_name, Direction dir, int level) {
                if (wx.m_fields.has(vector_name, dir, level)) {
                    return wx.m_fields.get(vector_name, dir, level);
                } else {
                    throw std::runtime_error("The vector field '" + vector_name + "' is unknown or is not allocated!");
                }
            },
            py::arg("vector_name"),
            py::arg("dir"),
            py::arg("level"),
            py::return_value_policy::reference_internal,
            R"doc(Return the component of a vector field (MultiFab) by name, direction, and level. The name is in the form like ``\"Efield_aux\"``, ``\"Efield_fp"``, etc. The direction is a Direction instance, Direction(idir) where idir is an integer 0, 1, or 2. The level is an integer with 0 being the lowest level.

The physical fields in WarpX have the following naming:

- ``_fp`` are the "fine" patches, the regular resolution of a current mesh-refinement level
- ``_aux`` are temporary (auxiliar) patches at the same resolution as ``_fp``.
  They usually include contributions from other levels and can be interpolated for gather routines of particles.
- ``_cp`` are "coarse" patches, at the same resolution (but not necessary values) as the ``_fp`` of ``level - 1``
  (only for level 1 and higher).)doc"
        )
        .def("multifab",
            [](WarpX & wx, std::string vector_name, int idir, int level) {
                Direction const dir{idir};
                if (wx.m_fields.has(vector_name, dir, level)) {
                    return wx.m_fields.get(vector_name, dir, level);
                } else {
                    throw std::runtime_error("The vector field '" + vector_name + "' is unknown or is not allocated!");
                }
            },
            py::arg("vector_name"),
            py::arg("idir"),
            py::arg("level"),
            py::return_value_policy::reference_internal,
            R"doc(Return the component of a vector field (MultiFab) by name, direction, and level. The name is in the form like ``\"Efield_aux\"``, ``\"Efield_fp"``, etc. The direction is an integer 0, 1, or 2. The level is an integer with 0 being the lowest level.

The physical fields in WarpX have the following naming:

- ``_fp`` are the "fine" patches, the regular resolution of a current mesh-refinement level
- ``_aux`` are temporary (auxiliar) patches at the same resolution as ``_fp``.
  They usually include contributions from other levels and can be interpolated for gather routines of particles.
- ``_cp`` are "coarse" patches, at the same resolution (but not necessary values) as the ``_fp`` of ``level - 1``
  (only for level 1 and higher).)doc"
        )
        .def("multi_particle_container",
            [](WarpX& wx){ return &wx.GetPartContainer(); },
            py::return_value_policy::reference_internal
        )
        .def("get_particle_boundary_buffer",
            [](WarpX& wx){ return &wx.GetParticleBoundaryBuffer(); },
            py::return_value_policy::reference_internal
        )

        // Expose functions used to sync the charge density multifab
        // accross tiles and apply appropriate boundary conditions
        .def("sync_rho",
            [](WarpX& wx){ wx.SyncRho(); }
        )
#ifdef WARPX_DIM_RZ
        .def("apply_inverse_volume_scaling_to_charge_density",
            [](WarpX& wx, amrex::MultiFab* rho, int const lev) {
                wx.ApplyInverseVolumeScalingToChargeDensity(rho, lev);
            },
            py::arg("rho"), py::arg("lev")
        )
#endif

        // Expose functions to get the current simulation step and time
        .def("getistep",
            [](WarpX const & wx, int lev){ return wx.getistep(lev); },
            py::arg("lev"),
            "Get the current step on mesh-refinement level ``lev``."
        )
        .def("gett_new",
            [](WarpX const & wx, int lev){ return wx.gett_new(lev); },
            py::arg("lev"),
            "Get the current physical time on mesh-refinement level ``lev``."
        )
        .def("getdt",
            [](WarpX const & wx, int lev){ return wx.getdt(lev); },
            py::arg("lev"),
            "Get the current physical time step size on mesh-refinement level ``lev``."
        )

        .def("set_potential_on_domain_boundary",
            [](WarpX& wx,
               std::string potential_lo_x, std::string potential_hi_x,
               std::string potential_lo_y, std::string potential_hi_y,
               std::string potential_lo_z, std::string potential_hi_z)
            {
                if (potential_lo_x != "") wx.GetElectrostaticSolver().m_poisson_boundary_handler->potential_xlo_str = potential_lo_x;
                if (potential_hi_x != "") wx.GetElectrostaticSolver().m_poisson_boundary_handler->potential_xhi_str = potential_hi_x;
                if (potential_lo_y != "") wx.GetElectrostaticSolver().m_poisson_boundary_handler->potential_ylo_str = potential_lo_y;
                if (potential_hi_y != "") wx.GetElectrostaticSolver().m_poisson_boundary_handler->potential_yhi_str = potential_hi_y;
                if (potential_lo_z != "") wx.GetElectrostaticSolver().m_poisson_boundary_handler->potential_zlo_str = potential_lo_z;
                if (potential_hi_z != "") wx.GetElectrostaticSolver().m_poisson_boundary_handler->potential_zhi_str = potential_hi_z;
                wx.GetElectrostaticSolver().m_poisson_boundary_handler->BuildParsers();
            },
            py::arg("potential_lo_x") = "",
            py::arg("potential_hi_x") = "",
            py::arg("potential_lo_y") = "",
            py::arg("potential_hi_y") = "",
            py::arg("potential_lo_z") = "",
            py::arg("potential_hi_z") = "",
            "Sets the domain boundary potential string(s) and updates the function parser."
        )
        .def("set_potential_on_eb",
            [](WarpX& wx, std::string potential) {
                wx.GetElectrostaticSolver().m_poisson_boundary_handler->setPotentialEB(potential);
            },
            py::arg("potential"),
            "Sets the EB potential string and updates the function parser."
        )
        .def_static("run_div_cleaner",
            [] () { WarpX::ProjectionCleanDivB(); },
            "Executes projection based divergence cleaner on loaded Bfield_fp_external."
        )
        .def("synchronize",
            [] (WarpX& wx) { wx.Synchronize(); },
            "Synchronize particle velocities and positions."
        )
    ;

    py::class_<warpx::Config>(m, "Config")
//        .def_property_readonly_static(
//            "warpx_version",
//            [](py::object) { return Version(); },
//            "WarpX version")
        .def_property_readonly_static(
            "have_mpi",
            [](py::object){
#ifdef AMREX_USE_MPI
                return true;
#else
                return false;
#endif
            })
        .def_property_readonly_static(
            "have_gpu",
            [](py::object){
#ifdef AMREX_USE_GPU
                return true;
#else
                return false;
#endif
            })
        .def_property_readonly_static(
            "have_omp",
            [](py::object){
#ifdef AMREX_USE_OMP
                return true;
#else
                return false;
#endif
            })
        .def_property_readonly_static(
            "gpu_backend",
            [](py::object){
#ifdef AMREX_USE_CUDA
                return "CUDA";
#elif defined(AMREX_USE_HIP)
                return "HIP";
#elif defined(AMREX_USE_DPCPP)
                return "SYCL";
#else
                return py::none();
#endif
            })
        ;
}
