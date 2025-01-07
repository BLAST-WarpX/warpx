/* Copyright 2021-2023 The WarpX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyWarpX.H"
#include "callbacks.H"

#include <WarpX.H>  // todo: move this out to Python/WarpX.cpp
#include <Utils/WarpXUtil.H>  // todo: move to its own Python/Utils.cpp
#include <Utils/WarpXVersion.H>
#include <Initialization/WarpXAMReXInit.H>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#define CONCAT_NAME(PRE, SUF) PRE ## SUF

// see: CMakeLists.txt, setup.py and __init__.py
#if defined(WARPX_DIM_1D_Z)
#  define PYWARPX_MODULE_NAME CONCAT_NAME(warpx_pybind_, 1d)
#elif defined(WARPX_DIM_XZ)
#  define PYWARPX_MODULE_NAME CONCAT_NAME(warpx_pybind_, 2d)
#elif defined(WARPX_DIM_RZ)
#  define PYWARPX_MODULE_NAME CONCAT_NAME(warpx_pybind_, rz)
#elif defined(WARPX_DIM_3D)
#  define PYWARPX_MODULE_NAME CONCAT_NAME(warpx_pybind_, 3d)
#endif

//using namespace warpx;


// forward declarations of exposed classes
void init_BoundaryBufferParIter (py::module&);
void init_MultiParticleContainer (py::module&);
void init_MultiFabRegister (py::module&);
void init_ParticleBoundaryBuffer (py::module&);
void init_PinnedMemoryParticleContainer (py::module&);
void init_WarpXParIter (py::module&);
void init_WarpXParticleContainer (py::module&);
void init_WarpX(py::module&);

PYBIND11_MODULE(PYWARPX_MODULE_NAME, m) {
    // make sure AMReX types are known
#if defined(WARPX_DIM_3D)
    auto amr = py::module::import("amrex.space3d");
#elif defined(WARPX_DIM_1D_Z)
    auto amr = py::module::import("amrex.space1d");
#else
    auto amr = py::module::import("amrex.space2d");
#endif

    m.doc() = R"pbdoc(
            warpx_pybind
            --------------
            .. currentmodule:: warpx_pybind_(1d|2d|3d|rz)

            .. autosummary::
               :toctree: _generate
               WarpX
    )pbdoc";

    // note: order from parent to child classes
    init_MultiFabRegister(m);
    init_PinnedMemoryParticleContainer(m);
    init_WarpXParticleContainer(m);
    init_WarpXParIter(m);
    init_BoundaryBufferParIter(m);
    init_ParticleBoundaryBuffer(m);
    init_MultiParticleContainer(m);
    init_WarpX(m);

    // expose our amrex module
    m.attr("amr") = amr;

    // API runtime version
    //   note PEP-440 syntax: x.y.zaN but x.y.z.devN
#ifdef PYWARPX_VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(PYWARPX_VERSION_INFO);
#else
    // note: not necessarily PEP-440 compliant
    m.attr("__version__") = warpx::Version();
#endif

    // authors
    m.attr("__author__") =
        "Jean-Luc Vay, David P. Grote, Maxence Thevenet, Remi Lehe, Andrew Myers, Weiqun Zhang, Axel Huebl, et al.";

    // API runtime build-time feature variants
    // m.attr("variants") = warpx::getVariants();
    // TODO allow to query runtime versions of all dependencies

    // license SPDX identifier
    m.attr("__license__") = "BSD-3-Clause-LBNL";

    // TODO broken numpy if not at least v1.15.0: raise warning
    // auto numpy = py::module::import("numpy");
    // auto npversion = numpy.attr("__version__");
    // std::cout << "numpy version: " << py::str(npversion) << std::endl;

    m.def("amrex_init",
        [](const py::list args) {
            amrex::Vector<std::string> cargs;
            amrex::Vector<char*> argv;

            // Populate the "command line"
            for (const auto& v: args)
                cargs.push_back(v.cast<std::string>());
            for (auto& v: cargs)
                argv.push_back(&v[0]);
            int argc = argv.size();

            // note: +1 since there is an extra char-string array element,
            //       that ANSII C requires to be a simple NULL entry
            //       https://stackoverflow.com/a/39096006/2719194
            argv.push_back(NULL);
            char** tmp = argv.data();

            const bool build_parm_parse = (cargs.size() > 1);
            // TODO: handle version with MPI
            return warpx::initialization::amrex_init(argc, tmp, build_parm_parse);
        }, py::return_value_policy::reference,
        "Initialize AMReX library");
    m.def("amrex_finalize", [] () { amrex::Finalize(); },
        "Close out the amrex related data");

    // Expose functions to get the processor number
    m.def("getNProcs", [](){return amrex::ParallelDescriptor::NProcs();} );
    m.def("getMyProc", [](){return amrex::ParallelDescriptor::MyProc();} );

    // Expose the python callback function installation and removal functions
    m.def("add_python_callback", &InstallPythonCallback);
    m.def("remove_python_callback", &ClearPythonCallback);
    m.def("execute_python_callback", &ExecutePythonCallback, py::arg("name"));
}
