import os
import platform
import re
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build import build
from setuptools.command.build_ext import build_ext


class CopyPreBuild(build):
    def initialize_options(self):
        build.initialize_options(self)
        # We just overwrite this because the default "build" (and "build/lib")
        # clashes with directories many developers have in their source trees;
        # this can create confusing results with "pip install .", which clones
        # the whole source tree by default
        self.build_base = os.path.join("_tmppythonbuild", "warpx")

    def run(self):
        # remove existing build directory
        #   by default, this stays around. we want to make sure generated
        #   files like libwarpx.(1d|2d|rz|3d).(so|pyd) are always only the
        #   ones we want to package and not ones from an earlier wheel's stage
        if os.path.exists(self.build_base):
            shutil.rmtree(self.build_base)

        # call superclass
        build.run(self)

        # matches: warpx_pybind_(1d|2d|rz|3d). ... .(so|pyd)
        re_libprefix = re.compile(r"warpx_pybind_..\..*\.(?:so|pyd)")
        libs_found = []
        for lib_name in os.listdir(PYWARPX_LIB_DIR):
            if re_libprefix.match(lib_name):
                lib_path = os.path.join(PYWARPX_LIB_DIR, lib_name)
                libs_found.append(lib_path)
        if len(libs_found) == 0:
            raise RuntimeError(
                "Error: no pre-build WarpX modules found in "
                "PYWARPX_LIB_DIR='{}'".format(PYWARPX_LIB_DIR)
            )

        # copy external libs into collection of files in a temporary build dir
        dst_path = os.path.join(self.build_lib, "pywarpx")
        for lib_path in libs_found:
            shutil.copy(lib_path, dst_path)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        from packaging.version import parse

        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake 3.24.0+ must be installed to build the following "
                + "extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        cmake_version = parse(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
        if cmake_version < parse("3.24.0"):
            raise RuntimeError("CMake >= 3.24.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        r_dim = re.search(r"warpx_(1|2|rz|3)(?:d*)", ext.name)
        dims = r_dim.group(1).upper()

        pyv = sys.version_info
        cmake_args = [
            # Python: use the calling interpreter in CMake
            # https://cmake.org/cmake/help/latest/module/FindPython.html#hints
            # https://cmake.org/cmake/help/latest/command/find_package.html#config-mode-version-selection
            f"-DPython_ROOT_DIR={sys.prefix}",
            f"-DPython_FIND_VERSION={pyv.major}.{pyv.minor}.{pyv.micro}",
            "-DPython_FIND_VERSION_EXACT=TRUE",
            "-DPython_FIND_STRATEGY=LOCATION",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + os.path.join(extdir, "pywarpx"),
            "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=" + extdir,
            "-DWarpX_DIMS=" + dims,
            "-DWarpX_APP:BOOL=OFF",
            ## variants
            "-DWarpX_COMPUTE=" + WARPX_COMPUTE,
            "-DWarpX_MPI:BOOL=" + WARPX_MPI,
            "-DWarpX_EB:BOOL=" + WARPX_EB,
            "-DWarpX_OPENPMD:BOOL=" + WARPX_OPENPMD,
            "-DWarpX_PRECISION=" + WARPX_PRECISION,
            "-DWarpX_PARTICLE_PRECISION=" + WARPX_PARTICLE_PRECISION,
            "-DWarpX_FFT:BOOL=" + WARPX_FFT,
            "-DWarpX_PYTHON:BOOL=ON",
            "-DWarpX_PYTHON_IPO:BOOL=" + WARPX_PYTHON_IPO,
            "-DWarpX_QED:BOOL=" + WARPX_QED,
            "-DWarpX_QED_TABLE_GEN:BOOL=" + WARPX_QED_TABLE_GEN,
            ## dependency control (developers & package managers)
            "-DWarpX_amrex_internal=" + WARPX_AMREX_INTERNAL,
            # PEP-440 conformant version from package
            "-DpyWarpX_VERSION_INFO=" + self.distribution.get_version(),
            #        see PICSAR and openPMD below
            ## static/shared libs
            "-DBUILD_SHARED_LIBS:BOOL=" + BUILD_SHARED_LIBS,
            ## Unix: rpath to current dir when packaged
            ##       needed for shared (here non-default) builds and ADIOS1
            ##       wrapper libraries
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=ON",
            "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=OFF",
            # Windows: has no RPath concept, all `.dll`s must be in %PATH%
            #          or same dir as calling executable
        ]
        if WARPX_QED.upper() in ["1", "ON", "TRUE", "YES"]:
            cmake_args.append("-DWarpX_picsar_internal=" + WARPX_PICSAR_INTERNAL)
        if WARPX_OPENPMD.upper() in ["1", "ON", "TRUE", "YES"]:
            cmake_args += [
                "-DHDF5_USE_STATIC_LIBRARIES:BOOL=" + HDF5_USE_STATIC_LIBRARIES,
                "-DADIOS_USE_STATIC_LIBS:BOOL=" + ADIOS_USE_STATIC_LIBS,
                "-DWarpX_openpmd_internal=" + WARPX_OPENPMD_INTERNAL,
            ]
        # further dependency control (developers & package managers)
        if WARPX_AMREX_SRC:
            cmake_args.append("-DWarpX_amrex_src=" + WARPX_AMREX_SRC)
        if WARPX_AMREX_REPO:
            cmake_args.append("-DWarpX_amrex_repo=" + WARPX_AMREX_REPO)
        if WARPX_AMREX_BRANCH:
            cmake_args.append("-DWarpX_amrex_branch=" + WARPX_AMREX_BRANCH)
        if WARPX_OPENPMD_SRC:
            cmake_args.append("-DWarpX_openpmd_src=" + WARPX_OPENPMD_SRC)
        if WARPX_PICSAR_SRC:
            cmake_args.append("-DWarpX_picsar_src=" + WARPX_PICSAR_SRC)
        if WARPX_PYAMREX_SRC:
            cmake_args.append("-DWarpX_pyamrex_src=" + WARPX_PYAMREX_SRC)
        if WARPX_PYAMREX_INTERNAL:
            cmake_args.append("-DWarpX_pyamrex_internal=" + WARPX_PYAMREX_INTERNAL)
        if WARPX_PYBIND11_SRC:
            cmake_args.append("-DWarpX_pybind11_src=" + WARPX_PYBIND11_SRC)
        if WARPX_PYBIND11_INTERNAL:
            cmake_args.append("-DWarpX_pybind11_internal=" + WARPX_PYBIND11_INTERNAL)
        if WARPX_CCACHE_PROGRAM is not None:
            cmake_args.append("-DCCACHE_PROGRAM=" + WARPX_CCACHE_PROGRAM)

        if sys.platform == "darwin":
            cmake_args.append("-DCMAKE_INSTALL_RPATH=@loader_path")
        else:
            # values: linux*, aix, freebsd, ...
            #   just as well win32 & cygwin (although Windows has no RPaths)
            cmake_args.append("-DCMAKE_INSTALL_RPATH=$ORIGIN")

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(
                    cfg.upper(), os.path.join(extdir, "pywarpx")
                )
            ]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]

        build_args += ["--parallel", BUILD_PARALLEL]

        build_dir = os.path.join(self.build_temp, dims)
        os.makedirs(build_dir, exist_ok=True)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_dir)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_dir)
        # note that this does not call install;
        # we pick up artifacts directly from the build output dirs


with open("./README.md", encoding="utf-8") as f:
    long_description = f.read()

# Allow to control options via environment vars.
#   Work-around for https://github.com/pypa/setuptools/issues/1712
# Pick up existing WarpX libraries or...
PYWARPX_LIB_DIR = os.environ.get("PYWARPX_LIB_DIR")

WARPX_PYTHON_IPO = os.environ.get("WARPX_PYTHON_IPO", "ON")

env = os.environ.copy()
# ... build WarpX libraries with CMake
#   note: changed default for SHARED, MPI, TESTING and EXAMPLES
#   note: we use all-uppercase variable names for environment control to be
#         consistent across platforms (especially Windows)
WARPX_COMPUTE = env.pop("WARPX_COMPUTE", "OMP")
WARPX_MPI = env.pop("WARPX_MPI", "OFF")
WARPX_EB = env.pop("WARPX_EB", "ON")
WARPX_OPENPMD = env.pop("WARPX_OPENPMD", "ON")
WARPX_PRECISION = env.pop("WARPX_PRECISION", "DOUBLE")
WARPX_PARTICLE_PRECISION = env.pop("WARPX_PARTICLE_PRECISION", WARPX_PRECISION)
WARPX_FFT = env.pop("WARPX_FFT", "OFF")
WARPX_QED = env.pop("WARPX_QED", "ON")
WARPX_QED_TABLE_GEN = env.pop("WARPX_QED_TABLE_GEN", "OFF")
WARPX_DIMS = env.pop("WARPX_DIMS", "1;2;RZ;3")
BUILD_PARALLEL = env.pop("BUILD_PARALLEL", "2")
BUILD_SHARED_LIBS = env.pop("WARPX_BUILD_SHARED_LIBS", "OFF")
# BUILD_TESTING = env.pop('WARPX_BUILD_TESTING',
#                               'OFF')
# BUILD_EXAMPLES = env.pop('WARPX_BUILD_EXAMPLES',
#                                'OFF')
# openPMD-api sub-control
HDF5_USE_STATIC_LIBRARIES = env.pop("HDF5_USE_STATIC_LIBRARIES", "OFF")
ADIOS_USE_STATIC_LIBS = env.pop("ADIOS_USE_STATIC_LIBS", "OFF")
# CMake dependency control (developers & package managers)
WARPX_AMREX_SRC = env.pop("WARPX_AMREX_SRC", "")
WARPX_AMREX_REPO = env.pop("WARPX_AMREX_REPO", "")
WARPX_AMREX_BRANCH = env.pop("WARPX_AMREX_BRANCH", "")
WARPX_AMREX_INTERNAL = env.pop("WARPX_AMREX_INTERNAL", "ON")
WARPX_OPENPMD_SRC = env.pop("WARPX_OPENPMD_SRC", "")
WARPX_OPENPMD_INTERNAL = env.pop("WARPX_OPENPMD_INTERNAL", "ON")
WARPX_PICSAR_SRC = env.pop("WARPX_PICSAR_SRC", "")
WARPX_PICSAR_INTERNAL = env.pop("WARPX_PICSAR_INTERNAL", "ON")
WARPX_PYAMREX_SRC = env.pop("WARPX_PYAMREX_SRC", "")
WARPX_PYAMREX_INTERNAL = env.pop("WARPX_PYAMREX_INTERNAL", "ON")
WARPX_PYBIND11_SRC = env.pop("WARPX_PYBIND11_SRC", "")
WARPX_PYBIND11_INTERNAL = env.pop("WARPX_PYBIND11_INTERNAL", "ON")
WARPX_CCACHE_PROGRAM = env.pop("WARPX_CCACHE_PROGRAM", None)

for key in env.keys():
    if key.lower().startswith("warpx"):
        print(
            f"\nWARNING: Found environment variable '{key}', which is not a recognized WarpX option\n"
        )


# https://cmake.org/cmake/help/v3.0/command/if.html
if WARPX_MPI.upper() in ["1", "ON", "TRUE", "YES"]:
    WARPX_MPI = "ON"
else:
    WARPX_MPI = "OFF"

# Include embedded boundary functionality
if WARPX_EB.upper() in ["1", "ON", "TRUE", "YES"]:
    WARPX_EB = "ON"
else:
    WARPX_EB = "OFF"


# for CMake
cxx_modules = []  # values: warpx_1d, warpx_2d, warpx_rz, warpx_3d
cmdclass = {}  # build extensions

# externally pre-built: pick up pre-built WarpX libraries
if PYWARPX_LIB_DIR:
    cmdclass = dict(build=CopyPreBuild)
# CMake: build WarpX libraries ourselves
else:
    cmdclass = dict(build_ext=CMakeBuild)
    for dim in [x.lower() for x in WARPX_DIMS.split(";")]:
        name = dim if dim == "rz" else dim + "d"
        cxx_modules.append(CMakeExtension("warpx_" + name))

# Get the package requirements from the requirements.txt file
install_requires = []
with open("./requirements.txt") as f:
    install_requires = [line.strip("\n") for line in f.readlines()]
    if WARPX_MPI == "ON":
        install_requires.append("mpi4py>=2.1.0")

# keyword reference:
#   https://packaging.python.org/guides/distributing-packages-using-setuptools
setup(
    name="pywarpx",
    # note PEP-440 syntax: x.y.zaN but x.y.z.devN
    version="25.01",
    packages=["pywarpx"],
    package_dir={"pywarpx": "Python/pywarpx"},
    author="Jean-Luc Vay, David P. Grote, Maxence Thévenet, Rémi Lehe, Andrew Myers, Weiqun Zhang, Axel Huebl, et al.",
    author_email="jlvay@lbl.gov, grote1@llnl.gov, maxence.thevenet@desy.de, rlehe@lbl.gov, atmyers@lbl.gov, WeiqunZhang@lbl.gov, axelhuebl@lbl.gov",
    maintainer="Axel Huebl, David P. Grote, Rémi Lehe",  # wheel/pypi packages
    maintainer_email="axelhuebl@lbl.gov, grote1@llnl.gov, rlehe@lbl.gov",
    description="WarpX is an advanced electromagnetic Particle-In-Cell code.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=(
        "WarpX openscience mpi hpc research pic particle-in-cell "
        "plasma laser-plasma accelerator modeling simulation"
    ),
    url="https://ecp-warpx.github.io",
    project_urls={
        "Documentation": "https://warpx.readthedocs.io",
        "Doxygen": "https://warpx.readthedocs.io/en/latest/_static/doxyhtml/index.html",
        #'Reference': 'https://doi.org/...', (Paper and/or Zenodo)
        "Source": "https://github.com/ECP-WarpX/WarpX",
        "Tracker": "https://github.com/ECP-WarpX/WarpX/issues",
    },
    # CMake: self-built as extension module
    ext_modules=cxx_modules,
    cmdclass=cmdclass,
    # scripts=['warpx_1d', 'warpx_2d', 'warpx_rz', 'warpx_3d'],
    zip_safe=False,
    python_requires=">=3.8",  # left for CI, truly ">=3.9"
    # tests_require=['pytest'],
    install_requires=install_requires,
    # see: src/bindings/python/cli
    # entry_points={
    #    'console_scripts': [
    #        'warpx_3d = warpx.3d.__main__:main'
    #    ]
    # },
    extras_require={
        "all": [
            "openPMD-api~=0.15.1",
            "openPMD-viewer~=1.1",
            "yt>=4.1.0",
            "matplotlib",
        ],
    },
    # cmdclass={'test': PyTest},
    # platforms='any',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        (
            "License :: OSI Approved :: BSD License"
        ),  # TODO: use real SPDX: BSD-3-Clause-LBNL
    ],
    # new PEP 639 format
    license="BSD-3-Clause-LBNL",
    license_files=["LICENSE.txt", "LEGAL.txt"],
)
