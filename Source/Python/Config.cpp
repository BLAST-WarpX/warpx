/* Copyright 2026 The WarpX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyWarpX.H"

#include <WarpX.H>

#include <AMReX.H>
#include <AMReX_SIMD.H>

#ifdef WARPX_USE_OPENPMD
#   include <openPMD/version.hpp>
#endif

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>


namespace warpx {
    struct Config {};
}

namespace
{
    using ConfigValue = std::variant<
        bool,
        int,
        std::string,
        std::map<std::string, bool>,
        std::optional<std::string>
    >;
    struct ConfigEntry
    {
        ConfigValue value;
        char const * doc;
    };
    using ConfigMap = std::map<std::string, ConfigEntry>;

    std::string config_repr (std::string const & module_name, ConfigMap const & config)
    {
        std::size_t name_width = 0;
        for (auto const & entry : config)
        {
            if (entry.first.size() > name_width)
            {
                name_width = entry.first.size();
            }
        }

        std::string repr = module_name + ".Config:";
        for (auto const & [name, entry] : config)
        {
            repr += "\n    " + name;
            repr.append(name_width - name.size(), ' ');
            repr += " = ";
            if (name == "openpmd_backends")
            {
                // show only the enabled backends
                py::list enabled_backends;
                auto const & backends = std::get<std::map<std::string, bool>>(entry.value);
                for (auto const & [backend, enabled] : backends)
                {
                    if (enabled)
                    {
                        enabled_backends.append(backend);
                    }
                }
                repr += py::repr(enabled_backends).cast<std::string>();
            }
            else
            {
                repr += py::repr(py::cast(entry.value)).cast<std::string>();
            }
        }
        return repr;
    }
}

void init_Config (py::module& m)
{
    std::optional<std::string> gpu_backend;
#ifdef AMREX_USE_CUDA
    gpu_backend = "CUDA";
#elif defined(AMREX_USE_HIP)
    gpu_backend = "HIP";
#elif defined(AMREX_USE_DPCPP)
    gpu_backend = "SYCL";
#endif

    std::shared_ptr<ConfigMap const> const config = std::make_shared<ConfigMap>(
        ConfigMap{
            {"amrex_version", {
                amrex::Version(),
                "AMReX library version used to build WarpX"}},

            {"gpu_backend", {
                gpu_backend,
                "GPU backend ('CUDA', 'HIP' or 'SYCL'), None without GPU support"}},

            {"have_fft", {
#ifdef WARPX_USE_FFT
                true,
#else
                false,
#endif
                "Build supports FFT-based (spectral) solvers and features"}},

            {"have_gpu", {
#ifdef AMREX_USE_GPU
                true,
#else
                false,
#endif
                "Build supports GPUs"}},

            {"have_mpi", {
#ifdef AMREX_USE_MPI
                true,
#else
                false,
#endif
                "Build supports MPI"}},

            {"have_omp", {
#ifdef AMREX_USE_OMP
                true,
#else
                false,
#endif
                "Build supports OpenMP"}},

            {"have_openpmd", {
#ifdef WARPX_USE_OPENPMD
                true,
#else
                false,
#endif
                "Build supports openPMD I/O"}},

            {"have_simd", {
#ifdef AMREX_USE_SIMD
                true,
#else
                false,
#endif
                "Build supports explicit SIMD vectorization"}},

            {"openpmd_backends", {
#ifdef WARPX_USE_OPENPMD
                openPMD::getVariants(),
#else
                std::map<std::string, bool>{},
#endif
                "Available openPMD-api backends and if they are enabled"}},

            {"precision", {
#ifdef AMREX_USE_FLOAT
                std::string{"SINGLE"},
#else
                std::string{"DOUBLE"},
#endif
                "Floating point precision of amrex::Real ('SINGLE' or 'DOUBLE')"}},

            {"precision_particles", {
#ifdef AMREX_SINGLE_PRECISION_PARTICLES
                std::string{"SINGLE"},
#else
                std::string{"DOUBLE"},
#endif
                "Floating point precision of amrex::ParticleReal ('SINGLE' or 'DOUBLE')"}},

            {"simd_size", {
                static_cast<int>(amrex::simd::native_simd_size_particlereal),
                "Number of amrex::ParticleReal elements in a native SIMD vector"}},

            {"warpx_version", {
                WarpX::Version(),
                "WarpX version"}}
        }
    );

    // create a custom metaclass deriving from pybind11's metaclass, so that
    // repr(Config) prints the full build configuration in interactive use
    py::dict config_metaclass_namespace;
    config_metaclass_namespace["__module__"] = m.attr("__name__");
    config_metaclass_namespace["__repr__"] = py::cpp_function(
        [config, module_name = py::cast<std::string>(m.attr("__name__"))]() {
            return config_repr(module_name, *config);
        }
    );
    py::object const warpx_class = m.attr("WarpX");
    py::object const pybind11_metaclass = py::type::of(warpx_class);
    py::object const config_metaclass = py::type::of(pybind11_metaclass)(
        "ConfigMeta",
        py::make_tuple(pybind11_metaclass),
        config_metaclass_namespace
    );

    py::class_<warpx::Config> pyWarpXConfig(
        m, "Config", py::metaclass(config_metaclass)
    );
    for (auto const & kv : *config)
    {
        std::string const & name = kv.first;
        ConfigEntry const & entry = kv.second;
        pyWarpXConfig.def_property_readonly_static(
            name.c_str(),
            [config, name](py::object const &) {
                return config->at(name).value;
            },
            entry.doc
        );
    }
    pyWarpXConfig.def_static(
        "to_dict",
        [config]() {
            py::dict d;
            for (auto const & [name, entry] : *config)
            {
                d[name.c_str()] = entry.value;
            }
            return d;
        },
        "Return the WarpX build configuration as a dictionary."
    );
}
