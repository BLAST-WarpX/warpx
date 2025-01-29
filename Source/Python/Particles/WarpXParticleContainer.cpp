/* Copyright 2021-2022 The WarpX Community
 *
 * Authors: Axel Huebl, Remi Lehe
 * License: BSD-3-Clause-LBNL
 */

#include "Python/pyWarpX.H"

#include <Particles/WarpXParticleContainer.H>


void init_WarpXParIter (py::module& m)
{
    py::class_<
        WarpXParIter, amrex::ParIterSoA<PIdx::nattribs, 0>
    >(m, "WarpXParIter")
        .def(py::init<amrex::ParIterSoA<PIdx::nattribs, 0>::ContainerType&, int>(),
            py::arg("particle_container"), py::arg("level"))
        .def(py::init<amrex::ParIterSoA<PIdx::nattribs, 0>::ContainerType&, int, amrex::MFItInfo&>(),
            py::arg("particle_container"), py::arg("level"),
            py::arg("info"))
    ;
}

void init_WarpXParticleContainer (py::module& m)
{
    py::class_<
        WarpXParticleContainer,
        amrex::ParticleContainerPureSoA<PIdx::nattribs, 0>
    > wpc (m, "WarpXParticleContainer");
    wpc
        .def("add_real_comp",
            [](WarpXParticleContainer& pc, const std::string& name, bool comm) { pc.NewRealComp(name, comm); },
            py::arg("name"), py::arg("comm")
        )
        .def("add_n_particles",
            [](WarpXParticleContainer& pc, int lev,
                int n, py::array_t<double> &x,
                py::array_t<double> &y,
                py::array_t<double> &z,
                py::array_t<double> &ux,
                py::array_t<double> &uy,
                py::array_t<double> &uz,
                const int nattr_real, py::array_t<double> &attr_real,
                const int nattr_int, py::array_t<int> &attr_int,
                int uniqueparticles, int id
            ) {
                amrex::Vector<amrex::ParticleReal> xp(x.data(), x.data() + n);
                amrex::Vector<amrex::ParticleReal> yp(y.data(), y.data() + n);
                amrex::Vector<amrex::ParticleReal> zp(z.data(), z.data() + n);
                amrex::Vector<amrex::ParticleReal> uxp(ux.data(), ux.data() + n);
                amrex::Vector<amrex::ParticleReal> uyp(uy.data(), uy.data() + n);
                amrex::Vector<amrex::ParticleReal> uzp(uz.data(), uz.data() + n);

                // create 2d arrays of real and in attributes
                amrex::Vector<amrex::Vector<amrex::ParticleReal>> attr;
                const double *attr_data = attr_real.data();
                for (int ii=0; ii<nattr_real; ii++) {
                    amrex::Vector<amrex::ParticleReal> attr_ii(n);
                    for (int jj=0; jj<n; jj++) {
                        attr_ii[jj] = attr_data[ii + jj*nattr_real];
                    }
                    attr.push_back(attr_ii);
                }

                amrex::Vector<amrex::Vector<int>> iattr;
                const int *iattr_data = attr_int.data();
                for (int ii=0; ii<nattr_int; ii++) {
                    amrex::Vector<int> attr_ii(n);
                    for (int jj=0; jj<n; jj++) {
                        attr_ii[jj] = iattr_data[ii + jj*nattr_int];
                    }
                    iattr.push_back(attr_ii);
                }

                pc.AddNParticles(
                    lev, n, xp, yp, zp, uxp, uyp, uzp, nattr_real, attr,
                    nattr_int, iattr, uniqueparticles, id
                );
            },
            py::arg("lev"), py::arg("n"),
            py::arg("x"), py::arg("y"), py::arg("z"),
            py::arg("ux"), py::arg("uy"), py::arg("uz"),
            py::arg("nattr_real"), py::arg("attr_real"),
            py::arg("nattr_int"), py::arg("attr_int"),
            py::arg("uniqueparticles"), py::arg("id")=-1
        )
        .def("get_comp_index",
            [](WarpXParticleContainer& pc, std::string comp_name)
            {
                auto particle_comps = pc.getParticleComps();
                return particle_comps.at(comp_name);
            },
            py::arg("comp_name")
        )
        .def("get_icomp_index",
            [](WarpXParticleContainer& pc, std::string comp_name)
            {
                auto particle_comps = pc.getParticleiComps();
                return particle_comps.at(comp_name);
            },
            py::arg("comp_name")
        )
        .def("num_local_tiles_at_level",
            &WarpXParticleContainer::numLocalTilesAtLevel,
            py::arg("level")
        )
        .def("total_number_of_particles",
            &WarpXParticleContainer::TotalNumberOfParticles,
            py::arg("valid_particles_only"), py::arg("local")
        )
        .def("sum_particle_weight",
            &WarpXParticleContainer::sumParticleWeight,
            py::arg("local")
        )
        .def("sum_particle_charge",
            &WarpXParticleContainer::sumParticleCharge,
            py::arg("local")
        )
        .def("deposit_charge",
            [](WarpXParticleContainer& pc,
            amrex::MultiFab* rho, const int lev)
            {
                for (WarpXParIter pti(pc, lev); pti.isValid(); ++pti)
                {
                    const long np = pti.numParticles();
                    auto& wp = pti.GetAttribs(PIdx::w);
                    pc.DepositCharge(pti, wp, nullptr, rho, 0, 0, np, 0, lev, lev);
                }
            },
            py::arg("rho"), py::arg("lev")
        )
        .def("get_charge_density",
            [](WarpXParticleContainer& pc, int lev, bool local)
            {
                return pc.GetChargeDensity(lev, local);
            },
            py::arg("lev"), py::arg("local")
        )
        .def("set_do_not_push",
            [](WarpXParticleContainer& pc, bool flag) { pc.setDoNotPush(flag); },
            py::arg("flag")
        )
    ;
}
