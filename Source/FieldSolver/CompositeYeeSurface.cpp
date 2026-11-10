/* Copyright 2026 The WarpX Community
 * License: BSD-3-Clause-LBNL
 */
#include "CompositeYeeSurface.H"
#include "FixedShapeCurrent.H"

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <vector>

using namespace amrex::literals;
using amrex::Real;
using warpx::fields::FieldType;

namespace {
std::string
checkpoint_identity (WarpX& w) {
    bool fixed = false, gather = false;
    amrex::ParmParse pp("warpx");
    pp.query("composite_yee_fixed_shape_current", fixed);
    pp.query("composite_yee_fixed_shape_gather", gather);
    std::ostringstream out;
    out << "WarpX conforming Yee surface 1\n"
        << w.refRatio(0) << ' ' << fixed << ' ' << gather << ' ' << WarpX::nox
        << ' ' << static_cast<int>(WarpX::current_deposition_algo) << ' '
        << WarpX::use_filter << ' ' << WarpX::filter_npass_each_dir << '\n';
    return out.str();
}
} // namespace

void
CompositeYeeSurface::WriteCheckpointIdentity (WarpX& w,
                                              std::string const& directory) {
    bool enabled = false;
    amrex::ParmParse("warpx").query("composite_yee_surface", enabled);
    if (enabled) {
        std::ofstream marker(std::filesystem::path(directory) /
                             "CompositeYeeSurface");
        marker << checkpoint_identity(w);
        marker.close();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            marker.good(),
            "Could not save conforming surface checkpoint identity.");
    }
}

void
CompositeYeeSurface::CheckRestartIdentity (WarpX& w,
                                           std::string const& directory,
                                           bool enabled) {
    auto const file = std::filesystem::path(directory) / "CompositeYeeSurface";
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::filesystem::exists(file) == enabled,
        "A conforming Yee checkpoint must be restarted with the same interface "
        "solver.");
    if (enabled) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !std::filesystem::exists(std::filesystem::path(directory) / "Level_1" / "pml_Bx_fp_H"),
            "This checkpoint evolved an interior absorber on its stepped Yee interface.");
        std::ifstream marker(file);
        std::ostringstream text;
        text << marker.rdbuf();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            text.str() == checkpoint_identity(w),
            "The conforming Yee checkpoint has a different interface or "
            "particle coupling.");
    }
}

#ifdef WARPX_DIM_3D
namespace {
using Point = std::array<int, 3>;
using Site = std::array<int, 4>; // component and native Yee index
using Row = std::map<int, Real>;
using Rows = std::vector<Row>;
using Vector = amrex::Gpu::DeviceVector<Real>;

Point
shift (Point p, int d, int n) {
    p[d] += n;
    return p;
}
Site
site (int d, Point const& p) {
    return {d, p[0], p[1], p[2]};
}
Point
coord (Site const& p) {
    return {p[1], p[2], p[3]};
}

template <class T>
void
upload (amrex::Gpu::DeviceVector<T>& dst, std::vector<T> const& src) {
    dst.resize(src.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, src.begin(), src.end(),
                     dst.begin());
}

struct Coverage {
    Point n;
    std::vector<unsigned char> values;
    explicit Coverage (WarpX& w) {
        auto const box = w.Geom(0).Domain();
        for (int d = 0; d < 3; ++d) {
            n[d] = box.length(d);
        }
        values.resize(static_cast<std::size_t>(n[0]) * n[1] * n[2]);
        for (int k = 0; k < w.boxArray(1).size(); ++k) {
            auto const& fb = w.boxArray(1)[k];
            auto const cb = amrex::coarsen(fb, w.refRatio(0));
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                amrex::refine(cb, w.refRatio(0)) == fb &&
                    box.strictly_contains(cb),
                "The Yee surface requires an interior, coarse-cell-aligned "
                "refinement.");
            for (int z = cb.smallEnd(2); z <= cb.bigEnd(2); ++z) {
                for (int y = cb.smallEnd(1); y <= cb.bigEnd(1); ++y) {
                    for (int x = cb.smallEnd(0); x <= cb.bigEnd(0); ++x) {
                        values[index({x, y, z})] = 1;
                    }
                }
            }
        }
    }
    std::size_t
    index (Point const& p) const {
        return p[0] + static_cast<std::size_t>(n[0]) *
                          (p[1] + static_cast<std::size_t>(n[1]) * p[2]);
    }
    bool
    operator()(Point const& p) const {
        return values[index(p)] != 0;
    }
    std::vector<Site>
    faces () const {
        std::vector<Site> out;
        for (int z = 1; z < n[2] - 1; ++z) {
            for (int y = 1; y < n[1] - 1; ++y) {
                for (int x = 1; x < n[0] - 1; ++x) {
                    Point const p{x, y, z};
                    if (!(*this)(p)) {
                        continue;
                    }
                    for (int d = 0; d < 3; ++d) {
                        if (!(*this)(shift(p, d, -1))) {
                            out.push_back(site(d, p));
                        }
                        if (!(*this)(shift(p, d, 1))) {
                            out.push_back(site(d, shift(p, d, 1)));
                        }
                    }
                }
            }
        }
        return out;
    }
};

// The current leaf operator represents vacuum; conducting or cut edges may
// occur on the exterior grid only. Also exclude a two-parent-cell stencil
// collar.
void
check_vacuum_patch (WarpX& w) {
    amrex::ParmParse const pp("warpx");
    std::vector<amrex::RealBox> ports;
    if (w.HasCurrentControlledPorts()) {
        int n = 1;
        bool const indexed = pp.query("current_controlled_port.n_ports", n);
        for (int i = 0; i < n; ++i) {
            auto const prefix =
                indexed ? "current_controlled_port.port_" + std::to_string(i)
                        : "current_controlled_port";
            std::array<Real, 3> lo{}, hi{};
            for (int end = 0; end < 2; ++end) {
                std::vector<Real> a, b;
                auto const terminal =
                    prefix + ".terminal_" + std::to_string(end);
                utils::parser::getArrWithParser(
                    pp, (terminal + ".lower_bound").c_str(), a);
                utils::parser::getArrWithParser(
                    pp, (terminal + ".upper_bound").c_str(), b);
                for (int d = 0; d < 3; ++d) {
                    lo[d] = end ? std::min(lo[d], a.at(d)) : a.at(d);
                    hi[d] = end ? std::max(hi[d], b.at(d)) : b.at(d);
                }
            }
            ports.emplace_back(lo.data(), hi.data());
        }
    }
    for (int k = 0; k < w.boxArray(1).size(); ++k) {
        auto const fb = w.boxArray(1)[k];
        auto const cb = amrex::grow(amrex::coarsen(fb, w.refRatio(0)), 2);
        auto const& geom = w.Geom(0);
        amrex::RealBox physical(cb, geom.CellSize(), geom.ProbLo());
        for (auto const& port : ports) {
            bool overlaps = true;
            for (int d = 0; d < 3; ++d) {
                overlaps &= port.lo(d) <= physical.hi(d) &&
                            physical.lo(d) <= port.hi(d);
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !overlaps, "Prescribed ports must lie outside the Yee patch "
                           "and its stencil collar.");
        }
        for (int lev = 0; lev < 2; ++lev) {
            auto const region = lev ? fb : cb;
            if (auto* mat = w.get_pointer_MacroscopicProperties(lev)) {
                auto constant = [&] (amrex::MultiFab const& field, Real value) {
                    auto const box = amrex::convert(region, field.ixType());
                    return field.min(box, 0) == value &&
                           field.max(box, 0) == value;
                };
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    constant(mat->getsigma_mf(), 0._rt) &&
                        constant(mat->getepsilon_mf(), PhysConst::epsilon_0) &&
                        constant(mat->getmu_mf(), PhysConst::mu0),
                    "The refined volume and its interface collar must have "
                    "vacuum material coefficients.");
            }
            if (EB::enabled()) {
                for (int d = 0; d < 3; ++d) {
                    for (auto* flag : {w.GetEBUpdateEFlag()[lev][d].get(),
                                       w.GetEBUpdateBFlag()[lev][d].get()}) {
                        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                            flag && flag->min(
                                        amrex::convert(region, flag->ixType()),
                                        0) > 0,
                            "The refined volume and its interface collar must "
                            "have active Yee edges and faces.");
                    }
                }
            }
        }
    }
}

struct Sites {
    std::map<Site, int> ids;
    std::vector<Site> ordered;
    int
    insert (Site const& key) {
        auto [it, added] = ids.emplace(key, static_cast<int>(ordered.size()));
        if (added) {
            ordered.push_back(key);
        }
        return it->second;
    }
};

struct Leaf {
    Sites edge, magnetic;
    Rows circulation;
    std::vector<Real> mass;
    Real volume;
    Leaf (Coverage const& coverage, std::vector<Site> const& faces,
          Point const& ratio, std::array<Real, 3> const& h, bool fine)
        : volume(h[0] * h[1] * h[2]) {
        auto active = [&] (Point p) {
            for (int d = 0; d < 3; ++d) {
                p[d] /= ratio[d];
            }
            return coverage(p) == fine ? 1 : 0;
        };
        for (auto const& face : faces) {
            int const n = face[0], u = (n + 1) % 3, v = (n + 2) % 3;
            for (int a = 0; a < ratio[u]; ++a) {
                for (int b = 0; b < ratio[v]; ++b) {
                    Point q = coord(face);
                    for (int d = 0; d < 3; ++d) {
                        q[d] *= ratio[d];
                    }
                    q[u] += a;
                    q[v] += b;
                    edge.insert(site(u, q));
                    edge.insert(site(u, shift(q, v, 1)));
                    edge.insert(site(v, q));
                    edge.insert(site(v, shift(q, u, 1)));
                }
            }
        }
        circulation.resize(edge.ordered.size());
        for (int i = 0; i < static_cast<int>(edge.ordered.size()); ++i) {
            auto const& e = edge.ordered[i];
            auto const p = coord(e);
            int const a = e[0], u = (a + 1) % 3, v = (a + 2) % 3;
            int const count = active(p) + active(shift(p, u, -1)) +
                              active(shift(p, v, -1)) +
                              active(shift(shift(p, u, -1), v, -1));
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                count > 0 && count < 4, "Invalid Yee surface edge ownership.");
            mass.push_back(volume * 0.25_rt * count);
            for (int d : {u, v}) {
                int const b = 3 - a - d;
                Real const sign = (b + 1) % 3 == d ? 1._rt : -1._rt;
                for (int offset : {0, -1}) {
                    auto const q = shift(p, d, offset);
                    Real const fraction =
                        0.5_rt * (active(q) + active(shift(q, b, -1)));
                    if (fraction == 0._rt) {
                        continue;
                    }
                    int const id = magnetic.insert(site(b, q));
                    circulation[i][id] +=
                        volume * fraction * (offset == 0 ? -sign : sign) / h[d];
                }
            }
        }
    }
};

Rows
trace (Leaf const& coarse, Leaf const& fine, Point const& ratio) {
    Rows rows(fine.edge.ordered.size());
    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        auto const e = fine.edge.ordered[i];
        Point origin = coord(e);
        std::array<Real, 3> frac{};
        for (int d = 0; d < 3; ++d) {
            frac[d] = (d == e[0] ? 0._rt
                                 : static_cast<Real>(origin[d] % ratio[d]) /
                                       ratio[d]);
            origin[d] /= ratio[d];
        }
        for (int mask = 0; mask < 8; ++mask) {
            Point p = origin;
            Real weight = 1._rt;
            for (int d = 0; d < 3; ++d) {
                if (mask & (1 << d)) {
                    ++p[d];
                    weight *= frac[d];
                } else {
                    weight *= 1._rt - frac[d];
                }
            }
            if (weight != 0._rt) {
                rows[i][coarse.edge.ids.at(site(e[0], p))] += weight;
            }
        }
    }
    return rows;
}

struct Matrix {
    amrex::Gpu::DeviceVector<int> offsets, columns;
    Vector values;
    void
    set (Rows const& rows) {
        std::vector<int> o{0}, c;
        std::vector<Real> v;
        for (auto const& row : rows) {
            for (auto const& [col, val] : row) {
                c.push_back(col);
                v.push_back(val);
            }
            o.push_back(static_cast<int>(c.size()));
        }
        upload(offsets, o);
        upload(columns, c);
        upload(values, v);
    }
    struct View {
        int const* offsets;
        int const* columns;
        Real const* values;
        AMREX_GPU_HOST_DEVICE Real
        apply (int i, Real const* x) const noexcept {
            Real sum = 0._rt;
            for (int k = offsets[i]; k < offsets[i + 1]; ++k) {
                sum += values[k] * x[columns[k]];
            }
            return sum;
        }
    };
    View
    view () const {
        return {offsets.data(), columns.data(), values.data()};
    }
};

// One valid owner supplies a read. Every overlapping nodal copy receives a
// write.
struct Bound {
    struct Ref {
        Real* addr;
        int i;
    };
    amrex::Gpu::DeviceVector<Ref> unique, all;
    void
    bind (ablastr::fields::VectorField const& fields, Sites const& sites) {
        std::vector<Ref> owner(sites.ordered.size(), {nullptr, 0}), copies;
        for (int d = 0; d < 3; ++d) {
            for (amrex::MFIter mfi(*fields[d]); mfi.isValid(); ++mfi) {
                auto& fab = (*fields[d])[mfi];
                auto const box = mfi.validbox();
                for (auto it =
                         sites.ids.lower_bound({d, box.smallEnd(0), 0, 0});
                     it != sites.ids.end() && it->first[0] == d &&
                     it->first[1] <= box.bigEnd(0);
                     ++it) {
                    auto const& s = it->first;
                    amrex::IntVect const p{s[1], s[2], s[3]};
                    if (!box.contains(p)) {
                        continue;
                    }
                    Ref const ref{fab.dataPtr() + fab.box().index(p),
                                  it->second};
                    copies.push_back(ref);
                    if (!owner[ref.i].addr) {
                        owner[ref.i] = ref;
                    }
                }
            }
        }
        for (auto const& r : owner) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                r.addr, "A surface degree of freedom has no native owner.");
        }
        upload(unique, owner);
        upload(all, copies);
    }
    void
    read (Vector& out) const {
        auto* v = out.data();
        auto const* ref = unique.data();
        amrex::ParallelFor(
            static_cast<int>(unique.size()),
            [=] AMREX_GPU_DEVICE(int i) noexcept { v[i] = *ref[i].addr; });
    }
    void
    write (Vector const& in) const {
        auto const* v = in.data();
        auto const* ref = all.data();
        amrex::ParallelFor(static_cast<int>(all.size()),
                           [=] AMREX_GPU_DEVICE(int i) noexcept {
                               *ref[i].addr = v[ref[i].i];
                           });
    }
};
// Match both tangential E and the integrated normal B at startup. Faraday
// subsequently preserves the normal condition when the tangential traces agree.
void
check_trace (Matrix const& map, Vector const& source, Vector const& target) {
    auto const mat = map.view();
    auto const* x = source.data();
    auto const* y = target.data();
    amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax> ops;
    amrex::ReduceData<Real, Real> data(ops);
    using Tuple = decltype(data)::Type;
    ops.eval(static_cast<int>(target.size()), data,
             [=] AMREX_GPU_DEVICE(int i) -> Tuple {
                 auto const prediction = mat.apply(i, x);
                 return {std::abs(y[i] - prediction),
                         amrex::max(std::abs(y[i]), std::abs(prediction))};
             });
    auto const result = data.value();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        amrex::get<0>(result) <= 128._rt *
                                     std::numeric_limits<Real>::epsilon() *
                                     amrex::get<1>(result),
        "Initial tangential electric and normal magnetic traces must agree "
        "across the Yee surface.");
}

void
check_normal_flux (WarpX& w, std::vector<Site> const& boundary,
                   Point const& ratio) {
    Sites coarse, fine;
    Rows average(boundary.size());
    for (auto const& face : boundary) {
        auto const c = coarse.insert(face);
        int const normal = face[0];
        auto const u = (normal + 1) % 3, v = (normal + 2) % 3;
        Point p = coord(face);
        for (int d = 0; d < 3; ++d) {
            p[d] *= ratio[d];
        }
        for (int a = 0; a < ratio[u]; ++a) {
            for (int b = 0; b < ratio[v]; ++b) {
                auto const f =
                    fine.insert(site(normal, shift(shift(p, u, a), v, b)));
                average[c][f] = 1._rt / (ratio[u] * ratio[v]);
            }
        }
    }
    Vector bc(coarse.ordered.size()), bf(fine.ordered.size());
    Bound readc, readf;
    readc.bind(w.m_fields.get_alldirs(FieldType::Bfield_fp, 0), coarse);
    readf.bind(w.m_fields.get_alldirs(FieldType::Bfield_fp, 1), fine);
    readc.read(bc);
    readf.read(bf);
    Matrix map;
    map.set(average);
    check_trace(map, bf, bc);
}

} // namespace
#endif

#ifdef WARPX_DIM_3D
struct CompositeYeeSurface::Implementation {
    WarpX& w;
    amrex::IntVect ratio;
    Bound edges[2], faces[2], currents[2], coarse_patch_current;
    Vector electric[2], magnetic[2], current[2], patch_current, rhs[2], base,
        delta, tmp, diagonal_inverse;
    Matrix curl[2], prolong, transpose, scaled_offdiag;
    int ne[2]{};
    Real volume[2]{};
    int iterations = 0;
    std::unique_ptr<FixedShapeCurrent> fixed_shape_current;
    std::array<std::array<std::unique_ptr<amrex::MultiFab>, 3>, 2> gather_patch;

    explicit Implementation (WarpX& warpx) : w(warpx), ratio(w.refRatio(0)) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            sizeof(Real) == sizeof(double),
            "Private surface controls require double precision.");
        Point r;
        std::array<Real, 3> hc{}, hf{};
        for (int d = 0; d < 3; ++d) {
            r[d] = ratio[d];
            hc[d] = w.Geom(0).CellSize(d);
            hf[d] = hc[d] / r[d];
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                w.Geom(0).Domain().smallEnd(d) == 0 && r[d] >= 1,
                "The Yee surface expects zero-origin native indices and "
                "positive refinement ratios.");
        }
        check_vacuum_patch(w);
        bool use_fixed_shape = false;
        amrex::ParmParse("warpx").query("composite_yee_fixed_shape_current",
                                        use_fixed_shape);
        if (use_fixed_shape) {
            fixed_shape_current = std::make_unique<FixedShapeCurrent>(w);
        }
        Coverage const covered(w);
        auto const boundary = covered.faces();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !boundary.empty(), "The Yee refinement boundary is empty.");
        Leaf const coarse(covered, boundary, {1, 1, 1}, hc, false);
        Leaf const fine(covered, boundary, r, hf, true);
        auto& fields = w.m_fields;
        for (int lev = 0; lev < 2; ++lev) {
            Leaf const& leaf = lev == 0 ? coarse : fine;
            volume[lev] = leaf.volume;
            ne[lev] = static_cast<int>(leaf.edge.ordered.size());
            electric[lev].resize(ne[lev]);
            current[lev].resize(ne[lev]);
            rhs[lev].resize(ne[lev]);
            magnetic[lev].resize(leaf.magnetic.ordered.size());
            edges[lev].bind(fields.get_alldirs(FieldType::Efield_fp, lev),
                            leaf.edge);
            faces[lev].bind(fields.get_alldirs(FieldType::Bfield_fp, lev),
                            leaf.magnetic);
            currents[lev].bind(fields.get_alldirs(FieldType::current_fp, lev),
                               leaf.edge);
            curl[lev].set(leaf.circulation);
        }
        coarse_patch_current.bind(fields.get_alldirs(FieldType::current_cp, 1),
                                  coarse.edge);
        patch_current.resize(ne[0]);
        base.resize(ne[0]);
        delta.resize(ne[0]);
        tmp.resize(ne[0]);

        Rows const p = trace(coarse, fine, r);
        Rows pt(ne[0]), mass(ne[0]);
        for (int i = 0; i < ne[0]; ++i) {
            mass[i][i] = coarse.mass[i];
        }
        for (int f = 0; f < ne[1]; ++f) {
            for (auto const& [i, a] : p[f]) {
                pt[i][f] = a;
                for (auto const& [j, b] : p[f]) {
                    mass[i][j] += a * fine.mass[f] * b;
                }
            }
        }
        prolong.set(p);
        edges[0].read(electric[0]);
        edges[1].read(electric[1]);
        check_trace(prolong, electric[0], electric[1]);
        check_normal_flux(w, boundary, r);
        transpose.set(pt);
        // Every fine surface edge interpolates a coarse tangential trace. The
        // positive coarse dual volumes make the Jacobi solve contractive.
        // Bound the tail of its Neumann series from the actual matrix rows.
        Real contraction = 0._rt;
        std::vector<Real> dinv(ne[0]);
        for (int i = 0; i < ne[0]; ++i) {
            dinv[i] = 1._rt / mass[i].at(i);
            mass[i].erase(i);
            Real sum = 0._rt;
            for (auto& [j, value] : mass[i]) {
                value *= dinv[i];
                sum += std::abs(value);
            }
            contraction = std::max(contraction, sum);
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            contraction < 1._rt,
            "The surface mass iteration does not contract.");
        if (contraction > 0._rt) {
            iterations = static_cast<int>(
                std::ceil(std::log(std::numeric_limits<Real>::epsilon() *
                                   (1._rt - contraction)) /
                          std::log(contraction)));
        }
        scaled_offdiag.set(mass);
        upload(diagonal_inverse, dinv);
        for (int s = 0; s < 2; ++s) {
            auto const aux = fields.get_alldirs(
                s ? FieldType::Bfield_aux : FieldType::Efield_aux, 1);
            for (int d = 0; d < 3; ++d) {
                auto const& a = *aux[d];
                auto const pad =
                    amrex::coarsen(a.nGrowVect() + ratio - amrex::IntVect(1),
                                   ratio) +
                    amrex::IntVect(1);
                auto const ba = amrex::coarsen(a.boxArray(), ratio);
                for (int k = 0; k < ba.size(); ++k) {
                    auto const domain =
                        amrex::convert(w.Geom(0).Domain(), a.ixType());
                    auto const buf_type =
                        s ? FieldType::Bfield_cax : FieldType::Efield_cax;
                    auto const width =
                        fields.has_vector(buf_type, 1)
                            ? amrex::elemwiseMax(
                                  pad,
                                  fields
                                      .get(buf_type,
                                           ablastr::fields::Direction{d}, 1)
                                      ->nGrowVect())
                            : pad;
                    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                        domain.contains(amrex::grow(ba[k], width)),
                        "The gather stencil needs a physical coarse collar "
                        "around each fine box.");
                }
                gather_patch[s][d] = std::make_unique<amrex::MultiFab>(
                    ba, a.DistributionMap(), 1, pad);
            }
        }
        amrex::Print() << "Conforming Yee surface: " << ne[0]
                       << " parent edges, " << ne[1] << " fine edges, "
                       << iterations << " mass iterations (bound "
                       << contraction << ").\n";
    }
};

CompositeYeeSurface::CompositeYeeSurface (WarpX& w)
    : m_impl(std::make_unique<Implementation>(w)) {}
CompositeYeeSurface::~CompositeYeeSurface () = default;

void
CompositeYeeSurface::RememberElectric () {
    m_impl->edges[0].read(m_impl->electric[0]);
}

void
CompositeYeeSurface::RestrictMagnetic () {
    auto& g = *m_impl;
    g.faces[0].read(g.magnetic[0]);
    auto const f = g.w.m_fields.get_alldirs(FieldType::Bfield_fp, 1);
    auto const c = g.w.m_fields.get_alldirs(FieldType::Bfield_fp, 0);
    for (int d = 0; d < 3; ++d) {
        amrex::average_down_faces(*f[d], *c[d], g.ratio);
    }
    g.faces[0].write(g.magnetic[0]);
}

void
CompositeYeeSurface::AdvanceElectric (Real dt) {
    auto& g = *m_impl;
    g.coarse_patch_current.read(g.patch_current);
    for (int lev = 0; lev < 2; ++lev) {
        g.faces[lev].read(g.magnetic[lev]);
        g.currents[lev].read(g.current[lev]);
        auto const curl = g.curl[lev].view();
        auto const* b = g.magnetic[lev].data();
        auto const* j = g.current[lev].data();
        auto const* cp = g.patch_current.data();
        auto* rhs = g.rhs[lev].data();
        auto const v = g.volume[lev];
        amrex::ParallelFor(g.ne[lev], [=] AMREX_GPU_DEVICE(int i) noexcept {
            Real const leaf_j = lev == 0 ? j[i] - cp[i] : j[i];
            rhs[i] = PhysConst::c * PhysConst::c * curl.apply(i, b) -
                     v * leaf_j / PhysConst::epsilon_0;
        });
    }
    auto const pt = g.transpose.view();
    auto const* rc = g.rhs[0].data();
    auto const* rf = g.rhs[1].data();
    auto const* inv = g.diagonal_inverse.data();
    auto* base = g.base.data();
    auto* x = g.delta.data();
    amrex::ParallelFor(g.ne[0], [=] AMREX_GPU_DEVICE(int i) noexcept {
        base[i] = inv[i] * (rc[i] + pt.apply(i, rf));
        x[i] = base[i];
    });
    auto const off = g.scaled_offdiag.view();
    for (int it = 0; it < g.iterations; ++it) {
        auto const* cur = g.delta.data();
        auto* next = g.tmp.data();
        amrex::ParallelFor(g.ne[0], [=] AMREX_GPU_DEVICE(int i) noexcept {
            next[i] = base[i] - off.apply(i, cur);
        });
        g.delta.swap(g.tmp);
    }
    auto const* de = g.delta.data();
    auto* ec = g.electric[0].data();
    amrex::ParallelFor(
        g.ne[0], [=] AMREX_GPU_DEVICE(int i) noexcept { ec[i] += dt * de[i]; });
    auto const p = g.prolong.view();
    auto* ef = g.electric[1].data();
    amrex::ParallelFor(g.ne[1], [=] AMREX_GPU_DEVICE(int i) noexcept {
        ef[i] = p.apply(i, ec);
    });
    auto const fine = g.w.m_fields.get_alldirs(FieldType::Efield_fp, 1);
    auto const coarse = g.w.m_fields.get_alldirs(FieldType::Efield_fp, 0);
    for (int d = 0; d < 3; ++d) {
        amrex::average_down_edges(*fine[d], *coarse[d], g.ratio);
    }
    g.edges[0].write(g.electric[0]);
    g.edges[1].write(g.electric[1]);
}

void
CompositeYeeSurface::AssembleParticleCurrent (Real dt) {
    if (m_impl->fixed_shape_current) {
        m_impl->fixed_shape_current->Assemble(dt);
    }
}

void
CompositeYeeSurface::UpdateGatherFields () {
    auto& g = *m_impl;
    auto& fields = g.w.m_fields;
    auto const ratio = g.ratio;
    for (int s = 0; s < 2; ++s) {
        auto const type = s ? FieldType::Bfield_fp : FieldType::Efield_fp;
        auto const aux_type = s ? FieldType::Bfield_aux : FieldType::Efield_aux;
        auto const fine = fields.get_alldirs(type, 1),
                   coarse = fields.get_alldirs(type, 0);
        auto const aux = fields.get_alldirs(aux_type, 1);
        for (int d = 0; d < 3; ++d) {
            auto& scratch = *g.gather_patch[s][d];
            scratch.ParallelCopy(*coarse[d], 0, 0, 1, amrex::IntVect(0),
                                 scratch.nGrowVect(),
                                 g.w.Geom(0).periodicity());
            auto const cax_type =
                s ? FieldType::Bfield_cax : FieldType::Efield_cax;
            if (fields.has_vector(cax_type, 1)) {
                auto* buffered =
                    fields.get(cax_type, ablastr::fields::Direction{d}, 1);
                // Buffered fine particles gather from the physical parent,
                // including guards.
                buffered->ParallelCopy(*coarse[d], 0, 0, 1, amrex::IntVect(0),
                                       buffered->nGrowVect(),
                                       g.w.Geom(0).periodicity());
            }
            auto const nodal = aux[d]->ixType().toIntVect();
            // Valid fine values and neighboring fine boxes take precedence over
            // this Whitney extension. Keep exterior coarse fields in the force
            // stencil even for a diagnostic particle momentum half push.
            for (amrex::MFIter mfi(*aux[d]); mfi.isValid(); ++mfi) {
                auto const src = scratch.const_array(mfi);
                auto const dst = aux[d]->array(mfi);
                auto const valid = mfi.validbox();
                amrex::ParallelFor(
                    mfi.fabbox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        amrex::IntVect const q{i, j, k};
                        if (valid.contains(q)) {
                            return;
                        }
                        auto const p = amrex::coarsen(q, ratio);
                        Real f[3];
                        int hi[3];
                        for (int a = 0; a < 3; ++a) {
                            int const rem = nodal[a] * (q[a] - p[a] * ratio[a]);
                            f[a] = static_cast<Real>(rem) / ratio[a];
                            hi[a] = static_cast<int>(rem != 0);
                        }
                        Real val = 0._rt;
                        for (int z = 0; z <= hi[2]; ++z) {
                            for (int y = 0; y <= hi[1]; ++y) {
                                for (int x = 0; x <= hi[0]; ++x) {
                                    val += (x ? f[0] : 1._rt - f[0]) *
                                           (y ? f[1] : 1._rt - f[1]) *
                                           (z ? f[2] : 1._rt - f[2]) *
                                           src(p[0] + x, p[1] + y, p[2] + z);
                                }
                            }
                        }
                        dst(i, j, k) = val;
                    });
            }
            aux[d]->ParallelCopy(*fine[d], 0, 0, 1, amrex::IntVect(0),
                                 aux[d]->nGrowVect(),
                                 g.w.Geom(1).periodicity());
        }
    }
    if (g.fixed_shape_current) {
        g.fixed_shape_current->UpdateGather();
    }
}
#else
struct CompositeYeeSurface::Implementation {};
CompositeYeeSurface::CompositeYeeSurface (WarpX&) {
    WARPX_ABORT_WITH_MESSAGE("The conforming Yee surface requires 3D.");
}
CompositeYeeSurface::~CompositeYeeSurface () = default;
void
CompositeYeeSurface::RememberElectric () {}
void
CompositeYeeSurface::AdvanceElectric (Real) {}
void
CompositeYeeSurface::RestrictMagnetic () {}
void
CompositeYeeSurface::AssembleParticleCurrent (Real) {}
void
CompositeYeeSurface::UpdateGatherFields () {}
#endif
