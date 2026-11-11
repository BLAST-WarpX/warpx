"""Check charge, Gauss's law and particle/field work while crossing both XZ slab faces."""

import argparse
from pathlib import Path

import numpy as np

from pywarpx import callbacks, geometry, libwarpx
from pywarpx.particle_containers import ParticleContainerWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--ratio", type=int, nargs=2, required=True)
parser.add_argument("--shape", type=int, required=True)
parser.add_argument("--deposition", choices=("esirkepov", "villasenor"), required=True)
parser.add_argument("--filter", action="store_true")
parser.add_argument("--native-source", action="store_true")
parser.add_argument("--periodic-corner", action="store_true")
parser.add_argument("--restart", type=Path)
parser.add_argument("--guarded-restart", action="store_true")
args = parser.parse_args()
rx, rz = args.ratio
nx, nz = 96, 16
lo, hi = 32, 64
h = 1.0e-3
dt = 0.4 * h / 299792458.0
steps = 10
initial_x = np.array([31.53, 32.39, 63.59, 64.47]) * h
initial_z = np.array([15.8, 0.2, 12.1, 6.7]) * h
if args.periodic_corner:
    initial_x[2:] = np.array([95.7, 0.35]) * h
    initial_z[2:] = np.array([15.82, 0.17]) * h
weights = np.array([1.0, 1.4, 0.7, 0.9]) * 1.0e6
elementary_charge = 1.602176634e-19
geometry.dims = 2
settings = [
    "crossing-slab",
    "geometry.dims=2",
    "geometry.prob_lo=0 0",
    "geometry.prob_hi=0.096 0.016",
    f"amr.n_cell={nx // rx} {nz // rz}",
    "amr.max_level=1",
    f"amr.ref_ratio_vect={rx} {rz}",
    "amr.max_grid_size=128",
    "amr.blocking_factor=2",
    "amr.n_error_buf=0",
    "warpx.fine_tag_lo=0.032 0",
    "warpx.fine_tag_hi=0.064 0.016",
    "warpx.composite_xz_interface=1",
    f"warpx.composite_xz_fixed_shape={int(not args.guarded_restart)}",
    "algo.maxwell_solver=yee",
    "algo.field_gathering=energy-conserving",
    f"algo.current_deposition={args.deposition}",
    f"algo.particle_shape={args.shape}",
    "warpx.n_field_gather_buffer=0",
    "warpx.n_current_deposition_buffer=0",
    "warpx.do_initial_div_cleaning=0",
    f"warpx.use_filter={int(args.filter)}",
    "warpx.filter_npass_each_dir=1 1",
    "boundary.field_lo=periodic periodic",
    "boundary.field_hi=periodic periodic",
    "boundary.particle_lo=periodic periodic",
    "boundary.particle_hi=periodic periodic",
    "particles.species_names=ions",
    "ions.species_type=proton",
    "ions.injection_style=multipleparticles",
    "ions.multiple_particles_pos_x=" + " ".join(map(str, initial_x)),
    "ions.multiple_particles_pos_z=" + " ".join(map(str, initial_z)),
    "ions.multiple_particles_pos_y=0 0 0 0",
    "ions.multiple_particles_ux=0.55 -0.51 0.50 -0.54",
    "ions.multiple_particles_uy=0.3 -0.2 -0.1 0.22",
    "ions.multiple_particles_uz=0.35 -0.36 0.41 -0.42",
    "ions.multiple_particles_weight=" + " ".join(map(str, weights)),
    f"warpx.const_dt={dt:.17g}",
    "warpx.verbose=0",
    f"max_step={steps}",
    "diagnostics.diags_names=chk",
    "chk.diag_type=Full",
    "chk.format=checkpoint",
    "chk.intervals=5",
    "chk.file_prefix=diags/chk",
]
if args.restart:
    settings.append(f"amr.restart={args.restart}")
libwarpx.initialize(settings)
register = libwarpx.warpx.multifab_register()
ions = ParticleContainerWrapper("ions")


def grid(name, direction, level, full=False):
    a = np.squeeze(np.asarray(register.get(name, dir=direction, level=level)[...]))
    if full:
        return a[:nx, :nz].copy()
    if level == 0:
        return a[: nx // rx, : nz // rz].copy()
    return a[:, :nz].copy()


def project(nodes):
    """Transpose scalar interpolation onto independent nodes, including both traces."""
    i, j = np.indices((nx, nz))
    exterior = (i <= lo) | (i >= hi)
    left = np.where(exterior, i // rx * rx, i)
    bottom = np.where(exterior, j // rz * rz, j)
    fx, fz = (i - left) / rx, (j - bottom) / rz
    result = np.zeros_like(nodes)
    for dx, wx in ((0, 1 - fx), (rx, fx)):
        for dz, wz in ((0, 1 - fz), (rz, fz)):
            np.add.at(result, ((left + dx) % nx, (bottom + dz) % nz), wx * wz * nodes)
    return result


def shape(index, x, n):
    a = abs((index - x / h + n / 2) % n - n / 2)
    if args.shape == 1:
        return np.maximum(1 - a, 0)
    if args.shape == 2:
        return np.where(a < 0.5, 0.75 - a**2, 0.5 * np.maximum(1.5 - a, 0) ** 2)
    return np.where(a < 1, 2 / 3 - a**2 + a**3 / 2, np.maximum(2 - a, 0) ** 3 / 6)


def charge(xs, zs, ws):
    result = np.zeros((nx, nz))
    for x, z, w in zip(xs, zs, ws):
        result += (
            w
            * elementary_charge
            * np.outer(shape(np.arange(nx), x, nx), shape(np.arange(nz), z, nz))
        )
    if args.filter:
        for axis in (0, 1):
            result = (
                np.roll(result, 1, axis) + 2 * result + np.roll(result, -1, axis)
            ) / 4
    return project(result)


def particles():
    xs, zs, ws = [], [], []
    for lev in (0, 1):
        for buf, name in ((xs, "x"), (zs, "z"), (ws, "weight")):
            buf.extend(
                getattr(ions, f"get_particle_{name}")(level=lev, copy_to_host=True)
            )
    return tuple(np.concatenate(a) for a in (xs, zs, ws))


def divergence(name, field=False):
    # The trace field has one half dual volume on each mesh. Each deposited
    # level current already carries its full contribution to the trace load.
    cx, cz = [grid(name, d, 0) for d in ("x", "z")]
    fx, fz = [grid(name, d, 1) for d in ("x", "z")]
    assert fx.shape == (hi - lo, nz) and fz.shape == (hi - lo + 1, nz)
    if field:
        cz[[lo // rx, hi // rx]] *= 0.5
        fz[[0, -1]] *= 0.5
    coarse = rz * h * (cx - np.roll(cx, 1, 0)) + rx * h * (cz - np.roll(cz, 1, 1))
    result = np.zeros((nx, nz))
    result[::rx, ::rz] = coarse
    result[lo : hi + 1] += h * (
        np.diff(np.pad(fx, ((1, 1), (0, 0))), axis=0) + fz - np.roll(fz, 1, 1)
    )
    return project(result)


def snapshot():
    return {
        f"{name}_{d}_{lev}": grid(name, d, lev)
        for name in ("Efield_fp", "Bfield_fp", "current_fp")
        for lev in (0, 1)
        for d in ("x", "y", "z")
    }


gathered, native = {}, {}


def save_gather():
    for d in ("x", "y", "z"):
        gathered[d] = grid("composite_fixed_E", d, 1, full=True)
        for lev in (0, 1):
            native[d, lev] = grid("Efield_fp", d, lev)


# Optional independent sources lie well away from each particle. This checks
# that assembling the kinetic current preserves sources on both native grids.
source_nodes = ((8, 3, 0.125), (16, 6, 0.25))


def add_native_sources():
    for lev, (i, j, amplitude) in enumerate(source_nodes):
        mf = register.get("current_fp", dir="y", level=lev)
        values = np.asarray(mf[...]).copy()
        values[i, j] += amplitude
        mf[...] = values


if args.native_source:
    assert not args.filter
    callbacks.installafterdeposition(add_native_sources)
callbacks.installbeforedeposition(save_gather)
q0 = charge(initial_x, initial_z, weights)
qprev = charge(*particles())
scale = np.max(weights) * elementary_charge
max_error = np.zeros(3)
for _ in range(libwarpx.warpx.getistep(0), steps):
    libwarpx.warpx.evolve(1)
    q = charge(*particles())
    flow = divergence("current_fp")
    gauss = 8.8541878188e-12 * divergence("Efield_fp", field=True)
    assert np.max(abs(dt * flow)) > 1.0e-3 * scale
    work = ref = magnitude = 0.0
    source = "xz_fixed_filtered_current" if args.filter else "xz_fixed_raw_current"
    for d in ("x", "y", "z"):
        product = grid(source, d, 1, full=True) * gathered[d]
        ref += np.sum(product)
        magnitude += np.sum(abs(product))
        for lev, area in ((0, rx * rz), (1, 1)):
            current = grid("current_fp", d, lev)
            if args.native_source and d == "y":
                i, j, amplitude = source_nodes[lev]
                np.testing.assert_allclose(current[i, j], amplitude, rtol=1e-12)
                current[i, j] -= amplitude
            work += area * np.sum(current * native[d, lev])
    err = [
        np.max(abs(q - qprev + dt * flow)) / scale,
        np.max(abs(gauss - q + q0)) / scale,
        abs(work - ref) / magnitude if magnitude else 0.0,
    ]
    max_error = np.maximum(max_error, err)
    qprev = q

xs, zs, ws = particles()
order = np.argsort(ws)
np.testing.assert_array_equal(ws[order], weights[np.argsort(weights)])
before = (initial_x > lo * h) & (initial_x < hi * h)
after = (xs > lo * h) & (xs < hi * h)
expected = ~before
if args.periodic_corner:
    expected[2:] = False
    assert (
        xs[ws == weights[2]].item() < 3 * h
        and xs[ws == weights[3]].item() > (nx - 3) * h
    )
    assert (
        zs[ws == weights[2]].item() < 3 * h
        and zs[ws == weights[3]].item() > (nz - 3) * h
    )
np.testing.assert_array_equal(after[order], expected[np.argsort(weights)])
assert (
    zs[ws == weights[0]].item() < 2 * h and zs[ws == weights[1]].item() > (nz - 2) * h
)
print("Maximum crossing continuity, Gauss and work residuals", max_error)
threshold = max(2.0e-11, 512 * np.finfo(grid("Efield_fp", "x", 0).dtype).eps)
assert np.max(max_error) < threshold
state = snapshot()
state.update(x=xs[order], z=zs[order], weight=ws[order])
if args.restart:
    with np.load(args.restart.parents[1] / "crossing_endpoint.npz") as expected:
        for name, value in state.items():
            np.testing.assert_allclose(
                value,
                expected[name],
                rtol=0,
                atol=64 * np.finfo(value.dtype).eps * np.max(abs(expected[name])),
            )
else:
    np.savez("crossing_endpoint.npz", **state)
libwarpx.finalize()
