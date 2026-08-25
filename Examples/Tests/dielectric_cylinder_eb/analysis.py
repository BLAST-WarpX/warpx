#!/usr/bin/env python3

import sys

import numpy as np
import yt


def analytic_fields(x, z, radius, epsilon_r, applied_field):
    r2 = x**2 + z**2
    r = np.sqrt(r2)
    inside = r < radius
    outside = ~inside

    phi = np.empty_like(x)
    ex = np.empty_like(x)
    ez = np.empty_like(x)

    interior_field = 2.0 * applied_field / (epsilon_r + 1.0)
    dielectric_factor = (epsilon_r - 1.0) / (epsilon_r + 1.0)

    phi[inside] = -interior_field * x[inside]
    ex[inside] = interior_field
    ez[inside] = 0.0

    dipole = applied_field * dielectric_factor * radius**2
    phi[outside] = -applied_field * x[outside] + dipole * x[outside] / r2[outside]
    ex[outside] = applied_field + dipole * (
        x[outside] ** 2 - z[outside] ** 2
    ) / r2[outside] ** 2
    ez[outside] = 2.0 * dipole * x[outside] * z[outside] / r2[outside] ** 2

    return phi, ex, ez


filename = sys.argv[1] if len(sys.argv) > 1 else "diags/diag1000001"

radius = 1.0
epsilon_r = 5.0
applied_field = 1.0
edge_margin = 1.0

ds = yt.load(filename)
data = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)

phi = data["phi"].to_ndarray()[:, :, 0]
ex = data["Ex"].to_ndarray()[:, :, 0]
ey = data["Ey"].to_ndarray()[:, :, 0]
ez = data["Ez"].to_ndarray()[:, :, 0]
epsilon = data["dielectric_epsilon"].to_ndarray()[:, :, 0]
dielectric_mask = data["dielectric_mask"].to_ndarray()[:, :, 0]

left_edge = ds.domain_left_edge.to_ndarray()
right_edge = ds.domain_right_edge.to_ndarray()
dims = ds.domain_dimensions
dx = (right_edge - left_edge) / dims

x = left_edge[0] + (np.arange(dims[0]) + 0.5) * dx[0]
z = left_edge[1] + (np.arange(dims[1]) + 0.5) * dx[1]
xx, zz = np.meshgrid(x, z, indexing="ij")
r = np.sqrt(xx**2 + zz**2)

phi_exact, ex_exact, ez_exact = analytic_fields(
    xx, zz, radius, epsilon_r, applied_field
)

interface_margin = 4.0 * min(dx[0], dx[1])
valid = (
    (np.abs(r - radius) > interface_margin)
    & (xx > left_edge[0] + edge_margin)
    & (xx < right_edge[0] - edge_margin)
    & (zz > left_edge[1] + edge_margin)
    & (zz < right_edge[1] - edge_margin)
)

field_error = np.sqrt(
    np.mean((ex[valid] - ex_exact[valid]) ** 2 + (ez[valid] - ez_exact[valid]) ** 2)
)
field_norm = np.sqrt(np.mean(ex_exact[valid] ** 2 + ez_exact[valid] ** 2))
field_rms_relative_error = field_error / field_norm

phi_error = phi[valid] - phi_exact[valid]
phi_error -= np.mean(phi_error)
phi_rms_relative_error = np.sqrt(np.mean(phi_error**2)) / np.sqrt(
    np.mean(phi_exact[valid] ** 2)
)

deep_inside = r < radius - interface_margin
far_outside = r > radius + interface_margin

print(f"Field RMS relative error: {field_rms_relative_error:.6e}")
print(f"Phi RMS relative error: {phi_rms_relative_error:.6e}")
print(f"Mean Ex inside cylinder: {np.mean(ex[deep_inside]):.6e}")

assert field_rms_relative_error < 0.04
assert phi_rms_relative_error < 0.04
assert np.max(np.abs(ey)) < 1.0e-14
assert np.allclose(epsilon[deep_inside], epsilon_r)
assert np.allclose(epsilon[far_outside], 1.0)
assert np.all((dielectric_mask >= -1.0e-12) & (dielectric_mask <= 1.0 + 1.0e-12))
