#!/usr/bin/env python3

from dataclasses import dataclass

import numpy as np
import yt

CONDUCTOR_RADIUS = 0.5
DIELECTRIC_RADIUS = 1.0
EPSILON_R = 5.0
APPLIED_FIELD = 1.0


@dataclass(frozen=True)
class ProblemParameters:
    conductor_radius: float = CONDUCTOR_RADIUS
    dielectric_radius: float = DIELECTRIC_RADIUS
    epsilon_r: float = EPSILON_R
    applied_field: float = APPLIED_FIELD


@dataclass(frozen=True)
class PlotfileData:
    filename: str
    left_edge: np.ndarray
    right_edge: np.ndarray
    dims: np.ndarray
    dx: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    xx: np.ndarray
    yy: np.ndarray
    zz: np.ndarray
    r: np.ndarray
    phi: np.ndarray
    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray
    epsilon: np.ndarray
    dielectric_mask: np.ndarray
    eb_covered: np.ndarray


@dataclass(frozen=True)
class AnalyticFields:
    phi: np.ndarray
    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray


@dataclass(frozen=True)
class ErrorMasks:
    valid: np.ndarray
    shell: np.ndarray
    deep_core: np.ndarray
    deep_shell: np.ndarray
    far_outside: np.ndarray
    outside_conductor: np.ndarray
    inside_dielectric_object: np.ndarray


@dataclass(frozen=True)
class ErrorMetrics:
    field_rms_relative_error: float
    shell_field_rms_relative_error: float
    phi_rms_relative_error: float


def analytic_fields(
    x,
    y,
    z,
    conductor_radius=CONDUCTOR_RADIUS,
    dielectric_radius=DIELECTRIC_RADIUS,
    epsilon_r=EPSILON_R,
    applied_field=APPLIED_FIELD,
):
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)

    alpha = (conductor_radius / dielectric_radius) ** 3
    shell_coeff = -3.0 * applied_field / (
        epsilon_r * (1.0 + 2.0 * alpha) + 2.0 * (1.0 - alpha)
    )
    shell_image_coeff = -shell_coeff * conductor_radius**3
    outer_image_coeff = applied_field * dielectric_radius**3 + shell_coeff * (
        dielectric_radius**3 - conductor_radius**3
    )

    phi = np.zeros_like(x)
    ex = np.zeros_like(x)
    ey = np.zeros_like(x)
    ez = np.zeros_like(x)

    shell = (r > conductor_radius) & (r < dielectric_radius)
    outside = r >= dielectric_radius

    for mask, potential_coeff, image_coeff in (
        (shell, shell_coeff, shell_image_coeff),
        (outside, -applied_field, outer_image_coeff),
    ):
        radius2 = r2[mask]
        radius3 = radius2 * np.sqrt(radius2)
        radius5 = radius2 * radius3
        phi[mask] = potential_coeff * x[mask] + image_coeff * x[mask] / radius3
        ex[mask] = (
            -potential_coeff
            + image_coeff * (3.0 * x[mask] ** 2 - radius2) / radius5
        )
        ey[mask] = 3.0 * image_coeff * x[mask] * y[mask] / radius5
        ez[mask] = 3.0 * image_coeff * x[mask] * z[mask] / radius5

    return phi, ex, ey, ez


def analytic_solution(data, params=ProblemParameters()):
    phi, ex, ey, ez = analytic_fields(
        data.xx,
        data.yy,
        data.zz,
        params.conductor_radius,
        params.dielectric_radius,
        params.epsilon_r,
        params.applied_field,
    )
    return AnalyticFields(phi=phi, ex=ex, ey=ey, ez=ez)


def load_plotfile(filename):
    ds = yt.load(filename)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )

    dims = np.asarray(ds.domain_dimensions, dtype=int)
    left_edge = ds.domain_left_edge.to_ndarray()
    right_edge = ds.domain_right_edge.to_ndarray()
    dx = (right_edge - left_edge) / dims

    x = left_edge[0] + (np.arange(dims[0]) + 0.5) * dx[0]
    y = left_edge[1] + (np.arange(dims[1]) + 0.5) * dx[1]
    z = left_edge[2] + (np.arange(dims[2]) + 0.5) * dx[2]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt(xx**2 + yy**2 + zz**2)

    return PlotfileData(
        filename=filename,
        left_edge=left_edge,
        right_edge=right_edge,
        dims=dims,
        dx=dx,
        x=x,
        y=y,
        z=z,
        xx=xx,
        yy=yy,
        zz=zz,
        r=r,
        phi=grid["phi"].to_ndarray(),
        ex=grid["Ex"].to_ndarray(),
        ey=grid["Ey"].to_ndarray(),
        ez=grid["Ez"].to_ndarray(),
        epsilon=grid["dielectric_epsilon"].to_ndarray(),
        dielectric_mask=grid["dielectric_mask"].to_ndarray(),
        eb_covered=grid["eb_covered"].to_ndarray(),
    )


def make_error_masks(
    data, params=ProblemParameters(), edge_margin=0.75, interface_cells=1.5
):
    interface_margin = interface_cells * np.min(data.dx)
    away_from_edges = (
        (data.xx > data.left_edge[0] + edge_margin)
        & (data.xx < data.right_edge[0] - edge_margin)
        & (data.yy > data.left_edge[1] + edge_margin)
        & (data.yy < data.right_edge[1] - edge_margin)
        & (data.zz > data.left_edge[2] + edge_margin)
        & (data.zz < data.right_edge[2] - edge_margin)
    )
    valid = (
        (data.r > params.conductor_radius + interface_margin)
        & (np.abs(data.r - params.dielectric_radius) > interface_margin)
        & away_from_edges
    )
    shell = valid & (data.r < params.dielectric_radius)
    deep_core = data.r < params.conductor_radius - interface_margin
    deep_shell = (data.r > params.conductor_radius + interface_margin) & (
        data.r < params.dielectric_radius - interface_margin
    )
    far_outside = data.r > params.dielectric_radius + interface_margin
    outside_conductor = data.r > params.conductor_radius + interface_margin
    inside_dielectric_object = data.r < params.dielectric_radius - interface_margin

    return ErrorMasks(
        valid=valid,
        shell=shell,
        deep_core=deep_core,
        deep_shell=deep_shell,
        far_outside=far_outside,
        outside_conductor=outside_conductor,
        inside_dielectric_object=inside_dielectric_object,
    )


def compute_error_metrics(data, exact, masks):
    field_error = np.sqrt(
        np.mean(
            (data.ex[masks.valid] - exact.ex[masks.valid]) ** 2
            + (data.ey[masks.valid] - exact.ey[masks.valid]) ** 2
            + (data.ez[masks.valid] - exact.ez[masks.valid]) ** 2
        )
    )
    field_norm = np.sqrt(
        np.mean(
            exact.ex[masks.valid] ** 2
            + exact.ey[masks.valid] ** 2
            + exact.ez[masks.valid] ** 2
        )
    )

    shell_field_error = np.sqrt(
        np.mean(
            (data.ex[masks.shell] - exact.ex[masks.shell]) ** 2
            + (data.ey[masks.shell] - exact.ey[masks.shell]) ** 2
            + (data.ez[masks.shell] - exact.ez[masks.shell]) ** 2
        )
    )
    shell_field_norm = np.sqrt(
        np.mean(
            exact.ex[masks.shell] ** 2
            + exact.ey[masks.shell] ** 2
            + exact.ez[masks.shell] ** 2
        )
    )

    phi_error = data.phi[masks.valid] - exact.phi[masks.valid]
    phi_norm = np.sqrt(np.mean(exact.phi[masks.valid] ** 2))

    return ErrorMetrics(
        field_rms_relative_error=field_error / field_norm,
        shell_field_rms_relative_error=shell_field_error / shell_field_norm,
        phi_rms_relative_error=np.sqrt(np.mean(phi_error**2)) / phi_norm,
    )
