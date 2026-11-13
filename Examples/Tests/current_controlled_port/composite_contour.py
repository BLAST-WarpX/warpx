"""Measure a physical Ampere contour from the completed simulation fields."""

import itertools

import numpy as np

from pywarpx import picmi


def measure_contours(simulation, lower, upper, ndim, planes, max_level):
    """Integrate interpolated H on a fixed rectangle, using leaf-grid data."""
    components = ("x", "y", "z") if ndim == 3 else ("y",)
    levels = []
    for level in range(max_level + 1):
        spacing = 1 / (32 * 2**level)
        fields = {}
        for component in components:
            wrapper = simulation.fields.get("Bfield_aux", dir=component, level=level)
            box = wrapper.box_array().minimal_box()
            start = np.array(box.small_end)
            end = np.array(box.big_end)
            values = np.squeeze(np.asarray(wrapper[()]))
            extent = end - start + 1
            guards = (np.array(values.shape) - extent) // 2
            assert np.all(np.array(values.shape) == extent + 2 * guards)
            nodal = np.zeros(ndim, dtype=int)
            if ndim == 3:
                nodal[components.index(component)] = 1
            fields[component] = (values, start - guards, nodal)
            if component == components[0]:
                cell_lower, cell_upper = start, end - nodal
        levels.append((spacing, cell_lower, cell_upper, fields))

    def h(point, component):
        for spacing, cell_lower, cell_upper, fields in reversed(levels):
            location = (np.array(point) - lower) / spacing
            cell = np.floor(location).astype(int)
            if not np.all((cell >= cell_lower) & (cell <= cell_upper)):
                continue
            values, origin, nodal = fields[component]
            location -= 0.5 * (1 - nodal)
            base = np.floor(location).astype(int)
            fraction = location - base
            result = 0.0
            for offset in itertools.product((0, 1), repeat=ndim):
                weight = np.prod(np.where(offset, fraction, 1 - fraction))
                if weight == 0:
                    continue
                index = base + offset
                mu = picmi.constants.mu0 * (
                    2 + 0.5 * (lower[-1] + (index[-1] + 0.5) * spacing)
                )
                result += weight * values[tuple(index - origin)] / mu
            return result
        raise AssertionError("contour point outside the grid")

    currents = []
    for z in planes:
        if ndim == 2:
            current = h([0.125, z], "y") - h([-0.125, z], "y")
        else:
            current = 0.0
            intervals = 8 * 2**max_level
            coordinates = np.linspace(-0.125, 0.125, intervals + 1)
            centers = 0.5 * (coordinates[:-1] + coordinates[1:])
            for coordinate in centers:
                current += (
                    h([coordinate, -0.125, z], "x")
                    - h([coordinate, 0.125, z], "x")
                    + h([0.125, coordinate, z], "y")
                    - h([-0.125, coordinate, z], "y")
                ) * (0.25 / intervals)
        currents.append(current)
    return np.array(currents)
