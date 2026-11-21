"""Small standard-library reader for level-zero AMReX plotfile fields."""

import glob
import math
import os
import re
import struct


class PlotfileGeometry:
    def __init__(self, dimensions, prob_lo, prob_hi):
        self.domain_dimensions = tuple(dimensions) + (1,) * (3 - len(dimensions))
        self.domain_left_edge = tuple(prob_lo) + (0.0,) * (3 - len(prob_lo))
        self.domain_right_edge = tuple(prob_hi) + (1.0,) * (3 - len(prob_hi))

    def cell_center(self, index):
        return tuple(
            self.domain_left_edge[axis]
            + (index[axis] + 0.5)
            * (self.domain_right_edge[axis] - self.domain_left_edge[axis])
            / self.domain_dimensions[axis]
            for axis in range(3)
        )


def plotfiles(base_dir="."):
    return sorted(
        path
        for path in glob.glob(os.path.join(base_dir, "diags", "diag1*"))
        if os.path.isdir(path) and re.fullmatch(r"diag1\d+", os.path.basename(path))
    )


def last_plotfile(base_dir="."):
    plots = plotfiles(base_dir)
    if not plots:
        raise RuntimeError(f"No plotfiles under {base_dir}/diags/")
    return plots[-1]


def _integers(text):
    return tuple(int(value) for value in text.split(","))


def _fab_header(line):
    byte_match = re.match(r"FAB \(\((\d+),", line)
    box_match = re.search(
        r"\(\(([-0-9,]+)\) \(([-0-9,]+)\) \(([-0-9,]+)\)\)\s+(\d+)\s*$",
        line,
    )
    if not byte_match or not box_match:
        raise RuntimeError(f"Could not parse FAB header: {line!r}")
    return (
        int(byte_match.group(1)),
        _integers(box_match.group(1)),
        _integers(box_match.group(2)),
        int(box_match.group(4)),
    )


def load_fields(plotfile, names):
    with open(os.path.join(plotfile, "Header")) as header_file:
        header = [line.strip() for line in header_file]
    component_count = int(header[1])
    component_names = header[2 : 2 + component_count]
    dimension = int(header[2 + component_count])
    prob_lo = [float(value) for value in header[5 + component_count].split()]
    prob_hi = [float(value) for value in header[6 + component_count].split()]
    domain_match = re.match(
        r"\(\(([-0-9,]+)\) \(([-0-9,]+)\)", header[8 + component_count]
    )
    if not domain_match:
        raise RuntimeError(f"Could not parse domain box in {plotfile}/Header")
    domain_lo = _integers(domain_match.group(1))
    domain_hi = _integers(domain_match.group(2))
    dimensions = [domain_hi[axis] - domain_lo[axis] + 1 for axis in range(dimension)]
    geometry = PlotfileGeometry(dimensions, prob_lo, prob_hi)

    requested = {}
    for name in names:
        if name not in component_names:
            raise RuntimeError(f"{name} is absent from {plotfile}: {component_names}")
        requested[component_names.index(name)] = name
    fields = {name: {} for name in names}

    level_dir = os.path.join(plotfile, "Level_0")
    with open(os.path.join(level_dir, "Cell_H")) as cell_header:
        entries = re.findall(r"FabOnDisk:\s+(\S+)\s+(\d+)", cell_header.read())
    if not entries:
        raise RuntimeError(f"No FAB entries in {level_dir}/Cell_H")

    for filename, offset in entries:
        with open(os.path.join(level_dir, filename), "rb") as fab_file:
            fab_file.seek(int(offset))
            byte_count, lo, hi, fab_components = _fab_header(
                fab_file.readline().decode("ascii")
            )
            shape = tuple(hi[axis] - lo[axis] + 1 for axis in range(dimension))
            points = math.prod(shape)
            value_count = points * fab_components
            code = {4: "f", 8: "d"}.get(byte_count)
            if code is None:
                raise RuntimeError(f"Unsupported {byte_count}-byte FAB values")
            raw = fab_file.read(value_count * byte_count)
            if len(raw) != value_count * byte_count:
                raise RuntimeError(f"Short FAB read from {filename} at {offset}")
            values = struct.unpack(f"={value_count}{code}", raw)

        padded_shape = shape + (1,) * (3 - dimension)
        padded_lo = lo + (0,) * (3 - dimension)
        for component, name in requested.items():
            base = component * points
            for k in range(padded_shape[2]):
                for j in range(padded_shape[1]):
                    for i in range(padded_shape[0]):
                        local = i + padded_shape[0] * (j + padded_shape[1] * k)
                        index = (padded_lo[0] + i, padded_lo[1] + j, padded_lo[2] + k)
                        fields[name][index] = float(values[base + local])
    return geometry, fields


def all_finite(field):
    return all(math.isfinite(value) for value in field.values())


def max_abs(field, predicate=None):
    values = (
        abs(value)
        for index, value in field.items()
        if predicate is None or predicate(index)
    )
    return max(values, default=0.0)


def relative_field_norms(reference, candidate):
    if reference.keys() != candidate.keys():
        raise RuntimeError("Field index sets differ")
    delta2 = sum((candidate[index] - value) ** 2 for index, value in reference.items())
    reference2 = sum(value * value for value in reference.values())
    delta_max = max(
        (abs(candidate[index] - value) for index, value in reference.items()),
        default=0.0,
    )
    reference_max = max((abs(value) for value in reference.values()), default=0.0)
    return (
        math.sqrt(delta2 / max(reference2, 1.0e-60)),
        delta_max / max(reference_max, 1.0e-30),
    )
