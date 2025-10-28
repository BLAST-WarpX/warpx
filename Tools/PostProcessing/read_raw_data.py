# Copyright 2017-2019 Andrew Myers, Axel Huebl, Maxence Thevenet
# Remi Lehe
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

from collections import namedtuple
from glob import glob

import numpy as np

HeaderInfo = namedtuple("HeaderInfo", ["version", "how", "ncomp", "nghost"])


def read_data(plt_file):
    """

    This function reads the raw (i.e. not averaged to cell centers) data
    from a WarpX plt file. The plt file must have been written with the
    plot_raw_fields option turned on, so that it contains a raw_data
    sub-directory. This is only really useful for single-level data.

    Arguments:

        plt_file : An AMReX plt_file file. Must contain a raw_data directory.

    Returns:

        A list of dictionaries where the keys are field name strings and the values
        are numpy arrays. Each entry in the list corresponds to a different level.

    Example:

        >>> data = read_data("plt00016")
        >>> print(data.keys())
        >>> print(data["Ex"].shape)

    """
    all_data = []
    raw_files = sorted(glob(plt_file + "/raw_fields/Level_*/"))
    for raw_file in raw_files:
        field_names = _get_field_names(raw_file)

        data = {}
        for field in field_names:
            data[field] = _read_field(raw_file, field)

        all_data.append(data)

    return all_data


def _get_field_names(raw_file):
    header_files = glob(raw_file + "*_H")
    return [hf.split("/")[-1][:-2] for hf in header_files]


def _string_to_numpy_array(s):
    return np.array([int(v) for v in s[1:-1].split(",")], dtype=np.int64)


def _line_to_numpy_arrays(line):
    lo_corner = _string_to_numpy_array(line[0][1:])
    hi_corner = _string_to_numpy_array(line[1][:])
    node_type = _string_to_numpy_array(line[2][:-1])
    return lo_corner, hi_corner, node_type


def _read_local_Header(header_file, dim):
    with open(header_file, "r") as f:
        t_snapshot = float(f.readline())
        if dim == 2:
            nx, nz = [int(x) for x in f.readline().split()]
            ny = 1
            xmin, zmin = [float(x) for x in f.readline().split()]
            ymin = 0
            xmax, zmax = [float(x) for x in f.readline().split()]
            ymax = 0
        if dim == 3:
            nx, ny, nz = [int(x) for x in f.readline().split()]
            xmin, ymin, zmin = [float(x) for x in f.readline().split()]
            xmax, ymax, zmax = [float(x) for x in f.readline().split()]
        field_names = f.readline().split()

    local_info = {
        "t_snapshot": t_snapshot,
        "field_names": field_names,
        "xmin": xmin,
        "ymin": ymin,
        "zmin": zmin,
        "xmax": xmax,
        "ymax": ymax,
        "zmax": zmax,
        "nx": nx,
        "ny": ny,
        "nz": nz,
    }
    return local_info


## ------------------------------------------------------------
## USE THIS INSTEAD OF THE PREVIOUS FUNCTION IF Header contains
## (x,y,z) min and max vectors instead of zmin and zmax
## ------------------------------------------------------------
# def _read_local_Header(header_file):
#     with open(header_file, "r") as f:
#         t_snapshot = float(f.readline())
#         axes_lo = [float(x) for x in f.readline().split()]
#         axes_hi = [float(x) for x in f.readline().split()]
#     local_info = {
#         't_snapshot' : t_snapshot,
#         'axes_lo' : axes_lo,
#         'axes_hi' : axes_hi
#         }
#     return local_info


def _read_global_Header(header_file):
    with open(header_file, "r") as f:
        nshapshots = int(f.readline())
        dt_between_snapshots = float(f.readline())
        gamma_boost = float(f.readline())
        beta_boost = float(f.readline())

    global_info = {
        "nshapshots": nshapshots,
        "dt_between_snapshots": dt_between_snapshots,
        "gamma_boost": gamma_boost,
        "beta_boost": beta_boost,
    }

    return global_info


def _read_header(header_file):
    with open(header_file, "r") as f:
        version = int(f.readline())
        how = int(f.readline())
        ncomp = int(f.readline())
        # If the number of ghost cells is the same in all directions,
        # s is a string of the form '16\n'.
        # If the number of ghost cells varies depending on the direction,
        # s is a string of the form '(9,8)\n' in 2D or '(9,8,9)\n' in 3D.
        s = f.readline()
        s = s.replace("(", "")  # remove left  parenthesis '(', if any
        s = s.replace(")", "")  # remove right parenthesis ')', if any
        nghost = np.fromstring(
            s, dtype=int, sep=","
        )  # convert from string to numpy array

        header = HeaderInfo(version, how, ncomp, nghost)

        # skip the next line
        f.readline()

        # read boxes
        boxes = []
        for line in f:
            clean_line = line.strip().split()
            if clean_line == [")"]:
                break
            lo_corner, hi_corner, node_type = _line_to_numpy_arrays(clean_line)
            boxes.append((lo_corner - nghost, hi_corner + nghost, node_type))

        # read the file and offset position for the corresponding box
        file_names = []
        offsets = []
        for line in f:
            if line.startswith("FabOnDisk:"):
                clean_line = line.strip().split()
                file_names.append(clean_line[1])
                offsets.append(int(clean_line[2]))

    return boxes, file_names, offsets, header


def _combine_boxes(boxes):
    lo_corners, hi_corners = zip(*[(box[0], box[1]) for box in boxes])
    domain_lo = np.min(lo_corners, axis=0)
    domain_hi = np.max(hi_corners, axis=0)
    return domain_lo, domain_hi


def _read_field(raw_file, field_name):
    header_file = raw_file + field_name + "_H"
    boxes, file_names, offsets, header = _read_header(header_file)

    dom_lo, dom_hi = _combine_boxes(boxes)
    data_shape = dom_hi - dom_lo + 1
    if header.ncomp > 1:
        data_shape = np.append(data_shape, header.ncomp)
    data = np.zeros(data_shape)

    for box, fn, offset in zip(boxes, file_names, offsets):
        lo = box[0] - dom_lo
        hi = box[1] - dom_lo
        shape = hi - lo + 1
        if header.ncomp > 1:
            shape = np.append(shape, header.ncomp)
        with open(raw_file + fn, "rb") as f:
            f.seek(offset)
            if header.version == 1:
                f.readline()  # skip the first line
            arr = np.fromfile(f, "float64", np.prod(shape))
            arr = arr.reshape(shape, order="F")
            box_shape = [slice(low, hig + 1) for low, hig in zip(lo, hi)]
            if header.ncomp > 1:
                box_shape += [slice(None)]
            data[tuple(box_shape)] = arr

    return data


def _read_buffer(snapshot, header_fn, _component_names):
    boxes, file_names, offsets, header = _read_header(header_fn)

    dom_lo, dom_hi = _combine_boxes(boxes)

    all_data = {}
    for i in range(header.ncomp):
        all_data[_component_names[i]] = np.zeros(dom_hi - dom_lo + 1)

    for box, fn, offset in zip(boxes, file_names, offsets):
        lo = box[0] - dom_lo
        hi = box[1] - dom_lo
        shape = hi - lo + 1
        size = np.prod(shape)
        with open(snapshot + "/Level_0/" + fn, "rb") as f:
            f.seek(offset)
            if header.version == 1:
                f.readline()  # skip the first line
            arr = np.fromfile(f, "float64", header.ncomp * size)
            for i in range(header.ncomp):
                comp_data = arr[i * size : (i + 1) * size].reshape(shape, order="F")
                data = all_data[_component_names[i]]
                data[tuple([slice(low, hig + 1) for low, hig in zip(lo, hi)])] = (
                    comp_data
                )
                all_data[_component_names[i]] = data
    return all_data


def read_reduced_diags(filename, delimiter=" "):
    """
    Read data written by WarpX Reduced Diagnostics, and return them into Python objects
    input:
    - filename name of file to open
    - delimiter (optional, default ',') delimiter between fields in header.
    output:
    - metadata_dict dictionary where first key is the type of metadata, second is the field
    - data dictionary with data
    """
    # Read header line
    unformatted_header = list(
        np.genfromtxt(
            filename, comments="@", max_rows=1, dtype="str", delimiter=delimiter
        )
    )
    # From header line, get field name, units and column number
    field_names = [s[s.find("]") + 1 : s.find("(")] for s in unformatted_header]
    field_units = [s[s.find("(") + 1 : s.find(")")] for s in unformatted_header]
    field_column = [s[s.find("[") + 1 : s.find("]")] for s in unformatted_header]
    # Load data and re-format to a dictionary
    data = np.loadtxt(filename, delimiter=delimiter)
    if data.ndim == 1:
        data_dict = {key: np.atleast_1d(data[i]) for i, key in enumerate(field_names)}
    else:
        data_dict = {key: data[:, i] for i, key in enumerate(field_names)}
    # Put header data into a dictionary
    metadata_dict = {}
    metadata_dict["units"] = {key: field_units[i] for i, key in enumerate(field_names)}
    metadata_dict["column"] = {
        key: field_column[i] for i, key in enumerate(field_names)
    }
    return metadata_dict, data_dict


def read_reduced_diags_histogram(filename, delimiter=" "):
    """
    Modified based on read_reduced_diags
    Two extra return objects:
    - bin_value: the values of bins
    - bin_data: the histogram data values of bins
    """
    # Read header line
    unformatted_header = list(
        np.genfromtxt(
            filename, comments="@", max_rows=1, dtype="str", delimiter=delimiter
        )
    )
    # From header line, get field name, units and column number
    field_names = [s[s.find("]") + 1 : s.find("(")] for s in unformatted_header]
    field_names[2:] = [s[s.find("b") : s.find("=")] for s in field_names[2:]]
    field_units = [s[s.find("(") + 1 : s.find(")")] for s in unformatted_header]
    field_column = [s[s.find("[") + 1 : s.find("]")] for s in unformatted_header]
    field_bin = [s[s.find("=") + 1 : s.find("(")] for s in unformatted_header]
    # Load data and re-format to a dictionary
    data = np.loadtxt(filename, delimiter=delimiter)
    if data.ndim == 1:
        data_dict = {key: data[i] for i, key in enumerate(field_names)}
    else:
        data_dict = {key: data[:, i] for i, key in enumerate(field_names)}
    # Put header data into a dictionary
    metadata_dict = {}
    metadata_dict["units"] = {key: field_units[i] for i, key in enumerate(field_names)}
    metadata_dict["column"] = {
        key: field_column[i] for i, key in enumerate(field_names)
    }
    # Save bin values
    bin_value = np.asarray(field_bin[2:], dtype=np.float64, order="C")
    if data.ndim == 1:
        bin_data = data[2:]
    else:
        bin_data = data[:, 2:]
    return metadata_dict, data_dict, bin_value, bin_data
