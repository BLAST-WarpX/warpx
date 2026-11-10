"""Read native Yee fields with their valid boxes and the fine coverage mask."""

import numpy as np


def native_blocks(fields, level, family, direction):
    field = fields.get(f"{family}field_fp", dir="xyz"[direction], level=level)
    blocks = []
    for mfi in field:
        box = mfi.validbox()
        lo, hi = np.array(box.small_end), np.array(box.big_end)
        offset = lo - np.array(mfi.fabbox().small_end)
        region = tuple(slice(a, a + b + 1) for a, b in zip(offset, hi - lo))
        view = field.array(mfi).to_xp(copy=False, order="F")[(*region, 0)]
        blocks.append((lo, hi, view))
    return blocks


def host(view):
    return view.get() if hasattr(view, "get") else np.asarray(view)


def mesh_fields(fields, n, ratio):
    blocks = {
        (level, family, d): native_blocks(fields, level, family, d)
        for level in range(2)
        for family in "EB"
        for d in range(3)
    }
    # Bx has one more x face than x cells in every grid box.
    boxes = [
        [(lo, hi - [1, 0, 0]) for lo, hi, _ in blocks[level, "B", 0]]
        for level in range(2)
    ]
    covered = np.zeros((n, n, n), dtype=bool)
    for lo, hi in boxes[1]:
        covered[tuple(slice(i, j + 1) for i, j in zip(lo // ratio, hi // ratio))] = True
    return blocks, boxes, covered
