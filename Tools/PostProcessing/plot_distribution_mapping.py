# Standard imports
import os
from collections import defaultdict
from itertools import product

# High-performance math
import numpy as np


class SimData:
    """
    Structure for easy access to load costs reduced diagnostics
    """

    def __init__(self, directory, prange, is_3D):
        """
        Set data-containing dir, data range; load data
        """
        self.is_3D = is_3D
        self._get_costs_reduced_diagnostics(directory, prange)

    def __call__(self, i):
        """
        Set data for current timestep
        """
        if not self.data_fields:
            print("No data_fields!")
            return

        if i not in self.keys:
            print("Index is out of range!")
            print("Valid keys are ", self.keys)
            return

        # Set field values at output i
        self.values = []
        for attr in list(self.data_fields[i].keys()):
            setattr(self, attr, self.data_fields[i][attr])
            self.values.append(attr)

        # Data_fields index currently set
        self.idx = i

    def _get_costs_reduced_diagnostics(self, directory, prange):
        """
        Read costs reduced diagnostics
        """
        # Load data
        self.directory, self.prange = directory, prange
        if not os.path.exists(directory):
            print("Directory " + directory + " does not exist")
            return

        data_fields = defaultdict(dict)
        self.data_fields, self.keys = data_fields, list(prange)

        data = np.genfromtxt(directory)
        if len(data.shape) == 1:
            data = data.reshape(-1, data.shape[0])

        steps = data[:, 0].astype(int)

        times = data[:, 1]
        data = data[:, 2:]

        # Compute the number of datafields saved per box
        n_data_fields = 0
        with open(directory) as f:
            h = f.readlines()[0]
            unique_headers = [
                "".join([ln for ln in w if not ln.isdigit()]) for w in h.split()
            ][2::]

        # Either 9 or 10 depending if GPU
        n_data_fields = 9 if len(set(unique_headers)) % 9 == 0 else 10
        f.close()

        # From data header, data layout is:
        #     [step, time,
        #      cost_box_0, proc_box_0, lev_box_0, i_low_box_0, j_low_box_0,
        #           k_low_box_0, num_cells_0, num_macro_particles_0,
        #           (, gpu_ID_box_0 if GPU run), hostname_box_0,
        #      cost_box_1, proc_box_1, lev_box_1, i_low_box_1, j_low_box_1,
        #           k_low_box_1, num_cells_1, num_macro_particles_1,
        #           (, gpu_ID_box_1 if GPU run), hostname_box_1
        #      ...
        #      cost_box_n, proc_box_n, lev_box_n, i_low_box_n, j_low_box_n,
        #           k_low_box_n, num_cells_n, num_macro_particles_n,
        #           (, gpu_ID_box_n if GPU run), hostname_box_n
        i, j, k = (
            data[0, 3::n_data_fields],
            data[0, 4::n_data_fields],
            data[0, 5::n_data_fields],
        )

        i_blocks = np.diff(np.array(sorted(i.astype(int))))
        j_blocks = np.diff(np.array(sorted(j.astype(int))))
        k_blocks = np.diff(np.array(sorted(k.astype(int))))

        i_non_zero = i_blocks[i_blocks != 0]
        j_non_zero = j_blocks[j_blocks != 0]
        k_non_zero = k_blocks[k_blocks != 0]

        #                   only one block in a dir - or smallest block size
        i_blocking_factor = 1 if len(i_non_zero) == 0 else i_non_zero.min()
        j_blocking_factor = 1 if len(j_non_zero) == 0 else j_non_zero.min()
        k_blocking_factor = 1 if len(k_non_zero) == 0 else k_non_zero.min()

        imax = i.astype(int).max() // i_blocking_factor
        jmax = j.astype(int).max() // j_blocking_factor
        kmax = k.astype(int).max() // k_blocking_factor

        for key in self.keys:
            row = np.where(key == steps)[0][0]
            costs = data[row, 0::n_data_fields].astype(float)
            ranks = data[row, 1::n_data_fields].astype(int)
            icoords = i.astype(int) // i_blocking_factor
            jcoords = j.astype(int) // j_blocking_factor
            kcoords = k.astype(int) // k_blocking_factor

            # Fill in cost array
            shape = (kmax + 1, jmax + 1, imax + 1)[: 2 + self.is_3D]
            coords = [
                coord[: 2 + self.is_3D] for coord in zip(kcoords, jcoords, icoords)
            ]

            cost_arr = np.full(shape, 0.0)
            rank_arr = np.full(shape, -1)
            for nc, cost in enumerate(costs):
                coord = coords[nc]
                cost_arr[coord] += cost
                rank_arr[coord] = ranks[nc]

            # For non-uniform blocks: fill with the corresponding cost/rank
            visited = np.full(shape, False)

            def dfs(corner, pos, prev):
                # Exit conditions
                if any([pos[i] >= shape[i] for i in range(len(shape))]):
                    return
                edges = (
                    list(rank_arr[corner[0] : pos[0] + 1, pos[1], pos[2]])
                    + list(rank_arr[pos[0], corner[1] : pos[1] + 1, pos[2]])
                    + list(rank_arr[pos[0], pos[1], corner[2] : pos[2] + 1])
                    if self.is_3D
                    else list(rank_arr[corner[0] : pos[0] + 1, pos[1]])
                    + list(rank_arr[pos[0], corner[1] : pos[1] + 1])
                )
                if visited[pos] or not set(edges).issubset(set([prev, -1])):
                    return

                visited[pos] = True
                if rank_arr[pos] not in [-1, prev]:
                    prev, corner = rank_arr[pos], pos
                else:
                    rank_arr[pos] = prev

                args = [[0, 1] for _ in range(len(shape))]
                neighbors = [
                    tuple(np.array(pos) + np.array(p))
                    for p in product(*args)
                    if not p == (0,) * len(shape)
                ]
                for n in neighbors:
                    dfs(corner, n, prev)

            for corner in coords:
                dfs(corner, corner, rank_arr[corner])

            self.data_fields[key]["cost_arr"] = cost_arr
            self.data_fields[key]["rank_arr"] = rank_arr

            # Compute load balance efficiency
            rank_to_cost_map = {r: 0.0 for r in set(ranks)}
            for c, r in zip(costs, ranks):
                rank_to_cost_map[r] += c

            efficiencies = np.array(list(rank_to_cost_map.values()))
            efficiencies /= efficiencies.max()
            self.data_fields[key]["ranks"] = np.array(list(rank_to_cost_map.keys()))
            self.data_fields[key]["lb_efficiencies"] = efficiencies
            self.data_fields[key]["lb_efficiency"] = efficiencies.mean()
            self.data_fields[key]["lb_efficiency_max"] = efficiencies.max()
            self.data_fields[key]["lb_efficiency_min"] = efficiencies.min()
            self.data_fields[key]["t"] = times[row]
            self.data_fields[key]["step"] = steps[row]
            # ...
