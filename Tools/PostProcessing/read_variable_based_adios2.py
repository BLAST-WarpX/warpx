import numpy as np
from adios2 import Stream


def species_path(name, field):
    """
    variable based adios2 particles data is stored in /data/particles/species_name/field/__data__
    this function is only to format this path

    Note : momenta are stored as 'momentum/x' and positions as 'position/x', but EM fields are stored as 'ex' (and not 'e/x')
    """
    return f"/data/particles/{name}/{field}/__data__"


def display_avail_variables(diag_path):
    """
    if you are not sure how the variable you are looking for is named, or if you
    want to check which variables are stored, this function displays every variable
    available
    """
    with Stream(diag_path, "r") as s:
        for _ in s.steps():
            if s.current_step() == 0:
                var_avail = s.available_variables()
                for name, _ in var_avail.items():
                    print(f"Found variable : {name}")
                break


def get_particle_data(
    diag_path, species_name, variables, n_iterations=-1, n_particles=-1
):
    """
    Retrieves the particle data for the given variables

    WARNING : depending on the case, RAM usage might be important

    INPUT:
    - diag_path : str
    - species names : str
    - variables : array of str
    optional:
    - n_iterations : int, will be computed if not given
    - n_particles : int, will be computed if not given

    OUTPUT:
    - one array of shape (n_iterations, n_particles) per variable given
    """
    if n_iterations == -1 or n_particles == -1:
        if n_iterations == -1:
            n_iterations = 0
            skip_for = False
        with Stream(diag_path, "r") as s:
            for _ in s.steps():
                if s.current_step() == 0 and n_particles == -1:
                    fieldPath = species_path(species_name, variables[0])
                    n_particles = (s.read(fieldPath)).shape[0]
                    if skip_for:
                        break
                n_iterations += 1

    data = np.zeros((len(variables), n_iterations, n_particles))

    with Stream(diag_path, "r") as s:
        for _ in s.steps():
            iteration = s.current_step()

            for i, variable in enumerate(variables):
                data[i, iteration, :] = s.read(species_path(species_name, variable))

                # progress bar
                if iteration % 100 == 0:
                    n_eg = int(20 * iteration / n_iterations)
                    print(
                        f"[{n_eg * '='}{(20 - n_eg) * ' '}] Iteration {iteration}",
                        end="\r" if iteration != n_iterations else "\n",
                    )

        return data


def get_single_traj(x, ids, particle_id):
    """
    Returns the trajectory of a single in the given space x

    INPUT :
    x :     numpy array
            shape = (n_iterations, n_particles)
    ids :   numpy array
            shape = (n_iterations, n_particles)
    particle_id : int
                  id of the particle to plot

    OUTPUT :
    (n_iterations,) numpy array
    """

    return x[ids == particle_id]


def select_particles(ids, iteration, *argv):
    """
    select particles by filtering values

    USAGE :
    select_ids = select_particles(id_t, 8000, (ux_t, [60e-6, 90e-6]))
    -> returns the ids off all particles with 60e-6 < ux < 90e-6 at iteration 8000

    select parameters can be stacked but need to be in tuple form (array, [range])

    INPUT :
    ids : numpy array (n_iterations, n_particles) float64
    iteration : int

    *argv : tuple (numpy array (n_iterations, n_particles) , 2-numbers array-like)

    OUTPUT:
    array with the selected ids

    """
    id_it = np.copy(ids[iteration, :])
    selected_ids = np.copy(ids[iteration, :])

    for arg in argv:
        # check if arg is in the expected shape
        if not isinstance(arg, tuple):
            raise TypeError(f"expected tuple argument, got {type(arg)}")
        if not isinstance(arg[0], np.ndarray):
            raise TypeError(f"expected np.ndarray, got {type(arg[0])}")
        if arg[0].shape != ids.shape:
            raise ValueError("Dimensions not matching")
        try:
            if len(arg[1]) != 2:
                raise ValueError(
                    f"limits should contain two values, but has {len(arg[1])}"
                )
        except TypeError:
            raise TypeError(f"Expected a 2 elements container, got {type(arg[1])}")

        x_it = (arg[0])[iteration, :]
        selected_ids = id_it[
            np.isin(id_it, selected_ids) & (x_it >= (arg[1])[0]) & (x_it <= (arg[1])[1])
        ]

    return selected_ids
