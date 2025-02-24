#!/usr/bin/env python

"""
This analysis script checks that there is no spurious charge build-up when a particle is absorbed by an embedded boundary.

More specifically, this test simulates two particles of oppposite charge that are initialized at
the same position and then move in opposite directions. The particles are surrounded by a cylindrical
embedded boundary, and are absorbed when their trajectory intersects this boundary. With an
electromagnetic solver, this can lead to spurious charge build-up (i.e., div(E)!= rho/epsion_0)
that remains at the position where particle was absorbed.

Note that, in this test, there will also be a (non-spurious) component of div(E) that propagates
along the embedded boundary, due to electromagnetic waves reflecting on this boundary.
When checking for static, spurious charge build-up, we average div(E) in time to remove this component.

The test is performed in 2D, 3D and RZ.
(In 2D, the cylindrical embedded boundary becomes two parallel plates)
"""

from openpmd_viewer import OpenPMDTimeSeries

ts = OpenPMDTimeSeries("./diags/diag1/")

divE_stacked = ts.iterate(
    lambda iteration: ts.get_field("divE", iteration=iteration)[0]
)
start_avg_iter = 25
end_avg_iter = 100
divE_avg = divE_stacked[start_avg_iter:end_avg_iter].mean(axis=0)

# Adjust the tolerance so that the remaining error due to the propagating
# div(E) (after averaging) is below this tolerance, but so that any typical
# spurious charge build-up is above this tolerance. This is dimension-dependent.
dim = ts.fields_metadata["divE"]["geometry"]
if dim == "3dcartesian":
    tolerance = 7e-11
elif dim == "2dcartesian":
    tolerance = 3.5e-10
elif dim == "thetaMode":
    # In RZ: there are issues with divE on axis
    # Set the few cells around the axis to 0 for this test
    divE_avg[:, 13:19] = 0
    tolerance = 4e-12


def check_tolerance(array, tolerance):
    assert abs(array).max() <= tolerance, (
        f"Test did not pass: the max error {abs(array).max()} exceeded the tolerance of {tolerance}."
    )
    print("All elements of are within the tolerance.")


check_tolerance(divE_avg, tolerance)
