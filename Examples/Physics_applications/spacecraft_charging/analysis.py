#!/usr/bin/env python

"""
This script tests the potential-time profile on the surface of
a conducting sphere (spacecraft) immersed in an initially static
thermal plasma. The potential on the spacecraft decreases over
the time to reach an equilibrium floating potential.

An input Python file inputs_test_rz_spacecraft_charging_picmi.py is used.

The test will check the curve fitting parameters v0 and tau defined
by the following exponential function: phi(t)=v0(1-exp(-t/tau))
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import yt
from openpmd_viewer import OpenPMDTimeSeries
from scipy.optimize import curve_fit

yt.funcs.mylog.setLevel(0)

filename = sys.argv[1]
ts = OpenPMDTimeSeries(filename)
dt = 1.27e-8
t = []
phi = []
it = ts.iterations

for i in it:
    phi_i = ts.get_field("phi", iteration=i, plot=False)
    # Find the minimum value among all grids for this iteration
    phi_min = np.min(phi_i[0])
    phi.append(phi_min)
    t.append(dt * i)


def func(x, v0, tau):
    return v0 * (1 - np.exp(-np.array(x) / tau))


popt, pcov = curve_fit(func, t, phi)

plt.plot(t, phi, label="modelisation")
plt.plot(t, func(t, *popt), "r-", label="fit: v0=%5.3f, tau=%5.9f" % (popt[0], popt[1]))
plt.legend()
plt.savefig("min_phi_analysis.png")

print(
    "fit parameters between the min(phi) curve over the time and the function v0(1-exp(-t/tau)):"
)
print("v0=%5.3f, tau=%5.9f" % (popt[0], popt[1]))


tolerance_v0 = 0.04
tolerance_tau = 0.04
print("tolerance for v0 = " + str(tolerance_v0 * 100) + "%")
print("tolerance for tau = " + str(tolerance_tau * 100) + "%")

mean_v0 = -151.347
mean_tau = 0.000004351

diff_v0 = np.abs((popt[0] - mean_v0) / mean_v0)
diff_tau = np.abs((popt[1] - mean_tau) / mean_tau)

print("percentage error for v0 = " + str(diff_v0 * 100) + "%")
print("percentage error for tau = " + str(diff_tau * 100) + "%")

assert (diff_v0 < tolerance_v0) and (diff_tau < tolerance_tau), (
    "Test spacecraft_charging did not pass"
)
