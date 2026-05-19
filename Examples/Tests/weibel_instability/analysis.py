#!/usr/bin/env python3

# Copyright 2023-2026 Juliette Pech, Edoardo Zoni
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
Analyse the Weibel instability: compare the temporal growth of magnetic-field energy against
the theoretical linear growth rate, fit an experimental growth rate via linear regression, and
animate the electron charge-density phase space over all output iterations.
"""

import math

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from openpmd_viewer import OpenPMDTimeSeries
from sklearn.linear_model import LinearRegression


def ln(x):
    # Vectorised natural log to handle both scalar and array inputs
    ln = np.vectorize(np.log)
    return ln(x)


# Theoretical growth curve using analytically estimated t_0
def f(t, t_0):
    exp = np.vectorize(math.exp)
    return exp(2 * gamma * (t - t_0))


# Regression-fitted growth curve using gamma_opt and t_0_opt
def g(t, t_0_opt):
    exp = np.vectorize(math.exp)
    return exp(2 * gamma_opt * (t - t_0_opt))


def animate(i):
    iter = iterations[i]
    cax.cla()
    data, info = ts.get_field(field="rho_electrons_1", iteration=iter, plot=False)
    im = ax.imshow(data, origin="lower")
    fig.colorbar(im, cax=cax)
    tx.set_text("Rho_electrons_1 \n Iteration {} - Time {:.2e} s".format(iter, time[i]))


# Load the openPMD time series written by the WarpX field diagnostics
ts = OpenPMDTimeSeries("diags/diag1")

t = ts.t
B_x_2 = []
B_y_2 = []
B_z_2 = []

for i in t:
    # Fetch B-field components and accumulate their spatially-summed squares
    B_x, info_x = ts.get_field(field="B", coord="x", t=i, plot=False)
    B_y, info_y = ts.get_field(field="B", coord="y", t=i, plot=False)
    B_z, info_z = ts.get_field(field="B", coord="z", t=i, plot=False)

    B_x_2 = np.append(B_x_2, np.sum(np.square(B_x)))
    B_y_2 = np.append(B_y_2, np.sum(np.square(B_y)))
    B_z_2 = np.append(B_z_2, np.sum(np.square(B_z)))

# Load field energy reduced diagnostics; columns: 0=step, 2=total, 3=electric, 4=magnetic energy
field_energy = np.loadtxt("diags/reducedfiles/EF.txt", skiprows=1)
assert field_energy.ndim == 2
assert field_energy.shape[0] >= 2

step = field_energy[:, 0]
total_energy = field_energy[:, 2]
electric_energy = field_energy[:, 3]
magnetic_energy = field_energy[:, 4]

# Consistency checks: monotone steps, finite values, and non-negative energies
assert np.all(np.diff(step) > 0)
assert np.all(np.isfinite(total_energy))
assert np.all(np.isfinite(electric_energy))
assert np.all(np.isfinite(magnetic_energy))
assert np.all(total_energy >= 0.0)
assert np.all(electric_energy >= 0.0)
assert np.all(magnetic_energy >= 0.0)

# Verify that magnetic energy grows over the run (Weibel instability signature)
early_magnetic_energy = np.mean(magnetic_energy[: field_energy.shape[0] // 2])
late_magnetic_energy = np.mean(magnetic_energy[field_energy.shape[0] // 2 :])
assert late_magnetic_energy > early_magnetic_energy

# Verify charge conservation: total rho must equal the sum of both electron populations
rho_1, _ = ts.get_field(
    field="rho_electrons_1", iteration=ts.iterations[-1], plot=False
)
rho_2, _ = ts.get_field(
    field="rho_electrons_2", iteration=ts.iterations[-1], plot=False
)
rho, _ = ts.get_field(field="rho", iteration=ts.iterations[-1], plot=False)

assert np.all(np.isfinite(rho_1))
assert np.all(np.isfinite(rho_2))
assert np.all(np.isfinite(rho))
np.testing.assert_allclose(rho, rho_1 + rho_2)

# Identify the time interval over which ln(B_x^2) grows linearly (linear-instability phase)
dt = t[1] - t[0]  # Uniform output time step (s)

# Time-derivative of ln(B_x^2); equals 2*gamma during the linear growth phase
dlnB_x_2dt = np.gradient(ln(B_x_2), dt)

# Remove non-finite values (inf/NaN) produced when B_x is zero
dlndt_cleaned = [x for x in dlnB_x_2dt if not (math.isinf(x) or math.isnan(x))]

# Indices where consecutive derivative values change by less than eps
indices = []
eps = 0.03e12  # Maximum allowed change between consecutive derivative values

for i in range(len(dlndt_cleaned) - 1):
    if abs(dlndt_cleaned[i + 1] - dlndt_cleaned[i]) < eps:
        indices.append(i)

# Group contiguous indices into (start, end) intervals
int_der_const = []
min = indices[0]
for i in range(len(indices) - 2):
    if indices[i + 1] - indices[i] < 4 and indices[i + 2] - indices[i + 1] >= 4:
        max = indices[i + 1]
        int_der_const.append((min, max))
    elif (
        indices[i + 1] - indices[i] < 4
        and indices[i + 2] - indices[i + 1] < 4
        and i + 2 != len(indices) - 1
    ):
        max = indices[i + 2]
    elif (
        indices[i + 1] - indices[i] < 4
        and indices[i + 2] - indices[i + 1] < 4
        and i + 2 == len(indices) - 1
    ):  # last segment: close and record the final interval
        max = indices[i + 2]
        int_der_const.append((min, max))
    else:
        min = indices[i + 1]

# Use the first constant-derivative interval as the fitting window
a = int_der_const[0][0]
b = int_der_const[0][1] + 1
X = t[a:b]
Y = B_x_2[a:b]

# Theoretical growth rate: B_x^2 ~ exp(2*gamma*(t - t_0)), with gamma = beta * w_p
beta = 0.01  # Beam normalised momentum p/(m_e*c)
# Plasma frequency: w_p = sqrt(n_0 * e^2 / (m_e * eps_0)), with n_0 = 2e25 1/m^3
w_p = math.sqrt(2e25 * (1.602e-19) ** 2 / (9.109e-31 * 8.854e-12))
# Weibel linear growth rate (non-relativistic limit)
gamma = beta * w_p

# Estimate t_0 from the mean offset between ln(B_x^2) and the theoretical slope
a0 = 2 * gamma
b0_opt = np.mean(ln(Y) - a0 * X)
t_0 = -b0_opt / a0

# Fit ln(B_x^2) vs t to extract the experimental growth rate
X = X.reshape((-1, 1))
model = LinearRegression().fit(X, ln(Y))
r_sq = model.score(X, ln(Y))

print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")
print("theoretical gamma times 2: {:e}".format(2 * gamma))

# Extract experimental growth rate and time offset from the regression coefficients
gamma_opt = model.coef_ / 2
t_0_opt = -model.intercept_ / (2 * gamma_opt)

# Semilog plot of B^2 components with theoretical and fitted growth curves
plt.figure()
plt.semilogy(t, B_x_2, label="B_x squared")
plt.semilogy(t, B_y_2, label="B_y squared")
plt.semilogy(t, B_z_2, label="B_z squared")
plt.semilogy(t, f(t, t_0), label="e^2*gamma*(t-t_0)")
plt.semilogy(t, g(t, t_0_opt), label="e^2*gamma*(t-t_0) optimized")

plt.title("Time evolution of energy in logarithmic scale")
plt.xlabel("time (s)")
plt.ylabel("B^2")
plt.xlim(0, 4e-12)
plt.ylim(0, 1e8)
plt.legend()
plt.savefig("weibel_magnetic_energy_growth.png")

# Animate the electron charge-density (rho_electrons_1) phase space over all output iterations
plt.rcParams["figure.autolayout"] = True

fig = plt.figure()

ax = fig.add_subplot(111)
div = make_axes_locatable(ax)
cax = div.append_axes("right", "10%", "10%")
data, info = ts.get_field(field="rho_electrons_1", iteration=0, plot=False)
im = ax.imshow(data)
cb = fig.colorbar(im, cax=cax)
tx = ax.set_title("Iteration 0", pad=20)
ax.set(xlabel="x", ylabel="z")

iterations = ts.iterations
time = ts.t

ani = animation.FuncAnimation(fig, animate, frames=len(iterations), interval=100)
ani.save("weibel_electron_charge_density.gif")

# Check that the fitted growth rate agrees with the theoretical value to within 10%
np.testing.assert_allclose(gamma_opt, gamma, rtol=0.1)
