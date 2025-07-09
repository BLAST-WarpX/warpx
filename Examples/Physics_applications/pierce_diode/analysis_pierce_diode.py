#!/usr/bin/env python3


import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import e, epsilon_0, m_u

# Parameters used in the corresponding input script
kV = 1000
cm = 0.01
extractor_voltage = -93.0 * kV
d_plate = 8.0 * cm
ion_mass = 39 * m_u

# Calculate Child-Langmuir limit for a given voltage
J_CL = (
    (4 / 9)
    * epsilon_0
    * np.sqrt(2 * abs(e) / ion_mass)
    * abs(extractor_voltage) ** (3 / 2)
    / d_plate**2
)

ts = OpenPMDTimeSeries("./diags/diag1/")

jz_t_last, meta = ts.get_field("j", "z", plot=False, iteration=ts.iterations[-1])
time_cur = ts.current_t
plt.figure()
plt.plot(meta.z, jz_t_last)
plt.axhline(y=J_CL, color="black", linestyle="-", label=r"$y = J_{CL \ limit}$")
plt.xlabel("z, mm")
plt.ylabel(
    r"$J_z, A/m^2$ at " + "t = " + str(np.round(time_cur / 1e-6, 5)) + r" $ \mu s$"
)
plt.legend()
plt.ylim([30, 40])
plt.savefig("pierce_diode_1d_analysis.png")
tolerance = 0.15  # corresponds to 15 %
assert np.all(np.abs(jz_t_last - J_CL) / J_CL <= tolerance), (
    "Child–Langmuir limit is violated!"
)
