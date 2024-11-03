import matplotlib.pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries

ts = OpenPMDTimeSeries("./diags/diag_labframe/")


def extract_peak_E(iteration):
    """
    Extract peak electric field and its position
    """
    Ex, info = ts.get_field("E", "x", iteration=iteration)
    Ex_max = abs(Ex).max()
    z_max = info.z[abs(Ex).argmax()]
    return z_max, Ex_max


# Loop through the lab-frame snapshots and extract the peak electric field
z_max, Ex_max = ts.iterate(extract_peak_E)

# Create a figure
plt.figure(figsize=(8, 4))

# Plot of the E field growth
plt.subplot(121)  # Span all rows in the first column
plt.semilogy(z_max, Ex_max)
plt.ylim(2e7, 2e9)
plt.xlabel("z (m)")
plt.ylabel("Peak $E_x$ (V/m)")
plt.title("Growth of the radiation field\n along the undulator")

# Plots of snapshot
iteration = 16
plt.subplot(122)  # Upper right panel


plt.ylabel("$E_x$ (V/m)")
plt.xlabel("")
ts.get_particle(["z"], iteration=iteration, nbins=300, species="electrons", plot=True)
plt.title("")
plt.ylim(0, 30e12)
plt.ylabel("Electron density (a. u.)", color="b")
plt.twinx()
Ex, info = ts.get_field("E", "x", iteration=iteration, plot=True)
plt.ylabel("$E_x$ (V/m)", color="r")
plt.plot(info.z, Ex, color="r")
plt.ylim(-0.6e9, 0.4e9)
plt.xlabel("z (m)")
plt.title("Snapshot 1.6 m into the undulator")

plt.tight_layout()

plt.savefig("FEL.png")
