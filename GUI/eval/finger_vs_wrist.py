import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
# Updated Parameters
l_sw = 0.521  # Length of shoulder to wrist
l_h = 0.187   # Length of hand
h = 1.345     # Height
delta = np.linspace(135, 215, 150)  # Updated Delta angles in degrees
gamma = np.linspace(0, 60, 150)  # Updated Gamma angles in degrees

# # Statistics
# gamma_mean = 50.6096
# gamma_std = 6.8302

# delta_mean = 181.9744
# delta_std = 7.6189

# def calculate_68_percentile_range(mean, std):
#     lower_bound = mean -  std
#     upper_bound = mean +  std
#     return lower_bound, upper_bound

# # 95th Percentile Range Calculation
# def calculate_95_percentile_range(mean, std):
#     lower_bound = mean - 1.96 * std
#     upper_bound = mean + 1.96 * std
#     return lower_bound, upper_bound

# # Calculate ranges
# gamma_range = calculate_68_percentile_range(gamma_mean, gamma_std)
# delta_range = calculate_95_percentile_range(delta_mean, delta_std)
# print(gamma_range)
# delta = np.linspace(delta_range[0], delta_range[1], 150)  # Updated Delta angles in degrees
# gamma = np.linspace(gamma_range[0], gamma_range[1], 150)  # Updated Gamma angles in degrees

# Convert to radians for calculations
delta_rad = np.deg2rad(delta)  # Delta in radians
gamma_rad = np.deg2rad(gamma)  # Gamma in radians

# Compute l_si and theta
l_si = np.sqrt(l_sw**2 + l_h**2 - 2 * l_sw * l_h * np.cos(delta_rad))
theta = np.arcsin(l_h * np.sin(delta_rad) / l_si)

# Create 2D mesh grids for gamma and delta
gamma_mesh, delta_mesh = np.meshgrid(gamma, delta)
gamma_mesh_rad, delta_mesh_rad = np.meshgrid(gamma_rad, delta_rad)

# Compute x values
theta_mesh = np.arcsin(l_h * np.sin(delta_mesh_rad) / np.sqrt(l_sw**2 + l_h**2 - 2 * l_sw * l_h * np.cos(delta_mesh_rad)))
x = h * (np.tan(gamma_mesh_rad + theta_mesh) - np.tan(gamma_mesh_rad))
print("--distance variation--")
print("min:", np.min(x), "max:",np.max(x), "average:", np.average(x), "standard deviation:", np.std(x))
x_df = pd.DataFrame(x)
print(x_df.quantile([0.25,0.5,0.75]))

print("--angle variation--")
theta_mesh_deg = np.rad2deg(theta_mesh)
print("min:", np.min(theta_mesh_deg), "max:",np.max(theta_mesh_deg), "average:", np.average(theta_mesh_deg), "standard deviation:", np.std(theta_mesh_deg))


# 3D Plot for x
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(delta_mesh, gamma_mesh, x.T, cmap='viridis', edgecolor='none')
ax.set_xlabel("Delta (degrees)")
ax.set_ylabel("Gamma (degrees)")
ax.set_zlabel("distance (m)")
ax.set_title("3D Plot of x as a Function of Delta and Gamma")

# 2D Plot for Theta (affected only by Delta)
ax2 = fig.add_subplot(122)
ax2.plot(delta, np.rad2deg(theta), color='blue', label='Theta (degrees)')
ax2.set_xlabel("Delta (degrees)")
ax2.set_ylabel("Theta (degrees)")
ax2.set_title("2D Plot of Theta vs Delta")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
