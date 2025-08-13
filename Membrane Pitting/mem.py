import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label

# ---- Parameters ----
filename = "membrane_surface_map.dat"
bin_size = 10.0  # Å per bin (match VMD script)

# ---- Load data ----
data = pd.read_csv(filename, delim_whitespace=True, comment="#", names=["i", "j", "avgZ"])
nx = data["i"].max() + 1
ny = data["j"].max() + 1

# Create Z grid
z_grid = np.full((nx, ny), np.nan)
for _, row in data.iterrows():
    i, j = int(row["i"]), int(row["j"])
    z_grid[i, j] = row["avgZ"]

# ---- Define pit threshold ----
mean_z = np.nanmean(z_grid)
std_z = np.nanstd(z_grid)
threshold = mean_z - 1.5 * std_z  # Pits = 1.5 std dev below mean

# Create binary mask: 1 = pit, 0 = not
pit_mask = (z_grid < threshold) & ~np.isnan(z_grid)

# Label connected pits
labeled_pits, num_features = label(pit_mask)

# ---- Analyze each pit ----
pits_info = []
for pit_id in range(1, num_features + 1):
    mask = labeled_pits == pit_id
    pit_values = z_grid[mask]
    area = np.sum(mask) * (bin_size ** 2)  # Å²
    depth = mean_z - np.nanmin(pit_values)
    pits_info.append({
        "Pit ID": pit_id,
        "Area (Å²)": area,
        "Depth (Å)": depth,
        "Min Z": np.nanmin(pit_values),
        "Mean Z": np.nanmean(pit_values),
        "Num bins": np.sum(mask)
    })

# ---- Convert to DataFrame and print ----
pits_df = pd.DataFrame(pits_info)
print(pits_df.sort_values("Depth (Å)", ascending=False))

# ---- Optional: plot with pit regions outlined ----
plt.figure(figsize=(8, 6))
plt.imshow(z_grid.T, origin='lower', cmap='viridis')
plt.contour(labeled_pits.T, levels=np.arange(1, num_features+1), colors='r', linewidths=0.5)
plt.colorbar(label="Z-height (Å)")
plt.title("Detected Membrane Pits")
plt.xlabel("X bin index")
plt.ylabel("Y bin index")
plt.tight_layout()
plt.show()
