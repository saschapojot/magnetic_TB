from datetime import datetime
import sys

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from name_conventions import  plotting_band_data_pkl_file_name

sp.init_printing(use_unicode=False, wrap_line=False)
from plot_energy_band.block_diagonalization import *
argErrCode = 20
data_non_existent_err_code=21

if (len(sys.argv) != 2):
    print("wrong number of arguments")
    print("example: python run_plot_band.py /path/to/mc.conf")
    exit(argErrCode)


confFileName = str(sys.argv[1])

def load_plotting_band_data(confFileName):
    conf_file_path = Path(confFileName)
    directory = conf_file_path.parent
    data_for_plotting_file_name=str(directory / plotting_band_data_pkl_file_name)
    try:
        with open(data_for_plotting_file_name, 'rb') as f:
            data_for_plotting = pickle.load(f)
        print(f"Successfully loaded data from {data_for_plotting_file_name}")
        return data_for_plotting
    except FileNotFoundError:
        print(f"Error: File not found at {data_for_plotting_file_name}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the pickle file: {e}")
        return None

data_for_plotting=load_plotting_band_data(confFileName)
if data_for_plotting is None:
    print("Error: Could not load data. Exiting.")
    exit(data_non_existent_err_code)
name=data_for_plotting["name"]
all_coords=data_for_plotting["all_coords"]
all_distances=data_for_plotting["all_distances"]
high_symmetry_indices=data_for_plotting["high_symmetry_indices"]
high_symmetry_labels=data_for_plotting["high_symmetry_labels"]
quantum_numbers_k=data_for_plotting["quantum_numbers_k"]
all_eigenvalues=data_for_plotting["all_eigenvalues"]
# --- Plotting bands ---

# Create a figure
plt.figure(figsize=(6, 5))
# all_eigenvalues is typically shape (num_k_points, num_bands)
# We iterate over the second dimension (bands) to plot each band line
num_bands = all_eigenvalues.shape[1]
for i in range(0,num_bands):
    # Plot the i-th column (band) against the k-path distance
    plt.plot(all_distances, all_eigenvalues[:, i], color='blue', linewidth=1.5)

# Add vertical lines for high symmetry points
for index in high_symmetry_indices:
    # Map the index to the actual distance value
    # Check bounds to avoid errors if index is out of range
    if index < len(all_distances):
        plt.axvline(x=all_distances[index], color='black', linestyle='--', linewidth=0.8)

# Set x-ticks to be the high symmetry points
valid_indices = [i for i in high_symmetry_indices if i < len(all_distances)]
tick_locations = [all_distances[i] for i in valid_indices]

plt.xticks(tick_locations, high_symmetry_labels, fontsize=14)


# Limit x-axis to the range of the path
plt.xlim(all_distances[0], all_distances[-1])
plt.ylim(-0.5,0.5)
# plt.yticks(fontsize=20)

# Y-axis label size
# plt.ylabel("Energy", fontsize=22)
# plt.yticks(ticks=plt.yticks()[0], labels=[])


# Title size
plt.title(f"{name}", fontsize=24)
plt.grid(alpha=0.3)
plt.tight_layout()
conf_file_path = Path(confFileName)
base_directory=conf_file_path.parent
out_pic_file_name=str(base_directory/"band.png")
plt.savefig(out_pic_file_name, bbox_inches='tight')


