# HW8_main.py
import os
import numpy as np
import scipy.io as sio
from self_py_fun.HW8Fun import produce_trun_mean_cov, plot_trunc_mean, plot_trunc_cov

# --------------------------
# Global variables
# --------------------------
save_dir = "K114"
subject_name = "K114"

# Load .mat data
mat_file_path = "/Users/sapnadavid/Documents/GitHub/BIOS-584/BIOS-584/data/K114_001_BCI_TRN_Truncated_Data_0.5_6.mat"
eeg_trunc_obj = sio.loadmat(mat_file_path)

# Check keys (optional)
print("MAT-file keys:", eeg_trunc_obj.keys())

# Extract relevant arrays
input_signal = eeg_trunc_obj['Signal']          # EEG data
input_type   = eeg_trunc_obj['Type'].flatten()  # Target labels

# Number of electrodes (update if different)
E_val = 16

# Calculate number of time points per electrode
length_per_electrode = input_signal.shape[1] // E_val
time_index = np.arange(length_per_electrode)

# Electrode names
electrode_name_ls = [f"E{i+1}" for i in range(E_val)]

# --------------------------
# Call functions
# --------------------------

# Compute truncated mean and covariance
signal_tar_mean, signal_ntar_mean, signal_tar_cov, signal_ntar_cov, signal_all_cov = produce_trun_mean_cov(
    input_signal, input_type, E_val
)

# Plot and save mean figure
plot_trunc_mean(
    signal_tar_mean, signal_ntar_mean, subject_name, time_index, E_val, electrode_name_ls,
    save_dir
)

# Plot and save covariance figures
plot_trunc_cov(signal_tar_cov, "Target", time_index, subject_name, E_val, electrode_name_ls, save_dir)
plot_trunc_cov(signal_ntar_cov, "Non-Target", time_index, subject_name, E_val, electrode_name_ls, save_dir)
plot_trunc_cov(signal_all_cov, "All", time_index, subject_name, E_val, electrode_name_ls, save_dir)

print(f"All figures saved in folder: {save_dir}")
