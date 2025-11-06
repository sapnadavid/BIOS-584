
import os
import numpy as np
import scipy.io as sio # This will be used to load an MATLAB file
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf
eeg_trunc_obj = sio.loadmat("/Users/sapnadavid/Documents/GitHub/BIOS-584/BIOS-584/data/K114_001_BCI_TRN_Truncated_Data_0.5_6.mat")

# Check the variable names stored in the .mat file
print(eeg_trunc_obj.keys())
def produce_trun_mean_cov(input_signal, input_type, E_val):

  # Derive dimensions
    sample_size_len, feature_len = input_signal.shape
    length_per_electrode = feature_len // E_val

    # Separate target and non-target signals
    target_signals = input_signal[input_type == 1]
    nontarget_signals = input_signal[input_type == 0]
    all_signals = input_signal.reshape(input_signal.shape[0], E_val, length_per_electrode)

    # Compute target mean and covariance
    if target_signals.shape[0] > 0:
        target_signals = target_signals.reshape(target_signals.shape[0], E_val, length_per_electrode)
        signal_tar_mean = np.mean(target_signals, axis=0)
        signal_tar_cov = np.array([np.cov(target_signals[:, e, :], rowvar=False) for e in range(E_val)])
    else:
        signal_tar_mean = np.full((E_val, length_per_electrode), np.nan)
        signal_tar_cov = np.full((E_val, length_per_electrode, length_per_electrode), np.nan)

    # Compute non-target mean and covariance
    if nontarget_signals.shape[0] > 0:
        nontarget_signals = nontarget_signals.reshape(nontarget_signals.shape[0], E_val, length_per_electrode)
        signal_ntar_mean = np.mean(nontarget_signals, axis=0)
        signal_ntar_cov = np.array([np.cov(nontarget_signals[:, e, :], rowvar=False) for e in range(E_val)])
    else:
        signal_ntar_mean = np.full((E_val, length_per_electrode), np.nan)
        signal_ntar_cov = np.full((E_val, length_per_electrode, length_per_electrode), np.nan)

    # Compute overall covariance
    signal_all_cov = np.array([np.cov(all_signals[:, e, :], rowvar=False) for e in range(E_val)])

    # Return all 5 arrays
    return [signal_tar_mean, signal_ntar_mean, signal_tar_cov, signal_ntar_cov, signal_all_cov]

def plot_trunc_mean(
        eeg_tar_mean, eeg_ntar_mean, subject_name, time_index, E_val, electrode_name_ls,
        save_dir, y_limit=np.array([-5, 8]), fig_size=(12, 12)
):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=fig_size)

    for i in range(E_val):
        row = i // 4
        col = i % 4
        ax = axes[row, col]

        ax.plot(time_index, eeg_tar_mean[i], color='red', label='Target')
        ax.plot(time_index, eeg_ntar_mean[i], color='blue', label='Non-Target')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(electrode_name_ls[i])
        ax.set_ylim(y_limit)
        ax.grid(True)
        if i == 0:
            ax.legend()

    fig.suptitle(f"Subject: {subject_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure instead of showing
    save_path = os.path.join(save_dir, "Mean.png")
    plt.savefig(save_path)
    plt.close()

def plot_trunc_cov(
        eeg_cov, cov_type, time_index, subject_name, E_val, electrode_name_ls,
        save_dir, fig_size=(14,12)
):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=fig_size)
    X, Y = np.meshgrid(time_index, time_index)

    for i in range(E_val):
        row = i // 4
        col = i % 4
        ax = axes[row, col]

        Z = eeg_cov[i]
        cf = ax.contourf(X, Y, Z, cmap='viridis')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Time (ms)')
        ax.set_title(electrode_name_ls[i])
        ax.invert_yaxis()
        fig.colorbar(cf, ax=ax)

    fig.suptitle(f"{subject_name} - {cov_type} Covariance", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure instead of showing
    save_name = f"Covariance_{cov_type}.png"
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    plt.close()