import numpy as np


def compute_D_correct(input_signal):
    r"""
    Computes D using consecutive differences, correcting typos from compute_D_partial.
    :param input_signal: 1D numpy array
    :return: float, D value
    """
    T_len = len(input_signal)
    # TYPO FIX: compute consecutive differences instead of last minus all others
    signal_diff_one = input_signal[1:] - input_signal[:-1]

    # Apply formula to consecutive differences
    D_val = np.sum(np.sqrt(1 + signal_diff_one ** 2)) / (T_len - 1)

    return D_val


# Example usage with sample_arr_2
sample_arr_2 = np.sin(np.arange(0, 2.1 * np.pi, np.pi / 10))
d_val_2 = compute_D_correct(sample_arr_2)
print("Correct D value for sample_arr_2:", d_val_2)
