import numpy as np
from scipy.fft import fft
from scipy.stats import entropy

def extract_magnitude_features(accel_data, sampling_rate=100):
    """
    Extracts statistical and frequency-domain features from raw accelerometer/gyroscope data.
    
    Parameters
    ----------
    accel_data : np.ndarray
        A 2D numpy array of shape (n_samples, 3), where each column corresponds to 
        accelerometer/gyroscope readings along X, Y, Z axes over time.
    sampling_rate : int, optional (default=100)
        The sampling frequency (Hz) of the sensor data. Used for FFT calculation.
    
    Returns
    -------
    features : dict
        A dictionary containing the extracted features:
            - 'Magnitude_mean'       : Mean of magnitude signal
            - 'Magnitude_std_dev'    : Standard deviation of magnitude
            - 'Magnitude_var'        : Variance of magnitude
            - 'Magnitude_rms'        : Root mean square of magnitude
            - 'Magnitude_maxmin_diff': Difference between max and min magnitude
            - 'Magnitude_fft_energy' : Total spectral energy
            - 'Magnitude_fft_entropy': Spectral entropy
            - 'Magnitude_fft_tot_power': Total power of FFT
            - 'Magnitude_fft_flatness': Spectral flatness measure
    """
    
    # Compute magnitude signal from 3 axes
    magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
    
    # Time-domain features
    mean_val = np.mean(magnitude)
    std_val = np.std(magnitude)
    var_val = np.var(magnitude)
    rms_val = np.sqrt(np.mean(magnitude**2))
    maxmin_diff = np.max(magnitude) - np.min(magnitude)
    
    # Frequency-domain features
    fft_vals = np.abs(fft(magnitude))
    fft_vals = fft_vals[:len(fft_vals)//2]  # Take only positive frequencies
    
    power_spectrum = np.square(fft_vals)
    total_power = np.sum(power_spectrum)
    
    # Normalize for entropy calculation
    ps_norm = power_spectrum / np.sum(power_spectrum) if np.sum(power_spectrum) != 0 else power_spectrum
    spectral_entropy = entropy(ps_norm) if np.sum(ps_norm) > 0 else 0
    
    spectral_energy = np.sum(power_spectrum)
    spectral_flatness = np.exp(np.mean(np.log(power_spectrum + 1e-12))) / (np.mean(power_spectrum) + 1e-12)
    
    return {
        "Magnitude_mean": mean_val,
        "Magnitude_std_dev": std_val,
        "Magnitude_var": var_val,
        "Magnitude_rms": rms_val,
        "Magnitude_maxmin_diff": maxmin_diff,
        "Magnitude_fft_energy": spectral_energy,
        "Magnitude_fft_entropy": spectral_entropy,
        "Magnitude_fft_tot_power": total_power,
        "Magnitude_fft_flatness": spectral_flatness
    }
