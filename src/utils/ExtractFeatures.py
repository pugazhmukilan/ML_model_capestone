import pandas as pd

# --- Your Final Feature List ---
TIME_DOMAIN_FEATURES = [
    "Magnitude_mean",
    "Magnitude_std_dev",
    "Magnitude_var",
    "Magnitude_rms",
    "Magnitude_maxmin_diff",
    "Magnitude_zero_cross_rt"
]

FREQ_DOMAIN_FEATURES = [
    "Magnitude_fft_energy",
    "Magnitude_fft_entropy",
    "Magnitude_fft_dom_freq",
    "Magnitude_fft_tot_power",
    "Magnitude_fft_pw_ar_dom_freq",
    "Magnitude_fft_flatness"
]

FINAL_FEATURES = [
    "Constancy_of_rest",
    "Kinetic_tremor",	
    "Postural_tremor",	
    "Rest_tremor"]


ALL_FEATURES = TIME_DOMAIN_FEATURES + FREQ_DOMAIN_FEATURES + FINAL_FEATURES


# --- Function 1: Select all final features ---
def select_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame containing only the 12 selected features.
    """
    return df[ALL_FEATURES]


# --- Function 2: Select only one feature ---
def select_feature(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """
    Returns DataFrame containing only the given feature column.
    Raises error if feature is not in the selected list.
    """
    if feature_name not in ALL_FEATURES:
        raise ValueError(f"'{feature_name}' is not in the selected feature list.")
    return df[[feature_name]]


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    # Example dummy DataFrame
    data = {
        "Magnitude_mean": [1, 2, 3],
        "Magnitude_std_dev": [0.1, 0.2, 0.3],
        "Magnitude_var": [0.01, 0.04, 0.09],
        "Magnitude_rms": [1.1, 2.1, 3.1],
        "Magnitude_maxmin_diff": [5, 6, 7],
        "Magnitude_zero_cross_rt": [10, 20, 30],
        "Magnitude_fft_energy": [100, 200, 300],
        "Magnitude_fft_entropy": [0.5, 0.6, 0.7],
        "Magnitude_fft_dom_freq": [50, 60, 70],
        "Magnitude_fft_tot_power": [400, 500, 600],
        "Magnitude_fft_pw_ar_dom_freq": [0.9, 0.8, 0.7],
        "Magnitude_fft_flatness": [0.2, 0.3, 0.4],
        "Other_column": [999, 999, 999]  # Extra col (will be ignored)
    }

    df = pd.DataFrame(data)

    # Get all selected features
    print("All Features:")
    print(select_all_features(df))

    # Get only one feature
    print("\nSingle Feature (Magnitude_rms):")
    print(select_feature(df, "Magnitude_rms"))
