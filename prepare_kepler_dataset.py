import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, Tuple
import time

from dynamic_baseline import (
    process_and_merge_csv_files,
    preprocess_data
)

from dynamic_baseline_plots import (
    _calculate_baselines_for_sensors,
    _calculate_baselines_for_sensors_adaptive,
)


def parse_point_coordinates(point_str: str) -> Tuple[float, float]:
    """
    Parse point string "(longitude,latitude)" to separate coordinates.
    
    Parameters
    ----------
    point_str : str
        String in format "(longitude,latitude)" or "longitude,latitude"
    
    Returns
    -------
    Tuple[float, float]
        (longitude, latitude)
    """
    if pd.isna(point_str):
        return (np.nan, np.nan)
    
    # Remove parentheses if present
    point_str = str(point_str).strip()
    if point_str.startswith('(') and point_str.endswith(')'):
        point_str = point_str[1:-1]
    
    # Split by comma
    parts = point_str.split(',')
    if len(parts) != 2:
        return (np.nan, np.nan)
    
    try:
        longitude = float(parts[0].strip())
        latitude = float(parts[1].strip())
        return (longitude, latitude)
    except ValueError:
        return (np.nan, np.nan)


def create_kepler_dataset(
    directory: str,
    reaper_lift_threshold: float = 0.1,
    speed_threshold: Optional[float] = None,
    strategy_params_value: Optional[Dict[str, Any]] = None,
    strategy_params_freq: Optional[Dict[str, Any]] = None,
    output_filename: str = "harvest_with_delta.csv",
    include_freq: bool = True,
    delta_method: str = "absolute",  # "absolute" or "positive_only"
    min_delta_threshold: float = 0.0,
    start_datetime : Optional[str] = None,
) -> pd.DataFrame:
    """
    Create a dataset for kepler.gl visualization with potato mass deltas.
    
    Parameters
    ----------
    directory : str
        Directory containing CSV files
    reaper_lift_threshold : float
        Minimum reaper lift percentage to consider (default 0.1 = 10%)
    speed_threshold : Optional[float]
        Maximum speed threshold (default None = no speed filtering)
    strategy_params_value : Optional[Dict]
        Parameters for value sensor baseline calculation
    strategy_params_freq : Optional[Dict]
        Parameters for frequency sensor baseline calculation
    output_filename : str
        Output CSV filename
    include_freq : bool
        Whether to include frequency delta column
    delta_method : str
        "absolute" for abs(value - baseline) or "positive_only" for max(0, value - baseline)
    min_delta_threshold : float
        Minimum delta value to include (filter out small deltas)
    start_datetime : Optional[str]
        Start datetime filter (format: "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD")
        Only include data from this datetime onwards
    
    Returns
    -------
    pd.DataFrame
        Dataset with columns: longitude, latitude, delta_value, [delta_freq], timestamp, device_id
    """
    
    start_time = time.time()
    
    # 1) Load and merge all CSVs
    print("Step 1: Loading data...")
    df_raw = process_and_merge_csv_files(directory)

    if start_datetime:
        start_dt = pd.to_datetime(start_datetime)
        df_raw = df_raw[df_raw['time'] >= start_dt]
        print(f"   Applied start filter: >= {start_dt}")
        
        filtered_count = len(df_raw)
        print(f"   After time filtering: {filtered_count:,} points ")
    
    # 2) Parse coordinates from 'point' column
    print("Step 2: Parsing coordinates...")
    if 'point' not in df_raw.columns:
        raise KeyError("Dataset must contain 'point' column with coordinates")
    
    # Parse all coordinates at once (vectorized)
    coords = df_raw['point'].apply(parse_point_coordinates)
    df_raw['longitude'] = [c[0] for c in coords]
    df_raw['latitude'] = [c[1] for c in coords]
    
    # Count valid coordinates
    valid_coords = df_raw['longitude'].notna() & df_raw['latitude'].notna()
    
    # 3) Preprocess data
    print("Step 3: Preprocessing...")
    df_preprocessed = preprocess_data(
        df_raw,
        reaper_lift_threshold=reaper_lift_threshold,
        speed_threshold=speed_threshold,
    )
    print(f"   Filtered to {len(df_preprocessed):,} points ")
    
    # 4) Calculate dynamic baselines
    print("Step 4: Calculating baselines...")
    baseline_start = time.time()
    df_with_baselines = _calculate_baselines_for_sensors_adaptive(
        df_preprocessed,
        strategy_params_value=strategy_params_value,
        strategy_params_freq=strategy_params_freq,
    )
    print(f"   Baseline calculation: {time.time() - baseline_start:.1f}s")
    
    # 5) Calculate deltas
    print("Step 5: Calculating deltas...")
    
    if delta_method == "absolute":
        df_with_baselines["delta_value"] = (
            df_with_baselines["value"] - df_with_baselines["baseline_value"]
        ).abs()
        if include_freq and 'baseline_freq' in df_with_baselines.columns:
            df_with_baselines["delta_freq"] = (
                df_with_baselines["freq"] - df_with_baselines["baseline_freq"]
            ).abs()
    
    elif delta_method == "positive_only":
        df_with_baselines["delta_value"] = np.maximum(
            0, df_with_baselines["value"] - df_with_baselines["baseline_value"]
        )
        if include_freq and 'baseline_freq' in df_with_baselines.columns:
            df_with_baselines["delta_freq"] = np.maximum(
                0, df_with_baselines["freq"] - df_with_baselines["baseline_freq"]
            )
    
    else:
        raise ValueError(f"Unknown delta_method: {delta_method}. "
                       f"Use 'absolute' or 'positive_only'")
    
    # Filter by minimum delta threshold
    if min_delta_threshold > 0:
        before_filter = len(df_with_baselines)
        df_with_baselines = df_with_baselines[
            df_with_baselines["delta_value"] >= min_delta_threshold
        ]
        print(f"   Filtered by delta threshold ({min_delta_threshold}): "
              f"{len(df_with_baselines):,} points remaining "
              f"({before_filter - len(df_with_baselines):,} removed)")
    
    # 6) Prepare final dataset for kepler.gl
    print("Step 6: Preparing kepler.gl dataset...")
    
    # Select and rename columns
    kepler_columns = {
        "longitude": df_with_baselines["longitude"],
        "latitude": df_with_baselines["latitude"],
        "delta_value": df_with_baselines["delta_value"],
        "timestamp": df_with_baselines["time"],
    }
    
    if include_freq and 'delta_freq' in df_with_baselines.columns:
        kepler_columns["delta_freq"] = df_with_baselines["delta_freq"]
    
    # Add device_id if available
    if "device_id" in df_with_baselines.columns:
        kepler_columns["device_id"] = df_with_baselines["device_id"]
    
    # Add original sensor values for reference
    kepler_columns["value_raw"] = df_with_baselines["value"]
    if "freq" in df_with_baselines.columns:
        kepler_columns["freq_raw"] = df_with_baselines["freq"]
    
    # Add baselines for reference
    kepler_columns["baseline_value"] = df_with_baselines["baseline_value"]
    if "baseline_freq" in df_with_baselines.columns:
        kepler_columns["baseline_freq"] = df_with_baselines["baseline_freq"]
    
    # Add speed for filtering in kepler.gl
    if "speed" in df_with_baselines.columns:
        kepler_columns["speed"] = df_with_baselines["speed"]
    
    # Add reaper lift for filtering
    if "reaper_lift_percent" in df_with_baselines.columns:
        kepler_columns["reaper_lift_percent"] = df_with_baselines["reaper_lift_percent"]
    
    kepler_df = pd.DataFrame(kepler_columns)
    
    # Remove rows with invalid coordinates
    kepler_df = kepler_df.dropna(subset=["longitude", "latitude"])
    
    # Sort by timestamp
    kepler_df = kepler_df.sort_values("timestamp").reset_index(drop=True)
    
    # 7) Save to CSV
    print(f"Step 7: Saving to {output_filename}...")
    kepler_df.to_csv(output_filename, index=False, encoding='utf-8')
    
    total_time = time.time() - start_time
    print(f"   Total time: {total_time:.1f}s")
    
    return kepler_df

def plot_zone_timeseries(csv_path,
                         time_col="timestamp",
                         raw_col="value_raw",
                         baseline_col="baseline_value",
                         delta_col="delta_value",
                         speed_threshold: Optional[float] = None,
                         output_path=None):
    """
    Loads exported Kepler.gl filtered CSV (selected zone)
    and plots raw value, baseline and delta over time, plus speed with threshold.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file exported from Kepler.gl
    time_col : str
        Name of time column (default: "timestamp")
    raw_col : str
        Name of raw sensor value column (default: "value_raw")
    baseline_col : str
        Name of baseline column (default: "baseline_value")
    delta_col : str
        Name of delta column (default: "delta_value")
    speed_threshold : Optional[float]
        Speed threshold to plot as horizontal line (default: None)
    output_path : Optional[str]
        Path to save the plot (default: None, shows plot)
    """

    df = pd.read_csv(csv_path)

    # Convert time column
    df[time_col] = pd.to_datetime(df[time_col])

    # Sort by time
    df = df.sort_values(time_col)

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, constrained_layout=True)

    # --- Top plot: Sensor data ---
    ax_top = axes[0]
    ax_top.plot(df[time_col], df[raw_col], label="Raw Value", alpha=0.6, color="tab:blue")
    ax_top.plot(df[time_col], df[baseline_col], label="Baseline", alpha=0.8, color="tab:purple")
    # Delta
    # ax_top.plot(df[time_col], df[delta_col], label="Delta", alpha=0.8)

    ax_top.set_title("Sensor / Baseline / Delta over Time for Selected Zone")
    ax_top.set_ylabel("Sensor Values")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="upper right")

    # --- Bottom plot: Speed with threshold ---
    ax_bottom = axes[1]
    if "speed" in df.columns:
        ax_bottom.plot(
            df[time_col],
            df["speed"],
            color="tab:green",
            label="Speed",
        )
        if speed_threshold is not None:
            ax_bottom.axhline(
                speed_threshold,
                color="red",
                linestyle="--",
                linewidth=1.2,
                label=f"Speed threshold ({speed_threshold})",
            )
        ax_bottom.set_title("Speed with Threshold")
        ax_bottom.set_ylabel("Speed")
        ax_bottom.grid(True, alpha=0.3)
        ax_bottom.legend(loc="upper right")
    else:
        ax_bottom.text(0.5, 0.5, "No speed data available", 
                      ha="center", va="center", transform=ax_bottom.transAxes)
        ax_bottom.set_title("Speed with Threshold")
        ax_bottom.set_ylabel("Speed")

    axes[-1].set_xlabel("Time")
    
    if output_path:
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()



# Example usage
if __name__ == "__main__":
    # kepler_data = create_kepler_dataset(
    #     directory="data/3431363532335116004a0032b",
    #     reaper_lift_threshold=None,
    #     speed_threshold=7.0,
    #     strategy_params_value={
    #         "min_diff": 100,
    #         "min_window1": 300,
    #         "min_window2": 600,
    #         "threshold_q_mins": 0.1,
    #         "final_smoothing_window": 5,
    #         "const_window": 300,
    #         "const_std_thresh": 20,
    #         "rolling_std_small": 20,
    #         "rolling_window": 400,
    #         "min_low_points": 5,
    #     },
    #     strategy_params_freq=,
    #     output_filename="harvest_with_delta.csv",
    #     include_freq=True,
    #     delta_method="absolute",
    #     min_delta_threshold=-1,  # Ignore small deltas
    #     start_datetime="2025-09-19T00:00:00Z"
    # )

    # Common baseline parameters for all folders
    common_params_value = {
        "min_diff": 100,
        "min_window1": 300,
        "min_window2": 600,
        "threshold_q_mins": 0.1,
        "final_smoothing_window": 5,
        "const_window": 300,
        "const_std_thresh": 20,
        "rolling_std_small": 20,  # Adjust based on sensor noise
        "rolling_window": 400,
        "min_low_points": 5,
    }
    common_params_freq = common_params_value.copy()

    kepler_data = create_kepler_dataset(
        directory="data/3431363532335116004a0032b",
        reaper_lift_threshold=None,
        speed_threshold=7.0,
        strategy_params_value=common_params_value,
        strategy_params_freq=common_params_freq,
        output_filename="harvest_with_delta.csv",
        include_freq=True,
        delta_method="absolute",
        min_delta_threshold=-1,  # Ignore small deltas
        start_datetime="2025-09-18T00:00:00Z"
    )
    plot_zone_timeseries(csv_path="test3.csv", speed_threshold=7.0)
    