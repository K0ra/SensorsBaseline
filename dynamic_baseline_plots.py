import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import os
import pandas as pd
import time

from dynamic_baseline import (
    process_and_merge_csv_files,
    preprocess_data,
    calculate_dynamic_baseline,
    calculate_dynamic_baseline_downsample,
    adaptive_window_sizes,
    calculate_dynamic_baseline_adaptive,
)


def _calculate_baselines_for_sensors(
    df_preprocessed: pd.DataFrame,
    strategy_params_value: Optional[Dict[str, Any]] = None,
    strategy_params_freq: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Helper that calculates dynamic baselines for both `value` and `freq`
    sensors and merges the results back into a single DataFrame.
    """
    if strategy_params_value is None:
        strategy_params_value = {}
    if strategy_params_freq is None:
        strategy_params_freq = {}

    # Calculate baseline for value
    value_baseline_df = calculate_dynamic_baseline_downsample(df_preprocessed,
                                                    param_col_name="value",
                                                    strategy_params=strategy_params_value,
                                                    downsample_factor=1)

    # Calculate baseline for freq
    freq_baseline_df = calculate_dynamic_baseline_downsample(df_preprocessed,
                                                    param_col_name="freq",
                                                    strategy_params=strategy_params_freq,
                                                    downsample_factor=1)

    merged = df_preprocessed.copy()
    merged["baseline_value"] = value_baseline_df["final_baseline"]
    merged["baseline_value_smoothed"] = value_baseline_df["super_smoothed"]

    merged["baseline_freq"] = freq_baseline_df["final_baseline"]
    merged["baseline_freq_smoothed"] = freq_baseline_df["super_smoothed"]

    return merged


def _calculate_baselines_for_sensors_adaptive(
    df_preprocessed: pd.DataFrame,
    strategy_params_value: Optional[Dict[str, Any]] = None,
    strategy_params_freq: Optional[Dict[str, Any]] = None,
    adaptation_method: str = "auto",
) -> pd.DataFrame:
    """
    Helper that calculates adaptive dynamic baselines for both `value` and `freq`
    sensors using calculate_dynamic_baseline_adaptive() and merges the results
    back into a single DataFrame.
    
    Parameters
    ----------
    df_preprocessed:
        Preprocessed DataFrame with sensor data.
    strategy_params_value:
        Strategy parameters for 'value' sensor baseline calculation.
    strategy_params_freq:
        Strategy parameters for 'freq' sensor baseline calculation.
    adaptation_method:
        Method for adaptation: "auto" (default) or "simple".
        - "auto": Uses adaptive windows, with optimization for large segments
        - "simple": Always uses simple adaptive windows
    
    Returns
    -------
    DataFrame with original data plus baseline columns:
        - baseline_value: Final baseline for value sensor
        - baseline_value_smoothed: Smoothed baseline for value sensor
        - baseline_freq: Final baseline for freq sensor
        - baseline_freq_smoothed: Smoothed baseline for freq sensor
    """
    if strategy_params_value is None:
        strategy_params_value = {}
    if strategy_params_freq is None:
        strategy_params_freq = {}

    # Calculate adaptive baseline for value
    value_baseline_df = calculate_dynamic_baseline_adaptive(
        df_preprocessed,
        param_col_name="value",
        strategy_params=strategy_params_value,
        adaptation_method=adaptation_method,
    )

    # Calculate adaptive baseline for freq
    freq_baseline_df = calculate_dynamic_baseline_adaptive(
        df_preprocessed,
        param_col_name="freq",
        strategy_params=strategy_params_freq,
        adaptation_method=adaptation_method,
    )

    merged = df_preprocessed.copy()
    merged["baseline_value"] = value_baseline_df["final_baseline"]
    merged["baseline_value_smoothed"] = value_baseline_df["super_smoothed"]

    merged["baseline_freq"] = freq_baseline_df["final_baseline"]
    merged["baseline_freq_smoothed"] = freq_baseline_df["super_smoothed"]

    return merged


def plot_raw_and_dynamic_baseline(
    directory: str,
    reaper_lift_threshold: Optional[float] = None,
    speed_threshold: Optional[float] = None,
    strategy_params_value: Optional[Dict[str, Any]] = None,
    strategy_params_freq: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Plot the first two graphs from the calibration analysis:

    1. Raw sensor data (value & freq).
    2. After preprocessing with dynamic baselines overlaid.
    """
    start = time.time()
    # 1) Load and merge all CSVs
    df_raw = process_and_merge_csv_files(directory)

    print("Raw data is loaded")

     # 2) Preprocess (filter by reaper lift and speed)
    df_preprocessed = preprocess_data(
        df_raw,
        reaper_lift_threshold=reaper_lift_threshold,
        speed_threshold=speed_threshold,
    )

    print("Data is pre-processed by the speed in range "
        f"[0; {speed_threshold}] and reaper-lift > {reaper_lift_threshold}")

    # 3) Calculate dynamic baselines for both sensors on preprocessed data
    df_with_baselines = _calculate_baselines_for_sensors(
        df_preprocessed,
        strategy_params_value=strategy_params_value,
        strategy_params_freq=strategy_params_freq,
    )
    print(f"Algorithm time: {time.time() - start}")

    print(df_raw.shape)
    print(df_preprocessed.shape)
    print(df_with_baselines.shape)

    # 4) Create figure with three subplots (raw, baselines, speed)
    speed_thr = speed_threshold if speed_threshold is not None else 1.0
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        constrained_layout=True,
    )

    # --- Top plot: Raw sensor data ---
    ax_top = axes[0]
    ax_top.plot(df_raw["time"], df_raw["value"], color="tab:blue", label="Raw Value")
    ax_top.plot(df_raw["time"], df_raw["freq"], color="tab:orange", label="Raw Freq")
    ax_top.set_title("1. Raw Sensor Data")
    ax_top.set_ylabel("Raw Sensor Values")
    ax_top.legend(loc="upper right")
    ax_top.grid(True, alpha=0.3)

    # --- Second plot: Preprocessed + dynamic baselines ---
    ax_mid = axes[1]
    ax_mid.plot(
        df_raw["time"],
        df_raw["value"],
        color="tab:blue",
        alpha=0.6,
        label="Value",
    )
    ax_mid.plot(
        df_raw["time"],
        df_raw["freq"],
        color="tab:orange",
        alpha=0.6,
        label="Freq",
    )

    ax_mid.plot(
        df_with_baselines["time"],
        df_with_baselines["baseline_value_smoothed"],
        color="tab:purple",
        linewidth=1.5,
        linestyle="--",
        label="Dynamic Baseline Value",
    )
    ax_mid.plot(
        df_with_baselines["time"],
        df_with_baselines["baseline_freq_smoothed"],
        color="tab:red",
        linewidth=1.5,
        linestyle="--",
        label="Dynamic Baseline Freq",
    )

    ax_mid.set_title("2. After Preprocessing & Dynamic Baseline Application")
    ax_mid.set_ylabel("Sensor Values")
    ax_mid.grid(True, alpha=0.3)
    ax_mid.legend(loc="upper right")
    
    # Add adaptive windows visualization if adaptive windows were used
    if strategy_params_value and strategy_params_value.get("adaptive_windows"):
        # Calculate adaptive windows for visualization
        value_windows = adaptive_window_sizes(df_preprocessed["value"].values)
        plot_with_adaptive_windows(
            df_with_baselines,
            value_windows,
            axes=ax_mid,
            param_col_name="value",
        )

    # --- Third plot: Speed with threshold ---
    ax_bottom = axes[2]
    ax_bottom.plot(
        df_raw["time"],
        df_raw["speed"],
        color="tab:green",
        label="Speed",
    )
    if speed_thr is not None:
        ax_bottom.axhline(
            speed_thr,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"Speed threshold ({speed_thr})",
        )
    ax_bottom.set_title("3. Speed with Threshold")
    ax_bottom.set_ylabel("Speed")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc="upper right")

    axes[-1].set_xlabel("Time")

    if output_path:
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()


def plot_with_adaptive_windows(
    df: pd.DataFrame,
    windows_info: Dict[str, Any],
    axes: Optional[Any] = None,
    param_col_name: str = "value",
) -> None:
    """
    Visualize adaptive window boundaries on the plot.
    
    Parameters
    ----------
    df:
        DataFrame with time and sensor data.
    windows_info:
        Dictionary containing window information:
        - min_window1: Small window size
        - min_window2: Large window size
        - rolling_window: Rolling statistics window
        - const_window: Constant region detection window
        - change_points: (optional) Array of indices where segments change
    axes:
        Matplotlib axes to plot on. If None, uses current axes.
    param_col_name:
        Name of the sensor column to plot.
    """
    if axes is None:
        axes = plt.gca()
    
    # Extract window sizes for title
    min_w1 = windows_info.get("min_window1", "N/A")
    min_w2 = windows_info.get("min_window2", "N/A")
    roll_w = windows_info.get("rolling_window", "N/A")
    const_w = windows_info.get("const_window", "N/A")
    
    # Update title with window information
    current_title = axes.get_title()
    window_info_str = (
        f" | Windows: min1={min_w1}, min2={min_w2}, "
        f"rolling={roll_w}, const={const_w}"
    )
    axes.set_title(current_title + window_info_str, fontsize=9)
    
    # Plot vertical lines at change points if provided
    change_points = windows_info.get("change_points")
    if change_points is not None and len(change_points) > 0:
        for cp_idx in change_points:
            if cp_idx < len(df):
                cp_time = df["time"].iloc[cp_idx]
                axes.axvline(
                    x=cp_time,
                    color="gray",
                    linestyle=":",
                    alpha=0.5,
                    linewidth=0.8,
                )


def plot_folder_to_file(
    base_data_dir: str,
    folder_name: str,
    reaper_lift_threshold: Optional[float] = None,
    speed_threshold: Optional[float] = None,
    strategy_params_value: Optional[Dict[str, Any]] = None,
    strategy_params_freq: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Convenience wrapper: plot one folder under `base_data_dir` and save
    the figure to `<folder_name>.png` in the project root (or caller's cwd).
    """
    folder_path = os.path.join(base_data_dir, folder_name)
    # output_filename = f"results/{folder_name}.png"

    print(f"Processing folder: {folder_path}")
    plot_raw_and_dynamic_baseline(
        folder_path,
        reaper_lift_threshold=reaper_lift_threshold,
        speed_threshold=speed_threshold,
        strategy_params_value=strategy_params_value,
        strategy_params_freq=strategy_params_freq,
        # output_path=output_filename,
    )
    # print(f"Saved figure: {output_filename}")


if __name__ == "__main__":
    data_root = "data"

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

    # start_time = time.time()

    # 343136353233511600220035
    # 3431363532335116004a0032b

    # plot_folder_to_file(
    #         base_data_dir=data_root,
    #         folder_name="3431363532335116004a0032b",
    #         speed_threshold=7.0,
    #         strategy_params_value=common_params_value,
    #         strategy_params_freq=common_params_freq,
    #     )
    
    # Load and preprocess data
    folder_path = os.path.join(data_root, "3431363532335116004a0032b")
    df_raw = process_and_merge_csv_files(folder_path)
    df_preprocessed = preprocess_data(
        df_raw,
        reaper_lift_threshold=20.0,
        speed_threshold=7.0,
    )
    
    # Calculate adaptive baseline for value sensor
    print("\nCalculating adaptive baseline for 'value' sensor...")
    adaptive_params_value = {
        "min_diff": 20,
        "min_window1": 10,
        "min_window2": 20,
        "threshold_q_mins": 0.05,
        "final_smoothing_window": 5,
        "const_std_thresh": 20,
        "rolling_std_small": 5,
        "min_low_points": 5,
    }
    
    df_adaptive_value = calculate_dynamic_baseline_adaptive(
        df_preprocessed,
        param_col_name="value",
        strategy_params=adaptive_params_value,
        adaptation_method="auto",
    )
    
    # Get adaptive window information for visualization
    value_windows = adaptive_window_sizes(df_preprocessed["value"].values)
    print(f"Adaptive windows calculated: {value_windows}")
    
    # Create a plot with adaptive windows visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        df_preprocessed["time"],
        df_preprocessed["value"],
        color="tab:blue",
        alpha=0.6,
        label="Value",
    )
    ax.plot(
        df_adaptive_value["time"],
        df_adaptive_value["final_baseline"],
        color="tab:purple",
        linewidth=1.5,
        linestyle="--",
        label="Adaptive Baseline",
    )
    ax.set_title("Adaptive Baseline Calculation Example")
    ax.set_ylabel("Sensor Values")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    # Visualize adaptive windows on the plot
    plot_with_adaptive_windows(
        df_adaptive_value,
        value_windows,
        axes=ax,
        param_col_name="value",
    )

    plt.show()
    
    # plt.tight_layout()
    # plt.savefig("results/adaptive_baseline_example.png", dpi=150)
    # print("Saved adaptive baseline plot to: results/adaptive_baseline_example.png")
    # plt.close()

    # print(f"Total time: {time.time() - start_time}")

    # Process each subfolder under `data`
    # for name in os.listdir(data_root):
    #     folder_path = os.path.join(data_root, name)
    #     print(folder_path)
    #     if not os.path.isdir(folder_path):
    #         continue

    #     plot_folder_to_file(
    #         base_data_dir=data_root,
    #         folder_name=name,
    #         speed_threshold=7.0,
    #         strategy_params_value=common_params_value,
    #         strategy_params_freq=common_params_freq,
    #     )

