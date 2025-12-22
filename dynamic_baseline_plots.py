from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import pandas as pd

from dynamic_baseline import (
    process_and_merge_csv_files,
    preprocess_data,
    calculate_dynamic_baseline,
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
    value_baseline_df = calculate_dynamic_baseline(
        df_preprocessed, param_col_name="value", strategy_params=strategy_params_value
    )

    # Calculate baseline for freq
    freq_baseline_df = calculate_dynamic_baseline(
        df_preprocessed, param_col_name="freq", strategy_params=strategy_params_freq
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
    # 1) Load and merge all CSVs
    df_raw = process_and_merge_csv_files(directory)
    
    print("strategy_params_value: ", strategy_params_value)
    print("strategy_params_freq: ", strategy_params_freq)

    # 3) Calculate dynamic baselines for both sensors on preprocessed data
    df_with_baselines = _calculate_baselines_for_sensors(
        df_raw,
        strategy_params_value=strategy_params_value,
        strategy_params_freq=strategy_params_freq,
    )

     # 2) Preprocess (filter by reaper lift and speed)
    df_preprocessed = preprocess_data(
        df_with_baselines,
        reaper_lift_threshold=reaper_lift_threshold,
        speed_threshold=speed_threshold,
    )

    print(df_raw.shape)
    print(df_with_baselines.shape)
    print(df_preprocessed.shape)

    # 4) Create figure with two subplots (matching style of provided image)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
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
        df_preprocessed["time"],
        df_preprocessed["baseline_value_smoothed"],
        color="tab:blue",
        linewidth=1.5,
        linestyle="--",
        label="Dynamic Baseline Value",
    )
    ax_mid.plot(
        df_preprocessed["time"],
        df_preprocessed["baseline_freq_smoothed"],
        color="tab:orange",
        linewidth=1.5,
        linestyle="--",
        label="Dynamic Baseline Freq",
    )

    ax_mid.set_title("2. After Preprocessing & Dynamic Baseline Application")
    ax_mid.set_ylabel("Sensor Values")
    ax_mid.grid(True, alpha=0.3)
    ax_mid.legend(loc="upper right")

    axes[-1].set_xlabel("Time")

    if output_path:
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    # 3431363532335116004a0032
    dir_path = "data/343136353233511600220035"
    plot_raw_and_dynamic_baseline(dir_path,
                                    strategy_params_value = {
                                        "min_diff": 100,
                                        "min_window1": 200,
                                        "min_window2": 500,
                                        "threshold_q_mins": 0.05,
                                        "final_smoothing_window": 5,
                                        "const_window": 100,
                                        "const_std_thresh": 5,
                                        "rolling_std_small": 5,    # Adjust based on sensor noise
                                        "rolling_window": 10},
                                    strategy_params_freq = {
                                        "min_diff": 100,
                                        "min_window1": 200,
                                        "min_window2": 500,
                                        "threshold_q_mins": 0.05,
                                        "final_smoothing_window": 5,
                                        "const_window": 100,
                                        "const_std_thresh": 5,
                                        "rolling_std_small": 5,
                                        "rolling_window": 10},
                                    output_path="result.png")


