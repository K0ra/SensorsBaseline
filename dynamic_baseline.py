import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


def process_and_merge_csv_files(directory: str) -> pd.DataFrame:
    """
    Read all CSV files from a directory and merge them into a single DataFrame.

    This mirrors the behaviour of `process_and_merge_csv_files` from
    `baseline_calculation.py`, but is kept local so this module can be used
    independently.
    """
    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    df_list = []

    for file in csv_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        if df.empty:
            continue
        df_list.append(df)

    if not df_list:
        raise ValueError(f"No non-empty CSV files found in directory: {directory}")

    # Pair each DataFrame with its first time value so we can sort correctly
    df_time_pairs = [
        (
            pd.to_datetime(df["time"].iloc[0]),
            df,
        )
        for df in df_list
        if "time" in df.columns and not df.empty
    ]

    if not df_time_pairs:
        raise ValueError(
            "No DataFrames with a 'time' column were found when merging CSV files."
        )

    df_time_pairs.sort(key=lambda x: x[0])
    df_list_sorted = [df for _, df in df_time_pairs]

    merged_df = pd.concat(df_list_sorted, ignore_index=True)

    if "time" in merged_df.columns:
        merged_df["time"] = pd.to_datetime(merged_df["time"])

    return merged_df


def preprocess_data(
    df: pd.DataFrame,
    reaper_lift_threshold: Optional[float] = None,
    speed_threshold: Optional[float] = 7.0,
    reaper_lift_column: str = "reaper_lift_percent",
    speed_column: str = "speed",
) -> pd.DataFrame:
    """
    Filter and preprocess data based on reaper lift percentage and speed.

    This follows the same logic as `PotatoHarvestBaselinePredictor.preprocess_data`
    from `baseline_calculation.py`, but implemented as a standalone function.
    """
    if reaper_lift_threshold is None:
        # Reasonable defaults if not provided
        reaper_lift_threshold = 20.0
    if speed_threshold is None:
        speed_threshold = 1.0

    if reaper_lift_column not in df.columns:
        raise KeyError(f"Column '{reaper_lift_column}' not found in DataFrame.")
    if speed_column not in df.columns:
        raise KeyError(f"Column '{speed_column}' not found in DataFrame.")

    mask = (df[reaper_lift_column] > reaper_lift_threshold) & (
        df[speed_column] > speed_threshold
    )
    df_filtered = df.loc[mask].copy()

    if not df_filtered.empty:
        df_filtered = df_filtered.sort_values("time")

    return df_filtered.reset_index(drop=True)


def calculate_dynamic_baseline(
    df_input: pd.DataFrame,
    param_col_name: str,
    strategy_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Calculate dynamic baseline using a local minimum search approach.
    Replaces the previous smoothing-based logic.

    Parameters
    ----------
    df_input:
        Input DataFrame that must contain a `time` column and the column specified
        by `param_col_name`.
    param_col_name:
        Name of the sensor column to use for baseline calculation (e.g. 'value'
        or 'freq').
    strategy_params:
        Dictionary with algorithm parameters. Supported keys:
          - min_window1 (default 3)
          - min_window2 (default 5)
          - min_diff (default 50)
          - threshold_q_mins (default 0.01)
          - fill_limit (default None)
          - final_smoothing_window (default 1)
    """
    # Validation
    required_cols = {"time", param_col_name}
    if not required_cols.issubset(df_input.columns):
        missing = required_cols - set(df_input.columns)
        raise KeyError(f"Missing columns: {missing}")

    # Setup parameters
    strategy_params = strategy_params or {}
    df = df_input.copy().sort_values("time").reset_index(drop=True)
    data_values = df[param_col_name].values
    n = len(data_values)

    # === 1. Calculating the rolling window statistics ===
    const_window = int(strategy_params.get("const_window", 50))
    const_std_thresh = strategy_params.get("const_std_thresh")
    rolling_window = int(strategy_params.get("rolling_window", 30))
    
    # Series for all rolling window calculations
    data_series = pd.Series(data_values)
    
    rolling_stats = {}
    
    # Standard deviation for different windows
    if const_std_thresh is not None:
        rolling_stats['std_const'] = data_series.rolling(
            const_window, center=True, min_periods=1
        ).std()
    
    rolling_stats['std_local'] = data_series.rolling(
        rolling_window, center=True, min_periods=1
    ).std()
    
    # Converting to NumPy for speed
    local_stds = rolling_stats['std_local'].values
    is_constant = (
        rolling_stats['std_const'].values <= float(const_std_thresh)
        if const_std_thresh is not None
        else np.zeros(n, dtype=bool)
    )

    # === 2. Algorithm parameters ===
    min_window1 = int(strategy_params.get("min_window1", 3))
    min_window2 = int(strategy_params.get("min_window2", 5))
    min_diff = float(strategy_params.get("min_diff", 50))
    threshold_q_mins = float(strategy_params.get("threshold_q_mins", 0.01))
    min_low_points = int(strategy_params.get("min_low_points", 3))
    rolling_std_small = float(strategy_params.get("rolling_std_small", 5))
    fill_limit = strategy_params.get("fill_limit")
    final_smoothing_window = int(strategy_params.get("final_smoothing_window", 1))

    # === 3. Auxiliary function for finding minima ===
    def _find_minimum_in_window(window_data: np.ndarray) -> Optional[float]:
        """Находит минимальное значение в окне с учетом условий"""
        if len(window_data) == 0:
            return None
        
        max_val = np.max(window_data)
        min_val = np.min(window_data)
        
        # Checking if there is enough variation
        if (max_val - min_val) <= min_diff:
            return None
        
        # Looking for low points
        low_points = window_data[window_data < (max_val - min_diff)]
        
        # Several low points are needed
        if len(low_points) < min_low_points:
            return None
        
        # Take the lowest values
        threshold_val = np.quantile(low_points, threshold_q_mins)
        selected_lows = low_points[low_points <= threshold_val]
        
        return float(np.mean(selected_lows)) if len(selected_lows) > 0 else None

    # === 4. Main loop with optimizations ===
    min_values = np.full(n, np.nan, dtype=float)
    
    # Pre-calculate the boundaries
    window1_half = min_window1 // 2
    window2_half = min_window2 // 2
    
    # Set boundaries where analysis is impossible
    start_idx = max(window1_half, window2_half)
    end_idx = n - start_idx
    
    for i in range(n):
        # 1. Constant regions - use the actual value
        if is_constant[i]:
            min_values[i] = data_values[i]
            continue
        
        # 2. Local plateaus - use the actual value
        if local_stds[i] < rolling_std_small:
            min_values[i] = data_values[i]
            continue
        
        # 3. Skip points at the boundaries where the windows are incomplete
        if i < start_idx or i >= end_idx:
            continue
        
        # 4. Check window 1
        start1 = i - window1_half
        end1 = i + window1_half + 1
        min_val1 = _find_minimum_in_window(data_values[start1:end1])
        
        if min_val1 is not None:
            min_values[i] = min_val1
            continue
        
        # 5. Check window 2 (if window 1 did not produce a result)
        start2 = i - window2_half
        end2 = i + window2_half + 1
        min_val2 = _find_minimum_in_window(data_values[start2:end2])
        
        if min_val2 is not None:
            min_values[i] = min_val2
    
    # === 5. Filling in the blanks (simplified approach) ===
    baseline_series = pd.Series(min_values)
    
    # If almost all are NaN, return the global minimum.
    if baseline_series.isna().all():
        baseline_series = pd.Series(data_values).fillna(np.nanmin(data_values))
    else:
        # Fill in the gaps with interpolation
        baseline_series = baseline_series.interpolate(
            method='linear', 
            limit=fill_limit
        )
        # Fill the extreme values ​​with the closest ones
        baseline_series = baseline_series.bfill().ffill()

    # === 6. Final smoothing ===
    if final_smoothing_window > 1:
        final_baseline = baseline_series.rolling(
            window=final_smoothing_window, 
            center=True, 
            min_periods=1
        ).mean()
    else:
        final_baseline = baseline_series

    result_cols = {
        "time": df["time"],
        param_col_name: data_values,
        "final_baseline": final_baseline.values,
        "super_smoothed": final_baseline.values  # Одна копия вместо двух
    }
    
    # Add auxiliary columns
    for col in ["speed", "reaper_lift_percent"]:
        if col in df.columns:
            result_cols[col] = df[col]
    
    return pd.DataFrame(result_cols)


def load_preprocess_and_calculate_baseline(
    directory: str,
    param_col_name: str,
    strategy_params: Optional[Dict[str, Any]] = None,
    reaper_lift_threshold: Optional[float] = None,
    speed_threshold: Optional[float] = None,
    reaper_lift_column: str = "reaper_lift_percent",
    speed_column: str = "speed",
) -> pd.DataFrame:
    """
    Convenience function that:
      1. Reads all CSVs from `directory` into a single DataFrame,
      2. Preprocesses the data (reaper lift + speed filters),
      3. Calculates a dynamic baseline for `param_col_name`.

    Returns the filtered DataFrame with the extra baseline columns.
    """
    merged_df = process_and_merge_csv_files(directory)
    preprocessed_df = preprocess_data(
        merged_df,
        reaper_lift_threshold=reaper_lift_threshold,
        speed_threshold=speed_threshold,
        reaper_lift_column=reaper_lift_column,
        speed_column=speed_column,
    )

    if preprocessed_df.empty:
        raise ValueError("No data left after preprocessing filters were applied.")

    baseline_df = calculate_dynamic_baseline(
        preprocessed_df, param_col_name=param_col_name, strategy_params=strategy_params
    )

    return baseline_df


