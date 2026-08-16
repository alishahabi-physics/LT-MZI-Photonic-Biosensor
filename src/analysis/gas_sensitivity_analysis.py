import csv
from pathlib import Path

import numpy as np
from scipy.stats import linregress

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "base"

INPUT_FILE = DATA_DIR / "gas_sensitivity_comparison.csv"


# ==========================================================
# LOAD DATA
# ==========================================================


def load_results(input_file: Path):

    rows = []

    with input_file.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(
                {
                    "ri": float(row["gas_refractive_index"]),
                    "lambda_exact": float(row["lambda_exact_um"]),
                    "s_shift": float(row["sensitivity_shift_um_per_RIU"])
                    if row["sensitivity_shift_um_per_RIU"]
                    else np.nan,
                    "s_current": float(row["sensitivity_current_lambda_um_per_RIU"])
                    if row["sensitivity_current_lambda_um_per_RIU"]
                    else np.nan,
                    "s_previous": float(row["sensitivity_previous_lambda_um_per_RIU"])
                    if row["sensitivity_previous_lambda_um_per_RIU"]
                    else np.nan,
                }
            )

    if not rows:
        raise RuntimeError("Input file is empty.")

    return rows


# ==========================================================
# LINEAR FIT
# ==========================================================


def linear_sensitivity(
    ri,
    wavelength,
):

    fit = linregress(
        ri,
        wavelength,
    )

    return {
        "sensitivity": fit.slope,
        "intercept": fit.intercept,
        "r2": fit.rvalue**2,
        "p_value": fit.pvalue,
        "std_error": fit.stderr,
    }


# ==========================================================
# STATISTICS
# ==========================================================


def calculate_statistics(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    return {
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values, ddof=1) if values.size > 1 else 0.0,
        "minimum": np.min(values),
        "maximum": np.max(values),
    }


# ==========================================================
# MAIN
# ==========================================================


def main():

    rows = load_results(INPUT_FILE)

    ri = np.array([row["ri"] for row in rows])

    lambda_exact = np.array([row["lambda_exact"] for row in rows])

    s_shift = np.array([row["s_shift"] for row in rows])

    s_current = np.array([row["s_current"] for row in rows])

    s_previous = np.array([row["s_previous"] for row in rows])

    # ======================================================
    # LINEAR FIT â€” ALL GAS POINTS
    # ======================================================

    fit_all = linear_sensitivity(
        ri,
        lambda_exact,
    )

    # ======================================================
    # LINEAR FIT â€” EXCLUDE FIRST RI
    #
    # n = 1.002 ... 1.009
    # ======================================================

    mask_after_first = ri >= 1.002

    fit_after_first = linear_sensitivity(
        ri[mask_after_first],
        lambda_exact[mask_after_first],
    )

    # ======================================================
    # LOCAL SENSITIVITY STATISTICS
    # ======================================================

    stats_shift = calculate_statistics(s_shift)

    stats_current = calculate_statistics(s_current)

    stats_previous = calculate_statistics(s_previous)

    # ======================================================
    # PRINT
    # ======================================================

    print()
    print("=" * 110)
    print("GAS SENSITIVITY ANALYSIS")
    print("=" * 110)

    # ------------------------------------------------------
    # Linear fit â€” all
    # ------------------------------------------------------

    print()
    print("-" * 110)
    print("LINEAR FIT: n = 1.000 ... 1.009")
    print("-" * 110)

    print(f"Sensitivity : {fit_all['sensitivity']:.12f} Âµm/RIU")

    print(f"Intercept   : {fit_all['intercept']:.12f} Âµm")

    print(f"RÂ²          : {fit_all['r2']:.12f}")

    print()

    # ------------------------------------------------------
    # Linear fit â€” after first point
    # ------------------------------------------------------

    print("-" * 110)
    print("LINEAR FIT: n = 1.002 ... 1.009")
    print("-" * 110)

    print(f"Sensitivity : {fit_after_first['sensitivity']:.12f} Âµm/RIU")

    print(f"Intercept   : {fit_after_first['intercept']:.12f} Âµm")

    print(f"RÂ²          : {fit_after_first['r2']:.12f}")

    print()

    # ======================================================
    # S_SHIFT
    # ======================================================

    print("-" * 110)
    print("LOCAL SENSITIVITY â€” S_shift")
    print("-" * 110)

    print(f"Mean   : {stats_shift['mean']:.12f} Âµm/RIU")

    print(f"Median : {stats_shift['median']:.12f} Âµm/RIU")

    print(f"Std    : {stats_shift['std']:.12f} Âµm/RIU")

    print(f"Min    : {stats_shift['minimum']:.12f} Âµm/RIU")

    print(f"Max    : {stats_shift['maximum']:.12f} Âµm/RIU")

    print()

    # ======================================================
    # S_CURRENT
    # ======================================================

    print("-" * 110)
    print("LOCAL SENSITIVITY â€” S_current")
    print("-" * 110)

    print(f"Mean   : {stats_current['mean']:.12f} Âµm/RIU")

    print(f"Median : {stats_current['median']:.12f} Âµm/RIU")

    print(f"Std    : {stats_current['std']:.12f} Âµm/RIU")

    print(f"Min    : {stats_current['minimum']:.12f} Âµm/RIU")

    print(f"Max    : {stats_current['maximum']:.12f} Âµm/RIU")

    print()

    # ======================================================
    # S_PREVIOUS
    # ======================================================

    print("-" * 110)
    print("LOCAL SENSITIVITY â€” S_previous")
    print("-" * 110)

    print(f"Mean   : {stats_previous['mean']:.12f} Âµm/RIU")

    print(f"Median : {stats_previous['median']:.12f} Âµm/RIU")

    print(f"Std    : {stats_previous['std']:.12f} Âµm/RIU")

    print(f"Min    : {stats_previous['minimum']:.12f} Âµm/RIU")

    print(f"Max    : {stats_previous['maximum']:.12f} Âµm/RIU")

    print()

    # ======================================================
    # COMPARISON
    # ======================================================

    print("=" * 110)
    print("COMPARISON")
    print("=" * 110)

    print()

    print(f"{'Method':<35}{'Sensitivity (Âµm/RIU)':>25}")

    print("-" * 65)

    print(f"{'Linear fit â€” all points':<35}{fit_all['sensitivity']:>25.12f}")

    print(f"{'Linear fit â€” n >= 1.002':<35}{fit_after_first['sensitivity']:>25.12f}")

    print(f"{'Mean S_shift':<35}{stats_shift['mean']:>25.12f}")

    print(f"{'Mean S_current':<35}{stats_current['mean']:>25.12f}")

    print(f"{'Mean S_previous':<35}{stats_previous['mean']:>25.12f}")

    print()

    print("=" * 110)


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()

