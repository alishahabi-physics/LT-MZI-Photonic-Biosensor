import csv
from pathlib import Path

import numpy as np
from scipy.stats import linregress

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "base"

SENSITIVITY_FILE = DATA_DIR / "gas_sensitivity_theory_check.csv"

FWHM_FILE = DATA_DIR / "gas_fwhm_results.csv"

OUTPUT_FILE = DATA_DIR / "gas_final_metrics.csv"


# ==========================================================
# LOAD SENSITIVITY DATA
# ==========================================================


def load_sensitivity():

    rows = []

    with SENSITIVITY_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(
                {
                    "ri": float(row["ri"]),
                    "lambda_exact_um": float(row["lambda_exact_um"]),
                    "s_shift": (float(row["s_shift"]) if row["s_shift"] else np.nan),
                    "s_implicit": (
                        float(row["s_implicit_ref_minus_sensor"])
                        if row["s_implicit_ref_minus_sensor"]
                        else np.nan
                    ),
                }
            )

    return rows


# ==========================================================
# LOAD FWHM
# ==========================================================


def load_fwhm():

    rows = []

    with FWHM_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(
                {
                    "ri": float(row["gas_refractive_index"]),
                    "fwhm_um": float(row["fwhm_um"]),
                    "fwhm_nm": float(row["fwhm_nm"]),
                }
            )

    return rows


# ==========================================================
# LINEAR FIT
# ==========================================================


def calculate_linear_sensitivity(
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
    }


# ==========================================================
# CALCULATE FINAL METRICS
# ==========================================================


def calculate_metrics(
    sensitivity_rows,
    fwhm_rows,
    ri_min,
):

    # ------------------------------------------------------
    # Select range
    # ------------------------------------------------------

    sensitivity_selected = [row for row in sensitivity_rows if row["ri"] >= ri_min]

    fwhm_selected = [row for row in fwhm_rows if row["ri"] >= ri_min]

    # ------------------------------------------------------
    # Arrays
    # ------------------------------------------------------

    ri = np.array(
        [row["ri"] for row in sensitivity_selected],
        dtype=float,
    )

    wavelength = np.array(
        [row["lambda_exact_um"] for row in sensitivity_selected],
        dtype=float,
    )

    # ------------------------------------------------------
    # Linear sensitivity
    # ------------------------------------------------------

    fit = calculate_linear_sensitivity(
        ri,
        wavelength,
    )

    sensitivity_linear = fit["sensitivity"]

    r2 = fit["r2"]

    intercept = fit["intercept"]

    # ------------------------------------------------------
    # Mean FWHM
    # ------------------------------------------------------

    fwhm_um = np.array(
        [row["fwhm_um"] for row in fwhm_selected],
        dtype=float,
    )

    mean_fwhm_um = np.mean(fwhm_um)

    mean_fwhm_nm = mean_fwhm_um * 1000.0

    # ------------------------------------------------------
    # FOM using linear sensitivity
    # ------------------------------------------------------

    fom_linear = abs(sensitivity_linear) / mean_fwhm_um

    # ------------------------------------------------------
    # Mean implicit sensitivity
    # Exclude first point automatically
    # ------------------------------------------------------

    implicit_values = np.array(
        [
            row["s_implicit"]
            for row in sensitivity_selected
            if np.isfinite(row["s_implicit"])
        ],
        dtype=float,
    )

    mean_implicit = np.mean(implicit_values)

    fom_implicit = abs(mean_implicit) / mean_fwhm_um

    # ------------------------------------------------------
    # Return
    # ------------------------------------------------------

    return {
        "ri_min": ri_min,
        "ri_max": float(np.max(ri)),
        "number_of_points": len(ri),
        "sensitivity_linear_um_per_RIU": sensitivity_linear,
        "intercept_um": intercept,
        "r2": r2,
        "mean_fwhm_um": mean_fwhm_um,
        "mean_fwhm_nm": mean_fwhm_nm,
        "fom_linear_RIU_inv": fom_linear,
        "mean_implicit_sensitivity_um_per_RIU": mean_implicit,
        "fom_implicit_RIU_inv": fom_implicit,
    }


# ==========================================================
# MAIN
# ==========================================================


def main():

    sensitivity_rows = load_sensitivity()

    fwhm_rows = load_fwhm()

    # ------------------------------------------------------
    # Two reporting ranges
    # ------------------------------------------------------

    metrics_all = calculate_metrics(
        sensitivity_rows,
        fwhm_rows,
        ri_min=1.000,
    )

    metrics_stable = calculate_metrics(
        sensitivity_rows,
        fwhm_rows,
        ri_min=1.002,
    )

    # ======================================================
    # PRINT
    # ======================================================

    print()
    print("=" * 110)
    print("FINAL GAS SENSOR METRICS")
    print("=" * 110)

    print()

    # ------------------------------------------------------
    # All range
    # ------------------------------------------------------

    print("-" * 110)
    print("RANGE: n = 1.000 ... 1.009")
    print("-" * 110)

    print(f"Points                     : {metrics_all['number_of_points']}")

    print(
        f"Sensitivity (linear)       : "
        f"{metrics_all['sensitivity_linear_um_per_RIU']:.12f} Âµm/RIU"
    )

    print(f"RÂ²                         : {metrics_all['r2']:.12f}")

    print(f"Mean FWHM                  : {metrics_all['mean_fwhm_nm']:.12f} nm")

    print(
        f"FOM (linear sensitivity)    : {metrics_all['fom_linear_RIU_inv']:.12f} RIU^-1"
    )

    print(
        f"Mean implicit sensitivity  : "
        f"{metrics_all['mean_implicit_sensitivity_um_per_RIU']:.12f} Âµm/RIU"
    )

    print(
        f"FOM (implicit sensitivity) : "
        f"{metrics_all['fom_implicit_RIU_inv']:.12f} RIU^-1"
    )

    print()

    # ------------------------------------------------------
    # Stable range
    # ------------------------------------------------------

    print("-" * 110)
    print("RANGE: n = 1.002 ... 1.009")
    print("-" * 110)

    print(f"Points                     : {metrics_stable['number_of_points']}")

    print(
        f"Sensitivity (linear)       : "
        f"{metrics_stable['sensitivity_linear_um_per_RIU']:.12f} Âµm/RIU"
    )

    print(f"RÂ²                         : {metrics_stable['r2']:.12f}")

    print(f"Mean FWHM                  : {metrics_stable['mean_fwhm_nm']:.12f} nm")

    print(
        f"FOM (linear sensitivity)    : "
        f"{metrics_stable['fom_linear_RIU_inv']:.12f} RIU^-1"
    )

    print(
        f"Mean implicit sensitivity  : "
        f"{metrics_stable['mean_implicit_sensitivity_um_per_RIU']:.12f} Âµm/RIU"
    )

    print(
        f"FOM (implicit sensitivity) : "
        f"{metrics_stable['fom_implicit_RIU_inv']:.12f} RIU^-1"
    )

    # ======================================================
    # SAVE
    # ======================================================

    output_rows = [
        {
            "range": "1.000-1.009",
            **metrics_all,
        },
        {
            "range": "1.002-1.009",
            **metrics_stable,
        },
    ]

    fieldnames = list(output_rows[0].keys())

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(output_rows)

    print()
    print("=" * 110)
    print(f"Saved to:\n{OUTPUT_FILE}")
    print("=" * 110)


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()

