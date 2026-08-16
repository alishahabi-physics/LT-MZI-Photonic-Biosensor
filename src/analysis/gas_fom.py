import csv
from pathlib import Path

import numpy as np

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "base"

SENSITIVITY_FILE = DATA_DIR / "gas_sensitivity_theory_check.csv"

FWHM_FILE = DATA_DIR / "gas_fwhm_results.csv"

OUTPUT_FILE = DATA_DIR / "gas_fom_results.csv"


# ==========================================================
# LOAD SENSITIVITY
# ==========================================================


def load_sensitivity(
    file_path: Path,
):

    rows = []

    with file_path.open(
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
                    "m": int(row["m"]),
                    "s_shift": (float(row["s_shift"]) if row["s_shift"] else np.nan),
                    "s_lt_simple": (
                        float(row["s_lt_ref_minus_sensor"])
                        if row["s_lt_ref_minus_sensor"]
                        else np.nan
                    ),
                    "s_implicit": (
                        float(row["s_implicit_ref_minus_sensor"])
                        if row["s_implicit_ref_minus_sensor"]
                        else np.nan
                    ),
                }
            )

    if not rows:
        raise RuntimeError("Sensitivity file is empty.")

    return rows


# ==========================================================
# LOAD FWHM
# ==========================================================


def load_fwhm(
    file_path: Path,
):

    rows = []

    with file_path.open(
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
                }
            )

    if not rows:
        raise RuntimeError("FWHM file is empty.")

    return rows


# ==========================================================
# MAIN
# ==========================================================


def main():

    sensitivity_rows = load_sensitivity(SENSITIVITY_FILE)

    fwhm_rows = load_fwhm(FWHM_FILE)

    if len(sensitivity_rows) != len(fwhm_rows):
        raise ValueError(
            "Sensitivity and FWHM files contain different numbers of rows."
        )

    results = []

    for sensitivity, fwhm in zip(
        sensitivity_rows,
        fwhm_rows,
    ):
        if not np.isclose(
            sensitivity["ri"],
            fwhm["ri"],
            atol=1e-12,
        ):
            raise ValueError("RI mismatch between sensitivity and FWHM data.")

        ri = sensitivity["ri"]

        fwhm_um = fwhm["fwhm_um"]

        if fwhm_um <= 0:
            raise ValueError(f"FWHM must be positive for RI={ri}.")

        # --------------------------------------------------
        # FOM from direct resonance shift
        # --------------------------------------------------

        if np.isfinite(sensitivity["s_shift"]):
            fom_shift = abs(sensitivity["s_shift"]) / fwhm_um

        else:
            fom_shift = np.nan

        # --------------------------------------------------
        # FOM from simple LT-MZI formula
        # --------------------------------------------------

        if np.isfinite(sensitivity["s_lt_simple"]):
            fom_simple = abs(sensitivity["s_lt_simple"]) / fwhm_um

        else:
            fom_simple = np.nan

        # --------------------------------------------------
        # FOM from implicit sensitivity
        # --------------------------------------------------

        if np.isfinite(sensitivity["s_implicit"]):
            fom_implicit = abs(sensitivity["s_implicit"]) / fwhm_um

        else:
            fom_implicit = np.nan

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        results.append(
            {
                "gas_refractive_index": ri,
                "m": sensitivity["m"],
                "lambda_exact_um": sensitivity["lambda_exact_um"],
                "fwhm_um": fwhm_um,
                "fwhm_nm": fwhm_um * 1000.0,
                "s_shift": sensitivity["s_shift"],
                "s_lt_simple": sensitivity["s_lt_simple"],
                "s_implicit": sensitivity["s_implicit"],
                "fom_shift": fom_shift,
                "fom_simple": fom_simple,
                "fom_implicit": fom_implicit,
            }
        )

    # ======================================================
    # PRINT RESULTS
    # ======================================================

    print()
    print("=" * 130)
    print("GAS FOM")
    print("=" * 130)

    print()

    print(
        f"{'RI':>7}"
        f"{'FWHM(nm)':>14}"
        f"{'S_shift':>15}"
        f"{'S_implicit':>15}"
        f"{'FOM_shift':>15}"
        f"{'FOM_implicit':>17}"
    )

    print("-" * 130)

    for result in results:

        def fmt(value):

            if np.isnan(value):
                return "N/A"

            return f"{value:.9f}"

        print(
            f"{result['gas_refractive_index']:>7.3f}"
            f"{result['fwhm_nm']:>14.6f}"
            f"{fmt(result['s_shift']):>15}"
            f"{fmt(result['s_implicit']):>15}"
            f"{fmt(result['fom_shift']):>15}"
            f"{fmt(result['fom_implicit']):>17}"
        )

    # ======================================================
    # STATISTICS
    # ======================================================

    fom_shift_values = np.array(
        [r["fom_shift"] for r in results if np.isfinite(r["fom_shift"])],
        dtype=float,
    )

    fom_implicit_values = np.array(
        [r["fom_implicit"] for r in results if np.isfinite(r["fom_implicit"])],
        dtype=float,
    )

    fom_simple_values = np.array(
        [r["fom_simple"] for r in results if np.isfinite(r["fom_simple"])],
        dtype=float,
    )

    print()
    print("=" * 130)
    print("FOM STATISTICS")
    print("=" * 130)

    print()

    print("FOM from S_shift")
    print(f"Mean   : {np.mean(fom_shift_values):.9f} RIU^-1")
    print(f"Median : {np.median(fom_shift_values):.9f} RIU^-1")
    print(f"Std    : {np.std(fom_shift_values, ddof=1):.9f} RIU^-1")
    print(f"Min    : {np.min(fom_shift_values):.9f} RIU^-1")
    print(f"Max    : {np.max(fom_shift_values):.9f} RIU^-1")

    print()

    print("FOM from S_implicit")
    print(f"Mean   : {np.mean(fom_implicit_values):.9f} RIU^-1")
    print(f"Median : {np.median(fom_implicit_values):.9f} RIU^-1")
    print(f"Std    : {np.std(fom_implicit_values, ddof=1):.9f} RIU^-1")
    print(f"Min    : {np.min(fom_implicit_values):.9f} RIU^-1")
    print(f"Max    : {np.max(fom_implicit_values):.9f} RIU^-1")

    print()

    print("FOM from simple LT-MZI sensitivity")

    print(f"Mean   : {np.mean(fom_simple_values):.9f} RIU^-1")
    print(f"Median : {np.median(fom_simple_values):.9f} RIU^-1")
    print(f"Std    : {np.std(fom_simple_values, ddof=1):.9f} RIU^-1")
    print(f"Min    : {np.min(fom_simple_values):.9f} RIU^-1")
    print(f"Max    : {np.max(fom_simple_values):.9f} RIU^-1")

    # ======================================================
    # SAVE CSV
    # ======================================================

    fieldnames = list(results[0].keys())

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

        writer.writerows(results)

    print()
    print(f"Saved to:\n{OUTPUT_FILE}")

    print()
    print("=" * 130)


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()

