import csv
from pathlib import Path

import numpy as np

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "base"

INPUT_FILE = DATA_DIR / "gas_fsr_results.csv"

OUTPUT_FILE = DATA_DIR / "gas_fwhm_results.csv"


# ==========================================================
# FWHM FORMULA
# ==========================================================


def calculate_fwhm(
    fsr_um: float,
) -> float:
    """
    FWHM = FSR / pi

    Input:
        FSR in Âµm

    Output:
        FWHM in Âµm
    """

    return fsr_um / np.pi


# ==========================================================
# LOAD FSR DATA
# ==========================================================


def load_fsr_results(
    input_file: Path,
):

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
                    "gas_refractive_index": float(row["gas_refractive_index"]),
                    "m": int(row["m"]),
                    "lambda_res_um": float(row["lambda_res_um"]),
                    "delta_neff": float(row["delta_neff"]),
                    "fsr_um": float(row["fsr_um"]),
                    "fsr_nm": float(row["fsr_nm"]),
                }
            )

    if not rows:
        raise RuntimeError("gas_fsr_results.csv is empty.")

    return rows


# ==========================================================
# MAIN
# ==========================================================


def main():

    rows = load_fsr_results(INPUT_FILE)

    results = []

    for row in rows:
        fwhm_um = calculate_fwhm(row["fsr_um"])

        results.append(
            {
                "gas_refractive_index": row["gas_refractive_index"],
                "m": row["m"],
                "lambda_res_um": row["lambda_res_um"],
                "delta_neff": row["delta_neff"],
                "fsr_um": row["fsr_um"],
                "fsr_nm": row["fsr_nm"],
                "fwhm_um": fwhm_um,
                "fwhm_nm": fwhm_um * 1000.0,
            }
        )

    # ======================================================
    # PRINT
    # ======================================================

    print()
    print("=" * 100)
    print("GAS FWHM")
    print("=" * 100)

    print()

    print(f"{'RI':>7}{'m':>6}{'FSR (nm)':>16}{'FWHM (Âµm)':>18}{'FWHM (nm)':>18}")

    print("-" * 100)

    for result in results:
        print(
            f"{result['gas_refractive_index']:>7.3f}"
            f"{result['m']:>6d}"
            f"{result['fsr_nm']:>16.9f}"
            f"{result['fwhm_um']:>18.12f}"
            f"{result['fwhm_nm']:>18.9f}"
        )

    # ======================================================
    # STATISTICS
    # ======================================================

    fwhm_values_um = np.array(
        [result["fwhm_um"] for result in results],
        dtype=float,
    )

    fwhm_values_nm = fwhm_values_um * 1000.0

    print()
    print("=" * 100)
    print("FWHM STATISTICS")
    print("=" * 100)

    print(f"Mean   : {np.mean(fwhm_values_um):.12f} Âµm")

    print(f"Median : {np.median(fwhm_values_um):.12f} Âµm")

    print(f"Std    : {np.std(fwhm_values_um, ddof=1):.12f} Âµm")

    print(f"Min    : {np.min(fwhm_values_um):.12f} Âµm")

    print(f"Max    : {np.max(fwhm_values_um):.12f} Âµm")

    print()

    print(f"Mean   : {np.mean(fwhm_values_nm):.9f} nm")

    print(f"Median : {np.median(fwhm_values_nm):.9f} nm")

    print(f"Std    : {np.std(fwhm_values_nm, ddof=1):.9f} nm")

    print(f"Min    : {np.min(fwhm_values_nm):.9f} nm")

    print(f"Max    : {np.max(fwhm_values_nm):.9f} nm")

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


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()

