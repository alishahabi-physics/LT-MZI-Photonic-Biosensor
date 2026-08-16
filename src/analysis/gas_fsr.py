import csv
from pathlib import Path

import numpy as np

from src.config.settings import DEVICE_LENGTH_UM

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "base"

INPUT_FILE = DATA_DIR / "gas_sensitivity_comparison.csv"

OUTPUT_FILE = DATA_DIR / "gas_fsr_results.csv"


# ==========================================================
# SETTINGS
# ==========================================================



# ==========================================================
# LOAD INPUT
# ==========================================================


def load_input(
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
                    "lambda_exact_um": float(row["lambda_exact_um"]),
                    "m": int(row["m"]),
                    "delta_neff": float(row["delta_neff_at_resonance"]),
                }
            )

    if not rows:
        raise RuntimeError("Input file is empty.")

    return rows


# ==========================================================
# CALCULATE FSR
# ==========================================================


def calculate_fsr(
    lambda_res_um: float,
    delta_neff: float,
    length_um: float,
) -> float:
    """
    FSR = lambda_res^2 / (2 * Delta_neff * L)

    Units:
        lambda_res : Âµm
        Delta_neff : dimensionless
        L           : Âµm

    Result:
        FSR         : Âµm
    """

    denominator = 2.0 * delta_neff * length_um

    if denominator == 0.0:
        raise ZeroDivisionError("FSR denominator is zero.")

    return lambda_res_um**2 / denominator


# ==========================================================
# MAIN
# ==========================================================


def main():

    rows = load_input(INPUT_FILE)

    results = []

    for row in rows:
        lambda_res = row["lambda_exact_um"]

        delta_neff = row["delta_neff"]

        fsr_um = calculate_fsr(
            lambda_res_um=lambda_res,
            delta_neff=delta_neff,
            length_um=DEVICE_LENGTH_UM,
        )

        results.append(
            {
                "gas_refractive_index": row["gas_refractive_index"],
                "m": row["m"],
                "lambda_res_um": lambda_res,
                "delta_neff": delta_neff,
                "fsr_um": fsr_um,
                "fsr_nm": fsr_um * 1000.0,
            }
        )

    # ======================================================
    # PRINT
    # ======================================================

    print()
    print("=" * 100)
    print("GAS FSR")
    print("=" * 100)

    print()

    print(
        f"{'RI':>7}"
        f"{'m':>6}"
        f"{'lambda_res (Âµm)':>20}"
        f"{'Delta_neff':>16}"
        f"{'FSR (Âµm)':>16}"
        f"{'FSR (nm)':>16}"
    )

    print("-" * 100)

    for result in results:
        print(
            f"{result['gas_refractive_index']:>7.3f}"
            f"{result['m']:>6d}"
            f"{result['lambda_res_um']:>20.12f}"
            f"{result['delta_neff']:>16.12f}"
            f"{result['fsr_um']:>16.12f}"
            f"{result['fsr_nm']:>16.6f}"
        )

    # ======================================================
    # STATISTICS
    # ======================================================

    fsr_values = np.array(
        [result["fsr_um"] for result in results],
        dtype=float,
    )

    print()
    print("=" * 100)
    print("FSR STATISTICS")
    print("=" * 100)

    print(f"Mean   : {np.mean(fsr_values):.12f} Âµm")

    print(f"Median : {np.median(fsr_values):.12f} Âµm")

    print(f"Std    : {np.std(fsr_values, ddof=1):.12f} Âµm")

    print(f"Min    : {np.min(fsr_values):.12f} Âµm")

    print(f"Max    : {np.max(fsr_values):.12f} Âµm")

    print(f"Mean   : {np.mean(fsr_values) * 1000.0:.6f} nm")

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


