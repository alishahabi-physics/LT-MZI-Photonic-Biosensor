import csv
from pathlib import Path

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "base"

INPUT_FILE = DATA_DIR / "gas_peak_results.csv"

OUTPUT_FILE = DATA_DIR / "gas_sensitivity_results.csv"


# ==========================================================
# Load peak results
# ==========================================================


def load_peak_results(
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
                    "lambda_find_peaks_um": float(row["lambda_find_peaks_um"]),
                    "m": int(row["m"]),
                }
            )

    if not rows:
        raise RuntimeError("gas_peak_results.csv is empty.")

    return rows


# ==========================================================
# Calculate sensitivity
# ==========================================================


def calculate_sensitivity(
    rows,
):

    results = []

    previous_ri = None
    previous_lambda = None

    for row in rows:
        current_ri = row["gas_refractive_index"]

        current_lambda = row["lambda_exact_um"]

        # --------------------------------------------------
        # First point
        # --------------------------------------------------

        if previous_ri is None:
            sensitivity = None

        else:
            delta_n = current_ri - previous_ri

            delta_lambda = current_lambda - previous_lambda

            if delta_n == 0:
                raise ZeroDivisionError(
                    "Consecutive refractive-index values are identical."
                )

            sensitivity = delta_lambda / delta_n

        results.append(
            {
                "gas_refractive_index": current_ri,
                "lambda_exact_um": current_lambda,
                "m": row["m"],
                "sensitivity_um_per_RIU": sensitivity,
            }
        )

        previous_ri = current_ri
        previous_lambda = current_lambda

    return results


# ==========================================================
# Save results
# ==========================================================


def save_results(
    results,
    output_file: Path,
):

    fieldnames = [
        "gas_refractive_index",
        "lambda_exact_um",
        "m",
        "sensitivity_um_per_RIU",
    ]

    with output_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    rows = load_peak_results(INPUT_FILE)

    results = calculate_sensitivity(rows)

    save_results(
        results,
        OUTPUT_FILE,
    )

    print("=" * 100)
    print("GAS SENSITIVITY")
    print("=" * 100)

    for result in results:
        ri = result["gas_refractive_index"]

        wavelength = result["lambda_exact_um"]

        m = result["m"]

        sensitivity = result["sensitivity_um_per_RIU"]

        if sensitivity is None:
            print(f"n={ri:.3f} | lambda={wavelength:.12f} µm | m={m} | sensitivity=N/A")

        else:
            print(
                f"n={ri:.3f} | "
                f"lambda={wavelength:.12f} µm | "
                f"m={m} | "
                f"sensitivity="
                f"{sensitivity:.12f} µm/RIU"
            )

    print()
    print(f"Saved to:\n{OUTPUT_FILE}")
