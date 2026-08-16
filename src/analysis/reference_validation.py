import csv
from pathlib import Path

import numpy as np

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "base"

FINAL_METRICS_FILE = DATA_DIR / "gas_final_metrics.csv"

FWHM_FILE = DATA_DIR / "gas_fwhm_results.csv"

OUTPUT_FILE = DATA_DIR / "reference_validation.csv"


# ==========================================================
# REFERENCE PAPER
# ==========================================================

REFERENCE_PAPER = {
    "title": "Compact Gas Sensor Using "
    "Silicon-on-Insulator "
    "Loop-Terminated Mach-Zehnder Interferometer",
    "wavelength_um": 1.55,
    "reference_length_um": 150.0,
    "sensitivity_nm_per_RIU": 1070.0,
    "fom_RIU_inv": 280.8,
    "silicon_layer_nm": 220.0,
    "slot_width_nm": 100.0,
    "dc_gap_nm": 100.0,
    "dc_length_um": 12.1,
    "cladding": "air",
}


# ==========================================================
# OUR DESIGN
# ==========================================================

OUR_LENGTH_UM = 50.0

OUR_WAVELENGTH_MIN_UM = 1.50
OUR_WAVELENGTH_MAX_UM = 1.65

OUR_STABLE_RI_MIN = 1.002
OUR_STABLE_RI_MAX = 1.009


# ==========================================================
# LOAD FINAL METRICS
# ==========================================================


def load_final_metrics():

    rows = []

    with FINAL_METRICS_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(
                {
                    "range": row["range"],
                    "ri_min": float(row["ri_min"]),
                    "ri_max": float(row["ri_max"]),
                    "number_of_points": int(row["number_of_points"]),
                    "sensitivity_nm_per_RIU": float(
                        row["sensitivity_linear_um_per_RIU"]
                    )
                    * 1000.0,
                    "r2": float(row["r2"]),
                    "mean_fwhm_nm": float(row["mean_fwhm_nm"]),
                    "fom_linear_RIU_inv": float(row["fom_linear_RIU_inv"]),
                    "mean_implicit_sensitivity_nm_per_RIU": float(
                        row["mean_implicit_sensitivity_um_per_RIU"]
                    )
                    * 1000.0,
                    "fom_implicit_RIU_inv": float(row["fom_implicit_RIU_inv"]),
                }
            )

    if not rows:
        raise RuntimeError("gas_final_metrics.csv is empty.")

    return rows


# ==========================================================
# LOAD FWHM
# ==========================================================


def load_fwhm_statistics():

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
                    "fwhm_nm": float(row["fwhm_nm"]),
                }
            )

    return rows


# ==========================================================
# MAIN
# ==========================================================


def main():

    metrics = load_final_metrics()

    fwhm_rows = load_fwhm_statistics()

    # ======================================================
    # REFERENCE VALUES
    # ======================================================

    paper_length = REFERENCE_PAPER["reference_length_um"]

    paper_sensitivity = REFERENCE_PAPER["sensitivity_nm_per_RIU"]

    paper_fom = REFERENCE_PAPER["fom_RIU_inv"]

    # ======================================================
    # LENGTH-NORMALIZED PAPER BASELINE
    #
    # Paper reports:
    #
    # S = 1070 nm/RIU at L = 150 Âµm
    #
    # Assuming linear length scaling:
    #
    # S(50) = 1070 * 50/150
    # ======================================================

    paper_sensitivity_50 = paper_sensitivity * OUR_LENGTH_UM / paper_length

    paper_fom_50 = paper_fom * OUR_LENGTH_UM / paper_length

    # ======================================================
    # GET OUR STABLE-RANGE RESULT
    # ======================================================

    stable_result = None

    for row in metrics:
        if np.isclose(
            row["ri_min"],
            OUR_STABLE_RI_MIN,
            atol=1e-12,
        ) and np.isclose(
            row["ri_max"],
            OUR_STABLE_RI_MAX,
            atol=1e-12,
        ):
            stable_result = row
            break

    if stable_result is None:
        raise RuntimeError("Stable-range result 1.002-1.009 was not found.")

    # ======================================================
    # OUR RESULTS
    # ======================================================

    our_sensitivity = stable_result["sensitivity_nm_per_RIU"]

    our_fom = stable_result["fom_linear_RIU_inv"]

    our_implicit_sensitivity = stable_result["mean_implicit_sensitivity_nm_per_RIU"]

    our_implicit_fom = stable_result["fom_implicit_RIU_inv"]

    our_fwhm = stable_result["mean_fwhm_nm"]

    our_r2 = stable_result["r2"]

    # ======================================================
    # IMPROVEMENT
    # ======================================================

    sensitivity_improvement = our_sensitivity / paper_sensitivity_50

    fom_improvement = our_fom / paper_fom_50

    sensitivity_improvement_percent = (sensitivity_improvement - 1.0) * 100.0

    fom_improvement_percent = (fom_improvement - 1.0) * 100.0

    # ======================================================
    # FWHM IMPLIED BY PAPER
    #
    # FOM = S/FWHM
    # ======================================================

    paper_fwhm_nm = paper_sensitivity / paper_fom

    paper_fwhm_50_nm = paper_sensitivity_50 / paper_fom_50

    # ======================================================
    # PRINT
    # ======================================================

    print()
    print("=" * 110)
    print("REFERENCE PAPER VALIDATION")
    print("=" * 110)

    print()
    print("REFERENCE PAPER")

    print(f"Title                : {REFERENCE_PAPER['title']}")

    print(f"Wavelength           : {REFERENCE_PAPER['wavelength_um']:.2f} Âµm")

    print(f"Reference L          : {paper_length:.1f} Âµm")

    print(f"Reported Sensitivity : {paper_sensitivity:.3f} nm/RIU")

    print(f"Reported FOM        : {paper_fom:.3f} RIU^-1")

    print(f"Implied FWHM        : {paper_fwhm_nm:.6f} nm")

    print()
    print("REFERENCE NORMALIZED TO OUR L = 50 Âµm")

    print(f"Sensitivity          : {paper_sensitivity_50:.6f} nm/RIU")

    print(f"FOM                  : {paper_fom_50:.6f} RIU^-1")

    print(f"Implied FWHM        : {paper_fwhm_50_nm:.6f} nm")

    print()
    print("OUR GAS SENSOR")

    print(f"RI range             : {OUR_STABLE_RI_MIN:.3f} - {OUR_STABLE_RI_MAX:.3f}")

    print(f"L                    : {OUR_LENGTH_UM:.1f} Âµm")

    print(f"Sensitivity          : {our_sensitivity:.6f} nm/RIU")

    print(f"Implicit sensitivity : {our_implicit_sensitivity:.6f} nm/RIU")

    print(f"Mean FWHM           : {our_fwhm:.6f} nm")

    print(f"FOM                 : {our_fom:.6f} RIU^-1")

    print(f"Implicit FOM        : {our_implicit_fom:.6f} RIU^-1")

    print(f"RÂ²                  : {our_r2:.9f}")

    print()
    print("IMPROVEMENT OVER LENGTH-NORMALIZED REFERENCE")

    print(f"Sensitivity ratio   : {sensitivity_improvement:.6f} Ã—")

    print(f"Sensitivity gain    : {sensitivity_improvement_percent:.3f} %")

    print(f"FOM ratio           : {fom_improvement:.6f} Ã—")

    print(f"FOM gain            : {fom_improvement_percent:.3f} %")

    print()
    print("=" * 110)

    # ======================================================
    # SAVE
    # ======================================================

    result = {
        "reference_title": REFERENCE_PAPER["title"],
        "paper_wavelength_um": REFERENCE_PAPER["wavelength_um"],
        "paper_length_um": paper_length,
        "our_length_um": OUR_LENGTH_UM,
        "paper_sensitivity_nm_per_RIU": paper_sensitivity,
        "paper_fom_RIU_inv": paper_fom,
        "paper_implied_fwhm_nm": paper_fwhm_nm,
        "paper_sensitivity_normalized_50um_nm_per_RIU": paper_sensitivity_50,
        "paper_fom_normalized_50um_RIU_inv": paper_fom_50,
        "our_ri_min": OUR_STABLE_RI_MIN,
        "our_ri_max": OUR_STABLE_RI_MAX,
        "our_sensitivity_nm_per_RIU": our_sensitivity,
        "our_implicit_sensitivity_nm_per_RIU": our_implicit_sensitivity,
        "our_mean_fwhm_nm": our_fwhm,
        "our_fom_RIU_inv": our_fom,
        "our_implicit_fom_RIU_inv": our_implicit_fom,
        "our_r2": our_r2,
        "sensitivity_ratio": sensitivity_improvement,
        "sensitivity_gain_percent": sensitivity_improvement_percent,
        "fom_ratio": fom_improvement,
        "fom_gain_percent": fom_improvement_percent,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(result.keys()),
        )

        writer.writeheader()

        writer.writerow(result)

    print()
    print(f"Saved to:\n{OUTPUT_FILE}")


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()

