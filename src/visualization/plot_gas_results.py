import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "base"

SENSITIVITY_FILE = DATA_DIR / "gas_sensitivity_theory_check.csv"

FOM_FILE = DATA_DIR / "gas_fom_results.csv"

OUTPUT_DIR = DATA_DIR / "plots"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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
# LOAD FOM DATA
# ==========================================================


def load_fom():

    rows = []

    with FOM_FILE.open(
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
                    "fom_shift": (
                        float(row["fom_shift"]) if row["fom_shift"] else np.nan
                    ),
                    "fom_implicit": (
                        float(row["fom_implicit"]) if row["fom_implicit"] else np.nan
                    ),
                }
            )

    return rows


# ==========================================================
# MAIN
# ==========================================================


def main():

    sensitivity_rows = load_sensitivity()

    fom_rows = load_fom()

    # ======================================================
    # ARRAYS
    # ======================================================

    ri = np.array(
        [row["ri"] for row in sensitivity_rows],
        dtype=float,
    )

    lambda_exact = np.array(
        [row["lambda_exact_um"] for row in sensitivity_rows],
        dtype=float,
    )

    s_shift = np.array(
        [row["s_shift"] for row in sensitivity_rows],
        dtype=float,
    )

    s_implicit = np.array(
        [row["s_implicit"] for row in sensitivity_rows],
        dtype=float,
    )

    fwhm_nm = np.array(
        [row["fwhm_nm"] for row in fom_rows],
        dtype=float,
    )

    fom_shift = np.array(
        [row["fom_shift"] for row in fom_rows],
        dtype=float,
    )

    fom_implicit = np.array(
        [row["fom_implicit"] for row in fom_rows],
        dtype=float,
    )

    # ======================================================
    # MASK FOR STABLE RANGE
    # ======================================================

    stable_mask = ri >= 1.002

    # ======================================================
    # FIGURE 1
    # Lambda resonance vs RI
    # ======================================================

    plt.figure(figsize=(10, 7))

    plt.plot(
        ri,
        lambda_exact,
        "o-",
        linewidth=1.5,
        markersize=5,
        label="Î»res",
    )

    # Linear fit for n >= 1.002
    coeff = np.polyfit(
        ri[stable_mask],
        lambda_exact[stable_mask],
        1,
    )

    ri_fit = np.linspace(
        1.002,
        1.009,
        200,
    )

    lambda_fit = np.polyval(
        coeff,
        ri_fit,
    )

    plt.plot(
        ri_fit,
        lambda_fit,
        "--",
        linewidth=1.5,
        label="Linear fit",
    )

    plt.xlabel("Analyte refractive index")

    plt.ylabel("Resonance wavelength (Âµm)")

    plt.title("Gas Sensor: Resonance Wavelength vs Refractive Index")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    figure_path = OUTPUT_DIR / "gas_lambda_res_vs_ri.png"

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    # ======================================================
    # FIGURE 2
    # Sensitivity vs RI
    # ======================================================

    plt.figure(figsize=(10, 7))

    plt.plot(
        ri,
        s_shift,
        "o-",
        linewidth=1.5,
        markersize=5,
        label="S from resonance shift",
    )

    plt.plot(
        ri,
        s_implicit,
        "s-",
        linewidth=1.5,
        markersize=5,
        label="S from implicit resonance equation",
    )

    plt.xlabel("Analyte refractive index")

    plt.ylabel("Sensitivity (Âµm/RIU)")

    plt.title("Gas Sensor: Sensitivity vs Refractive Index")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    figure_path = OUTPUT_DIR / "gas_sensitivity_vs_ri.png"

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    # ======================================================
    # FIGURE 3
    # FOM vs RI
    # ======================================================

    plt.figure(figsize=(10, 7))

    plt.plot(
        ri,
        fom_shift,
        "o-",
        linewidth=1.5,
        markersize=5,
        label="FOM from resonance shift",
    )

    plt.plot(
        ri,
        fom_implicit,
        "s-",
        linewidth=1.5,
        markersize=5,
        label="FOM from implicit sensitivity",
    )

    plt.xlabel("Analyte refractive index")

    plt.ylabel("FOM (RIU$^{-1}$)")

    plt.title("Gas Sensor: FOM vs Refractive Index")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    figure_path = OUTPUT_DIR / "gas_fom_vs_ri.png"

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    # ======================================================
    # FIGURE 4
    # FWHM vs RI
    # ======================================================

    plt.figure(figsize=(10, 7))

    plt.plot(
        ri,
        fwhm_nm,
        "o-",
        linewidth=1.5,
        markersize=5,
    )

    plt.xlabel("Analyte refractive index")

    plt.ylabel("FWHM (nm)")

    plt.title("Gas Sensor: FWHM vs Refractive Index")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    figure_path = OUTPUT_DIR / "gas_fwhm_vs_ri.png"

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    # ======================================================
    # FINISHED
    # ======================================================

    print()
    print("=" * 100)
    print("GAS PLOTS CREATED")
    print("=" * 100)

    print()
    print(f"Output directory:\n{OUTPUT_DIR}")

    print()
    print("Created:")

    print(" - gas_lambda_res_vs_ri.png")

    print(" - gas_sensitivity_vs_ri.png")

    print(" - gas_fom_vs_ri.png")

    print(" - gas_fwhm_vs_ri.png")


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()

