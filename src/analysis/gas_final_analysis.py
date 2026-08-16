from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from src.calculations.delta_neff import DeltaNeffCalculator
from src.config.settings import DEVICE_LENGTH_UM, GAS_RI_STEP
from src.io.loader import ModeDataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "base"
OUTPUT_DIR = DATA_DIR / "plots"

REFERENCE_FILE = DATA_DIR / "reference.mat"
GAS_FILES = sorted(DATA_DIR.glob("sensor-gas-*.mat"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def transmission(wavelength, delta_neff):
    return np.cos(
        (2.0 * np.pi * delta_neff * DEVICE_LENGTH_UM) / wavelength
    ) ** 2


def exact_resonance(wavelength, delta_neff, target_m=56.0):
    m = 2.0 * delta_neff * DEVICE_LENGTH_UM / wavelength
    crossings = np.where((m[:-1] - target_m) * (m[1:] - target_m) <= 0.0)[0]

    if len(crossings) == 0:
        raise RuntimeError("No m=56 crossing found.")

    # Choose the crossing closest to the strongest transmission peak.
    t = transmission(wavelength, delta_neff)
    peak_index = int(np.argmax(t))

    crossing_index = min(
        crossings,
        key=lambda i: abs(i - peak_index),
    )

    i = crossing_index

    return np.interp(
        target_m,
        [m[i], m[i + 1]],
        [wavelength[i], wavelength[i + 1]],
    )


def calculate_fwhm(wavelength, delta_neff, lambda_res):
    def t_of_lambda(lam):
        dn = np.interp(lam, wavelength, delta_neff)
        return np.cos(
            (2.0 * np.pi * dn * DEVICE_LENGTH_UM) / lam
        ) ** 2

    t_peak = t_of_lambda(lambda_res)
    half_level = t_peak / 2.0

    span = 0.04  # ±40 nm search window
    grid_left = np.linspace(lambda_res - span, lambda_res, 2000)
    grid_right = np.linspace(lambda_res, lambda_res + span, 2000)

    left_values = np.array(
        [t_of_lambda(x) - half_level for x in grid_left]
    )
    right_values = np.array(
        [t_of_lambda(x) - half_level for x in grid_right]
    )

    left_crossings = np.where(
        left_values[:-1] * left_values[1:] <= 0.0
    )[0]

    right_crossings = np.where(
        right_values[:-1] * right_values[1:] <= 0.0
    )[0]

    if len(left_crossings) == 0 or len(right_crossings) == 0:
        raise RuntimeError(
            f"Could not determine FWHM at λ={lambda_res:.9f} µm."
        )

    li = left_crossings[-1]
    ri = right_crossings[0]

    lambda_left = brentq(
        lambda x: t_of_lambda(x) - half_level,
        grid_left[li],
        grid_left[li + 1],
    )

    lambda_right = brentq(
        lambda x: t_of_lambda(x) - half_level,
        grid_right[ri],
        grid_right[ri + 1],
    )

    return (lambda_right - lambda_left) * 1000.0


def main():
    reference = ModeDataLoader.load(REFERENCE_FILE)

    wavelength = np.asarray(
        reference.wavelength_neff,
        dtype=float,
    ).ravel()

    results = []

    previous_lambda = None

    for sensor_file in GAS_FILES:
        sensor = ModeDataLoader.load(sensor_file)

        delta_neff = DeltaNeffCalculator.calculate(
            reference,
            sensor,
        )

        lambda_res = exact_resonance(
            wavelength,
            delta_neff,
        )

        fwhm_nm = calculate_fwhm(
            wavelength,
            delta_neff,
            lambda_res,
        )

        ri = int(sensor_file.stem.split("-")[-1]) / 1000.0

        if previous_lambda is None:
            sensitivity = np.nan
            fom = np.nan
        else:
            sensitivity = (
                (lambda_res - previous_lambda)
                * 1000.0
                / GAS_RI_STEP
            )
            fom = sensitivity / fwhm_nm

        results.append(
            {
                "ri": ri,
                "lambda_nm": lambda_res * 1000.0,
                "sensitivity_nm_per_riu": sensitivity,
                "fwhm_nm": fwhm_nm,
                "fom_riu_inv": fom,
            }
        )

        previous_lambda = lambda_res

    print()
    print("=" * 90)
    print("FINAL GAS ANALYSIS")
    print("=" * 90)
    print()
    print(
        f"{'RI':>7}"
        f"{'Lambda (nm)':>16}"
        f"{'S (nm/RIU)':>16}"
        f"{'FWHM (nm)':>14}"
        f"{'FOM (RIU^-1)':>16}"
    )
    print("-" * 90)

    for row in results:
        if np.isnan(row["sensitivity_nm_per_riu"]):
            print(
                f"{row['ri']:7.3f}"
                f"{row['lambda_nm']:16.6f}"
                f"{'N/A':>16}"
                f"{row['fwhm_nm']:14.6f}"
                f"{'N/A':>16}"
            )
        else:
            print(
                f"{row['ri']:7.3f}"
                f"{row['lambda_nm']:16.6f}"
                f"{row['sensitivity_nm_per_riu']:16.4f}"
                f"{row['fwhm_nm']:14.6f}"
                f"{row['fom_riu_inv']:16.6f}"
            )

    ri_values = np.array([r["ri"] for r in results])
    lambda_values = np.array([r["lambda_nm"] for r in results])
    sensitivity_values = np.array(
        [r["sensitivity_nm_per_riu"] for r in results]
    )
    fwhm_values = np.array([r["fwhm_nm"] for r in results])
    fom_values = np.array([r["fom_riu_inv"] for r in results])

    valid = np.isfinite(sensitivity_values)

    # λres vs RI
    plt.figure(figsize=(9, 6))
    plt.plot(ri_values, lambda_values, marker="o")
    plt.xlabel("Gas Refractive Index")
    plt.ylabel("Resonance Wavelength (nm)")
    plt.title("Gas Resonance Wavelength vs RI")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "gas_resonance_vs_ri.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Sensitivity vs RI
    plt.figure(figsize=(9, 6))
    plt.plot(
        ri_values[valid],
        sensitivity_values[valid],
        marker="o",
    )
    plt.xlabel("Gas Refractive Index")
    plt.ylabel("Sensitivity (nm/RIU)")
    plt.title("Gas Sensitivity vs RI")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "gas_sensitivity_vs_ri.png",
        dpi=300,
        bbox_inches="tight",
    )

    # FOM vs RI
    plt.figure(figsize=(9, 6))
    plt.plot(
        ri_values[valid],
        fom_values[valid],
        marker="o",
    )
    plt.xlabel("Gas Refractive Index")
    plt.ylabel("FOM (RIU^-1)")
    plt.title("Gas FOM vs RI")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "gas_fom_vs_ri.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()
