import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import linregress

from src.calculations.delta_neff import DeltaNeffCalculator
from src.calculations.peak_detector import PeakDetector
from src.calculations.transmission import TransmissionCalculator
from src.config.settings import DEVICE_LENGTH_UM, GAS_RI_STEP
from src.io.loader import ModeDataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "base"
OUTPUT_DIR = DATA_DIR / "plots"

REFERENCE_FILE = DATA_DIR / "reference.mat"
OUTPUT_FILE = DATA_DIR / "gas_final_results.csv"

GAS_FILES = sorted(
    DATA_DIR.glob("sensor-gas-*.mat")
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def extract_gas_ri(file):
    return int(file.stem.split("-")[-1]) / 1000.0


def interpolate_value(
    wavelength,
    values,
    target,
):
    return float(
        np.interp(
            target,
            wavelength,
            values,
        )
    )


def calculate_fwhm(
    wavelength,
    delta_neff,
    lambda_res,
):
    def transmission_at(lam):
        dn = np.interp(
            lam,
            wavelength,
            delta_neff,
        )

        return np.cos(
            (2.0 * np.pi * dn * DEVICE_LENGTH_UM) / lam
        ) ** 2

    peak = transmission_at(lambda_res)
    half = peak / 2.0

    span = 0.04

    left_grid = np.linspace(
        lambda_res - span,
        lambda_res,
        2000,
    )

    right_grid = np.linspace(
        lambda_res,
        lambda_res + span,
        2000,
    )

    left_values = np.array(
        [
            transmission_at(x) - half
            for x in left_grid
        ]
    )

    right_values = np.array(
        [
            transmission_at(x) - half
            for x in right_grid
        ]
    )

    left_crossings = np.where(
        left_values[:-1] * left_values[1:] <= 0
    )[0]

    right_crossings = np.where(
        right_values[:-1] * right_values[1:] <= 0
    )[0]

    if (
        len(left_crossings) == 0
        or len(right_crossings) == 0
    ):
        raise RuntimeError(
            f"FWHM could not be determined at "
            f"{lambda_res:.9f} um."
        )

    li = left_crossings[-1]
    ri = right_crossings[0]

    lambda_left = brentq(
        lambda x: transmission_at(x) - half,
        left_grid[li],
        left_grid[li + 1],
    )

    lambda_right = brentq(
        lambda x: transmission_at(x) - half,
        right_grid[ri],
        right_grid[ri + 1],
    )

    return (
        lambda_right - lambda_left
    ) * 1000.0


def solve_lambda_for_m(
    wavelength,
    delta_neff,
    m_target,
    lambda_reference,
):
    """
    Solve:

        2 * Delta_neff(lambda) * L / lambda = m_target

    and select the root closest to the tracked resonance.
    """

    def equation(lam):
        dn = np.interp(
            lam,
            wavelength,
            delta_neff,
        )

        return (
            2.0 * dn * DEVICE_LENGTH_UM / lam
            - m_target
        )

    values = np.array(
        [
            equation(lam)
            for lam in wavelength
        ],
        dtype=float,
    )

    roots = []

    for i in range(len(wavelength) - 1):

        left = values[i]
        right = values[i + 1]

        if not (
            np.isfinite(left)
            and np.isfinite(right)
        ):
            continue

        if left == 0.0:
            roots.append(
                float(wavelength[i])
            )
            continue

        if left * right < 0.0:

            root = brentq(
                equation,
                wavelength[i],
                wavelength[i + 1],
            )

            roots.append(
                float(root)
            )

    if not roots:
        return None

    return float(
        min(
            roots,
            key=lambda root: abs(
                root - lambda_reference
            ),
        )
    )


def calculate_exact_fsr(
    wavelength,
    delta_neff,
    m_center,
    lambda_center,
):
    """
    Calculate FSR from exact adjacent fringe-order roots.

    Prefer m+1. If unavailable, use m-1.
    """

    lambda_plus = solve_lambda_for_m(
        wavelength=wavelength,
        delta_neff=delta_neff,
        m_target=m_center + 1,
        lambda_reference=lambda_center,
    )

    if lambda_plus is not None:
        return (
            abs(lambda_plus - lambda_center)
            * 1000.0,
            m_center + 1,
        )

    lambda_minus = solve_lambda_for_m(
        wavelength=wavelength,
        delta_neff=delta_neff,
        m_target=m_center - 1,
        lambda_reference=lambda_center,
    )

    if lambda_minus is not None:
        return (
            abs(lambda_center - lambda_minus)
            * 1000.0,
            m_center - 1,
        )

    return np.nan, None

def calculate_results():
    reference = ModeDataLoader.load(
        REFERENCE_FILE
    )

    wavelength = np.asarray(
        reference.wavelength_neff,
        dtype=float,
    ).ravel()

    reference_neff = np.asarray(
        reference.neff,
        dtype=float,
    ).ravel()

    if not GAS_FILES:
        raise RuntimeError(
            "No gas sensor files were found."
        )

    results = []

    previous_sensor_neff = None
    previous_ri = None
    previous_lambda_exact = None
    previous_lambda_find = None

    for sensor_file in GAS_FILES:

        sensor = ModeDataLoader.load(
            sensor_file
        )

        sensor_neff = np.asarray(
            sensor.neff,
            dtype=float,
        ).ravel()

        if sensor_neff.shape != wavelength.shape:
            raise ValueError(
                f"Shape mismatch in "
                f"{sensor_file.name}"
            )

        delta_neff = (
            reference_neff
            - sensor_neff
        )

        transmission = (
            TransmissionCalculator.calculate(
                wavelength=wavelength,
                delta_neff=delta_neff,
                length=DEVICE_LENGTH_UM,
            )
        )

        detector = PeakDetector(
            wavelength=wavelength,
            transmission=transmission,
            delta_neff=delta_neff,
            length=DEVICE_LENGTH_UM,
        )

        peak = detector.detect(
            previous_lambda=previous_lambda_find
        )

        ri = extract_gas_ri(
            sensor_file
        )

        lambda_exact = (
            peak.lambda_exact
        )

        delta_neff_res = interpolate_value(
            wavelength,
            delta_neff,
            lambda_exact,
        )

        fwhm_nm = calculate_fwhm(
            wavelength,
            delta_neff,
            lambda_exact,
        )

        # Exact FSR from adjacent fringe-order roots.
        fsr_nm, adjacent_order = calculate_exact_fsr(
            wavelength=wavelength,
            delta_neff=delta_neff,
            m_center=peak.m,
            lambda_center=lambda_exact,
        )

        if previous_lambda_exact is None:
            sensitivity_shift = np.nan
            waveguide_sensitivity = np.nan
            sensitivity_formula_current = np.nan
            fom = np.nan
            delta_lambda_nm = np.nan
        else:
            delta_ri = (
                ri - previous_ri
            )

            if not np.isclose(
                delta_ri,
                GAS_RI_STEP,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Unexpected RI step: "
                    f"{previous_ri} -> {ri}"
                )

            delta_lambda_nm = (
                lambda_exact
                - previous_lambda_exact
            ) * 1000.0

            sensitivity_shift = (
                delta_lambda_nm
                / delta_ri
            )

            waveguide_array = (
                sensor_neff
                - previous_sensor_neff
            ) / delta_ri

            waveguide_sensitivity = (
                interpolate_value(
                    wavelength,
                    waveguide_array,
                    lambda_exact,
                )
            )

            sensitivity_formula_current = (
                lambda_exact
                / delta_neff_res
                * waveguide_sensitivity
            )

            fom = (
                abs(sensitivity_shift)
                / fwhm_nm
            )

        results.append(
            {
                "ri": ri,
                "m": peak.m,
                "lambda_find_peaks_um": (
                    peak.lambda_find_peaks
                ),
                "lambda_exact_um": (
                    lambda_exact
                ),
                "peak_correction_nm": (
                    peak.difference_nm
                ),
                "delta_neff": (
                    delta_neff_res
                ),
                "waveguide_sensitivity": (
                    waveguide_sensitivity
                ),
                "sensitivity_shift_nm_per_riu": (
                    sensitivity_shift
                ),
                "sensitivity_formula_nm_per_riu": (
                    sensitivity_formula_current
                ),
                "fwhm_nm": fwhm_nm,
                "fsr_nm": fsr_nm,
                "fsr_adjacent_order": (
                    adjacent_order
                ),
                "fom_riu_inv": fom,
            }
        )

        previous_sensor_neff = (
            sensor_neff.copy()
        )

        previous_ri = ri
        previous_lambda_exact = (
            lambda_exact
        )
        previous_lambda_find = (
            peak.lambda_find_peaks
        )

    return results


def save_results(results):
    fieldnames = list(
        results[0].keys()
    )

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


def print_summary(results):

    print()
    print("=" * 120)
    print("CANONICAL GAS ANALYSIS")
    print("=" * 120)

    print()
    print(
        f"{'RI':>7}"
        f"{'m':>6}"
        f"{'lambda_exact(nm)':>20}"
        f"{'S_shift':>14}"
        f"{'FWHM(nm)':>12}"
        f"{'FSR(nm)':>12}"
        f"{'FOM':>12}"
    )

    print("-" * 120)

    for row in results:

        sensitivity = row[
            "sensitivity_shift_nm_per_riu"
        ]

        fom = row[
            "fom_riu_inv"
        ]

        s_text = (
            "N/A"
            if np.isnan(sensitivity)
            else f"{sensitivity:.4f}"
        )

        fom_text = (
            "N/A"
            if np.isnan(fom)
            else f"{fom:.4f}"
        )

        print(
            f"{row['ri']:7.3f}"
            f"{row['m']:6d}"
            f"{row['lambda_exact_um'] * 1000:20.6f}"
            f"{s_text:>14}"
            f"{row['fwhm_nm']:12.4f}"
            f"{row['fsr_nm']:12.4f}"
            f"{fom_text:>12}"
        )

    print()
    print("-" * 120)

    stable = [
        row
        for row in results
        if (
            row["ri"] >= 1.002
            and np.isfinite(
                row[
                    "sensitivity_shift_nm_per_riu"
                ]
            )
        )
    ]

    ri = np.array(
        [row["ri"] for row in stable],
        dtype=float,
    )

    wavelength = np.array(
        [
            row["lambda_exact_um"]
            for row in stable
        ],
        dtype=float,
    )

    if len(ri) >= 2:

        fit = linregress(
            ri,
            wavelength,
        )

        mean_fwhm = np.mean(
            [
                row["fwhm_nm"]
                for row in stable
            ]
        )

        linear_fom = (
            abs(fit.slope * 1000.0)
            / mean_fwhm
        )

        print(
            f"Stable range: "
            f"RI = {ri.min():.3f} ... {ri.max():.3f}"
        )

        print(
            f"Linear sensitivity : "
            f"{fit.slope * 1000.0:.4f} nm/RIU"
        )

        print(
            f"R2                 : "
            f"{fit.rvalue ** 2:.8f}"
        )

        print(
            f"Mean FWHM          : "
            f"{mean_fwhm:.4f} nm"
        )

        print(
            f"Linear-fit FOM     : "
            f"{linear_fom:.4f} RIU^-1"
        )

    print()
    print(
        f"Results saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print()
    print("=" * 120)


def make_plots(results):

    ri = np.array(
        [row["ri"] for row in results],
        dtype=float,
    )

    lambda_nm = np.array(
        [
            row["lambda_exact_um"] * 1000.0
            for row in results
        ]
    )

    sensitivity = np.array(
        [
            row[
                "sensitivity_shift_nm_per_riu"
            ]
            for row in results
        ]
    )

    fom = np.array(
        [
            row["fom_riu_inv"]
            for row in results
        ]
    )

    valid = np.isfinite(
        sensitivity
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        ri,
        lambda_nm,
        marker="o",
    )

    plt.xlabel(
        "Gas Refractive Index"
    )

    plt.ylabel(
        "Resonance Wavelength (nm)"
    )

    plt.title(
        "Gas Resonance Wavelength vs RI"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "gas_resonance_vs_ri.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        ri[valid],
        sensitivity[valid],
        marker="o",
    )

    plt.xlabel(
        "Gas Refractive Index"
    )

    plt.ylabel(
        "Sensitivity (nm/RIU)"
    )

    plt.title(
        "Gas Sensitivity vs RI"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "gas_sensitivity_vs_ri.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        ri[valid],
        fom[valid],
        marker="o",
    )

    plt.xlabel(
        "Gas Refractive Index"
    )

    plt.ylabel(
        "FOM (RIU^-1)"
    )

    plt.title(
        "Gas FOM vs RI"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "gas_fom_vs_ri.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main():

    results = calculate_results()

    save_results(
        results
    )

    print_summary(
        results
    )

    make_plots(
        results
    )


if __name__ == "__main__":
    main()

