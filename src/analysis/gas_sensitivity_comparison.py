import csv
from pathlib import Path

import numpy as np

from src.calculations.peak_detector import PeakDetector
from src.calculations.transmission import TransmissionCalculator
from src.config.settings import DEVICE_LENGTH_UM, GAS_RI_STEP
from src.io.loader import ModeDataLoader

# ==========================================================
# SETTINGS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "base"

REFERENCE_FILE = DATA_DIR / "reference.mat"

OUTPUT_FILE = DATA_DIR / "gas_sensitivity_comparison.csv"


# Gas RI step
DELTA_N_MEDIUM = GAS_RI_STEP


# ==========================================================
# GAS RI
# ==========================================================


def extract_gas_refractive_index(
    file: Path,
) -> float:

    value = file.stem.split("-")[-1]

    return int(value) / 1000.0


# ==========================================================
# INTERPOLATION
# ==========================================================


def interpolate_value(
    wavelength,
    values,
    target_wavelength,
):
    """
    Linear interpolation at the requested wavelength.
    """

    return float(
        np.interp(
            target_wavelength,
            wavelength,
            values,
        )
    )


# ==========================================================
# MAIN
# ==========================================================


def main():

    # ------------------------------------------------------
    # Load reference
    # ------------------------------------------------------

    reference = ModeDataLoader.load(REFERENCE_FILE)

    wavelength = np.asarray(
        reference.wavelength_neff,
        dtype=float,
    ).ravel()

    reference_neff = np.asarray(
        reference.neff,
        dtype=float,
    ).ravel()

    # ------------------------------------------------------
    # Gas files
    # ------------------------------------------------------

    gas_files = sorted(DATA_DIR.glob("sensor-gas-*.mat"))

    if not gas_files:
        raise RuntimeError("No gas sensor files were found.")

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    results = []

    previous_lambda_find = None
    previous_lambda_exact = None

    previous_sensor_neff = None
    previous_ri = None

    # ======================================================
    # PROCESS GAS SWEEP
    # ======================================================

    for file in gas_files:
        sensor = ModeDataLoader.load(file)

        sensor_neff = np.asarray(
            sensor.neff,
            dtype=float,
        ).ravel()

        # --------------------------------------------------
        # Validate wavelength / neff
        # --------------------------------------------------

        if sensor_neff.shape != wavelength.shape:
            raise ValueError(
                f"Shape mismatch in {file.name}: "
                f"reference wavelength={wavelength.shape}, "
                f"sensor neff={sensor_neff.shape}"
            )

        # --------------------------------------------------
        # Delta neff
        #
        # Previous project convention:
        #
        # Î”neff = neff_reference - neff_sensor
        # --------------------------------------------------

        delta_neff = reference_neff - sensor_neff

        # --------------------------------------------------
        # Transmission
        # --------------------------------------------------

        transmission = TransmissionCalculator.calculate(
            wavelength=wavelength,
            delta_neff=delta_neff,
            length=DEVICE_LENGTH_UM,
        )

        # --------------------------------------------------
        # Peak detection
        # --------------------------------------------------

        detector = PeakDetector(
            wavelength=wavelength,
            transmission=transmission,
            delta_neff=delta_neff,
            length=DEVICE_LENGTH_UM,
        )

        result = detector.detect(previous_lambda=previous_lambda_find)

        # --------------------------------------------------
        # Current RI
        # --------------------------------------------------

        current_ri = extract_gas_refractive_index(file)

        # --------------------------------------------------
        # Current Î»
        # --------------------------------------------------

        lambda_find = result.lambda_find_peaks

        lambda_exact = result.lambda_exact

        # --------------------------------------------------
        # First file
        # --------------------------------------------------

        if previous_sensor_neff is None:
            swg_sensor = np.nan
            swg_reference = 0.0

            delta_neff_at_resonance = interpolate_value(
                wavelength,
                delta_neff,
                lambda_exact,
            )

            sensitivity_shift = np.nan

            sensitivity_current_lambda = np.nan

            sensitivity_previous_lambda = np.nan

            delta_lambda = np.nan

            delta_ri = np.nan

        # --------------------------------------------------
        # Following files
        # --------------------------------------------------

        else:
            delta_ri = current_ri - previous_ri

            if not np.isclose(
                delta_ri,
                DELTA_N_MEDIUM,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Unexpected RI step between "
                    f"{previous_ri} and {current_ri}: "
                    f"{delta_ri}"
                )

            # --------------------------------------------------
            # Waveguide sensitivity of sensor arm
            #
            # Swg,sens = dneff_sensor / dn_medium
            # --------------------------------------------------

            swg_array = (sensor_neff - previous_sensor_neff) / delta_ri

            # --------------------------------------------------
            # Evaluate quantities at Î»_exact
            # --------------------------------------------------

            swg_sensor = interpolate_value(
                wavelength,
                swg_array,
                lambda_exact,
            )

            delta_neff_at_resonance = interpolate_value(
                wavelength,
                delta_neff,
                lambda_exact,
            )

            # --------------------------------------------------
            # Reference-arm sensitivity
            #
            # Reference does not change with analyte RI
            # --------------------------------------------------

            swg_reference = 0.0

            # --------------------------------------------------
            # Resonance wavelength shift
            # --------------------------------------------------

            delta_lambda = lambda_exact - previous_lambda_exact

            sensitivity_shift = delta_lambda / delta_ri

            # --------------------------------------------------
            # LT-MZI sensitivity using CURRENT Î»res
            # --------------------------------------------------

            sensitivity_current_lambda = (
                lambda_exact / delta_neff_at_resonance * (swg_sensor - swg_reference)
            )

            # --------------------------------------------------
            # LT-MZI sensitivity using PREVIOUS Î»res
            #
            # User-requested comparison
            # --------------------------------------------------

            sensitivity_previous_lambda = (
                previous_lambda_exact
                / delta_neff_at_resonance
                * (swg_sensor - swg_reference)
            )

        # ==================================================
        # SAVE RESULT
        # ==================================================

        results.append(
            {
                "gas_refractive_index": current_ri,
                "peak_index": result.peak_index,
                "m": result.m,
                "lambda_find_peaks_um": (lambda_find),
                "lambda_exact_um": (lambda_exact),
                "peak_correction_nm": (result.difference_nm),
                "delta_neff_at_resonance": (delta_neff_at_resonance),
                "swg_sensor": (swg_sensor),
                "swg_reference": (swg_reference),
                "delta_lambda_um": (delta_lambda),
                "delta_ri": (delta_ri),
                "sensitivity_shift_um_per_RIU": (sensitivity_shift),
                "sensitivity_current_lambda_um_per_RIU": (sensitivity_current_lambda),
                "sensitivity_previous_lambda_um_per_RIU": (sensitivity_previous_lambda),
            }
        )

        # --------------------------------------------------
        # Update previous data
        # --------------------------------------------------

        previous_sensor_neff = sensor_neff.copy()

        previous_ri = current_ri

        previous_lambda_find = lambda_find

        previous_lambda_exact = lambda_exact

    # ======================================================
    # PRINT RESULTS
    # ======================================================

    print()
    print("=" * 120)
    print("GAS SENSITIVITY COMPARISON")
    print("=" * 120)

    print()

    header = (
        f"{'RI':>7}"
        f"{'Î»_exact':>14}"
        f"{'Swg,sens':>14}"
        f"{'Î”neff':>14}"
        f"{'S_shift':>16}"
        f"{'S_current':>18}"
        f"{'S_previous':>18}"
    )

    print(header)
    print("-" * 120)

    for row in results:

        def fmt(value, digits=6):

            if value is None:
                return "N/A"

            if isinstance(value, float):
                if np.isnan(value):
                    return "N/A"

                return f"{value:.{digits}f}"

            return str(value)

        print(
            f"{row['gas_refractive_index']:>7.3f}"
            f"{fmt(row['lambda_exact_um'], 9):>14}"
            f"{fmt(row['swg_sensor'], 6):>14}"
            f"{fmt(row['delta_neff_at_resonance'], 9):>14}"
            f"{fmt(row['sensitivity_shift_um_per_RIU'], 6):>16}"
            f"{fmt(row['sensitivity_current_lambda_um_per_RIU'], 6):>18}"
            f"{fmt(row['sensitivity_previous_lambda_um_per_RIU'], 6):>18}"
        )

    print()
    print("=" * 120)

    # ======================================================
    # SAVE CSV
    # ======================================================

    fieldnames = list(results[0].keys())

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(results)

    print()
    print(f"Results saved to:\n{OUTPUT_FILE}")

    print()


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()



