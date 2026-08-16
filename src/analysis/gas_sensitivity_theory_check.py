import csv
from pathlib import Path

import numpy as np

from src.io.loader import ModeDataLoader

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "base"

REFERENCE_FILE = DATA_DIR / "reference.mat"

PEAK_RESULTS_FILE = DATA_DIR / "gas_sensitivity_comparison.csv"

OUTPUT_FILE = DATA_DIR / "gas_sensitivity_theory_check.csv"


# ==========================================================
# LOAD PEAK RESULTS
# ==========================================================


def load_peak_results():

    rows = []

    with PEAK_RESULTS_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(
                {
                    "ri": float(row["gas_refractive_index"]),
                    "lambda_exact": float(row["lambda_exact_um"]),
                    "lambda_find": float(row["lambda_find_peaks_um"]),
                    "m": int(row["m"]),
                }
            )

    if not rows:
        raise RuntimeError("gas_sensitivity_comparison.csv is empty.")

    return rows


# ==========================================================
# INTERPOLATION
# ==========================================================


def interpolate(
    wavelength,
    values,
    target,
):
    """
    Linear interpolation.
    """

    return float(
        np.interp(
            target,
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
    # Load previously calculated peak results
    # ------------------------------------------------------

    peak_rows = load_peak_results()

    results = []

    previous_sensor_neff = None
    previous_ri = None
    previous_lambda = None

    # ======================================================
    # GAS SWEEP
    # ======================================================

    for row in peak_rows:
        ri = row["ri"]

        lambda_exact = row["lambda_exact"]

        lambda_find = row["lambda_find"]

        sensor_file = DATA_DIR / f"sensor-gas-{int(round(ri * 1000)):04d}.mat"

        sensor = ModeDataLoader.load(sensor_file)

        sensor_neff = np.asarray(
            sensor.neff,
            dtype=float,
        ).ravel()

        # --------------------------------------------------
        # Shape validation
        # --------------------------------------------------

        if sensor_neff.shape != wavelength.shape:
            raise ValueError(
                f"Shape mismatch in {sensor_file.name}: "
                f"wavelength={wavelength.shape}, "
                f"sensor_neff={sensor_neff.shape}"
            )

        # --------------------------------------------------
        # Two definitions of Delta neff
        # --------------------------------------------------

        delta_ref_minus_sensor = reference_neff - sensor_neff

        delta_sensor_minus_ref = sensor_neff - reference_neff

        # ==================================================
        # FIRST GAS POINT
        # ==================================================

        if previous_sensor_neff is None:
            delta_ref_minus_sensor_at_res = interpolate(
                wavelength,
                delta_ref_minus_sensor,
                lambda_exact,
            )

            delta_sensor_minus_ref_at_res = interpolate(
                wavelength,
                delta_sensor_minus_ref,
                lambda_exact,
            )

            results.append(
                {
                    "ri": ri,
                    "lambda_exact_um": (lambda_exact),
                    "lambda_find_peaks_um": (lambda_find),
                    "m": row["m"],
                    "delta_ref_minus_sensor": (delta_ref_minus_sensor_at_res),
                    "delta_sensor_minus_ref": (delta_sensor_minus_ref_at_res),
                    "swg_sensor": np.nan,
                    "s_lt_ref_minus_sensor": np.nan,
                    "s_lt_sensor_minus_ref": np.nan,
                    "s_lt_previous_ref_minus_sensor": np.nan,
                    "s_lt_previous_sensor_minus_ref": np.nan,
                    "d_delta_d_lambda_ref_minus_sensor": np.nan,
                    "s_implicit_ref_minus_sensor": np.nan,
                    "s_implicit_sensor_minus_ref": np.nan,
                    "s_shift": np.nan,
                }
            )

            previous_sensor_neff = sensor_neff.copy()

            previous_ri = ri

            previous_lambda = lambda_exact

            continue

        # ==================================================
        # FOLLOWING GAS POINTS
        # ==================================================

        delta_ri = ri - previous_ri

        if delta_ri == 0:
            raise ZeroDivisionError(
                f"Zero RI difference between {previous_ri} and {ri}."
            )

        # --------------------------------------------------
        # Waveguide sensitivity of sensing arm
        #
        # Swg,sens = d(neff_sensor) / dn_medium
        # --------------------------------------------------

        swg_array = (sensor_neff - previous_sensor_neff) / delta_ri

        swg_sensor = interpolate(
            wavelength,
            swg_array,
            lambda_exact,
        )

        # --------------------------------------------------
        # Delta neff at resonance
        # --------------------------------------------------

        delta_ref_minus_sensor_at_res = interpolate(
            wavelength,
            delta_ref_minus_sensor,
            lambda_exact,
        )

        delta_sensor_minus_ref_at_res = interpolate(
            wavelength,
            delta_sensor_minus_ref,
            lambda_exact,
        )

        # --------------------------------------------------
        # User's LT-MZI formula
        #
        # S = lambda / Delta_neff
        #     * (Swg,sens - Swg,ref)
        #
        # S_wg,ref = 0
        #
        # Case A:
        # Delta = reference - sensor
        #
        # dDelta/dn = -Swg
        # --------------------------------------------------

        s_lt_ref_minus_sensor = (
            lambda_exact / delta_ref_minus_sensor_at_res * (-swg_sensor)
        )

        # --------------------------------------------------
        # Case B:
        #
        # Delta = sensor - reference
        #
        # dDelta/dn = +Swg
        # --------------------------------------------------

        s_lt_sensor_minus_ref = (
            lambda_exact / delta_sensor_minus_ref_at_res * swg_sensor
        )

        # --------------------------------------------------
        # Previous lambda comparison
        # --------------------------------------------------

        s_lt_previous_ref_minus_sensor = (
            previous_lambda / delta_ref_minus_sensor_at_res * (-swg_sensor)
        )

        s_lt_previous_sensor_minus_ref = (
            previous_lambda / delta_sensor_minus_ref_at_res * swg_sensor
        )

        # --------------------------------------------------
        # d(Delta neff) / d(lambda)
        # --------------------------------------------------

        d_delta_ref_minus_sensor = np.gradient(
            delta_ref_minus_sensor,
            wavelength,
        )

        d_delta_d_lambda = interpolate(
            wavelength,
            d_delta_ref_minus_sensor,
            lambda_exact,
        )

        # --------------------------------------------------
        # Implicit resonance sensitivity
        #
        # Resonance:
        #
        # 2 L Delta(lambda,n) / lambda = m
        #
        # Therefore:
        #
        # d(lambda)/dn =
        #
        # lambda * dDelta/dn
        # -----------------------------
        # Delta - lambda*dDelta/dlambda
        # --------------------------------------------------

        s_implicit_ref_minus_sensor = (
            lambda_exact
            * (-swg_sensor)
            / (delta_ref_minus_sensor_at_res - lambda_exact * d_delta_d_lambda)
        )

        # --------------------------------------------------
        # Equivalent sign convention
        # Delta = sensor - reference
        # --------------------------------------------------

        d_delta_sensor_minus_ref = -d_delta_d_lambda

        s_implicit_sensor_minus_ref = (
            lambda_exact
            * swg_sensor
            / (delta_sensor_minus_ref_at_res - lambda_exact * d_delta_sensor_minus_ref)
        )

        # --------------------------------------------------
        # Direct resonance shift sensitivity
        # --------------------------------------------------

        s_shift = (lambda_exact - previous_lambda) / delta_ri

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        results.append(
            {
                "ri": ri,
                "lambda_exact_um": (lambda_exact),
                "lambda_find_peaks_um": (lambda_find),
                "m": row["m"],
                "delta_ref_minus_sensor": (delta_ref_minus_sensor_at_res),
                "delta_sensor_minus_ref": (delta_sensor_minus_ref_at_res),
                "swg_sensor": (swg_sensor),
                "s_lt_ref_minus_sensor": (s_lt_ref_minus_sensor),
                "s_lt_sensor_minus_ref": (s_lt_sensor_minus_ref),
                "s_lt_previous_ref_minus_sensor": (s_lt_previous_ref_minus_sensor),
                "s_lt_previous_sensor_minus_ref": (s_lt_previous_sensor_minus_ref),
                "d_delta_d_lambda_ref_minus_sensor": (d_delta_d_lambda),
                "s_implicit_ref_minus_sensor": (s_implicit_ref_minus_sensor),
                "s_implicit_sensor_minus_ref": (s_implicit_sensor_minus_ref),
                "s_shift": (s_shift),
            }
        )

        # --------------------------------------------------
        # Update previous values
        # --------------------------------------------------

        previous_sensor_neff = sensor_neff.copy()

        previous_ri = ri

        previous_lambda = lambda_exact

    # ======================================================
    # PRINT RESULTS
    # ======================================================

    print()
    print("=" * 150)
    print("GAS SENSITIVITY THEORY CHECK")
    print("=" * 150)

    print()

    print(
        f"{'RI':>6}"
        f"{'S_shift':>14}"
        f"{'S_LT A':>14}"
        f"{'S_LT B':>14}"
        f"{'S_prev A':>14}"
        f"{'S_prev B':>14}"
        f"{'S_impl A':>18}"
        f"{'S_impl B':>18}"
    )

    print("-" * 150)

    for row in results:

        def fmt(value):

            if np.isnan(value):
                return "N/A"

            return f"{value:.8f}"

        print(
            f"{row['ri']:>6.3f}"
            f"{fmt(row['s_shift']):>14}"
            f"{fmt(row['s_lt_ref_minus_sensor']):>14}"
            f"{fmt(row['s_lt_sensor_minus_ref']):>14}"
            f"{fmt(row['s_lt_previous_ref_minus_sensor']):>14}"
            f"{fmt(row['s_lt_previous_sensor_minus_ref']):>14}"
            f"{fmt(row['s_implicit_ref_minus_sensor']):>18}"
            f"{fmt(row['s_implicit_sensor_minus_ref']):>18}"
        )

    print()
    print("=" * 150)

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
    print(f"Results saved to:\n{OUTPUT_FILE}")


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()

