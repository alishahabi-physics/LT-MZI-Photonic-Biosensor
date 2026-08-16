from pathlib import Path

from src.calculations.delta_neff import DeltaNeffCalculator
from src.calculations.peak_detector import PeakDetector
from src.config.settings import DEVICE_LENGTH_UM
from src.calculations.transmission import TransmissionCalculator
from src.io.loader import ModeDataLoader

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "base"

REFERENCE_FILE = DATA_DIR / "reference.mat"

LENGTH = DEVICE_LENGTH_UM


# ==========================================================
# PROCESS ONE SWEEP
# ==========================================================


def process_sweep(
    files,
    sweep_name,
):

    print()
    print("=" * 110)
    print(sweep_name)
    print("=" * 110)

    # ------------------------------------------------------
    # Load reference
    # ------------------------------------------------------

    reference = ModeDataLoader.load(REFERENCE_FILE)

    wavelength = reference.wavelength_neff

    # ------------------------------------------------------
    # Previous selected peak
    # ------------------------------------------------------

    previous_lambda = None

    # ------------------------------------------------------
    # Process files
    # ------------------------------------------------------

    for file in files:
        print()
        print("-" * 110)
        print(f"File : {file.name}")

        # --------------------------------------------------
        # Load sensor data
        # --------------------------------------------------

        sensor = ModeDataLoader.load(file)

        # --------------------------------------------------
        # Delta neff
        # --------------------------------------------------

        delta_neff = DeltaNeffCalculator.calculate(
            reference,
            sensor,
        )

        # --------------------------------------------------
        # Transmission
        # --------------------------------------------------

        transmission = TransmissionCalculator.calculate(
            wavelength=wavelength,
            delta_neff=delta_neff,
            length=LENGTH,
        )

        # --------------------------------------------------
        # Peak detector
        # --------------------------------------------------

        detector = PeakDetector(
            wavelength=wavelength,
            transmission=transmission,
            delta_neff=delta_neff,
            length=LENGTH,
        )

        # --------------------------------------------------
        # Detect peak
        # --------------------------------------------------

        result = detector.detect(previous_lambda=previous_lambda)

        # --------------------------------------------------
        # All detected peaks
        # --------------------------------------------------

        print()
        print("ALL find_peaks PEAKS")

        for number, (
            peak_index,
            peak_lambda,
        ) in enumerate(
            zip(
                result.all_peak_indices,
                result.all_peak_wavelengths,
            ),
            start=1,
        ):
            print(
                f"Peak {number:02d} | "
                f"Index = {peak_index:3d} | "
                f"Lambda = "
                f"{peak_lambda:.12f} Âµm"
            )

        # --------------------------------------------------
        # Previous peak
        # --------------------------------------------------

        print()

        if previous_lambda is None:
            print("Previous peak : NONE (first gas file)")

        else:
            print(f"Previous selected peak : {previous_lambda:.12f} Âµm")

        # --------------------------------------------------
        # Selected peak
        # --------------------------------------------------

        print()
        print("SELECTED PEAK")

        print(f"Peak index        : {result.peak_index}")

        print(f"Lambda            : {result.lambda_find_peaks:.12f} Âµm")

        print(f"Transmission      : {result.transmission_peak:.15f}")

        # --------------------------------------------------
        # Fringe order
        # --------------------------------------------------

        print()
        print("FRINGE ORDER")

        print(f"m calculated      : {result.m_float:.12f}")

        print(f"m center          : {result.m_center}")

        print(f"m candidates      : {result.m_candidates.tolist()}")

        print(f"Valid m           : {result.valid_m_candidates.tolist()}")

        # --------------------------------------------------
        # Exact wavelength candidates
        # --------------------------------------------------

        print()
        print("EXACT WAVELENGTH CANDIDATES")

        for m, lambda_exact in zip(
            result.valid_m_candidates,
            result.lambda_exact_candidates,
        ):
            difference = abs(lambda_exact - result.lambda_find_peaks)

            print(
                f"m = {m:3d} | "
                f"lambda_exact = "
                f"{lambda_exact:.12f} Âµm | "
                f"difference = "
                f"{difference:.12e} Âµm | "
                f"{difference * 1000:.12e} nm"
            )

        # --------------------------------------------------
        # Final result
        # --------------------------------------------------

        print()
        print("FINAL RESULT")

        print(f"Selected m       : {result.m}")

        print(f"lambda find_peaks: {result.lambda_find_peaks:.12f} Âµm")

        print(f"lambda exact     : {result.lambda_exact:.12f} Âµm")

        print(f"Difference        : {result.difference:.12e} Âµm")

        print(f"Difference        : {result.difference_nm:.12e} nm")

        # --------------------------------------------------
        # IMPORTANT:
        # Use the find_peaks wavelength as the previous
        # selected peak for the next gas file.
        # --------------------------------------------------

        previous_lambda = result.lambda_find_peaks


# ==========================================================
# GAS FILES ONLY
# ==========================================================

gas_files = sorted(DATA_DIR.glob("sensor-gas-*.mat"))


# ==========================================================
# CHECK GAS FILES
# ==========================================================

print()
print("=" * 110)
print(f"Gas files found : {len(gas_files)}")
print("=" * 110)

for file in gas_files:
    print(file.name)


# ==========================================================
# RUN GAS SWEEP
# ==========================================================

process_sweep(
    gas_files,
    "GAS SWEEP",
)


# ==========================================================
# FINISHED
# ==========================================================

print()
print("=" * 110)
print("GAS SWEEP FINISHED SUCCESSFULLY")
print("=" * 110)


