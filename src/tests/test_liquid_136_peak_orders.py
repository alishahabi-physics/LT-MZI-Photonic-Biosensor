from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from src.calculations.delta_neff import DeltaNeffCalculator
from src.calculations.transmission import TransmissionCalculator
from src.config.settings import DEVICE_LENGTH_UM
from src.io.loader import ModeDataLoader

# ==========================================================
# Settings
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "base"

REFERENCE_FILE = DATA / "reference.mat"
SENSOR_FILE = DATA / "sensor-liquid-136.mat"

LENGTH = DEVICE_LENGTH_UM


# ==========================================================
# Load data
# ==========================================================

reference = ModeDataLoader.load(REFERENCE_FILE)

sensor = ModeDataLoader.load(SENSOR_FILE)


# ==========================================================
# Calculate Delta neff
# ==========================================================

delta_neff = DeltaNeffCalculator.calculate(
    reference,
    sensor,
)


# ==========================================================
# Transmission
# ==========================================================

wavelength = reference.wavelength_neff

transmission = TransmissionCalculator.calculate(
    wavelength=wavelength,
    delta_neff=delta_neff,
    length=LENGTH,
)


# ==========================================================
# Find all Transmission maxima
# ==========================================================

peaks, _ = find_peaks(transmission)


# ==========================================================
# Output
# ==========================================================

print("=" * 100)
print("LIQUID-136 PEAK ORDER ANALYSIS")
print("=" * 100)

print()

for number, index in enumerate(peaks, start=1):
    lambda_peak = wavelength[index]

    delta_peak = delta_neff[index]

    m_float = 2.0 * delta_peak * LENGTH / lambda_peak

    m = int(np.rint(m_float))

    print("-" * 100)

    print(f"Peak number       : {number}")

    print(f"Index             : {index}")

    print(f"Lambda            : {lambda_peak:.12f} Âµm")

    print(f"Delta neff        : {delta_peak:.15f}")

    print(f"m (calculated)    : {m_float:.12f}")

    print(f"m (rounded)       : {m}")

    print(f"Transmission      : {transmission[index]:.15f}")

print()

print("=" * 100)
print("PREVIOUS LIQUID-135 PEAK")
print("=" * 100)

print("Lambda : 1.639072847682 Âµm")

print("m      : 42")

print("=" * 100)



