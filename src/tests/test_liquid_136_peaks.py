from pathlib import Path

from scipy.signal import find_peaks

from src.calculations.delta_neff import DeltaNeffCalculator
from src.calculations.transmission import TransmissionCalculator
from src.io.loader import ModeDataLoader

# ==========================================================
# Settings
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "base"

REFERENCE_FILE = DATA / "reference.mat"
SENSOR_FILE = DATA / "sensor-liquid-136.mat"

LENGTH = 50.0  # Âµm


# ==========================================================
# Load data
# ==========================================================

reference = ModeDataLoader.load(REFERENCE_FILE)

sensor = ModeDataLoader.load(SENSOR_FILE)


# ==========================================================
# Delta neff
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

peaks, properties = find_peaks(transmission)


# ==========================================================
# Output
# ==========================================================

print("=" * 100)
print("ALL TRANSMISSION PEAKS")
print("=" * 100)

print()

print(f"Number of peaks : {len(peaks)}")

print()

for i, index in enumerate(peaks):
    print(
        f"Peak {i + 1:02d} | "
        f"Index = {index:3d} | "
        f"Lambda = {wavelength[index]:.12f} Âµm | "
        f"Transmission = {transmission[index]:.15f}"
    )

print()

print("=" * 100)
print("PREVIOUS PEAK")
print("=" * 100)

print("Previous liquid-135 peak = 1.639072847682 Âµm")

print()

print("=" * 100)

