from pathlib import Path

import matplotlib.pyplot as plt

from src.calculations.delta_neff import DeltaNeffCalculator
from src.calculations.transmission import TransmissionCalculator
from src.config.settings import DEVICE_LENGTH_UM, WAVELENGTH_MAX_UM, WAVELENGTH_MIN_UM
from src.io.loader import ModeDataLoader

# ==========================================================
# Data directory
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "base"

REFERENCE_FILE = DATA / "reference.mat"

# ==========================================================
# Load reference
# ==========================================================

reference = ModeDataLoader.load(REFERENCE_FILE)

# ==========================================================
# Device length (Âµm)
# ==========================================================
L = DEVICE_LENGTH_UM
# ==========================================================
# Plot function
# ==========================================================


def plot_group(sensor_files, title):

    plt.figure(figsize=(10, 6))

    for file in sensor_files:
        sensor = ModeDataLoader.load(file)

        delta_neff = DeltaNeffCalculator.calculate(
            reference,
            sensor,
        )

        transmission = TransmissionCalculator.calculate(
            wavelength=reference.wavelength_neff,
            delta_neff=delta_neff,
            length=L,
        )

        plt.plot(
            reference.wavelength_neff,
            transmission,
            linewidth=2,
            label=file.stem,
        )

    plt.title(title, fontsize=14)

    plt.xlabel("Wavelength (Âµm)", fontsize=12)
    plt.ylabel("Transmission", fontsize=12)

    plt.xlim(WAVELENGTH_MIN_UM, WAVELENGTH_MAX_UM)
    plt.ylim(0.0, 1.05)

    plt.grid(True)

    plt.legend(
        fontsize=8,
        ncol=2,
    )

    plt.tight_layout()


# ==========================================================
# Gas files
# ==========================================================

gas_files = sorted(DATA.glob("sensor-gas-*.mat"))

print(f"Gas files : {len(gas_files)}")

plot_group(gas_files, "LT-MZI Transmission Spectrum (Gas)")


# ==========================================================


# ==========================================================
# Show
# ==========================================================

plt.show()



