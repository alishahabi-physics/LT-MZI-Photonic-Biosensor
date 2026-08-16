from pathlib import Path

import matplotlib.pyplot as plt

from src.calculations.delta_neff import DeltaNeffCalculator
from src.calculations.insertion_loss import InsertionLossCalculator
from src.calculations.transmission import TransmissionCalculator
from src.config.settings import DEVICE_LENGTH_UM, WAVELENGTH_MAX_UM, WAVELENGTH_MIN_UM
from src.io.loader import ModeDataLoader

# ==========================================================
# Data
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "base"

REFERENCE_FILE = DATA / "reference.mat"

DEVICE_LENGTH = DEVICE_LENGTH_UM


# ==========================================================
# Load reference
# ==========================================================

reference = ModeDataLoader.load(REFERENCE_FILE)


# ==========================================================
# Plot function
# ==========================================================


def plot_group(files, title):

    plt.figure(figsize=(10, 6))

    for file in files:
        sensor = ModeDataLoader.load(file)

        delta_neff = DeltaNeffCalculator.calculate(
            reference,
            sensor,
        )

        transmission = TransmissionCalculator.calculate(
            wavelength=reference.wavelength_neff,
            delta_neff=delta_neff,
            length=DEVICE_LENGTH,
        )

        insertion_loss = InsertionLossCalculator.calculate(transmission)

        plt.plot(
            reference.wavelength_neff,
            insertion_loss,
            linewidth=2,
            label=file.stem,
        )

    plt.title(title)

    plt.xlabel("Wavelength (Âµm)")
    plt.ylabel("Insertion Loss (dB)")

    plt.xlim(WAVELENGTH_MIN_UM, WAVELENGTH_MAX_UM)

    plt.grid(True)

    plt.legend(
        fontsize=8,
        ncol=2,
    )

    plt.tight_layout()


# ==========================================================
# Gas
# ==========================================================

gas_files = sorted(DATA.glob("sensor-gas-*.mat"))

print(f"Gas files : {len(gas_files)}")

plot_group(gas_files, "Insertion Loss - Gas")


# ==========================================================
# Liquid
# ==========================================================

liquid_files = sorted(DATA.glob("sensor-liquid-*.mat"))

print(f"Liquid files : {len(liquid_files)}")

plot_group(liquid_files, "Insertion Loss - Liquid")


# ==========================================================
# Show
# ==========================================================

plt.show()



