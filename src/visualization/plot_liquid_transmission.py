from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.calculations.delta_neff import DeltaNeffCalculator
from src.calculations.transmission import TransmissionCalculator
from src.config.settings import DEVICE_LENGTH_UM, WAVELENGTH_MAX_UM, WAVELENGTH_MIN_UM
from src.io.loader import ModeDataLoader

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "base"

REFERENCE_FILE = DATA_DIR / "reference.mat"


# ==========================================================
# LIQUID FILES
# ==========================================================

LIQUID_FILES = [
    DATA_DIR / "sensor-liquid-133.mat",
    DATA_DIR / "sensor-liquid-134.mat",
    DATA_DIR / "sensor-liquid-135.mat",
    DATA_DIR / "sensor-liquid-136.mat",
    DATA_DIR / "sensor-liquid-137.mat",
    DATA_DIR / "sensor-liquid-138.mat",
    DATA_DIR / "sensor-liquid-139.mat",
    DATA_DIR / "sensor-liquid-140.mat",
]


# ==========================================================
# SENSOR LENGTH
# ==========================================================

LENGTH = DEVICE_LENGTH_UM


# ==========================================================
# LOAD REFERENCE
# ==========================================================

reference = ModeDataLoader.load(REFERENCE_FILE)

wavelength = reference.wavelength_neff


# ==========================================================
# CHECK WAVELENGTH UNIT
# ==========================================================

print("=" * 100)
print("LIQUID TRANSMISSION PLOT")
print("=" * 100)

print()

print("Wavelength unit : Âµm")

print(f"First wavelength : {wavelength[0]:.12f} Âµm")

print(f"Last wavelength  : {wavelength[-1]:.12f} Âµm")

print(f"Number of points : {len(wavelength)}")

print()


# ==========================================================
# CREATE FIGURE
# ==========================================================

plt.figure(figsize=(14, 8))


# ==========================================================
# PROCESS ALL LIQUID FILES
# ==========================================================

for sensor_file in LIQUID_FILES:
    print(f"Processing : {sensor_file.name}")

    # ------------------------------------------------------
    # Load sensor
    # ------------------------------------------------------

    sensor = ModeDataLoader.load(sensor_file)

    # ------------------------------------------------------
    # Calculate Delta neff
    # ------------------------------------------------------

    delta_neff = DeltaNeffCalculator.calculate(
        reference,
        sensor,
    )

    # ------------------------------------------------------
    # Calculate Transmission
    # ------------------------------------------------------

    transmission = TransmissionCalculator.calculate(
        wavelength=wavelength,
        delta_neff=delta_neff,
        length=LENGTH,
    )

    # ------------------------------------------------------
    # Check shape
    # ------------------------------------------------------

    if transmission.shape != wavelength.shape:
        raise ValueError(
            f"Shape mismatch for {sensor_file.name}: "
            f"wavelength={wavelength.shape}, "
            f"transmission={transmission.shape}"
        )

    # ------------------------------------------------------
    # Find minimum and maximum
    # ------------------------------------------------------

    max_index = np.argmax(transmission)

    max_lambda = wavelength[max_index]

    max_transmission = transmission[max_index]

    print(f"Maximum Transmission : {max_transmission:.15f}")

    print(f"Maximum wavelength   : {max_lambda:.12f} Âµm")

    print()

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------

    plt.plot(
        wavelength,
        transmission,
        linewidth=1.5,
        label=sensor_file.stem,
    )


# ==========================================================
# AXES
# ==========================================================

plt.xlabel(
    "Wavelength (Âµm)",
    fontsize=13,
)

plt.ylabel(
    "Transmission",
    fontsize=13,
)


# ==========================================================
# TITLE
# ==========================================================

plt.title(
    "Transmission Spectra of Liquid Samples",
    fontsize=15,
)


# ==========================================================
# WAVELENGTH RANGE
# ==========================================================

plt.xlim(
    WAVELENGTH_MIN_UM,
    WAVELENGTH_MAX_UM,
)


# ==========================================================
# TRANSMISSION RANGE
# ==========================================================

plt.ylim(
    0.99,
    1.00001,
)


# ==========================================================
# GRID
# ==========================================================

plt.grid(
    True,
    alpha=0.3,
)


# ==========================================================
# LEGEND
# ==========================================================

plt.legend(
    fontsize=9,
    loc="lower left",
)


# ==========================================================
# LAYOUT
# ==========================================================

plt.tight_layout()


# ==========================================================
# SHOW
# ==========================================================

plt.show()


# ==========================================================
# FINISHED
# ==========================================================

print("=" * 100)
print("PLOT FINISHED")
print("=" * 100)



