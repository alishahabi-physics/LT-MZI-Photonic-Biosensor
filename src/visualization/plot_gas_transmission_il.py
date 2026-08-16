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

OUTPUT_DIR = DATA_DIR / "plots"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# SETTINGS
# ==========================================================

LENGTH_UM = DEVICE_LENGTH_UM


# ==========================================================
# LOAD REFERENCE
# ==========================================================

reference = ModeDataLoader.load(REFERENCE_FILE)

wavelength = np.asarray(
    reference.wavelength_neff,
    dtype=float,
).ravel()


# ==========================================================
# GAS FILES
# ==========================================================

gas_files = sorted(DATA_DIR.glob("sensor-gas-*.mat"))

if not gas_files:
    raise RuntimeError("No gas sensor files were found.")


# ==========================================================
# FIGURE 1
# TRANSMISSION
# ==========================================================

plt.figure(figsize=(12, 8))


# ==========================================================
# FIGURE 2
# INSERTION LOSS
# ==========================================================

plt.figure(figsize=(12, 8))


# ==========================================================
# PROCESS GAS FILES
# ==========================================================

for sensor_file in gas_files:
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
        length=LENGTH_UM,
    )

    # ------------------------------------------------------
    # Calculate Insertion Loss
    #
    # IL = 10 log10(T)
    # ------------------------------------------------------

    transmission_safe = np.clip(
        transmission,
        np.finfo(float).tiny,
        None,
    )

    insertion_loss = 10.0 * np.log10(transmission_safe)

    # ------------------------------------------------------
    # Gas refractive index
    #
    # Example:
    # sensor-gas-1000.mat -> 1.000
    # sensor-gas-1009.mat -> 1.009
    # ------------------------------------------------------

    gas_index = int(sensor_file.stem.split("-")[-1])

    gas_ri = gas_index / 1000.0

    label = f"n={gas_ri:.3f}"

    # ------------------------------------------------------
    # Plot Transmission
    # ------------------------------------------------------

    plt.figure(1)

    plt.plot(
        wavelength,
        transmission,
        linewidth=1.3,
        label=label,
    )

    # ------------------------------------------------------
    # Plot Insertion Loss
    # ------------------------------------------------------

    plt.figure(2)

    plt.plot(
        wavelength,
        insertion_loss,
        linewidth=1.3,
        label=label,
    )


# ==========================================================
# TRANSMISSION FIGURE SETTINGS
# ==========================================================

plt.figure(1)

plt.xlabel(
    "Wavelength (Âµm)",
    fontsize=13,
)

plt.ylabel(
    "Transmission",
    fontsize=13,
)

plt.title(
    "Gas Sensor Transmission Spectra",
    fontsize=15,
)

plt.xlim(
    WAVELENGTH_MIN_UM,
    WAVELENGTH_MAX_UM,
)

plt.ylim(
    0.0,
    1.01,
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend(
    fontsize=9,
    ncol=2,
)

plt.tight_layout()


# ==========================================================
# SAVE TRANSMISSION
# ==========================================================

transmission_path = OUTPUT_DIR / "gas_transmission.png"

plt.savefig(
    transmission_path,
    dpi=300,
    bbox_inches="tight",
)


# ==========================================================
# INSERTION LOSS FIGURE SETTINGS
# ==========================================================

plt.figure(2)

plt.xlabel(
    "Wavelength (Âµm)",
    fontsize=13,
)

plt.ylabel(
    "Insertion Loss (dB)",
    fontsize=13,
)

plt.title(
    "Gas Sensor Insertion Loss Spectra",
    fontsize=15,
)

plt.xlim(
    WAVELENGTH_MIN_UM,
    WAVELENGTH_MAX_UM,
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend(
    fontsize=9,
    ncol=2,
)

plt.tight_layout()


# ==========================================================
# SAVE INSERTION LOSS
# ==========================================================

il_path = OUTPUT_DIR / "gas_insertion_loss.png"

plt.savefig(
    il_path,
    dpi=300,
    bbox_inches="tight",
)


# ==========================================================
# SHOW FIGURES
# ==========================================================

plt.figure(1)
plt.show()

plt.figure(2)
plt.show()


# ==========================================================
# FINISHED
# ==========================================================

print()
print("=" * 100)
print("GAS TRANSMISSION / IL PLOTS CREATED")
print("=" * 100)

print()
print(f"Transmission:\n{transmission_path}")

print()
print(f"Insertion Loss:\n{il_path}")


