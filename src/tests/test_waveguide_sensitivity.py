from pathlib import Path

import matplotlib.pyplot as plt

from src.config.settings import GAS_RI_STEP
from src.calculations.waveguide_sensitivity import WaveguideSensitivityCalculator
from src.io.loader import ModeDataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "base"


def plot_group(files, delta_medium, title, zoom=False):

    plt.figure(figsize=(10, 6))

    previous_file = files[0]
    previous = ModeDataLoader.load(previous_file)

    for file in files[1:]:
        current = ModeDataLoader.load(file)

        swg = WaveguideSensitivityCalculator.calculate(
            previous_neff=previous.neff,
            current_neff=current.neff,
            delta_medium=delta_medium,
        )

        label = f"{previous_file.stem} -> {file.stem}"

        plt.plot(
            current.wavelength_neff,
            swg,
            linewidth=2,
            label=label,
        )

        previous = current
        previous_file = file

    plt.title(title)

    plt.xlabel("Wavelength (Âµm)")
    plt.ylabel("Waveguide Sensitivity")

    plt.grid(True)

    plt.legend(fontsize=7)

    if zoom:
        plt.ylim(0.45, 0.73)

    plt.tight_layout()


# ==========================================================
# Gas
# ==========================================================

gas_files = sorted(DATA.glob("sensor-gas-*.mat"))

plot_group(
    gas_files,
    delta_medium=GAS_RI_STEP,
    title="Waveguide Sensitivity (Gas)",
)

# ==========================================================
# Gas Zoom
# ==========================================================

plot_group(
    gas_files[1:],
    delta_medium=GAS_RI_STEP,
    title="Waveguide Sensitivity (Gas) - Zoom",
    zoom=True,
)
# ==========================================================
# Liquid
# ==========================================================

liquid_files = sorted(DATA.glob("sensor-liquid-*.mat"))

plot_group(
    liquid_files,
    delta_medium=0.01,
    title="Waveguide Sensitivity (Liquid)",
)

plt.show()








