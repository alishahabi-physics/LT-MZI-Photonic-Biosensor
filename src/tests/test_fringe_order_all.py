from pathlib import Path

import numpy as np

from src.calculations.delta_neff import DeltaNeffCalculator
from src.config.settings import DEVICE_LENGTH_UM
from src.io.loader import ModeDataLoader

# ==========================================================
# Settings
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "base"

L = 50.0  # Âµm

REFERENCE_FILE = DATA / "reference.mat"

TEST_FILES = [
    DATA / "sensor-gas-1000.mat",
    DATA / "sensor-gas-1009.mat",
    DATA / "sensor-liquid-133.mat",
    DATA / "sensor-liquid-140.mat",
]


# ==========================================================
# Load reference
# ==========================================================

reference = ModeDataLoader.load(REFERENCE_FILE)


# ==========================================================
# Calculate fringe order
# ==========================================================

print("=" * 80)
print("FRINGE ORDER TEST")
print("=" * 80)

global_min = np.inf
global_max = -np.inf

for file in TEST_FILES:
    sensor = ModeDataLoader.load(file)

    delta_neff = DeltaNeffCalculator.calculate(
        reference,
        sensor,
    )

    wavelength = reference.wavelength_neff

    m = (2.0 * delta_neff * L) / wavelength

    m_min = np.min(m)
    m_max = np.max(m)

    global_min = min(global_min, m_min)
    global_max = max(global_max, m_max)

    print()
    print("-" * 80)
    print(file.name)
    print("-" * 80)

    print(f"Minimum m : {m_min:.10f}")
    print(f"Maximum m : {m_max:.10f}")

    print(f"Rounded minimum : {int(np.floor(m_min))}")
    print(f"Rounded maximum : {int(np.ceil(m_max))}")

print()
print("=" * 80)
print("GLOBAL RESULT")
print("=" * 80)

print(f"Global minimum : {global_min:.10f}")
print(f"Global maximum : {global_max:.10f}")

print()
print(f"Suggested M_MIN = {int(np.floor(global_min))}")
print(f"Suggested M_MAX = {int(np.ceil(global_max))}")

print("=" * 80)


