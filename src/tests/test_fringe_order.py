from pathlib import Path

import numpy as np

from src.calculations.delta_neff import DeltaNeffCalculator
from src.config.settings import DEVICE_LENGTH_UM
from src.io.loader import ModeDataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "base"

L = 50.0  # Âµm


reference = ModeDataLoader.load(DATA / "reference.mat")

sensor = ModeDataLoader.load(DATA / "sensor-gas-1000.mat")


delta_neff = DeltaNeffCalculator.calculate(
    reference,
    sensor,
)

wavelength = reference.wavelength_neff


m = (2.0 * delta_neff * L) / wavelength


print("=" * 80)
print("FRINGE ORDER")
print("=" * 80)

print()

print("Minimum m :", np.min(m))
print("Maximum m :", np.max(m))

print()

print("Rounded minimum :", np.floor(np.min(m)))
print("Rounded maximum :", np.ceil(np.max(m)))

print()

print("First 10 values")

for value in m[:10]:
    print(value)

print()

print("Last 10 values")

for value in m[-10:]:
    print(value)


