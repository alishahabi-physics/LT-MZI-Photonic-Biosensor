from pathlib import Path

from src.calculations.delta_neff import DeltaNeffCalculator
from src.io.loader import ModeDataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "base"

reference = ModeDataLoader.load(DATA / "reference.mat")

sensor = ModeDataLoader.load(DATA / "sensor-liquid-133.mat")

delta_neff = DeltaNeffCalculator.calculate(
    reference,
    sensor,
)

print()

print("=" * 80)
print("DELTA NEFF")
print("=" * 80)

print("Shape :", delta_neff.shape)

print()

print("First 10 values")

for value in delta_neff[:10]:
    print(value)

print()

print("Minimum :", delta_neff.min())
print("Maximum :", delta_neff.max())

