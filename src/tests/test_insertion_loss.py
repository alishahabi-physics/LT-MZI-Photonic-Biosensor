from pathlib import Path

from src.calculations.delta_neff import DeltaNeffCalculator
from src.calculations.insertion_loss import InsertionLossCalculator
from src.calculations.transmission import TransmissionCalculator
from src.io.loader import ModeDataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "base"

reference = ModeDataLoader.load(DATA / "reference.mat")

sensor = ModeDataLoader.load(DATA / "sensor-gas-1000.mat")

delta_neff = DeltaNeffCalculator.calculate(
    reference,
    sensor,
)

transmission = TransmissionCalculator.calculate(
    wavelength=reference.wavelength_neff,
    delta_neff=delta_neff,
    length=50.0,
)

insertion_loss = InsertionLossCalculator.calculate(
    transmission,
)

print("=" * 80)
print("INSERTION LOSS")
print("=" * 80)

print()

print("Shape :", insertion_loss.shape)

print()

print("Minimum :", insertion_loss.min())
print("Maximum :", insertion_loss.max())

print()

print("First 10 values:")

for value in insertion_loss[:10]:
    print(value)

