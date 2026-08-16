from pathlib import Path

from src.io.loader import ModeDataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILE = PROJECT_ROOT / "data" / "base" / "reference.mat"

mode = ModeDataLoader.load(FILE)

print()

print("Effective Index")
print("----------------------------")
print(mode.wavelength_neff.shape)
print(mode.neff.shape)

print()

print("Group Index")
print("----------------------------")
print(mode.wavelength_ng.shape)
print(mode.ng.shape)

print()

print("Dispersion")
print("----------------------------")
print(mode.wavelength_dispersion.shape)
print(mode.dispersion.shape)

print()

print("First wavelength :", mode.wavelength_neff[0])
print("Last wavelength  :", mode.wavelength_neff[-1])

print()

print("First neff :", mode.neff[0])
print("Last neff  :", mode.neff[-1])
