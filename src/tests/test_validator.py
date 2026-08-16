from pathlib import Path

from src.core.validator import ModeDataValidator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "base"

files = sorted(DATA.glob("*.mat"))

print()

print("=" * 80)
print("VALIDATING FILES")
print("=" * 80)

for file in files:
    ModeDataValidator.validate(file)

    print(f"OK   {file.name}")

print()

print("=" * 80)
print("ALL FILES ARE VALID")
print("=" * 80)

