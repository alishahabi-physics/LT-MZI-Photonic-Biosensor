from pathlib import Path

from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILE = PROJECT_ROOT / "data" / "base" / "reference.mat"

data = loadmat(FILE)

print("=" * 80)
print("VARIABLES INSIDE MAT FILE")
print("=" * 80)

for key, value in data.items():
    if key.startswith("__"):
        continue

    print(f"\nVariable : {key}")
    print(f"Type     : {type(value)}")
    print(f"Shape    : {value.shape}")
    print(f"Dtype    : {value.dtype}")

print("\n")
print("=" * 80)
print("STRUCT DETAILS")
print("=" * 80)

for variable in ["neff", "ng", "dispersion"]:
    print("\n")
    print("=" * 80)
    print(variable.upper())
    print("=" * 80)

    struct = data[variable][0, 0]

    print(f"Struct type : {type(struct)}")
    print(f"Fields      : {struct.dtype.names}")

    for field in struct.dtype.names:
        value = struct[field]

        print(f"\nField : {field}")
        print(f"Shape : {value.shape}")
        print(f"Dtype : {value.dtype}")

        if value.size > 0:
            array = value.squeeze()

            print(f"Array shape : {array.shape}")

            if array.ndim == 1:
                print("First 5 values:")
                print(array[:5])

                print("Last 5 values:")
                print(array[-5:])

            else:
                print("Nested array")
                print(array)

