import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from src.calculations.delta_neff import DeltaNeffCalculator
from src.calculations.peak_detector import PeakDetector
from src.calculations.transmission import TransmissionCalculator
from src.config.settings import DEVICE_LENGTH_UM
from src.io.loader import ModeDataLoader

# ==========================================================
# Data model
# ==========================================================


@dataclass
class GasPeakResult:
    gas_refractive_index: float

    peak_index: int

    transmission_peak: float

    m_float: float
    m_center: int
    m: int

    lambda_find_peaks_um: float
    lambda_exact_um: float

    difference_um: float
    difference_nm: float


# ==========================================================
# Settings
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "base"

REFERENCE_FILE = DATA_DIR / "reference.mat"

OUTPUT_FILE = DATA_DIR / "gas_peak_results.csv"



# ==========================================================
# Gas RI from filename
# ==========================================================


def extract_gas_refractive_index(
    file: Path,
) -> float:

    # Example:
    # sensor-gas-1000.mat
    # 1000 -> 1.000

    value = file.stem.split("-")[-1]

    return int(value) / 1000.0


# ==========================================================
# Process gas sweep
# ==========================================================


def calculate_gas_peak_results():

    reference = ModeDataLoader.load(REFERENCE_FILE)

    wavelength = reference.wavelength_neff

    gas_files = sorted(DATA_DIR.glob("sensor-gas-*.mat"))

    if not gas_files:
        raise RuntimeError("No gas sensor files were found.")

    results = []

    previous_lambda = None

    for file in gas_files:
        sensor = ModeDataLoader.load(file)

        delta_neff = DeltaNeffCalculator.calculate(
            reference,
            sensor,
        )

        transmission = TransmissionCalculator.calculate(
            wavelength=wavelength,
            delta_neff=delta_neff,
            length=DEVICE_LENGTH_UM,
        )

        detector = PeakDetector(
            wavelength=wavelength,
            transmission=transmission,
            delta_neff=delta_neff,
            length=DEVICE_LENGTH_UM,
        )

        result = detector.detect(previous_lambda=previous_lambda)

        gas_ri = extract_gas_refractive_index(file)

        results.append(
            GasPeakResult(
                gas_refractive_index=gas_ri,
                peak_index=result.peak_index,
                transmission_peak=(result.transmission_peak),
                m_float=result.m_float,
                m_center=result.m_center,
                m=result.m,
                lambda_find_peaks_um=(result.lambda_find_peaks),
                lambda_exact_um=(result.lambda_exact),
                difference_um=(result.difference),
                difference_nm=(result.difference_nm),
            )
        )

        previous_lambda = result.lambda_find_peaks

    return results


# ==========================================================
# Save CSV
# ==========================================================


def save_results_csv(
    results,
    output_file: Path,
):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [asdict(result) for result in results]

    fieldnames = list(rows[0].keys())

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    results = calculate_gas_peak_results()

    save_results_csv(
        results,
        OUTPUT_FILE,
    )

    print("=" * 100)
    print("GAS PEAK RESULTS")
    print("=" * 100)

    for result in results:
        print(
            f"n={result.gas_refractive_index:.3f} | "
            f"peak_index={result.peak_index} | "
            f"m={result.m} | "
            f"lambda={result.lambda_exact_um:.12f} Âµm | "
            f"error={result.difference_nm:.6f} nm"
        )

    print()
    print(f"Saved to:\n{OUTPUT_FILE}")


