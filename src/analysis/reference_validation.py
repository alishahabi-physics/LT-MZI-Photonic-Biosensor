import csv
from pathlib import Path

import numpy as np
from scipy.stats import linregress

from src.config.settings import DEVICE_LENGTH_UM


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "base"

FINAL_RESULTS_FILE = (
    DATA_DIR / "gas_final_results.csv"
)

OUTPUT_FILE = (
    DATA_DIR / "reference_validation.csv"
)


REFERENCE_PAPER = {
    "title": (
        "Compact Gas Sensor Using "
        "Silicon-on-Insulator "
        "Loop-Terminated Mach-Zehnder Interferometer"
    ),
    "wavelength_um": 1.55,
    "reference_length_um": 150.0,
    "sensitivity_nm_per_RIU": 1070.0,
    "fom_RIU_inv": 280.8,
}


STABLE_RI_MIN = 1.002
STABLE_RI_MAX = 1.009


def load_final_results():

    rows = []

    with FINAL_RESULTS_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            rows.append(
                {
                    "ri": float(row["ri"]),
                    "lambda_exact_um": float(
                        row["lambda_exact_um"]
                    ),
                    "sensitivity": (
                        float(
                            row[
                                "sensitivity_shift_nm_per_riu"
                            ]
                        )
                        if row[
                            "sensitivity_shift_nm_per_riu"
                        ]
                        else np.nan
                    ),
                    "fwhm_nm": float(
                        row["fwhm_nm"]
                    ),
                    "fom": (
                        float(
                            row["fom_riu_inv"]
                        )
                        if row["fom_riu_inv"]
                        else np.nan
                    ),
                }
            )

    if not rows:
        raise RuntimeError(
            "gas_final_results.csv is empty."
        )

    return rows


def main():

    rows = load_final_results()

    stable = [
        row
        for row in rows
        if (
            STABLE_RI_MIN
            <= row["ri"]
            <= STABLE_RI_MAX
        )
    ]

    if len(stable) < 2:
        raise RuntimeError(
            "Not enough stable-range points."
        )

    ri = np.array(
        [row["ri"] for row in stable],
        dtype=float,
    )

    wavelength = np.array(
        [
            row["lambda_exact_um"]
            for row in stable
        ],
        dtype=float,
    )

    fwhm = np.array(
        [
            row["fwhm_nm"]
            for row in stable
        ],
        dtype=float,
    )

    fom = np.array(
        [
            row["fom"]
            for row in stable
        ],
        dtype=float,
    )

    fit = linregress(
        ri,
        wavelength,
    )

    our_sensitivity = (
        fit.slope * 1000.0
    )

    our_r2 = fit.rvalue ** 2

    our_mean_fwhm = np.mean(fwhm)

    our_linear_fom = (
        abs(our_sensitivity)
        / our_mean_fwhm
    )

    our_mean_local_fom = np.mean(fom)

    paper_sensitivity = (
        REFERENCE_PAPER[
            "sensitivity_nm_per_RIU"
        ]
    )

    paper_fom = (
        REFERENCE_PAPER[
            "fom_RIU_inv"
        ]
    )

    # Direct numerical ratios only.
    # No length normalization is applied.
    sensitivity_ratio = (
        our_sensitivity
        / paper_sensitivity
    )

    fom_ratio = (
        our_linear_fom
        / paper_fom
    )

    print()
    print("=" * 110)
    print("REFERENCE PAPER VALIDATION")
    print("=" * 110)

    print()
    print("REFERENCE PAPER")

    print(
        f"Title                : "
        f"{REFERENCE_PAPER['title']}"
    )

    print(
        f"Reference L          : "
        f"{REFERENCE_PAPER['reference_length_um']:.1f} um"
    )

    print(
        f"Reported Sensitivity : "
        f"{paper_sensitivity:.3f} nm/RIU"
    )

    print(
        f"Reported FOM         : "
        f"{paper_fom:.3f} RIU^-1"
    )

    print()
    print("OUR GAS SENSOR")

    print(
        f"RI range             : "
        f"{STABLE_RI_MIN:.3f} - "
        f"{STABLE_RI_MAX:.3f}"
    )

    print(
        f"L                    : "
        f"{DEVICE_LENGTH_UM:.1f} um"
    )

    print(
        f"Linear sensitivity   : "
        f"{our_sensitivity:.6f} nm/RIU"
    )

    print(
        f"Mean FWHM           : "
        f"{our_mean_fwhm:.6f} nm"
    )

    print(
        f"Linear-fit FOM      : "
        f"{our_linear_fom:.6f} RIU^-1"
    )

    print(
        f"Mean local FOM      : "
        f"{our_mean_local_fom:.6f} RIU^-1"
    )

    print(
        f"R2                  : "
        f"{our_r2:.9f}"
    )

    print()
    print("DIRECT NUMERICAL COMPARISON")
    print(
        "Note: device-length scaling is NOT applied."
    )

    print(
        f"Sensitivity ratio   : "
        f"{sensitivity_ratio:.6f} x"
    )

    print(
        f"FOM ratio           : "
        f"{fom_ratio:.6f} x"
    )

    print()
    print("=" * 110)

    result = {
        "reference_title": REFERENCE_PAPER["title"],
        "paper_reference_length_um": (
            REFERENCE_PAPER["reference_length_um"]
        ),
        "our_length_um": DEVICE_LENGTH_UM,
        "paper_sensitivity_nm_per_RIU": (
            paper_sensitivity
        ),
        "paper_fom_RIU_inv": paper_fom,
        "our_ri_min": STABLE_RI_MIN,
        "our_ri_max": STABLE_RI_MAX,
        "our_sensitivity_nm_per_RIU": (
            our_sensitivity
        ),
        "our_mean_fwhm_nm": (
            our_mean_fwhm
        ),
        "our_linear_fom_RIU_inv": (
            our_linear_fom
        ),
        "our_mean_local_fom_RIU_inv": (
            our_mean_local_fom
        ),
        "our_r2": our_r2,
        "direct_sensitivity_ratio": (
            sensitivity_ratio
        ),
        "direct_fom_ratio": fom_ratio,
        "length_scaling_applied": False,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(result.keys()),
        )

        writer.writeheader()
        writer.writerow(result)

    print()
    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
