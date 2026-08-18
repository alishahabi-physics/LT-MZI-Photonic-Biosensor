from pathlib import Path

import numpy as np

from src.calculations.delta_neff import DeltaNeffCalculator
from src.calculations.insertion_loss import InsertionLossCalculator
from src.calculations.transmission import TransmissionCalculator
from src.calculations.waveguide_sensitivity import WaveguideSensitivityCalculator
from src.calculations.peak_detector import PeakDetector
from src.config.settings import DEVICE_LENGTH_UM, GAS_RI_STEP
from src.io.loader import ModeDataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "base"


def test_loader_reference():
    mode = ModeDataLoader.load(DATA / "reference.mat")

    assert mode.wavelength_neff.shape == mode.neff.shape
    assert mode.wavelength_ng.shape == mode.ng.shape
    assert mode.wavelength_dispersion.shape == mode.dispersion.shape
    assert len(mode.wavelength_neff) > 0


def test_delta_neff():
    reference = ModeDataLoader.load(DATA / "reference.mat")
    sensor = ModeDataLoader.load(DATA / "sensor-gas-1000.mat")

    delta = DeltaNeffCalculator.calculate(reference, sensor)

    assert delta.shape == reference.neff.shape
    assert np.allclose(delta, reference.neff - sensor.neff)


def test_transmission_formula():
    wavelength = np.array([1.55, 1.56, 1.57])
    delta_neff = np.array([0.8, 0.81, 0.82])
    length = DEVICE_LENGTH_UM

    expected = np.cos(
        (2.0 * np.pi * delta_neff * length) / wavelength
    ) ** 2

    actual = TransmissionCalculator.calculate(
        wavelength=wavelength,
        delta_neff=delta_neff,
        length=length,
    )

    assert np.allclose(actual, expected)
    assert np.all(actual >= 0.0)
    assert np.all(actual <= 1.0)


def test_insertion_loss_formula():
    transmission = np.array([1.0, 0.5, 0.1])

    actual = InsertionLossCalculator.calculate(transmission)
    expected = 10.0 * np.log10(transmission)

    assert np.allclose(actual, expected)


def test_waveguide_sensitivity():
    previous = np.array([2.0, 2.1, 2.2])
    current = np.array([2.01, 2.12, 2.23])
    delta_medium = 0.001

    actual = WaveguideSensitivityCalculator.calculate(
        previous_neff=previous,
        current_neff=current,
        delta_medium=delta_medium,
    )

    expected = (current - previous) / delta_medium

    assert np.allclose(actual, expected)


def test_peak_detector_gas_first_file():
    reference = ModeDataLoader.load(DATA / "reference.mat")
    sensor = ModeDataLoader.load(DATA / "sensor-gas-1000.mat")

    delta_neff = DeltaNeffCalculator.calculate(reference, sensor)

    transmission = TransmissionCalculator.calculate(
        wavelength=reference.wavelength_neff,
        delta_neff=delta_neff,
        length=DEVICE_LENGTH_UM,
    )

    detector = PeakDetector(
        wavelength=reference.wavelength_neff,
        transmission=transmission,
        delta_neff=delta_neff,
        length=DEVICE_LENGTH_UM,
    )

    result = detector.detect()

    assert result.peak_index >= 0
    assert np.isfinite(result.lambda_find_peaks)
    assert np.isfinite(result.lambda_exact)
    assert result.m == 56


def test_gas_analysis_result_count():
    from src.analysis.gas_final_analysis import calculate_results

    results = calculate_results()

    assert len(results) == 10
    assert results[0]["ri"] == 1.0
    assert results[-1]["ri"] == 1.009

    stable = [
        row
        for row in results
        if 1.002 <= row["ri"] <= 1.009
    ]

    assert len(stable) == 8

    # The resonance and sensitivity remain valid across
    # RI = 1.002 ... 1.009.
    for row in stable:
        assert np.isfinite(
            row["sensitivity_formula_nm_per_riu"]
        )

    # With the current 1500-1600 nm wavelength window,
    # FWHM is fully observable only for RI=1.002 ... 1.005.
    fwhm_valid = [
        row
        for row in stable
        if np.isfinite(row["fwhm_nm"])
    ]

    assert len(fwhm_valid) == 4

    assert [
        row["ri"]
        for row in fwhm_valid
    ] == [
        1.002,
        1.003,
        1.004,
        1.005,
    ]

    # FOM is valid exactly when FWHM is valid.
    for row in stable:

        if np.isfinite(row["fwhm_nm"]):
            assert np.isfinite(
                row["fom_riu_inv"]
            )

        else:
            assert np.isnan(
                row["fom_riu_inv"]
            )
