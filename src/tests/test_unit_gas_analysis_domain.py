
import numpy as np

from src.analysis.gas_final_analysis import calculate_fwhm


def test_fwhm_returns_nan_when_right_half_crossing_is_outside_domain():

    # Construct a resonance extremely close to the upper
    # wavelength boundary. The left side contains enough
    # wavelength range to reach half maximum, while the
    # right side does not.
    wavelength = np.linspace(
        1.5800,
        1.6000,
        401,
    )

    lambda_res = 1.5999

    # Choose delta_neff so that:
    #
    #   2 * delta_neff * L / lambda_res = 55
    #
    # Therefore lambda_res is an exact transmission maximum.
    delta_neff_value = (
        55.0
        * lambda_res
        / (2.0 * 50.0)
    )

    delta_neff = np.full(
        wavelength.shape,
        delta_neff_value,
        dtype=float,
    )

    result = calculate_fwhm(
        wavelength,
        delta_neff,
        lambda_res,
    )

    assert np.isnan(result)


def test_fwhm_is_finite_when_both_crossings_are_inside_domain():

    wavelength = np.linspace(
        1.50,
        1.65,
        601,
    )

    lambda_res = 1.555

    delta_neff = (
        0.825
        + 0.001
        * (
            wavelength - 1.55
        )
    )

    result = calculate_fwhm(
        wavelength,
        delta_neff,
        lambda_res,
    )

    assert np.isfinite(result)
    assert result > 0.0
