import numpy as np

from src.models.mode_data import ModeData


class DeltaNeffCalculator:
    """
    Calculates

        Δneff = neff_reference - neff_sensor
    """

    @staticmethod
    def calculate(
        reference: ModeData,
        sensor: ModeData,
    ) -> np.ndarray:

        if len(reference.neff) != len(sensor.neff):
            raise ValueError("Reference and sensor have different neff lengths.")

        if not np.allclose(
            reference.wavelength_neff,
            sensor.wavelength_neff,
            atol=1e-15,
        ):
            raise ValueError("Wavelength grids are different.")

        delta_neff = reference.neff - sensor.neff

        return delta_neff
