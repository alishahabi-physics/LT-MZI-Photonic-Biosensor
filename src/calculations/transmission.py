import numpy as np


class TransmissionCalculator:
    """
    Calculates the transmission spectrum.

    MATLAB equivalent:

    T = cos((2*pi*dn*L)./lam).^2
    """

    @staticmethod
    def calculate(
        wavelength: np.ndarray,
        delta_neff: np.ndarray,
        length: float,
    ) -> np.ndarray:

        if len(wavelength) != len(delta_neff):
            raise ValueError("Wavelength and delta_neff must have the same length.")

        transmission = np.cos((2.0 * np.pi * delta_neff * length) / wavelength) ** 2

        return transmission
