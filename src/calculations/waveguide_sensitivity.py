import numpy as np


class WaveguideSensitivityCalculator:
    """
    Waveguide sensitivity.

    Swg = d(neff) / d(n_medium)

    Finite Forward Difference.
    """

    @staticmethod
    def calculate(
        previous_neff: np.ndarray,
        current_neff: np.ndarray,
        delta_medium: float,
    ) -> np.ndarray:

        return (current_neff - previous_neff) / delta_medium
