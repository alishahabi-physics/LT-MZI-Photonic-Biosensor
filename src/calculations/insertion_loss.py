import numpy as np


class InsertionLossCalculator:
    """
    Calculates insertion loss.

    MATLAB equivalent:

    IL = 10 * log10(T)
    """

    @staticmethod
    def calculate(
        transmission: np.ndarray,
    ) -> np.ndarray:

        transmission = np.clip(
            transmission,
            np.finfo(float).eps,
            None,
        )

        insertion_loss = 10.0 * np.log10(transmission)

        return insertion_loss
