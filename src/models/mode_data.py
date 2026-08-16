from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ModeData:
    """
    Container for Lumerical MODE results.
    """

    wavelength_neff: np.ndarray
    neff: np.ndarray

    wavelength_ng: np.ndarray
    ng: np.ndarray

    wavelength_dispersion: np.ndarray
    dispersion: np.ndarray
