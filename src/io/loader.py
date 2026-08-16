from pathlib import Path

import numpy as np
from scipy.io import loadmat

from src.models.mode_data import ModeData


class ModeDataLoader:
    """
    Loads a Lumerical MODE .mat file.
    """

    @staticmethod
    def load(file_path: str | Path) -> ModeData:

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        data = loadmat(file_path)

        neff_struct = data["neff"][0, 0]
        ng_struct = data["ng"][0, 0]
        dispersion_struct = data["dispersion"][0, 0]

        wavelength_neff = np.asarray(neff_struct["wavelength"]).squeeze()

        neff = np.asarray(neff_struct["neff"]).squeeze()

        wavelength_ng = np.asarray(ng_struct["wavelength"]).squeeze()

        ng = np.asarray(ng_struct["ng"]).squeeze()

        wavelength_dispersion = np.asarray(dispersion_struct["wavelength"]).squeeze()

        dispersion = np.asarray(dispersion_struct["dispersion"]).squeeze()

        return ModeData(
            wavelength_neff=wavelength_neff,
            neff=neff,
            wavelength_ng=wavelength_ng,
            ng=ng,
            wavelength_dispersion=wavelength_dispersion,
            dispersion=dispersion,
        )
