from pathlib import Path

import numpy as np
from scipy.io import loadmat

from src.models.mode_data import ModeData


class ModeDataLoader:
    """
    Loads consolidated MATLAB MAT files.

    Expected variables:

        wavelength_neff
        neff

        wavelength_ng
        ng

        wavelength_dispersion
        dispersion

    The three wavelength axes are independent.
    No interpolation or resampling is performed.
    """

    @staticmethod
    def load(path: str | Path) -> ModeData:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"MAT file not found: {path}")

        data = loadmat(
            path,
            squeeze_me=True,
            struct_as_record=False,
        )

        required = (
            "wavelength_neff",
            "neff",
            "wavelength_ng",
            "ng",
            "wavelength_dispersion",
            "dispersion",
        )

        missing = [
            name
            for name in required
            if name not in data
        ]

        if missing:
            raise KeyError(
                f"Missing variables in {path.name}: {missing}"
            )

        wavelength_neff = ModeDataLoader._read_array(
            data["wavelength_neff"],
            "wavelength_neff",
        )

        neff = ModeDataLoader._read_array(
            data["neff"],
            "neff",
        )

        wavelength_ng = ModeDataLoader._read_array(
            data["wavelength_ng"],
            "wavelength_ng",
        )

        ng = ModeDataLoader._read_array(
            data["ng"],
            "ng",
        )

        wavelength_dispersion = ModeDataLoader._read_array(
            data["wavelength_dispersion"],
            "wavelength_dispersion",
        )

        dispersion = ModeDataLoader._read_array(
            data["dispersion"],
            "dispersion",
        )

        ModeDataLoader._validate_pair(
            wavelength_neff,
            neff,
            "wavelength_neff",
            "neff",
        )

        ModeDataLoader._validate_pair(
            wavelength_ng,
            ng,
            "wavelength_ng",
            "ng",
        )

        ModeDataLoader._validate_pair(
            wavelength_dispersion,
            dispersion,
            "wavelength_dispersion",
            "dispersion",
        )

        return ModeData(
            wavelength_neff=wavelength_neff,
            neff=neff,
            wavelength_ng=wavelength_ng,
            ng=ng,
            wavelength_dispersion=wavelength_dispersion,
            dispersion=dispersion,
        )

    @staticmethod
    def _read_array(
        value: np.ndarray,
        name: str,
    ) -> np.ndarray:

        array = np.asarray(value)
        array = np.squeeze(array)

        if array.ndim != 1:
            raise ValueError(
                f"{name} must be 1-D, "
                f"got shape {array.shape}"
            )

        array = array.astype(np.float64)

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} contains non-finite values"
            )

        return array

    @staticmethod
    def _validate_pair(
        wavelength: np.ndarray,
        data: np.ndarray,
        wavelength_name: str,
        data_name: str,
    ) -> None:

        if len(wavelength) != len(data):
            raise ValueError(
                f"Length mismatch: "
                f"{wavelength_name}={len(wavelength)}, "
                f"{data_name}={len(data)}"
            )
