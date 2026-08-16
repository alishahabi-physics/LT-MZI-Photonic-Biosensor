from pathlib import Path

import numpy as np

from src.io.loader import ModeDataLoader


class ModeDataValidator:
    """
    Validates a Lumerical MODE data file.
    """

    @staticmethod
    def validate(file_path: str | Path) -> bool:

        mode = ModeDataLoader.load(file_path)

        # ---------- neff ----------

        assert mode.wavelength_neff.ndim == 1
        assert mode.neff.ndim == 1

        assert len(mode.wavelength_neff) == len(mode.neff)

        assert np.all(np.isfinite(mode.wavelength_neff))
        assert np.all(np.isfinite(mode.neff))

        assert np.all(np.diff(mode.wavelength_neff) > 0)

        # ---------- ng ----------

        assert mode.wavelength_ng.ndim == 1
        assert mode.ng.ndim == 1

        assert len(mode.wavelength_ng) == len(mode.ng)

        assert np.all(np.isfinite(mode.wavelength_ng))
        assert np.all(np.isfinite(mode.ng))

        assert np.all(np.diff(mode.wavelength_ng) > 0)

        # ---------- dispersion ----------

        assert mode.wavelength_dispersion.ndim == 1
        assert mode.dispersion.ndim == 1

        assert len(mode.wavelength_dispersion) == len(mode.dispersion)

        assert np.all(np.isfinite(mode.wavelength_dispersion))
        assert np.all(np.isfinite(mode.dispersion))

        assert np.all(np.diff(mode.wavelength_dispersion) > 0)

        return True
