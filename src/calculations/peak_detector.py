from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks


@dataclass
class PeakDetectionResult:
    peak_index: int

    transmission_peak: float

    lambda_find_peaks: float
    lambda_exact: float

    difference: float
    difference_nm: float

    m_float: float
    m_center: int
    m: int

    m_candidates: np.ndarray
    valid_m_candidates: np.ndarray
    lambda_exact_candidates: np.ndarray

    all_peak_indices: np.ndarray
    all_peak_wavelengths: np.ndarray


class PeakDetector:
    """
    Detect the Transmission peak according to the following
    project-specific rule:

    1. Find all local maxima using scipy.signal.find_peaks.

    2. For the first sensor file:
           select the FIRST peak.

    3. For every following sensor file:
           starting from the FIRST detected peak, select the
           FIRST peak whose wavelength is greater than the
           previously selected peak wavelength.

    4. The largest Transmission value is NOT used for peak
       selection.

    5. After selecting the peak, calculate the fringe order m.

    6. Construct seven integer m values centered around m_center:

           m_center - 3 ... m_center + 3

    7. Solve the exact wavelength for every m that has a
       solution inside the available wavelength range.

    8. Select the exact wavelength closest to the wavelength
       obtained from find_peaks.

    9. peak_index always remains the index returned by
       find_peaks.
    """

    def __init__(
        self,
        wavelength: np.ndarray,
        transmission: np.ndarray,
        delta_neff: np.ndarray,
        length: float,
    ):

        self.wavelength = np.asarray(
            wavelength,
            dtype=float,
        ).ravel()

        self.transmission = np.asarray(
            transmission,
            dtype=float,
        ).ravel()

        self.delta_neff = np.asarray(
            delta_neff,
            dtype=float,
        ).ravel()

        self.length = float(length)

        self._validate_inputs()

    # ==========================================================
    # Validation
    # ==========================================================

    def _validate_inputs(self) -> None:

        if self.wavelength.size < 3:
            raise ValueError("At least 3 wavelength points are required.")

        if not (self.wavelength.size == self.transmission.size == self.delta_neff.size):
            raise ValueError(
                "wavelength, transmission and delta_neff must have the same length."
            )

        if self.length <= 0:
            raise ValueError("length must be positive.")

        if not np.all(np.isfinite(self.wavelength)):
            raise ValueError("wavelength contains non-finite values.")

        if not np.all(np.isfinite(self.transmission)):
            raise ValueError("transmission contains non-finite values.")

        if not np.all(np.isfinite(self.delta_neff)):
            raise ValueError("delta_neff contains non-finite values.")

        if not np.all(np.diff(self.wavelength) > 0):
            raise ValueError("wavelength must be strictly increasing.")

    # ==========================================================
    # Main detection
    # ==========================================================

    def detect(
        self,
        previous_lambda: float | None = None,
    ) -> PeakDetectionResult:

        # ------------------------------------------------------
        # STEP 1
        # Find ALL local Transmission maxima
        # ------------------------------------------------------

        peaks, _ = find_peaks(self.transmission)

        if peaks.size == 0:
            raise RuntimeError("No Transmission peaks were found by find_peaks.")

        # ------------------------------------------------------
        # STEP 2
        # Select the peak according to the project rule
        # ------------------------------------------------------

        if previous_lambda is None:
            # --------------------------------------------------
            # First file:
            # FIRST peak is selected.
            # --------------------------------------------------

            peak_index = int(peaks[0])

        else:
            # --------------------------------------------------
            # Following files:
            #
            # Start from the first detected peak and select
            # the FIRST peak whose wavelength is greater than
            # the previous selected peak.
            # --------------------------------------------------

            selected_peak = None

            for peak_index_candidate in peaks:
                current_lambda = float(self.wavelength[peak_index_candidate])

                if current_lambda > previous_lambda:
                    selected_peak = int(peak_index_candidate)

                    break

            if selected_peak is None:
                raise RuntimeError(
                    "\n"
                    "No Transmission peak was found "
                    "at a wavelength greater than the "
                    "previous selected peak.\n\n"
                    f"Previous peak wavelength : "
                    f"{previous_lambda:.12f} µm\n"
                    f"Available peak wavelengths : "
                    f"{self.wavelength[peaks]}"
                )

            peak_index = selected_peak

        # ------------------------------------------------------
        # STEP 3
        # Selected peak wavelength
        # ------------------------------------------------------

        lambda_find_peaks = float(self.wavelength[peak_index])

        transmission_peak = float(self.transmission[peak_index])

        # ------------------------------------------------------
        # STEP 4
        # Delta neff at selected peak
        # ------------------------------------------------------

        delta_neff_at_peak = float(self.delta_neff[peak_index])

        # ------------------------------------------------------
        # STEP 5
        # Calculate continuous m
        # ------------------------------------------------------

        m_float = 2.0 * delta_neff_at_peak * self.length / lambda_find_peaks

        # ------------------------------------------------------
        # STEP 6
        # Central integer m
        # ------------------------------------------------------

        m_center = int(np.rint(m_float))

        # ------------------------------------------------------
        # STEP 7
        # Seven-value m range
        #
        # Example:
        #
        # m_center = 42
        #
        # 39 40 41 42 43 44 45
        #          ↑
        #        center
        # ------------------------------------------------------

        m_candidates = np.arange(
            m_center - 3,
            m_center + 4,
            dtype=int,
        )

        # ------------------------------------------------------
        # STEP 8
        # Solve exact wavelength for valid m values
        # ------------------------------------------------------

        valid_m_candidates = []

        lambda_exact_candidates = []

        for m_candidate in m_candidates:
            lambda_exact = self._solve_lambda_for_m(int(m_candidate))

            if lambda_exact is None:
                continue

            valid_m_candidates.append(int(m_candidate))

            lambda_exact_candidates.append(float(lambda_exact))

        if len(valid_m_candidates) == 0:
            raise RuntimeError(
                "\n"
                "None of the seven m candidates "
                "has an exact wavelength inside the "
                "available wavelength range.\n\n"
                f"m center : {m_center}\n"
                f"m candidates : "
                f"{m_candidates.tolist()}\n"
                f"Wavelength range : "
                f"[{self.wavelength[0]}, "
                f"{self.wavelength[-1]}] µm"
            )

        valid_m_candidates = np.asarray(
            valid_m_candidates,
            dtype=int,
        )

        lambda_exact_candidates = np.asarray(
            lambda_exact_candidates,
            dtype=float,
        )

        # ------------------------------------------------------
        # STEP 9
        # Compare exact wavelengths with find_peaks wavelength
        # ------------------------------------------------------

        differences = np.abs(lambda_exact_candidates - lambda_find_peaks)

        # ------------------------------------------------------
        # STEP 10
        # Select nearest exact wavelength
        # ------------------------------------------------------

        selected_index = int(np.argmin(differences))

        selected_m = int(valid_m_candidates[selected_index])

        lambda_exact = float(lambda_exact_candidates[selected_index])

        difference = float(differences[selected_index])

        difference_nm = difference * 1000.0

        # ------------------------------------------------------
        # STEP 11
        # Return complete result
        # ------------------------------------------------------

        return PeakDetectionResult(
            peak_index=peak_index,
            transmission_peak=(transmission_peak),
            lambda_find_peaks=(lambda_find_peaks),
            lambda_exact=(lambda_exact),
            difference=difference,
            difference_nm=difference_nm,
            m_float=m_float,
            m_center=m_center,
            m=selected_m,
            m_candidates=m_candidates,
            valid_m_candidates=(valid_m_candidates),
            lambda_exact_candidates=(lambda_exact_candidates),
            all_peak_indices=(peaks),
            all_peak_wavelengths=(self.wavelength[peaks]),
        )

    # ==========================================================
    # Solve exact lambda for a given m
    # ==========================================================

    def _solve_lambda_for_m(
        self,
        m: int,
    ) -> float | None:
        """
        Solve:

            2 * Delta_neff(lambda) * L
            --------------------------- = m
                    lambda

        inside the available wavelength range.

        Returns:
            float:
                exact wavelength in µm

            None:
                if no solution exists inside the wavelength
                range.
        """

        # ------------------------------------------------------
        # Interpolated Delta neff
        # ------------------------------------------------------

        def delta_neff_interp(
            lambda_value: float,
        ) -> float:

            return float(
                np.interp(
                    lambda_value,
                    self.wavelength,
                    self.delta_neff,
                )
            )

        # ------------------------------------------------------
        # Equation
        # ------------------------------------------------------

        def equation(
            lambda_value: float,
        ) -> float:

            delta = delta_neff_interp(lambda_value)

            return 2.0 * delta * self.length / lambda_value - m

        # ------------------------------------------------------
        # Evaluate equation on simulation grid
        # ------------------------------------------------------

        f_values = np.array(
            [equation(lambda_value) for lambda_value in self.wavelength],
            dtype=float,
        )

        roots = []

        # ------------------------------------------------------
        # Search for sign changes
        # ------------------------------------------------------

        for i in range(len(self.wavelength) - 1):
            f_left = f_values[i]
            f_right = f_values[i + 1]

            if not (np.isfinite(f_left) and np.isfinite(f_right)):
                continue

            # Exact grid point
            if f_left == 0.0:
                roots.append(float(self.wavelength[i]))

                continue

            # Sign change
            if f_left * f_right < 0.0:
                root = brentq(
                    equation,
                    self.wavelength[i],
                    self.wavelength[i + 1],
                )

                roots.append(float(root))

        # ------------------------------------------------------
        # No root inside wavelength range
        # ------------------------------------------------------

        if len(roots) == 0:
            return None

        # ------------------------------------------------------
        # Remove numerical duplicates
        # ------------------------------------------------------

        roots = np.asarray(
            roots,
            dtype=float,
        )

        roots = np.unique(
            np.round(
                roots,
                decimals=14,
            )
        )

        # ------------------------------------------------------
        # If more than one root exists, select the root closest
        # to the selected find_peaks wavelength.
        # ------------------------------------------------------

        closest_root_index = int(
            np.argmin(np.abs(roots - self.wavelength[np.argmax(self.transmission)]))
        )

        return float(roots[closest_root_index])
