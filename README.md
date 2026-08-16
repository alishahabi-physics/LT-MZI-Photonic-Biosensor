# LT-MZI Photonic Biosensor

## Overview

Python-based analysis pipeline for a Loop-Terminated Mach-Zehnder Interferometer (LT-MZI) photonic biosensor.

The project processes wavelength-dependent effective-index data extracted from Lumerical MODE and evaluates gas-sensing performance from the resulting interference spectrum.

## Research Focus

- RI = 1.000–1.009
- RI step = 0.001
- Wavelength range = 1.50–1.65 µm
- Device length = 50 µm
- TE-mode data from Lumerical MODE

## Analysis Pipeline

```text
Lumerical MODE .mat files
        ↓
ModeDataLoader
        ↓
Δneff
        ↓
Transmission
        ↓
Peak Detection
        ↓
Exact Resonance
        ↓
Fringe Order
        ↓
FWHM / FSR
        ↓
Sensitivity
        ↓
FOM
        ↓
CSV Results + Plots
```

## Sensitivity

The canonical local sensitivity is calculated using the group index:

\[
S_\lambda = \frac{\lambda_{res}}{\Delta n_g}\n\frac{\partial \Delta n_{eff}}{\partial n_a}
\]

where:

\[
\Delta n_g = n_{g,reference} - n_{g,sensor}
\]

Central finite-difference sensitivity is retained for independent numerical validation.

## Current Gas Results

Stable range:

```text
RI = 1.002–1.009
```

Linear resonance sensitivity:

```text
3221.13 nm/RIU
```

R²:

```text
0.99945137
```

RI = 1.001 is treated as a boundary point of the available spectral window. Its adjacent fringe orders are outside the simulated wavelength range, so FSR is reported as NaN rather than extrapolated.

## Repository Structure

```text
LT-MZI-Photonic-Biosensor/
├── data/
│   └── base/
├── src/
│   ├── analysis/
│   ├── calculations/
│   ├── config/
│   ├── core/
│   ├── io/
│   ├── models/
│   ├── tests/
│   └── visualization/
├── pytest.ini
├── requirements.txt
├── LICENSE
└── README.md
```

## Validation

Run the test suite:

```powershell
python -m pytest -q
```

Compile the source tree:

```powershell
python -m compileall src
```

Run the canonical gas analysis:

```powershell
python -m src.analysis.gas_final_analysis
```

## Requirements

See `requirements.txt`.

## License

See `LICENSE`.
