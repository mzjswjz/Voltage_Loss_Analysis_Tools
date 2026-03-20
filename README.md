# Voltage Loss Analysis Tools

Tools for analyzing voltage losses in photovoltaic devices. This repository contains scripts for calculating and visualizing various loss mechanisms in solar cells.

## Overview

Voltage loss analysis is essential for understanding the efficiency limits of photovoltaic devices. This toolkit provides methods to:

- Calculate open-circuit voltage (V_OC) losses
- Analyze radiative and non-radiative recombination
- Compare experimental data against theoretical limits (Shockley-Queisser)

## Key Metrics

- **V_OC**: Open-circuit voltage
- **J_SC**: Short-circuit current density
- **FF**: Fill factor
- **ΔV_OC**: Voltage loss relative to radiative limit

## Analysis Methods

1. **Radiative Loss Analysis** - Calculate radiative recombination limits
2. **Non-Radiative Loss Analysis** - Estimate Shockley-Read-Hall recombination
3. **SQ Limit Comparison** - Compare against theoretical maximum efficiency
4. **Loss Spectrum Analysis** - Identify wavelength-dependent losses

## Usage

```python
python main.py --input data.txt --analysis radiative
```

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- SciPy

## Author

mzjswjz
