# Voltage Loss Analysis Tools

A collection of Python tools for analyzing photovoltaic device EQE data.

## Files

| File | Description |
|------|-------------|
| `main.py` | Entry point with interactive prompts for running analysis workflows |
| `AM15G.txt` | AM1.5G solar spectrum data file |
| `Functions/Calculate_Jsc.py` | `Jsc` class — calculates short-circuit current density from EQE and AM1.5G spectrum |
| `Functions/Find_inflection_point.py` | `Inflection_Points` class — finds inflection points in EQE data using second derivative analysis |
| `Functions/Plot_Urbach_app_E.py` | `Urbach_E` class — plots Urbach energy and calculates Urbach energy at the band edge |
| `Functions/__init__.py` | Module initializer |

## What It Does

- **Calculate Jsc**: Integrates EQE data with AM1.5G solar spectrum to compute short-circuit current density (mA/cm²)
- **Find Inflection Points**: Uses Savitzky-Golay smoothing and second derivative analysis to locate band gap/CT state energies
- **Urbach Energy Analysis**: Fits exponential tail region of EQE to extract Urbach energy parameter

## Usage

Uncomment the relevant block in `main.py` and run:

```bash
python main.py
```

The script will prompt for file paths and analysis parameters interactively.

---
*Author: mzjswjz*
