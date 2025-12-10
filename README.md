# The Conjecture of Evolutionary Direction: A Multi-scale Explanatory Framework of Fractals and Entropy

## Overview

This repository contains the theoretical framework, simulation code, and suggested validation tools for the paper **"The Conjecture of Evolutionary Direction: A Multi-scale Explanatory Framework of Fractals and Entropy"**.

We propose that under specific boundary conditions (continuous energy input, entropy pressure), biological and ecological systems exhibit a probabilistic tendency towards increasing complexity, characterized by fractal geometry and information processing capacity. This framework is built on theoretical predictions, with one simulation-based verification (H3) and two testable empirical predictions (H1, H2).

## Theoretical Predictions 

1.  **H1 (Landscape Scale):** Predicts a power-law relationship between energy flux (proxied by Net Primary Productivity, NPP) and Landscape Fractal Dimension (\(D_L\)), with a theoretical scaling exponent \(\gamma \approx 1/4\).
2.  **H2 (Evolutionary Time):** Predicts that Morphological Fractal Dimension (\(D_M\)) increases with a lag of \(0.5–2.0 \, \text{Myr}\) following energy surges (e.g., oxygenation events) in geological records.
3.  **H3 (Simulation):** An Agent-Based Model (ABM) demonstrates a critical line \(E_c(\lambda) \propto \lambda^{0.5}\) separating regions of complexity growth from stagnation under energy and disturbance constraints.

## Structure 

* `src/simulation_H3.py`: Agent-Based Model verifying the critical line \(E_c(\lambda)\) under energy and disturbance constraints. **This is the main simulation result presented in the paper.**
* `tools/analysis_H1_H2.py`: Suggested statistical models (LME and Panel Regression) for future empirical testing of H1 and H2.
* `tools/utils_fractal.py`: Algorithm for Box-counting fractal dimension calculation, provided as a tool for future empirical studies.

## Usage 

### Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Simulation (H3)

To replicate the simulation results (H3), run:

```bash
python src/simulation_H3.py
```

This will generate the critical energy-disturbance relationship and output the results in `results/H3/`.

### Using the Suggested Validation Tools (H1 & H2)

The scripts in `tools/` are provided as methodological references for future empirical studies. They are not intended as validated results but as starting points for testing the theoretical predictions.

Example usage:

```bash
python tools/analysis_H1_H2.py --help
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use this code or the theoretical framework in your research, please cite our paper:

> Author(s). (Year). "The Conjecture of Evolutionary Direction: A Multi-scale Explanatory Framework of Fractals and Entropy". *Journal Name*, Volume(Issue), Pages. DOI: xxxxxx

---

**Note:** The H1 and H2 predictions are theoretical and require future empirical validation. The simulation (H3) is the primary computational contribution of this work.