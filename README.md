# The Conjecture of Evolutionary Direction: A Multi-scale Explanatory Framework of Fractals and Entropy
# 生物演化的方向的猜想——分形与熵的跨尺度解释框架

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxx)

## Overview (项目概述)

This repository contains the simulation code, data analysis scripts, and supplementary materials for the paper **"The Conjecture of Evolutionary Direction: A Multi-scale Explanatory Framework of Fractals and Entropy"**.

We propose that under specific boundary conditions (continuous energy input, entropy pressure), biological and ecological systems exhibit a probabilistic tendency towards increasing complexity, characterized by fractal geometry and information processing capacity.

## Key Predictions Verified (核心验证)

1.  **H1 (Landscape Scale):** A significant power-law relationship between Net Primary Productivity (NPP) and Landscape Fractal Dimension ($D_L$), with a scaling exponent $\gamma \approx 0.23$.
2.  **H2 (Evolutionary Time):** Morphological Fractal Dimension ($D_M$) increases with a lag of ~1.2 Myr following energy surges (e.g., oxygenation events).
3.  **H3 (Simulation):** An Agent-Based Model (ABM) confirms a critical line $E_c(\lambda) \propto \lambda^{0.5}$ separating regions of complexity growth from stagnation.

## Structure (代码结构)

* `src/simulation_H3.py`: Agent-Based Model verifying the critical line $E_c(\lambda)$ under energy and disturbance constraints.
* `src/analysis_H1_H2.py`: Statistical models (LME and Panel Regression) for empirical verification.
* `src/utils_fractal.py`: Algorithm for Box-counting fractal dimension calculation.

## Usage (使用方法)

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
