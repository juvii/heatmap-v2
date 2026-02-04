# Heatmap Generator V2

This repository contains the core logic for the multi-layer heatmap generation system used at [RaysAtlas.com](https://raysatlas.com).

## Overview

`generate_heatmap_v2.py` is a high-performance Python script designed to compute signal strengths across a large grid. It utilizes:
- **Numba** for JIT compilation and parallel execution.
- **Memory Mapping** for handling large datasets efficiently.
- **Checkpoint/Resume** functionality for long-running jobs.

## Usage

This script is part of a larger processing pipeline. It requires specific configuration files (`heatmap_v2_config.py`), kernel definitions (`heatmap_v2_kernels.py`), and utility modules (`utils.py`) to execute.

## About RaysAtlas

[RaysAtlas.com](https://raysatlas.com) provides interactive signal strength maps and tools.
