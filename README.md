# Heatmap Generator V2

This repository contains the core logic for the multi-layer heatmap generation system used at [RaysAtlas.com](https://raysatlas.com).

## Overview

`generate_heatmap_v2.py` is a high-performance Python script designed to compute signal strengths across a large grid. It utilizes:
- **Numba** for JIT compilation and parallel execution.
- **Memory Mapping** for handling large datasets efficiently.
- **Checkpoint/Resume** functionality for long-running jobs.

The system computes 20 signal layers simultaneously:
- **Top 5** strongest signals at every point.
- **Best signal** for each of the 10 major networks (ABC, NBC, CBS, FOX, PBS, etc.).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/juvii/heatmap-v2.git
   cd heatmap-v2
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

This repository contains the *code* but not the *data*. To run the simulation, you need to provide:

1.  **Tower Data**:
    - Place a CSV file named `towers_with_networks.csv` in the root directory.
    - Required columns: `id`, `lat`, `lon`, `height_m`, `power_kw`, `freq_mhz`, `network_id`, `network_name`.

2.  **Terrain Data (SRTM)**:
    - Create a directory named `srtm_raw/` in the root.
    - Populate it with `.hgt` files covering your area of interest (SRTM1 or SRTM3 format).

## Usage

Run the generator:
```bash
python generate_heatmap_v2.py
```

### Options
- `--restart`: Ignore the checkpoint file and start from the beginning.
- Check `heatmap_v2_config.py` to adjust grid resolution, bounds, and threaded workers.

## Output

The script produces a memory-mapped binary file `heatmap_v2_output/heatmap_multilayer.dat` containing the raw signal data for every grid point, along with a metadata JSON file.

## License

MIT License. See [LICENSE](LICENSE) for details.

## About RaysAtlas

[RaysAtlas.com](https://raysatlas.com) provides interactive signal strength maps, coverage analysis, and tools for cord-cutters and antenna installers.
