#!/usr/bin/env python3
"""
Multi-layer heatmap generator (v2) - Numba accelerated with checkpoint/resume.
Generates 20 signal layers: top 5 strongest + best per 10 major networks.

Output: Memory-mapped binary file with 45 bytes per grid point (mixed types):
  - Bytes 0-9:   uint16[5]  - Tower IDs of top 5 strongest (65535 = none)
  - Bytes 10-14: int8[5]    - Top 5 signal strengths (compressed: dBm + 128)
  - Bytes 15-34: uint16[10] - Best tower IDs per network (ABC/NBC/CBS/FOX/PBS/CW/MyN/ION/Telemundo/Univision)
  - Bytes 35-44: int8[10]   - Network signal strengths (compressed: dBm + 128)

Checkpoint/Resume:
  - Progress saved to checkpoint file every N chunks
  - On restart, resumes from last checkpoint
  - Use --restart flag to force fresh start

Threading:
  - Uses ThreadPoolExecutor (threads share memory naturally)
  - Numba releases GIL so threads run in parallel
  - No shared_memory complexity needed
"""

import os
import sys
import time
import json
import signal
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import KDTree

# Local imports
import heatmap_v2_config as config
from heatmap_v2_kernels import compute_point_signals_batch
from utils import preload_terrain_to_ram, get_elevation_from_ram

# Checkpoint configuration
CHECKPOINT_FILE = os.path.join(config.OUTPUT_DIR, "checkpoint.json")
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N chunks

# Global flag for graceful shutdown
shutdown_requested = False

# Module-level data (shared by all threads naturally)
_terrain_data = None
_tower_data = None


def load_tower_data():
    """Load tower data with network affiliations.
    
    Note: With multicast support, a physical tower may appear multiple times
    if it carries multiple networks. This is intentional - each row represents
    a tower-network pair that will update the corresponding network layer.
    """
    print("Loading tower data...")
    towers = pd.read_csv(config.TOWERS_CSV)
    
    unique_towers = towers['id'].nunique()
    print(f"  Loaded {len(towers)} tower-network entries ({unique_towers} unique towers)")
    
    # Show multicast stats
    multicast = towers.groupby('id').filter(lambda x: len(x) > 1)
    if len(multicast) > 0:
        print(f"  Multicast: {multicast['id'].nunique()} towers carry multiple networks")
    
    print(f"  Network distribution:")
    network_dist = towers.groupby('network_name').size().sort_values(ascending=False)
    for network, count in network_dist.items():
        print(f"    {network}: {count}")
    
    return towers


def build_spatial_index(towers):
    """Build KD-tree for fast spatial queries."""
    print("Building spatial index...")
    coords = np.column_stack([towers['lat'].values, towers['lon'].values])
    kdtree = KDTree(coords)
    print("  Spatial index built")
    return kdtree


TERRAIN_CACHE_FILE = os.path.join(config.OUTPUT_DIR, "terrain_grid_cache.npz")

def create_terrain_grid():
    """
    Create a coarse terrain grid for fast lookups.
    Caches to disk after first build - subsequent runs are instant.
    """
    print("Creating terrain grid...")
    
    # Use terrain_lats, we can get min/max from config
    lat_min = config.CONUS_BOUNDS['lat_min']
    lat_max = config.CONUS_BOUNDS['lat_max']
    lon_min = config.CONUS_BOUNDS['lon_min']
    lon_max = config.CONUS_BOUNDS['lon_max']
    
    # Tier 4: 0.00075° (~75m) terrain grid - max that fits in 32GB with 17GB SRTM
    terrain_res_deg = 0.000833
    
    terrain_lats = np.arange(lat_min, lat_max, terrain_res_deg)
    terrain_lons = np.arange(lon_min, lon_max, terrain_res_deg)
    
    print(f"  Terrain grid: {len(terrain_lats)} × {len(terrain_lons)} = {len(terrain_lats) * len(terrain_lons):,} points")
    
    # Check for cached terrain grid
    if os.path.exists(TERRAIN_CACHE_FILE):
        print(f"  Loading cached terrain grid from {TERRAIN_CACHE_FILE}...")
        cached = np.load(TERRAIN_CACHE_FILE)
        cached_lats = cached['lats']
        cached_lons = cached['lons']
        
        # Verify dimensions match
        if len(cached_lats) == len(terrain_lats) and len(cached_lons) == len(terrain_lons):
            print(f"  Loaded cached terrain grid ({cached['data'].nbytes / 1e9:.2f} GB)")
            return cached_lats.astype(np.float64), cached_lons.astype(np.float64), cached['data']
        else:
            print(f"  Cache dimensions mismatch, rebuilding...")
    
    # Build terrain grid (this takes ~1 hour first time)
    print(f"  Building terrain grid (first run only, will cache to disk)...")
    terrain_data = np.zeros((len(terrain_lats), len(terrain_lons)), dtype=np.float32)
    
    n_rows = len(terrain_lats)
    last_pct = 0
    
    for i, lat in enumerate(terrain_lats):
        for j, lon in enumerate(terrain_lons):
            terrain_data[i, j] = get_elevation_from_ram(lat, lon)
        
        pct = int((i + 1) / n_rows * 100)
        if pct > last_pct and pct % 5 == 0:
            print(f"    {pct}% complete ({i+1:,}/{n_rows:,} rows)...")
            last_pct = pct
    
    # Save to cache
    print(f"  Saving terrain grid to cache...")
    np.savez_compressed(TERRAIN_CACHE_FILE, 
                        lats=terrain_lats, 
                        lons=terrain_lons, 
                        data=terrain_data)
    print(f"  Cached to {TERRAIN_CACHE_FILE}")
    
    print("  Terrain grid created")
    return terrain_lats, terrain_lons, terrain_data


def generate_grid_points():
    """Generate the output grid coordinates."""
    print("Generating output grid...")
    
    lat_min = config.CONUS_BOUNDS['lat_min']
    lat_max = config.CONUS_BOUNDS['lat_max']
    lon_min = config.CONUS_BOUNDS['lon_min']
    lon_max = config.CONUS_BOUNDS['lon_max']
    
    lats = np.arange(lat_min, lat_max, config.LAT_STEP)
    lons = np.arange(lon_min, lon_max, config.LON_STEP)
    
    print(f"  Grid dimensions: {len(lats)} × {len(lons)} = {len(lats) * len(lons):,} points")
    print(f"  Resolution: ~{config.GRID_RESOLUTION_M}m")
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Flatten for processing
    grid_lats = lat_grid.flatten()
    grid_lons = lon_grid.flatten()
    
    return lats, lons, grid_lats, grid_lons


def process_chunk(args):
    """
    Process a chunk of grid points.
    Uses module-level data (shared by all threads).
    
    Returns:
    - chunk_id, top5_ids, top5_signals, network_ids, network_signals
    """
    global _terrain_data, _tower_data
    
    chunk_id, chunk_lats, chunk_lons, rx_height = args
    
    # Use module-level arrays (threads share memory naturally)
    tower_lats, tower_lons, tower_heights, tower_powers, tower_freqs, tower_ids, tower_networks = _tower_data
    terrain_lats, terrain_lons, terrain_grid = _terrain_data
    
    n_points = len(chunk_lats)
    
    # Allocate output arrays with new types (10 networks)
    output_top5_ids = np.full((n_points, 5), 65535, dtype=np.uint16)       # NO_TOWER_ID
    output_top5_signals = np.zeros((n_points, 5), dtype=np.int8)            # NO_SIGNAL
    output_network_ids = np.full((n_points, 10), 65535, dtype=np.uint16)   # NO_TOWER_ID for 10 networks
    output_network_signals = np.zeros((n_points, 10), dtype=np.int8)        # NO_SIGNAL
    
    # Call Numba kernel
    compute_point_signals_batch(
        chunk_lats, chunk_lons, rx_height,
        tower_lats, tower_lons, tower_heights, tower_powers, tower_freqs, tower_ids, tower_networks,
        terrain_lats, terrain_lons, terrain_grid,
        output_top5_ids, output_top5_signals,
        output_network_ids, output_network_signals
    )
    
    return chunk_id, output_top5_ids, output_top5_signals, output_network_ids, output_network_signals


def initialize_output_file(n_points, resume=False):
    """Create or open memory-mapped output file with mixed-type structured array."""
    print("Initializing output file...")
    
    # Define structured dtype for 40-byte record (10 networks)
    record_dtype = np.dtype([
        ('top5_ids', np.uint16, 5),
        ('top5_signals', np.int8, 5),
        ('network_ids', np.uint16, 10),
        ('network_signals', np.int8, 10)
    ])
    
    # Verify 45 bytes (5*2 + 5 + 10*2 + 10 = 45)
    assert record_dtype.itemsize == 45, f"Expected 45 bytes, got {record_dtype.itemsize}"
    
    mmap_file = config.OUTPUT_MMAP_FILE
    
    if resume and os.path.exists(mmap_file):
        # Open existing file for resume
        print(f"  Resuming with existing file: {mmap_file}")
        output_array = np.memmap(mmap_file, dtype=record_dtype, mode='r+', shape=(n_points,))
    else:
        # Create new file
        if os.path.exists(mmap_file):
            print(f"  Warning: Removing existing output file {mmap_file}")
            os.remove(mmap_file)
        
        output_array = np.memmap(mmap_file, dtype=record_dtype, mode='w+', shape=(n_points,))
        
        # Initialize with "no data" values
        output_array['top5_ids'][:] = 65535       # NO_TOWER_ID
        output_array['top5_signals'][:] = 0       # NO_SIGNAL
        output_array['network_ids'][:] = 65535    # NO_TOWER_ID
        output_array['network_signals'][:] = 0    # NO_SIGNAL
        output_array.flush()
        print(f"  Created new file: {mmap_file}")
    
    file_size_mb = os.path.getsize(mmap_file) / 1024 / 1024
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Record size: {record_dtype.itemsize} bytes")
    
    return output_array


def load_checkpoint():
    """Load checkpoint file if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
            print(f"  Found checkpoint: {len(checkpoint['completed_chunks'])} chunks completed")
            return checkpoint
        except Exception as e:
            print(f"  Warning: Failed to load checkpoint: {e}")
    return None


def save_checkpoint(completed_chunks, start_time, total_chunks):
    """Save checkpoint to disk."""
    checkpoint = {
        'completed_chunks': list(completed_chunks),
        'total_chunks': total_chunks,
        'last_save_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': time.time() - start_time
    }
    
    # Write atomically (write to temp, then rename)
    temp_file = CHECKPOINT_FILE + '.tmp'
    with open(temp_file, 'w') as f:
        json.dump(checkpoint, f)
    os.replace(temp_file, CHECKPOINT_FILE)


def clear_checkpoint():
    """Remove checkpoint file after successful completion."""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("  Checkpoint file removed")


def signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global shutdown_requested
    if shutdown_requested:
        print("\n  Force quit requested, exiting immediately...")
        sys.exit(1)
    print("\n  Shutdown requested, finishing current chunks...")
    print("  (Press Ctrl+C again to force quit)")
    shutdown_requested = True


def save_metadata(lats, lons, towers):
    """Save metadata JSON for downstream processing."""
    metadata = {
        'version': 2,
        'schema': {
            'record_size_bytes': 40,
            'top5_ids': {'offset': 0, 'dtype': 'uint16', 'shape': 5, 'no_data': 65535},
            'top5_signals': {'offset': 10, 'dtype': 'int8', 'shape': 5, 'no_data': 0, 'decode': 'value - 128'},
            'network_ids': {'offset': 15, 'dtype': 'uint16', 'shape': 10, 'no_data': 65535},
            'network_signals': {'offset': 35, 'dtype': 'int8', 'shape': 10, 'no_data': 0, 'decode': 'value - 128'}
        },
        'networks': {
            1: 'ABC', 2: 'NBC', 3: 'CBS', 4: 'FOX', 5: 'PBS',
            6: 'CW', 7: 'MyN', 8: 'ION', 9: 'Telemundo', 10: 'Univision'
        },
        'grid': {
            'n_lats': len(lats),
            'n_lons': len(lons),
            'lat_min': float(lats[0]),
            'lat_max': float(lats[-1]),
            'lon_min': float(lons[0]),
            'lon_max': float(lons[-1]),
            'lat_step': float(config.LAT_STEP),
            'lon_step': float(config.LON_STEP),
            'resolution_m': config.GRID_RESOLUTION_M,
            'total_points': len(lats) * len(lons)
        },
        'towers': {
            'total': len(towers),
            'by_network': towers.groupby('network_name').size().to_dict()
        },
        'networks': {
            'order': ['ABC', 'NBC', 'CBS', 'FOX', 'PBS'],
            'ids': config.MAJOR_NETWORKS
        },
        'signal_encoding': {
            'compression': 'dBm + 128',
            'min_dbm': -127,
            'max_dbm': 127,
            'no_signal_value': 0
        }
    }
    
    with open(config.OUTPUT_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {config.OUTPUT_METADATA_FILE}")


def main():
    """Main processing pipeline with checkpoint/resume support."""
    global shutdown_requested
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate multi-layer TV signal heatmap')
    parser.add_argument('--restart', action='store_true', 
                        help='Force fresh start, ignore existing checkpoint')
    args = parser.parse_args()
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    start_time = time.time()
    
    print("=" * 60)
    print("Multi-Layer TV Signal Heatmap Generator v2")
    print("  with Checkpoint/Resume Support")
    print("=" * 60)
    
    # Check for existing checkpoint
    checkpoint = None
    completed_chunks = set()
    
    if not args.restart:
        checkpoint = load_checkpoint()
        if checkpoint:
            completed_chunks = set(checkpoint['completed_chunks'])
            print(f"  Resuming from checkpoint with {len(completed_chunks)} chunks already done")
    else:
        print("  Fresh start requested, ignoring any existing checkpoint")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
    
    # Step 1: Load terrain into RAM
    preload_terrain_to_ram()
    
    # Step 2: Load towers
    towers = load_tower_data()
    kdtree = build_spatial_index(towers)
    
    # Step 3: Create terrain grid for fast lookups
    terrain_lats, terrain_lons, terrain_grid = create_terrain_grid()
    
    # Step 4: Generate output grid
    lats, lons, grid_lats, grid_lons = generate_grid_points()
    n_points = len(grid_lats)
    
    # Step 5: Initialize output file (resume mode if checkpoint exists)
    resume_mode = len(completed_chunks) > 0
    output_array = initialize_output_file(n_points, resume=resume_mode)
    
    # Step 6: Set up module-level data (threads share memory naturally)
    global _terrain_data, _tower_data
    print("Preparing data arrays for threads...")
    
    # Tower arrays
    tower_lats_arr = towers['lat'].values.astype(np.float64)
    tower_lons_arr = towers['lon'].values.astype(np.float64)
    tower_heights_arr = towers['height_m'].values.astype(np.float32)
    tower_powers_arr = towers['power_kw'].values.astype(np.float32)
    tower_freqs_arr = towers['freq_mhz'].values.astype(np.float32)
    tower_ids_arr = towers['id'].values.astype(np.uint16)
    tower_networks_arr = towers['network_id'].values.astype(np.int16)
    
    # Store in module-level globals (threads share these)
    _terrain_data = (terrain_lats, terrain_lons, terrain_grid)
    _tower_data = (tower_lats_arr, tower_lons_arr, tower_heights_arr, 
                   tower_powers_arr, tower_freqs_arr, tower_ids_arr, tower_networks_arr)
    
    terrain_mem_mb = (terrain_lats.nbytes + terrain_lons.nbytes + terrain_grid.nbytes) / 1024 / 1024
    tower_mem_mb = sum(arr.nbytes for arr in _tower_data) / 1024 / 1024
    print(f"  Terrain data: {terrain_mem_mb:.1f} MB")
    print(f"  Tower data: {tower_mem_mb:.1f} MB")
    
    # Step 7: Build chunk list
    rx_height = 10.0
    rx_height = 10.0
    all_chunks = []
    for i in range(0, n_points, config.CHUNK_SIZE):
        chunk_id = i // config.CHUNK_SIZE
        end_idx = min(i + config.CHUNK_SIZE, n_points)
        chunk_lats = grid_lats[i:end_idx]
        chunk_lons = grid_lons[i:end_idx]
        all_chunks.append((chunk_id, chunk_lats, chunk_lons, rx_height))
    
    total_chunks = len(all_chunks)
    
    # Filter out already-completed chunks
    pending_chunks = [(cid, clats, clons, rxh) for cid, clats, clons, rxh in all_chunks 
                      if cid not in completed_chunks]
    
    print(f"\nProcessing {n_points:,} points in {total_chunks} chunks...")
    print(f"  Chunk size: {config.CHUNK_SIZE}")
    print(f"  Already completed: {len(completed_chunks)}")
    print(f"  Pending: {len(pending_chunks)}")
    print(f"  Using {config.THREADS} threads (Numba releases GIL)")
    print(f"  Checkpoint interval: every {CHECKPOINT_INTERVAL} chunks")
    
    if len(pending_chunks) == 0:
        print("\n  All chunks already completed!")
    else:
        # Step 8: Process pending chunks using ThreadPoolExecutor
        # Threads share memory naturally - no copying!
        # Numba releases the GIL so threads run in parallel
        completed_this_run = 0
        last_checkpoint_count = len(completed_chunks)
        last_pct = int(len(completed_chunks) / total_chunks * 100) if total_chunks > 0 else 0
        
        with ThreadPoolExecutor(max_workers=config.THREADS) as executor:
            # Submit all pending chunks
            futures = {executor.submit(process_chunk, chunk): chunk[0] 
                       for chunk in pending_chunks}
            
            try:
                for future in as_completed(futures):
                    if shutdown_requested:
                        print("\n  Shutdown requested, cancelling...")
                        for f in futures:
                            f.cancel()
                        break
                    
                    chunk_id, top5_ids, top5_signals, network_ids, network_signals = future.result()
                    
                    # Write to memory-mapped file
                    start_idx = chunk_id * config.CHUNK_SIZE
                    end_idx = start_idx + len(top5_ids)
                    
                    output_array['top5_ids'][start_idx:end_idx] = top5_ids
                    output_array['top5_signals'][start_idx:end_idx] = top5_signals
                    output_array['network_ids'][start_idx:end_idx] = network_ids
                    output_array['network_signals'][start_idx:end_idx] = network_signals
                    
                    # Track completion
                    completed_chunks.add(chunk_id)
                    completed_this_run += 1
                    
                    # Progress reporting
                    pct = int(len(completed_chunks) / total_chunks * 100)
                    
                    if pct > last_pct:
                        elapsed = time.time() - start_time
                        rate = completed_this_run / elapsed if elapsed > 0 else 0
                        remaining_chunks = total_chunks - len(completed_chunks)
                        remaining_time = remaining_chunks / rate if rate > 0 else 0
                        
                        print(f"  Progress: {pct}% ({len(completed_chunks)}/{total_chunks} chunks) "
                              f"[{elapsed/60:.1f}m elapsed, {remaining_time/60:.1f}m remaining]")
                        last_pct = pct
                    
                    # Periodic checkpoint save
                    if len(completed_chunks) - last_checkpoint_count >= CHECKPOINT_INTERVAL:
                        output_array.flush()
                        save_checkpoint(completed_chunks, start_time, total_chunks)
                        last_checkpoint_count = len(completed_chunks)
                        
            except KeyboardInterrupt:
                print("\n  Interrupted! Saving checkpoint...")
            
            except Exception as e:
                print(f"\n  Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Save final checkpoint
        output_array.flush()
        if len(completed_chunks) < total_chunks:
            save_checkpoint(completed_chunks, start_time, total_chunks)
            print(f"\n  Checkpoint saved: {len(completed_chunks)}/{total_chunks} chunks complete")
            print(f"  Run again to resume from checkpoint")
        else:
            # All done, remove checkpoint
            clear_checkpoint()
    
    # Step 9: Finalize
    print("\nFinalizing output...")
    output_array.flush()
    del output_array
    
    # Step 10: Save metadata (only if complete)
    if len(completed_chunks) == total_chunks:
        save_metadata(lats, lons, towers)
        
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Processing COMPLETE in {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
        print(f"Output: {config.OUTPUT_MMAP_FILE}")
        print(f"Size: {os.path.getsize(config.OUTPUT_MMAP_FILE) / 1024 / 1024:.1f} MB")
        print("=" * 60)
    else:
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Processing PAUSED after {elapsed/60:.1f} minutes")
        print(f"  Completed: {len(completed_chunks)}/{total_chunks} chunks ({len(completed_chunks)/total_chunks*100:.1f}%)")
        print(f"  Run 'python generate_heatmap_v2.py' to resume")
        print("=" * 60)


if __name__ == '__main__':
    main()
