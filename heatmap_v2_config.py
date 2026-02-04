"""
Configuration for multi-layer heatmap generation (v2).
Outputs 20 signal layers: top 5 strongest + best per 10 major networks.
"""
import os
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "srtm_cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "heatmap_v2_output")
TILES_DIR = os.path.join(BASE_DIR, "public", "tiles")
TOWERS_CSV = os.path.join(BASE_DIR, "towers_with_networks.csv")

# Ensure directories exist
for directory in [OUTPUT_DIR, TILES_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Physics Constants
MAX_DISTANCE_KM = 120.0
EARTH_RADIUS_KM = 6371.0

# Network IDs (matching network_affiliates_final.csv)
NETWORK_ABC = 1
NETWORK_NBC = 2
NETWORK_CBS = 3
NETWORK_FOX = 4
NETWORK_PBS = 5
NETWORK_CW = 6
NETWORK_MYN = 7
NETWORK_ION = 8
NETWORK_TELEMUNDO = 9
NETWORK_UNIVISION = 10

# Network names for layer output
NETWORK_NAMES = {
    0: "Other",
    NETWORK_ABC: "ABC",
    NETWORK_NBC: "NBC", 
    NETWORK_CBS: "CBS",
    NETWORK_FOX: "FOX",
    NETWORK_PBS: "PBS",
    NETWORK_CW: "CW",
    NETWORK_MYN: "MyN",
    NETWORK_ION: "ION",
    NETWORK_TELEMUNDO: "Telemundo",
    NETWORK_UNIVISION: "Univision"
}

# Major networks to track (in fixed order for output bytes)
MAJOR_NETWORKS = [NETWORK_ABC, NETWORK_NBC, NETWORK_CBS, NETWORK_FOX, NETWORK_PBS,
                  NETWORK_CW, NETWORK_MYN, NETWORK_ION, NETWORK_TELEMUNDO, NETWORK_UNIVISION]

# Grid Configuration - Tier 4 quality
GRID_RESOLUTION_M = 160  # meters between grid points

# CONUS bounds
CONUS_BOUNDS = {
    'lat_min': 24.0,   # Southern tip of Florida
    'lat_max': 50.0,   # Canadian border
    'lon_min': -125.0, # West coast
    'lon_max': -66.0   # East coast
}

# Compute grid dimensions
def meters_to_degrees_lat(meters):
    """Convert meters to degrees latitude (constant)."""
    return meters / 111320.0

def meters_to_degrees_lon(meters, latitude):
    """Convert meters to degrees longitude (varies by latitude)."""
    import math
    return meters / (111320.0 * math.cos(math.radians(latitude)))

# Calculate grid size
lat_range = CONUS_BOUNDS['lat_max'] - CONUS_BOUNDS['lat_min']
lon_range = CONUS_BOUNDS['lon_max'] - CONUS_BOUNDS['lon_min']

# Use mid-latitude for longitude calculation
mid_lat = (CONUS_BOUNDS['lat_min'] + CONUS_BOUNDS['lat_max']) / 2.0

# Grid steps in degrees
LAT_STEP = meters_to_degrees_lat(GRID_RESOLUTION_M)
LON_STEP = meters_to_degrees_lon(GRID_RESOLUTION_M, mid_lat)

# Grid dimensions
N_LATS = int(np.ceil(lat_range / LAT_STEP))
N_LONS = int(np.ceil(lon_range / LON_STEP))
TOTAL_POINTS = N_LATS * N_LONS

# Output data structure (40 bytes per point) - Mixed Types
# Bytes 0-9:   uint16[5]  - Tower IDs of top 5 strongest (65535 = no tower)
# Bytes 10-14: int8[5]    - Top 5 signal strengths, compressed: int(dBm + 128), 0 = no signal
# Bytes 15-34: uint16[10] - Best tower ID per network (ABC/NBC/CBS/FOX/PBS/CW/MyN/ION/Telemundo/Univision)
# Bytes 35-44: int8[10]   - Network signal strengths, compressed: int(dBm + 128), 0 = no signal
#
# Signal compression: dBm ranges -128 to +127 after offset
#   -100 dBm -> 28,  -50 dBm -> 78,  0 dBm -> 128,  +20 dBm -> 148
#   To decode: dBm = value - 128
BYTES_PER_POINT = 40
SIGNAL_OFFSET = 128  # Add to dBm to get stored value
NO_TOWER_ID = 65535  # uint16 max = no tower
NO_SIGNAL = 0        # int8 value for no signal (represents -128 dBm)

# Output layer structure
NUM_TOP_SIGNALS = 5
NUM_MAJOR_NETWORKS = 10
TOTAL_LAYERS = NUM_TOP_SIGNALS * 2 + NUM_MAJOR_NETWORKS * 2  # 5 IDs + 5 signals + 10 net IDs + 10 net signals

# Memory-mapped output file
OUTPUT_MMAP_FILE = os.path.join(OUTPUT_DIR, "heatmap_multilayer.dat")
OUTPUT_METADATA_FILE = os.path.join(OUTPUT_DIR, "heatmap_multilayer.json")

# Multiprocessing
THREADS = 32  # Use 32 of 32 available threads
CHUNK_SIZE = 10000  # Points per chunk (increased from 1000 for better throughput)

# Signal strength thresholds (dBm)
MIN_SIGNAL_DBM = -100.0  # Minimum detectable signal
NOISE_FLOOR_DBM = -110.0  # Below this is pure noise

# GeoTIFF output configuration
GEOTIFF_OUTPUT = os.path.join(OUTPUT_DIR, "heatmap_multilayer.tif")
GEOTIFF_COMPRESSION = "DEFLATE"
GEOTIFF_TILED = True
GEOTIFF_TILE_SIZE = 512

# Tile generation
TILE_MIN_ZOOM = 4
TILE_MAX_ZOOM = 10
TILE_FORMAT = "png"
