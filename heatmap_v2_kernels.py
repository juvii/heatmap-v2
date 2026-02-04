"""
Numba-optimized signal computation kernels for multi-layer heatmap generation.
Computes top 5 strongest signals + best per 10 major networks.
(ABC/NBC/CBS/FOX/PBS/CW/MyN/ION/Telemundo/Univision)
"""

import numpy as np
import numba
from numba import jit, prange
import math

# Network IDs (1-10)
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

# Constants
EARTH_RADIUS_KM = 6371.0
EARTH_RADIUS_EFFECTIVE_KM = 6371.0 * (4.0 / 3.0)  # 4/3 Earth for atmospheric refraction
SPEED_OF_LIGHT = 299792458.0  # m/s
MIN_SIGNAL_DBM = -100.0
NOISE_FLOOR_DBM = -110.0


@jit(nopython=True)
def haversine_fast(lat1, lon1, lat2, lon2):
    """Fast Haversine distance calculation in km."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    
    a = (math.sin(dlat / 2.0) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


@jit(nopython=True)
def free_space_path_loss(distance_km, freq_mhz):
    """Calculate free space path loss in dB."""
    if distance_km <= 0.001:
        distance_km = 0.001
    return 32.45 + 20.0 * math.log10(freq_mhz) + 20.0 * math.log10(distance_km)


@jit(nopython=True)
def earth_bulge_clearance(distance_km, h_tx, h_rx):
    """
    Check if line-of-sight is blocked by Earth's curvature.
    Uses 4/3 Earth radius to account for atmospheric refraction.
    Returns True if clear, False if blocked.
    """
    if distance_km < 1.0:
        return True
    
    # Mid-point bulge height (using effective 4/3 Earth radius)
    bulge_height = (distance_km * 1000.0) ** 2 / (8.0 * EARTH_RADIUS_EFFECTIVE_KM * 1000.0)
    
    # Effective height considering both ends
    effective_height = (h_tx + h_rx) / 2.0
    
    return effective_height > bulge_height


@jit(nopython=True)
def terrain_profile_loss(elevations, distance_km, h_tx, h_rx, freq_mhz):
    """
    Terrain loss calculation using ITU-R P.526 knife-edge diffraction.
    Returns additional loss in dB from terrain obstruction.
    
    Uses actual tower frequency for accurate Fresnel zone calculation.
    """
    if len(elevations) < 3:
        return 0.0
    
    # Wavelength from actual frequency
    wavelength = SPEED_OF_LIGHT / (freq_mhz * 1e6)
    
    # Find worst obstruction using Fresnel-Kirchhoff diffraction parameter (nu)
    max_nu = -999.0  # Fresnel diffraction parameter (positive = obstructed)
    num_samples = len(elevations)
    
    for i in range(1, num_samples - 1):
        fraction = i / (num_samples - 1)
        
        # Distance from TX to this point, and from this point to RX
        d1 = distance_km * fraction * 1000.0  # meters
        d2 = distance_km * 1000.0 - d1
        
        if d1 <= 0 or d2 <= 0:
            continue
        
        # Expected LOS height at this point (linear interpolation)
        # Account for Earth curvature using 4/3 Earth radius
        earth_bulge = (d1 * d2) / (2.0 * EARTH_RADIUS_EFFECTIVE_KM * 1000.0)
        expected_height = h_tx + (h_rx - h_tx) * fraction - earth_bulge
        
        # Terrain clearance (positive = clear, negative = obstructed)
        clearance = expected_height - elevations[i]
        
        # First Fresnel zone radius at this point
        fresnel_radius = math.sqrt(wavelength * d1 * d2 / (d1 + d2))
        
        # Fresnel-Kirchhoff diffraction parameter: nu = h * sqrt(2/(lambda * d1 * d2 / (d1+d2)))
        # Simplified: nu = -clearance / (fresnel_radius / sqrt(2))
        if fresnel_radius > 0:
            nu = -clearance * math.sqrt(2.0) / fresnel_radius
            if nu > max_nu:
                max_nu = nu
    
    # Convert nu to diffraction loss using ITU-R P.526 approximation
    if max_nu < -0.78:
        # Clear LOS with good Fresnel clearance
        return 0.0
    elif max_nu < 0:
        # Partial Fresnel zone obstruction (0 to ~6 dB)
        return 6.02 + 9.11 * max_nu - 1.27 * max_nu * max_nu
    elif max_nu < 2.4:
        # Knife-edge diffraction region (6 to ~20 dB)
        return 6.02 + 9.11 * max_nu - 1.27 * max_nu * max_nu
    else:
        # Deep shadow (>20 dB loss)
        loss = 13.0 + 20.0 * math.log10(max_nu)
        return min(loss, 40.0)  # Cap at 40 dB


@jit(nopython=True)
def compute_signal_strength(
    rx_lat, rx_lon, rx_height,
    tx_lat, tx_lon, tx_height, tx_power_kw, freq_mhz,
    terrain_elevations
):
    """
    Compute received signal strength in dBm.
    
    Parameters:
    - rx_lat, rx_lon: Receiver coordinates
    - rx_height: Receiver antenna height (meters)
    - tx_lat, tx_lon: Transmitter coordinates  
    - tx_height: Transmitter antenna height (meters)
    - tx_power_kw: Transmitter power (kW)
    - freq_mhz: Frequency (MHz)
    - terrain_elevations: Array of elevation samples between TX and RX
    
    Returns:
    - Signal strength in dBm (or NOISE_FLOOR_DBM if no signal)
    """
    # Calculate distance
    distance_km = haversine_fast(rx_lat, rx_lon, tx_lat, tx_lon)
    
    # Beyond horizon
    if distance_km > 120.0:
        return NOISE_FLOOR_DBM
    
    # ERP in dBm
    tx_power_dbm = 10.0 * math.log10(tx_power_kw * 1000.0) + 30.0
    
    # Free space path loss
    fspl = free_space_path_loss(distance_km, freq_mhz)
    
    # Earth curvature check
    if not earth_bulge_clearance(distance_km, tx_height, rx_height):
        return NOISE_FLOOR_DBM
    
    # Terrain loss (using actual frequency for Fresnel calculation)
    terrain_loss = terrain_profile_loss(terrain_elevations, distance_km, tx_height, rx_height, freq_mhz)
    
    # Received signal
    rx_dbm = tx_power_dbm - fspl - terrain_loss
    
    if rx_dbm < MIN_SIGNAL_DBM:
        return NOISE_FLOOR_DBM
    
    return rx_dbm


# Constants for signal compression
SIGNAL_OFFSET = 128  # Add to dBm to get stored int8 value
NO_TOWER_ID = 65535  # uint16 max = no tower
NO_SIGNAL = 0        # int8 value for no signal


@jit(nopython=True)
def compress_signal(dbm):
    """Compress dBm to int8: value = int(dBm + 128), clamped to 1-255 (0 = no signal)."""
    val = int(dbm + SIGNAL_OFFSET)
    if val < 1:
        return 1  # Minimum valid signal
    if val > 255:
        return 255
    return val


@jit(nopython=True)
def insert_into_top5(tower_ids, signal_strengths, new_tower_id, new_signal):
    """
    Insert a new tower into top 5 strongest list if it qualifies.
    Arrays are sorted in descending order of signal strength.
    
    Parameters:
    - tower_ids: uint16[5] - Current top 5 tower IDs (65535 = empty)
    - signal_strengths: int8[5] - Current top 5 signal strengths (compressed)
    - new_tower_id: uint16 - New tower ID to potentially insert
    - new_signal: int8 - New signal strength (compressed: dBm + 128)
    
    Modifies arrays in-place.
    """
    # Check if weaker than 5th place
    if new_signal <= signal_strengths[4]:
        return
    
    # Find insertion position
    insert_pos = 4
    for i in range(5):
        if new_signal > signal_strengths[i]:
            insert_pos = i
            break
    
    # Shift lower entries down
    for i in range(4, insert_pos, -1):
        tower_ids[i] = tower_ids[i - 1]
        signal_strengths[i] = signal_strengths[i - 1]
    
    # Insert new entry
    tower_ids[insert_pos] = new_tower_id
    signal_strengths[insert_pos] = new_signal


@jit(nopython=True)
def update_best_network(best_network_towers, best_network_signals, 
                        network_id, tower_id, signal_strength):
    """
    Update best tower for a specific network if this signal is stronger.
    
    Parameters:
    - best_network_towers: uint16[10] - Tower IDs for 10 networks (65535 = none)
    - best_network_signals: int8[10] - Signal strengths for each network (compressed)
    - network_id: Network ID (1=ABC, 2=NBC, 3=CBS, 4=FOX, 5=PBS, 6=CW, 7=MyN, 8=ION, 9=Telemundo, 10=Univision)
    - tower_id: uint16 - Tower ID
    - signal_strength: int8 - Signal (compressed: dBm + 128)
    
    Modifies arrays in-place.
    """
    # Map network_id to array index (1-based to 0-based)
    if network_id < 1 or network_id > 10:
        return
    
    idx = network_id - 1
    
    if signal_strength > best_network_signals[idx]:
        best_network_towers[idx] = tower_id
        best_network_signals[idx] = signal_strength


@jit(nopython=True)
def sample_terrain_elevations(rx_lat, rx_lon, tx_lat, tx_lon, 
                               terrain_lats, terrain_lons, terrain_data,
                               num_samples=500):
    """
    Sample terrain elevations between two points using bilinear interpolation.
    
    Parameters:
    - rx_lat, rx_lon: Receiver coordinates
    - tx_lat, tx_lon: Transmitter coordinates
    - terrain_lats: 1D array of latitude grid
    - terrain_lons: 1D array of longitude grid
    - terrain_data: 2D array of elevation data [lat, lon]
    - num_samples: Number of elevation samples (500 for Tier 4)
    
    Returns:
    - Array of elevation samples
    """
    elevations = np.zeros(num_samples, dtype=np.float32)
    
    for i in range(num_samples):
        fraction = i / (num_samples - 1) if num_samples > 1 else 0.5
        sample_lat = rx_lat + (tx_lat - rx_lat) * fraction
        sample_lon = rx_lon + (tx_lon - rx_lon) * fraction
        
        # Simple nearest-neighbor lookup (faster than bilinear for now)
        lat_idx = np.searchsorted(terrain_lats, sample_lat)
        lon_idx = np.searchsorted(terrain_lons, sample_lon)
        
        # Clamp to bounds
        lat_idx = max(0, min(lat_idx, len(terrain_lats) - 1))
        lon_idx = max(0, min(lon_idx, len(terrain_lons) - 1))
        
        elevations[i] = terrain_data[lat_idx, lon_idx]
    
    return elevations


@jit(nopython=True, parallel=True)
def compute_point_signals_batch(
    rx_lats, rx_lons, rx_height,
    tower_lats, tower_lons, tower_heights, tower_powers, tower_freqs, tower_ids, tower_networks,
    terrain_lats, terrain_lons, terrain_data,
    output_top5_ids, output_top5_signals,
    output_network_ids, output_network_signals
):
    """
    Compute signals for a batch of receiver points.
    
    Parameters:
    - rx_lats, rx_lons: Arrays of receiver coordinates
    - rx_height: Receiver antenna height
    - tower_*: Tower parameter arrays
    - terrain_*: Terrain grid data
    - output_*: Pre-allocated output arrays
    
    Output arrays (modified in-place) - Mixed types for 40-byte record:
    - output_top5_ids: uint16[n_points, 5] - Top 5 tower IDs (65535 = none)
    - output_top5_signals: int8[n_points, 5] - Top 5 signals (compressed: dBm + 128)
    - output_network_ids: uint16[n_points, 10] - Best tower per network (10 networks)
    - output_network_signals: int8[n_points, 10] - Network signals (compressed: dBm + 128)
    """
    n_points = len(rx_lats)
    n_towers = len(tower_lats)
    
    for point_idx in prange(n_points):
        rx_lat = rx_lats[point_idx]
        rx_lon = rx_lons[point_idx]
        
        # Initialize scorecards with "no data" values
        top5_ids = np.full(5, 65535, dtype=np.uint16)      # NO_TOWER_ID
        top5_signals = np.zeros(5, dtype=np.int8)           # NO_SIGNAL (0)
        
        network_ids = np.full(10, 65535, dtype=np.uint16)   # NO_TOWER_ID for 10 networks
        network_signals = np.zeros(10, dtype=np.int8)        # NO_SIGNAL (0)
        
        # Process each tower
        for tower_idx in range(n_towers):
            # Quick distance filter
            distance_km = haversine_fast(rx_lat, rx_lon, 
                                        tower_lats[tower_idx], tower_lons[tower_idx])
            
            if distance_km > 120.0:
                continue
            
            # Sample terrain
            elevations = sample_terrain_elevations(
                rx_lat, rx_lon,
                tower_lats[tower_idx], tower_lons[tower_idx],
                terrain_lats, terrain_lons, terrain_data
            )
            
            # Convert rx_height from AGL to AMSL using terrain at RX location
            # elevations[0] is the ground elevation at the receiver
            rx_height_amsl = elevations[0] + rx_height
            
            # Compute signal (tx height is already AMSL from tower data)
            signal_dbm = compute_signal_strength(
                rx_lat, rx_lon, rx_height_amsl,
                tower_lats[tower_idx], tower_lons[tower_idx], 
                tower_heights[tower_idx], tower_powers[tower_idx], tower_freqs[tower_idx],
                elevations
            )
            
            if signal_dbm <= NOISE_FLOOR_DBM:
                continue
            
            # Compress signal to int8: value = dBm + 128, clamped to 1-255
            signal_compressed = compress_signal(signal_dbm)
            tower_id = tower_ids[tower_idx]
            network_id = tower_networks[tower_idx]
            
            # Update top 5 strongest
            insert_into_top5(top5_ids, top5_signals, tower_id, signal_compressed)
            
            # Update best per network
            update_best_network(network_ids, network_signals, network_id, tower_id, signal_compressed)
        
        # Write results
        output_top5_ids[point_idx] = top5_ids
        output_top5_signals[point_idx] = top5_signals
        output_network_ids[point_idx] = network_ids
        output_network_signals[point_idx] = network_signals
