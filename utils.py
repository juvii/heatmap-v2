import math
import numpy as np
import pandas as pd
import struct
import os
import config

def get_bearing(lat1, lon1, lat2, lon2):
    """Calculates angle (0-360) from User to Tower."""
    dLon = (lon2 - lon1)
    y = math.sin(math.radians(dLon)) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
        math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def haversine(lat1, lon1, lat2, lon2):
    """Calculates Haversine distance in km."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return config.EARTH_RADIUS_KM * c

def haversine_vectorized(lat1, lon1, lat2_array, lon2_array):
    """Fast distance calculation for filtering (Vectorized)."""
    dlat = np.radians(lat2_array - lat1)
    dlon = np.radians(lon2_array - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2_array)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return config.EARTH_RADIUS_KM * c

def get_freq_from_channel(channel):
    """Converts TV Channel to Frequency (MHz)."""
    try:
        ch = int(float(channel))
        if 2 <= ch <= 6: return 54 + (ch - 2) * 6 + 3
        if 7 <= ch <= 13: return 174 + (ch - 7) * 6 + 3
        if 14 <= ch <= 36: return 470 + (ch - 14) * 6 + 3
        return 500.0
    except:
        return 500.0

# --- RAM TERRAIN CACHE ---
# Global cache: {filename: numpy_array}
_TERRAIN_CACHE = {}

def preload_terrain_to_ram():
    """Load all .hgt files into RAM (~17GB). Call once at startup."""
    import glob
    print("Preloading terrain tiles into RAM...")
    
    hgt_files = glob.glob(os.path.join(config.RAW_DIR, "*.hgt"))
    print(f"   Found {len(hgt_files)} tiles")
    
    total_size = 0
    for i, filepath in enumerate(hgt_files):
        filename = os.path.basename(filepath)
        
        try:
            # Read entire file at once
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # Store raw bytes (faster than unpacking everything)
            _TERRAIN_CACHE[filename] = data
            total_size += len(data)
            
            if (i + 1) % 100 == 0:
                print(f"   Loaded {i+1}/{len(hgt_files)} tiles ({total_size / 1e9:.1f} GB)")
        except Exception as e:
            print(f"   Warning: Failed to load {filename}: {e}")
    
    print(f"   Complete! {len(_TERRAIN_CACHE)} tiles in RAM ({total_size / 1e9:.2f} GB)")

def get_elevation_from_ram(lat, lon):
    """Reads elevation from RAM cache (fast). Falls back to disk if not cached."""
    lat_floor = math.floor(lat)
    lon_floor = math.floor(lon)
    
    ns = 'N' if lat_floor >= 0 else 'S'
    ew = 'E' if lon_floor >= 0 else 'W'
    filename = f"{ns}{int(abs(lat_floor)):02d}{ew}{int(abs(lon_floor)):03d}.hgt"
    
    # Check RAM cache first
    if filename not in _TERRAIN_CACHE:
        return get_elevation_from_disk(lat, lon)  # Fallback
    
    try:
        data = _TERRAIN_CACHE[filename]
        
        # Detect file size (SRTM1 vs SRTM3)
        samples = 3601 if len(data) > 20000000 else 1201
        
        row = int(round((1 - (lat - lat_floor)) * (samples - 1)))
        col = int(round((lon - lon_floor) * (samples - 1)))
        
        # Calculate byte offset
        offset = (row * samples + col) * 2
        
        # Unpack directly from bytes
        val = struct.unpack('>h', data[offset:offset+2])[0]
        if val < -10000: return 0.0
        return float(val)
    except:
        return 0.0

def get_elevation_from_disk(lat, lon):
    """Reads 2 bytes from SSD. Uses 0 RAM."""
    lat_floor = math.floor(lat)
    lon_floor = math.floor(lon)
    
    ns = 'N' if lat_floor >= 0 else 'S'
    ew = 'E' if lon_floor >= 0 else 'W'
    filename = f"{ns}{int(abs(lat_floor)):02d}{ew}{int(abs(lon_floor)):03d}.hgt"
    filepath = os.path.join(config.RAW_DIR, filename)
    
    if not os.path.exists(filepath): return 0.0

    try:
        # Detect file size (SRTM1 vs SRTM3)
        filesize = os.path.getsize(filepath)
        samples = 3601 if filesize > 20000000 else 1201
        
        row = int(round((1 - (lat - lat_floor)) * (samples - 1)))
        col = int(round((lon - lon_floor) * (samples - 1)))
        
        # Calculate byte offset
        offset = (row * samples + col) * 2
        
        with open(filepath, 'rb') as f:
            f.seek(offset)
            val = struct.unpack('>h', f.read(2))[0]
            if val < -10000: return 0.0 # Void data
            return float(val)
    except: return 0.0

def get_profile_disk(lat1, lon1, lat2, lon2, num_points):
    """Manually samples elevation points between User and Tower using RAM cache."""
    profile = []
    for i in range(num_points):
        fraction = i / (num_points - 1)
        lat = lat1 + (lat2 - lat1) * fraction
        lon = lon1 + (lon2 - lon1) * fraction
        profile.append(get_elevation_from_ram(lat, lon))
    return profile

# Replaced with disk version but keeping signature consistent for now if needed elsewhere
def get_terrain_profile_manual(geo_data, lat1, lon1, lat2, lon2, num_points):
    """Legacy wrapper for compatibility or if srtm obj is passed."""
    return get_profile_disk(lat1, lon1, lat2, lon2, num_points)
