"""
Data preprocessing module for the Oil Spill Trajectory Analysis Engine.

This module handles preprocessing of environmental data:
- DEM resampling and slope calculation
- Wind/current data interpolation to spill location
- Initialization of particle positions (for Lagrangian model)
- Normalization and preparation of data for modeling
"""

import logging
import numpy as np
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class for preprocessing environmental data for oil spill modeling."""
    
    def __init__(self):
        """Initialize the data preprocessor."""
        pass
    
    def preprocess_wind_data(self, wind_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess wind data for use in the model.
        
        Args:
            wind_data: Raw wind data from the fetch_data module
            
        Returns:
            Preprocessed wind data ready for the model
        """
        processed_data = {
            'timestamps': [],
            'wind_speeds': [],
            'wind_directions': [],
            'wind_vectors_u': [],
            'wind_vectors_v': [],
            'latitude': None,
            'longitude': None
        }
        
        # Check if we have valid wind data
        if not wind_data or 'data' not in wind_data:
            logger.warning("Invalid wind data format")
            return processed_data
        
        try:
            # Extract data based on format
            data = wind_data['data']
            
            # Store location information if available
            if 'latitude' in data and 'longitude' in data:
                processed_data['latitude'] = data['latitude']
                processed_data['longitude'] = data['longitude']
            
            # Handle Open-Meteo format
            if 'hourly' in data and 'time' in data['hourly']:
                # Extract timestamps
                timestamps = data['hourly']['time']
                processed_data['timestamps'] = timestamps
                
                # Extract wind components if available
                if 'windspeed_10m' in data['hourly'] and 'winddirection_10m' in data['hourly']:
                    wind_speeds = data['hourly']['windspeed_10m']
                    wind_directions = data['hourly']['winddirection_10m']
                    
                    # Store wind speeds and directions
                    processed_data['wind_speeds'] = wind_speeds
                    processed_data['wind_directions'] = wind_directions
                    
                    # Calculate U and V components (vector components)
                    # Convert wind direction from meteorological to mathematical convention
                    # Meteorological: direction FROM which the wind is blowing
                    # Mathematical: direction TO which the wind is blowing
                    for speed, direction in zip(wind_speeds, wind_directions):
                        # Convert direction to mathematical convention (add 180 degrees and take modulo 360)
                        math_direction = (direction + 180) % 360
                        
                        # Convert to radians
                        direction_rad = np.radians(math_direction)
                        
                        # Calculate U and V components
                        # U is positive eastward, V is positive northward
                        u = speed * np.sin(direction_rad)
                        v = speed * np.cos(direction_rad)
                        
                        processed_data['wind_vectors_u'].append(u)
                        processed_data['wind_vectors_v'].append(v)
            
            # Handle direct vector format
            elif 'times' in data and 'u' in data and 'v' in data:
                # Extract timestamps
                processed_data['timestamps'] = data['times']
                
                # Extract u and v components
                u_components = data['u']
                v_components = data['v']
                
                # Calculate speeds and directions
                for u, v in zip(u_components, v_components):
                    # Calculate speed (magnitude of the vector)
                    speed = np.sqrt(u**2 + v**2)
                    
                    # Calculate direction (in degrees, 0 = north, clockwise)
                    # arctan2 returns angle in radians, convert to degrees
                    # Adjust for meteorological convention (direction FROM)
                    direction = np.degrees(np.arctan2(u, v))
                    if direction < 0:
                        direction += 360.0
                    
                    processed_data['wind_speeds'].append(speed)
                    processed_data['wind_directions'].append(direction)
                    processed_data['wind_vectors_u'].append(u)
                    processed_data['wind_vectors_v'].append(v)
            
            # Handle grid-based wind data
            elif 'grid' in data:
                grid = data['grid']
                
                # Extract unique timestamps if available
                timestamps = set()
                for point in grid:
                    if 'timestamp' in point:
                        timestamps.add(point['timestamp'])
                
                processed_data['timestamps'] = sorted(list(timestamps))
                
                # Group data by timestamp
                data_by_time = {}
                for point in grid:
                    timestamp = point.get('timestamp', processed_data['timestamps'][0] if processed_data['timestamps'] else None)
                    if timestamp not in data_by_time:
                        data_by_time[timestamp] = []
                    data_by_time[timestamp].append(point)
                
                # Process each timestamp
                for timestamp in processed_data['timestamps']:
                    points = data_by_time.get(timestamp, [])
                    if not points:
                        continue
                    
                    # Calculate average u, v components for this timestamp
                    u_sum = sum(point.get('u', 0) for point in points)
                    v_sum = sum(point.get('v', 0) for point in points)
                    count = len(points)
                    
                    u_avg = u_sum / count if count > 0 else 0
                    v_avg = v_sum / count if count > 0 else 0
                    
                    # Calculate speed and direction
                    speed = np.sqrt(u_avg**2 + v_avg**2)
                    direction = np.degrees(np.arctan2(u_avg, v_avg))
                    if direction < 0:
                        direction += 360.0
                    
                    processed_data['wind_speeds'].append(speed)
                    processed_data['wind_directions'].append(direction)
                    processed_data['wind_vectors_u'].append(u_avg)
                    processed_data['wind_vectors_v'].append(v_avg)
        except Exception as e:
            logger.error(f"Error preprocessing wind data: {e}")
        
        return processed_data
    
    def preprocess_ocean_currents(self, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess ocean current data for use in the model.
        
        Args:
            current_data: Raw ocean current data from the fetch_data module
            
        Returns:
            Preprocessed ocean current data ready for the model
        """
        processed_data = {
            'timestamps': [],
            'current_speeds': [],
            'current_directions': [],
            'current_vectors_u': [],
            'current_vectors_v': [],
            'depth_levels': [],
            'latitude': None,
            'longitude': None
        }
        
        # Check if we have valid current data
        if not current_data or 'data' not in current_data:
            logger.warning("Invalid ocean current data format")
            return processed_data
        
        try:
            # Extract data based on format
            data = current_data['data']
            
            # Store location information
            if 'latitude' in data and 'longitude' in data:
                processed_data['latitude'] = data['latitude']
                processed_data['longitude'] = data['longitude']
            
            # Handle NOAA ERDDAP format
            if 'times' in data and 'u' in data and 'v' in data:
                # Extract timestamps
                processed_data['timestamps'] = data['times']
                
                # Extract depth information if available
                if 'depth' in data:
                    processed_data['depth_levels'] = [data['depth']]
                
                # Extract u and v components
                u_components = data['u']
                v_components = data['v']
                
                # Calculate speeds and directions
                for u, v in zip(u_components, v_components):
                    # Calculate speed (magnitude of the vector)
                    speed = np.sqrt(u**2 + v**2)
                    
                    # Calculate direction (in degrees, 0 = north, clockwise)
                    # arctan2 returns angle in radians, convert to degrees
                    # Adjust for oceanographic convention (direction TO)
                    direction = np.degrees(np.arctan2(v, u))
                    if direction < 0:
                        direction += 360.0
                    
                    processed_data['current_speeds'].append(speed)
                    processed_data['current_directions'].append(direction)
                    processed_data['current_vectors_u'].append(u)
                    processed_data['current_vectors_v'].append(v)
            
            # Handle grid-based current data (e.g., from HYCOM or CMEMS)
            elif 'grid' in data:
                grid = data['grid']
                
                # Extract unique timestamps if available
                timestamps = set()
                for point in grid:
                    if 'timestamp' in point:
                        timestamps.add(point['timestamp'])
                
                processed_data['timestamps'] = sorted(list(timestamps))
                
                # Group data by timestamp
                data_by_time = {}
                for point in grid:
                    timestamp = point.get('timestamp', processed_data['timestamps'][0] if processed_data['timestamps'] else None)
                    if timestamp not in data_by_time:
                        data_by_time[timestamp] = []
                    data_by_time[timestamp].append(point)
                
                # Process each timestamp
                for timestamp in processed_data['timestamps']:
                    points = data_by_time.get(timestamp, [])
                    if not points:
                        continue
                    
                    # Calculate average u, v components for this timestamp
                    u_sum = sum(point.get('u', 0) for point in points)
                    v_sum = sum(point.get('v', 0) for point in points)
                    count = len(points)
                    
                    u_avg = u_sum / count if count > 0 else 0
                    v_avg = v_sum / count if count > 0 else 0
                    
                    # Calculate speed and direction
                    speed = np.sqrt(u_avg**2 + v_avg**2)
                    direction = np.degrees(np.arctan2(v_avg, u_avg))
                    if direction < 0:
                        direction += 360.0
                    
                    processed_data['current_speeds'].append(speed)
                    processed_data['current_directions'].append(direction)
                    processed_data['current_vectors_u'].append(u_avg)
                    processed_data['current_vectors_v'].append(v_avg)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing ocean current data: {e}")
            return processed_data
    
    def preprocess_elevation_data(self, elevation_data: Dict[str, Any], target_resolution: float = 30.0) -> Dict[str, Any]:
        """
        Preprocess elevation data, resample to target resolution, and calculate slopes.
        
        Args:
            elevation_data: Raw elevation data from the fetch_data module
            target_resolution: Target resolution in meters (default: 30m)
            
        Returns:
            Preprocessed elevation data with slope and aspect
        """
        processed_data = {
            'elevation': None,  # 2D numpy array
            'slope': None,      # 2D numpy array
            'aspect': None,     # 2D numpy array
            'resolution': None, # (x_res, y_res) in meters
            'bounds': None      # (min_x, min_y, max_x, max_y)
        }
        
        # Check if we have valid elevation data
        if not elevation_data or 'data' not in elevation_data or 'grid' not in elevation_data['data']:
            logger.warning("Invalid elevation data format")
            return processed_data
        
        try:
            # Extract grid data
            grid = elevation_data['data']['grid']
            
            # Extract bounds
            if 'bbox' in elevation_data['data']:
                bounds = elevation_data['data']['bbox']  # [min_lon, min_lat, max_lon, max_lat]
                processed_data['bounds'] = bounds
            else:
                # Calculate bounds from grid points
                lats = [point['lat'] for point in grid]
                lons = [point['lon'] for point in grid]
                bounds = [min(lons), min(lats), max(lons), max(lats)]
                processed_data['bounds'] = bounds
            
            # Calculate original resolution
            min_lon, min_lat, max_lon, max_lat = bounds
            original_resolution = elevation_data['data'].get('resolution', None)
            
            if original_resolution is None:
                # Estimate resolution from grid points
                # Find unique latitudes and longitudes to determine grid dimensions
                unique_lats = sorted(set([point['lat'] for point in grid]))
                unique_lons = sorted(set([point['lon'] for point in grid]))
                
                if len(unique_lats) > 1 and len(unique_lons) > 1:
                    lat_res = (max_lat - min_lat) / (len(unique_lats) - 1)
                    lon_res = (max_lon - min_lon) / (len(unique_lons) - 1)
                    original_resolution = (lon_res, lat_res)
                else:
                    logger.warning("Cannot determine resolution from grid points")
                    return processed_data
            elif isinstance(original_resolution, (int, float)):
                original_resolution = (original_resolution, original_resolution)
            
            # Create regular grid from potentially irregular data points
            # Determine grid dimensions based on bounds and target resolution
            # Convert target_resolution from meters to degrees (approximate)            
            # 1 degree of latitude is approximately 111 km (varies slightly with latitude)
            # 1 degree of longitude varies with latitude: 111 km * cos(latitude)
            avg_lat = (min_lat + max_lat) / 2
            target_lat_res = target_resolution / 111000  # Convert meters to degrees
            target_lon_res = target_resolution / (111000 * np.cos(np.radians(avg_lat)))
            
            # Calculate grid dimensions
            lat_steps = max(2, int(np.ceil((max_lat - min_lat) / target_lat_res)))
            lon_steps = max(2, int(np.ceil((max_lon - min_lon) / target_lon_res)))
            
            # Create regular grid
            grid_lons = np.linspace(min_lon, max_lon, lon_steps)
            grid_lats = np.linspace(min_lat, max_lat, lat_steps)
            grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lons, grid_lats)
            
            # Initialize elevation grid
            elevation_grid = np.zeros((lat_steps, lon_steps))
            
            # Create dictionary for fast lookup of elevation values
            elevation_lookup = {}
            for point in grid:
                lat_key = round(point['lat'], 6)  # Round to 6 decimal places for lookup
                lon_key = round(point['lon'], 6)
                elevation_lookup[(lat_key, lon_key)] = point['elevation']
            
            # Interpolate elevation values to regular grid
            # For each point in the regular grid, find the nearest points in the original grid
            from scipy.interpolate import griddata
            
            # Extract original grid points and values
            orig_points = np.array([(point['lon'], point['lat']) for point in grid])
            orig_values = np.array([point['elevation'] for point in grid])
            
            # Create target grid points
            target_points = np.vstack([grid_lon_mesh.flatten(), grid_lat_mesh.flatten()]).T
            
            # Interpolate using nearest neighbor for edges, linear for interior
            elevation_values = griddata(orig_points, orig_values, target_points, method='linear', 
                                        fill_value=None)
            
            # Fill NaN values with nearest neighbor interpolation
            if np.any(np.isnan(elevation_values)):
                nan_mask = np.isnan(elevation_values)
                elevation_values[nan_mask] = griddata(orig_points, orig_values, 
                                                     target_points[nan_mask], method='nearest')
            
            # Reshape to grid
            elevation_grid = elevation_values.reshape(grid_lat_mesh.shape)
            
            # Calculate slope and aspect
            slope_grid, aspect_grid = self._calculate_slope_aspect(elevation_grid, 
                                                                 (target_lon_res, target_lat_res))
            
            # Store results
            processed_data['elevation'] = elevation_grid
            processed_data['slope'] = slope_grid
            processed_data['aspect'] = aspect_grid
            processed_data['resolution'] = (target_lon_res, target_lat_res)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing elevation data: {e}")
            return processed_data
    
    def _calculate_slope_aspect(self, elevation_grid: np.ndarray, resolution: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate slope and aspect from elevation data.
        
        Args:
            elevation_grid: 2D numpy array of elevation values
            resolution: (x_res, y_res) in degrees
            
        Returns:
            Tuple of (slope_grid, aspect_grid) as 2D numpy arrays
        """
        # Convert resolution from degrees to meters (approximate)
        lon_res, lat_res = resolution
        avg_lat = np.mean(np.linspace(0, elevation_grid.shape[0] - 1, elevation_grid.shape[0]) * lat_res)
        x_res_meters = lon_res * 111000 * np.cos(np.radians(avg_lat))  # meters per longitude degree at this latitude
        y_res_meters = lat_res * 111000  # meters per latitude degree
        
        # Get grid dimensions
        rows, cols = elevation_grid.shape
        
        # Initialize slope and aspect grids
        slope_grid = np.zeros_like(elevation_grid)
        aspect_grid = np.zeros_like(elevation_grid)
        
        # Calculate slope and aspect using 3x3 window
        # We'll use the Horn method (also used by ESRI and GDAL)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # Get the 3x3 window of elevation values
                window = elevation_grid[i-1:i+2, j-1:j+2]
                
                # Calculate rate of change in x and y directions
                # Using Horn's formula for 3x3 window
                dz_dx = ((window[0, 2] + 2*window[1, 2] + window[2, 2]) - 
                         (window[0, 0] + 2*window[1, 0] + window[2, 0])) / (8 * x_res_meters)
                
                dz_dy = ((window[2, 0] + 2*window[2, 1] + window[2, 2]) - 
                         (window[0, 0] + 2*window[0, 1] + window[0, 2])) / (8 * y_res_meters)
                
                # Calculate slope (in degrees)
                slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
                
                # Calculate aspect (in degrees, 0 = north, 90 = east, etc.)
                aspect = np.degrees(np.arctan2(dz_dy, -dz_dx))
                
                # Convert to 0-360 degrees, with 0 at north
                if aspect < 0:
                    aspect += 360.0
                
                # Store values
                slope_grid[i, j] = slope
                aspect_grid[i, j] = aspect
        
        # Handle edges by extending the nearest calculated values
        # Top and bottom edges
        if rows > 2:
            slope_grid[0, 1:-1] = slope_grid[1, 1:-1]
            slope_grid[-1, 1:-1] = slope_grid[-2, 1:-1]
            aspect_grid[0, 1:-1] = aspect_grid[1, 1:-1]
            aspect_grid[-1, 1:-1] = aspect_grid[-2, 1:-1]
        
        # Left and right edges
        if cols > 2:
            slope_grid[1:-1, 0] = slope_grid[1:-1, 1]
            slope_grid[1:-1, -1] = slope_grid[1:-1, -2]
            aspect_grid[1:-1, 0] = aspect_grid[1:-1, 1]
            aspect_grid[1:-1, -1] = aspect_grid[1:-1, -2]
        
        # Corners
        if rows > 2 and cols > 2:
            slope_grid[0, 0] = slope_grid[1, 1]
            slope_grid[0, -1] = slope_grid[1, -2]
            slope_grid[-1, 0] = slope_grid[-2, 1]
            slope_grid[-1, -1] = slope_grid[-2, -2]
            aspect_grid[0, 0] = aspect_grid[1, 1]
            aspect_grid[0, -1] = aspect_grid[1, -2]
            aspect_grid[-1, 0] = aspect_grid[-2, 1]
            aspect_grid[-1, -1] = aspect_grid[-2, -2]
        
        return slope_grid, aspect_grid
    
    def initialize_particles(self, 
                            spill_location: Tuple[float, float], 
                            spill_volume: float,
                            particle_count: int = 1000) -> List[Dict[str, Any]]:
        """
        Initialize particles for the Lagrangian model.
        
        Args:
            spill_location: (latitude, longitude) of the spill center
            spill_volume: Volume of the spill in liters
            particle_count: Number of particles to generate
            
        Returns:
            List of particle dictionaries with initial positions and properties
        """
        particles = []
        
        # Calculate mass per particle
        # Assuming medium crude oil with density ~870 kg/m³
        # 1 liter = 0.001 m³, so mass = volume * 0.001 * 870 kg
        mass_per_particle = spill_volume * 0.001 * 870 / particle_count
        
        # Initialize particles at the spill location with small random offsets
        lat, lon = spill_location
        
        for i in range(particle_count):
            # Add small random offset (within ~10m)
            lat_offset = np.random.normal(0, 0.0001)
            lon_offset = np.random.normal(0, 0.0001)
            
            particle = {
                'id': i,
                'latitude': lat + lat_offset,
                'longitude': lon + lon_offset,
                'mass': mass_per_particle,
                'age': 0,  # Age in hours
                'status': 'active',  # active, beached, evaporated, etc.
                'depth': 0,  # Depth in meters (0 = surface)
                'velocity': (0, 0)  # Initial velocity vector (u, v)
            }
            
            particles.append(particle)
        
        return particles
    
    def interpolate_wind_to_location(self, 
                                   wind_data: Dict[str, Any],
                                   location: Tuple[float, float],
                                   timestamp: datetime) -> Tuple[float, float]:
        """
        Interpolate wind data to a specific location and time.
        
        Args:
            wind_data: Preprocessed wind data
            location: (latitude, longitude) to interpolate to
            timestamp: Time to interpolate to
            
        Returns:
            Tuple of (u, v) wind vector components at the location
        """
        # Check if we have valid wind data
        if not wind_data['timestamps'] or not wind_data['wind_vectors_u'] or not wind_data['wind_vectors_v']:
            logger.warning("No wind data available for interpolation")
            return (0, 0)  # No data available
        
        try:
            # Convert timestamp strings to datetime objects if needed
            time_objects = []
            for time_str in wind_data['timestamps']:
                try:
                    # Handle different ISO format variants
                    if 'Z' in time_str:
                        time_obj = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                    elif '+' in time_str or '-' in time_str and 'T' in time_str:
                        time_obj = datetime.fromisoformat(time_str)
                    else:
                        # Add UTC timezone if not specified
                        time_obj = datetime.fromisoformat(time_str + '+00:00')
                    time_objects.append(time_obj)
                except ValueError:
                    logger.warning(f"Could not parse timestamp: {time_str}")
                    time_objects.append(None)
            
            # Filter out None values
            valid_times = [(i, t) for i, t in enumerate(time_objects) if t is not None]
            if not valid_times:
                return (0, 0)  # No valid timestamps
            
            # Calculate time differences in seconds
            time_diffs = [(i, abs((t - timestamp).total_seconds())) for i, t in valid_times]
            
            # Sort by time difference
            time_diffs.sort(key=lambda x: x[1])
            
            # If the timestamp is exactly at or very close to a data point, use that point
            if time_diffs[0][1] < 60:  # Within 1 minute
                nearest_idx = time_diffs[0][0]
                return (wind_data['wind_vectors_u'][nearest_idx], wind_data['wind_vectors_v'][nearest_idx])
            
            # For temporal interpolation, find the two closest points in time
            # First, find all timestamps and sort them
            sorted_times = sorted([(i, t) for i, t in valid_times], key=lambda x: x[1])
            
            # Find the timestamps immediately before and after the target time
            before_idx = None
            after_idx = None
            
            for i, t in sorted_times:
                if t <= timestamp:
                    before_idx = i
                if t >= timestamp and after_idx is None:
                    after_idx = i
            
            # If we don't have points before and after, use nearest neighbor
            if before_idx is None or after_idx is None:
                nearest_idx = time_diffs[0][0]
                return (wind_data['wind_vectors_u'][nearest_idx], wind_data['wind_vectors_v'][nearest_idx])
            
            # Perform linear interpolation between the two points
            t_before = time_objects[before_idx]
            t_after = time_objects[after_idx]
            
            # Calculate weights based on time difference
            total_diff = (t_after - t_before).total_seconds()
            if total_diff == 0:  # Avoid division by zero
                weight_after = 0.5
            else:
                weight_after = (timestamp - t_before).total_seconds() / total_diff
            
            weight_before = 1 - weight_after
            
            # Get the wind vectors at the two times
            u_before = wind_data['wind_vectors_u'][before_idx]
            v_before = wind_data['wind_vectors_v'][before_idx]
            u_after = wind_data['wind_vectors_u'][after_idx]
            v_after = wind_data['wind_vectors_v'][after_idx]
            
            # Interpolate the wind vectors
            u_interp = u_before * weight_before + u_after * weight_after
            v_interp = v_before * weight_before + v_after * weight_after
            
            return (u_interp, v_interp)
            
        except Exception as e:
            logger.error(f"Error interpolating wind data: {e}")
            
            # Fallback to nearest neighbor if there's an error
            if wind_data['timestamps']:
                # Simple nearest-time lookup
                time_strings = wind_data['timestamps']
                time_diffs = []
                
                for time_str in time_strings:
                    try:
                        time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                        time_diffs.append(abs((time - timestamp).total_seconds()))
                    except ValueError:
                        time_diffs.append(float('inf'))
        
        if not time_diffs:
            return (0, 0)
            
        nearest_idx = np.argmin(time_diffs)
        
        # Return the wind vector at the nearest time
        return (wind_data['wind_vectors_u'][nearest_idx], wind_data['wind_vectors_v'][nearest_idx])
    
    def interpolate_currents_to_location(self,
                                      current_data: Dict[str, Any],
                                      location: Tuple[float, float],
                                      depth: float,
                                      timestamp: datetime) -> Tuple[float, float]:
        """
        Interpolate ocean current data to a specific location, depth, and time.
        
        Args:
            current_data: Preprocessed ocean current data
            location: (latitude, longitude) to interpolate to
            depth: Depth in meters to interpolate to
            timestamp: Time to interpolate to
            
        Returns:
            Tuple of (u, v) current vector components at the location
        """
        # Placeholder - this will be implemented in a future task
        # Would do 3D interpolation (lat, lon, depth) and temporal interpolation
        logger.warning("Current interpolation not fully implemented")
        return (0, 0)  # Placeholder
    
    def get_elevation_and_slope(self,
                              elevation_data: Dict[str, Any],
                              location: Tuple[float, float]) -> Tuple[float, float, float]:
        """
        Get elevation and slope at a specific location.
        
        Args:
            elevation_data: Preprocessed elevation data
            location: (latitude, longitude) to get data for
            
        Returns:
            Tuple of (elevation, slope, aspect) at the location
        """
        # Placeholder - this will be implemented in a future task
        # Would do spatial interpolation on the DEM and derived slope/aspect
        logger.warning("Elevation and slope interpolation not fully implemented")
        return (0, 0, 0)  # Placeholder


def preprocess_all_data(wind_data: Dict[str, Any],
                       current_data: Dict[str, Any],
                       elevation_data: Dict[str, Any],
                       spill_location: Tuple[float, float],
                       spill_volume: float,
                       start_time: Optional[datetime] = None,
                       simulation_duration_hours: int = 48,
                       target_resolution: float = 30.0,
                       particle_count: int = 1000) -> Dict[str, Any]:
    """
    Preprocess all environmental data and initialize particles.
    
    Args:
        wind_data: Raw wind data from fetch_data module
        current_data: Raw ocean current data from fetch_data module
        elevation_data: Raw elevation data from fetch_data module
        spill_location: (latitude, longitude) of the spill center
        spill_volume: Volume of the spill in liters
        start_time: Start time for the simulation (default: current time)
        simulation_duration_hours: Duration of the simulation in hours (default: 48)
        target_resolution: Target resolution for elevation data in meters (default: 30m)
        particle_count: Number of particles to generate (default: 1000)
        
    Returns:
        Dictionary containing all preprocessed data ready for modeling
    """
    # Set default start time if not provided
    if start_time is None:
        start_time = datetime.now()
    
    # Calculate end time
    end_time = start_time + timedelta(hours=simulation_duration_hours)
    
    # Create preprocessor instance
    preprocessor = DataPreprocessor()
    
    # Preprocess individual data types
    processed_wind = preprocessor.preprocess_wind_data(wind_data)
    processed_currents = preprocessor.preprocess_ocean_currents(current_data)
    processed_elevation = preprocessor.preprocess_elevation_data(elevation_data, target_resolution)
    
    # Initialize particles
    particles = preprocessor.initialize_particles(spill_location, spill_volume, particle_count)
    
    # Create simulation time steps
    time_step_hours = 1  # 1-hour time steps by default
    time_steps = []
    current_time = start_time
    
    while current_time <= end_time:
        time_steps.append(current_time)
        current_time += timedelta(hours=time_step_hours)
    
    # Normalize data for the model
    normalized_data = normalize_data_for_model(
        processed_wind, 
        processed_currents, 
        processed_elevation, 
        particles,
        spill_location,
        time_steps
    )
    
    # Combine all data
    preprocessed_data = {
        'raw': {
            'wind': processed_wind,
            'currents': processed_currents,
            'elevation': processed_elevation,
            'particles': particles
        },
        'normalized': normalized_data,
        'metadata': {
            'spill_location': spill_location,
            'spill_volume': spill_volume,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'time_step_hours': time_step_hours,
            'target_resolution': target_resolution,
            'particle_count': particle_count,
            'processing_timestamp': datetime.now().isoformat()
        }
    }
    
    return preprocessed_data


def normalize_data_for_model(wind_data: Dict[str, Any],
                           current_data: Dict[str, Any],
                           elevation_data: Dict[str, Any],
                           particles: List[Dict[str, Any]],
                           spill_location: Tuple[float, float],
                           time_steps: List[datetime]) -> Dict[str, Any]:
    """
    Normalize all preprocessed data for use in the modeling engine.
    
    Args:
        wind_data: Preprocessed wind data
        current_data: Preprocessed ocean current data
        elevation_data: Preprocessed elevation data
        particles: Initialized particles
        spill_location: (latitude, longitude) of the spill center
        time_steps: List of simulation time steps
        
    Returns:
        Dictionary containing normalized data ready for the modeling engine
    """
    # Create preprocessor instance for interpolation methods
    preprocessor = DataPreprocessor()
    
    # Initialize normalized data structure
    normalized_data = {
        'time_steps': [t.isoformat() for t in time_steps],
        'reference_location': spill_location,
        'wind_vectors': [],  # Will contain (u, v) for each time step
        'current_vectors': [],  # Will contain (u, v) for each time step
        'elevation': None,  # Will contain normalized elevation grid
        'slope': None,  # Will contain normalized slope grid
        'particles': []  # Will contain normalized particle data
    }
    
    # Normalize elevation data
    if elevation_data['elevation'] is not None:
        # Get min/max values for normalization
        elev_min = np.min(elevation_data['elevation'])
        elev_max = np.max(elevation_data['elevation'])
        elev_range = max(1.0, elev_max - elev_min)  # Avoid division by zero
        
        # Normalize elevation to [0, 1] range
        normalized_data['elevation'] = (elevation_data['elevation'] - elev_min) / elev_range
        
        # Normalize slope (typically in degrees, 0-90)
        if elevation_data['slope'] is not None:
            normalized_data['slope'] = elevation_data['slope'] / 90.0  # Normalize to [0, 1]
    
    # Normalize wind and current data for each time step
    for time_step in time_steps:
        # Interpolate wind data to spill location at this time step
        wind_u, wind_v = preprocessor.interpolate_wind_to_location(
            wind_data, spill_location, time_step)
        
        # Interpolate current data to spill location at this time step (surface level)
        current_u, current_v = preprocessor.interpolate_currents_to_location(
            current_data, spill_location, 0.0, time_step)
        
        # Store interpolated vectors
        normalized_data['wind_vectors'].append((wind_u, wind_v))
        normalized_data['current_vectors'].append((current_u, current_v))
    
    # Normalize particle data
    for particle in particles:
        # Convert lat/lon to relative coordinates from spill center
        rel_lat = particle['latitude'] - spill_location[0]
        rel_lon = particle['longitude'] - spill_location[1]
        
        # Create normalized particle
        norm_particle = {
            'id': particle['id'],
            'rel_position': (rel_lat, rel_lon),  # Relative position from spill center
            'mass_fraction': particle['mass'] / spill_volume,  # Normalize mass to fraction of total
            'status': particle['status'],
            'age': 0,  # Initial age is 0
            'depth': particle['depth']  # Keep depth as is (in meters)
        }
        
        normalized_data['particles'].append(norm_particle)
    
    return normalized_data
