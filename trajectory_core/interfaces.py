"""
Module interfaces and utility functions for the Oil Spill Trajectory Analysis Engine.

This module defines the interfaces between different components of the simulation
and provides utility functions for common operations such as:
- Coordinate transformations
- Unit conversions
- Data validation
- Error handling

These interfaces and utilities ensure consistent behavior across the simulation
and facilitate modular development and testing.
"""

from typing import Protocol, Dict, List, Tuple, Any, Optional, Union, TypeVar, Callable
import datetime
import math
from dataclasses import dataclass

# Import local modules
from trajectory_core.data_models import SpillConfig, EnvironmentalData, Particle, SimulationResults


# Type variables for generic functions
T = TypeVar('T')


# Protocol classes for module interfaces
class DataFetcherInterface(Protocol):
    """Interface for data fetching components."""
    
    def fetch_data(self, data_type: str, source: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch data from the specified source.
        
        Args:
            data_type: Type of data to fetch (e.g., 'wind', 'ocean_currents', 'elevation')
            source: Source of the data (e.g., 'open_meteo', 'noaa_erddap')
            params: Parameters for the data fetch (e.g., location, time range)
            
        Returns:
            Dictionary containing the fetched data and metadata
        """
        ...


class PreprocessorInterface(Protocol):
    """Interface for data preprocessing components."""
    
    def preprocess_data(self, data: Dict[str, Any], config: SpillConfig) -> EnvironmentalData:
        """
        Preprocess raw data into a format suitable for simulation.
        
        Args:
            data: Raw data from data fetchers
            config: Simulation configuration
            
        Returns:
            Processed environmental data ready for simulation
        """
        ...


class ModelInterface(Protocol):
    """Interface for simulation model components."""
    
    def initialize(self, config: SpillConfig, env_data: EnvironmentalData) -> List[Particle]:
        """
        Initialize the simulation model.
        
        Args:
            config: Simulation configuration
            env_data: Environmental data
            
        Returns:
            List of initial particles
        """
        ...
    
    def step(self, particles: List[Particle], env_data: EnvironmentalData, timestep: float) -> List[Particle]:
        """
        Advance the simulation by one timestep.
        
        Args:
            particles: Current particle states
            env_data: Environmental data
            timestep: Timestep in seconds
            
        Returns:
            Updated particle states
        """
        ...
    
    def run(self, config: SpillConfig, env_data: EnvironmentalData) -> SimulationResults:
        """
        Run the complete simulation.
        
        Args:
            config: Simulation configuration
            env_data: Environmental data
            
        Returns:
            Simulation results
        """
        ...


class ExporterInterface(Protocol):
    """Interface for result export components."""
    
    def export_results(self, results: SimulationResults, format: str, filepath: str) -> None:
        """
        Export simulation results to the specified format.
        
        Args:
            results: Simulation results
            format: Output format (e.g., 'geojson', 'csv', 'netcdf')
            filepath: Path to the output file
        """
        ...


# Utility functions for coordinate transformations
def lat_lon_to_meters(lat: float, lon: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    """
    Convert latitude and longitude to meters from a reference point.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        ref_lat: Reference latitude in degrees
        ref_lon: Reference longitude in degrees
        
    Returns:
        Tuple of (x, y) coordinates in meters from the reference point
    """
    # Earth's radius in meters
    earth_radius = 6371000
    
    # Convert degrees to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)
    
    # Calculate the distance in meters
    x = earth_radius * math.cos(ref_lat_rad) * (lon_rad - ref_lon_rad)
    y = earth_radius * (lat_rad - ref_lat_rad)
    
    return x, y


def meters_to_lat_lon(x: float, y: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    """
    Convert meters from a reference point to latitude and longitude.
    
    Args:
        x: X coordinate in meters from the reference point
        y: Y coordinate in meters from the reference point
        ref_lat: Reference latitude in degrees
        ref_lon: Reference longitude in degrees
        
    Returns:
        Tuple of (latitude, longitude) in degrees
    """
    # Earth's radius in meters
    earth_radius = 6371000
    
    # Convert reference point to radians
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)
    
    # Calculate the latitude and longitude in radians
    lat_rad = ref_lat_rad + y / earth_radius
    lon_rad = ref_lon_rad + x / (earth_radius * math.cos(ref_lat_rad))
    
    # Convert radians to degrees
    lat = math.degrees(lat_rad)
    lon = math.degrees(lon_rad)
    
    return lat, lon


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on the Earth.
    
    Args:
        lat1: Latitude of the first point in degrees
        lon1: Longitude of the first point in degrees
        lat2: Latitude of the second point in degrees
        lon2: Longitude of the second point in degrees
        
    Returns:
        Distance in meters
    """
    # Earth's radius in meters
    earth_radius = 6371000
    
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = earth_radius * c
    
    return distance


# Utility functions for unit conversions
def knots_to_ms(knots: float) -> float:
    """Convert wind speed from knots to meters per second."""
    return knots * 0.514444


def ms_to_knots(ms: float) -> float:
    """Convert wind speed from meters per second to knots."""
    return ms / 0.514444


def celsius_to_kelvin(celsius: float) -> float:
    """Convert temperature from Celsius to Kelvin."""
    return celsius + 273.15


def kelvin_to_celsius(kelvin: float) -> float:
    """Convert temperature from Kelvin to Celsius."""
    return kelvin - 273.15


def barrels_to_cubic_meters(barrels: float) -> float:
    """Convert oil volume from barrels to cubic meters."""
    return barrels * 0.158987


def cubic_meters_to_barrels(cubic_meters: float) -> float:
    """Convert oil volume from cubic meters to barrels."""
    return cubic_meters / 0.158987


# Utility functions for data validation
def validate_in_range(value: float, min_value: float, max_value: float, name: str) -> None:
    """
    Validate that a value is within the specified range.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        name: Name of the value for error messages
        
    Raises:
        ValueError: If the value is outside the allowed range
    """
    if not (min_value <= value <= max_value):
        raise ValueError(f"{name} must be between {min_value} and {max_value}, got {value}")


def validate_positive(value: float, name: str) -> None:
    """
    Validate that a value is positive.
    
    Args:
        value: Value to validate
        name: Name of the value for error messages
        
    Raises:
        ValueError: If the value is not positive
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(value: float, name: str) -> None:
    """
    Validate that a value is non-negative.
    
    Args:
        value: Value to validate
        name: Name of the value for error messages
        
    Raises:
        ValueError: If the value is negative
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_latitude(lat: float) -> None:
    """
    Validate that a latitude value is within the valid range.
    
    Args:
        lat: Latitude value to validate
        
    Raises:
        ValueError: If the latitude is outside the valid range
    """
    validate_in_range(lat, -90, 90, "Latitude")


def validate_longitude(lon: float) -> None:
    """
    Validate that a longitude value is within the valid range.
    
    Args:
        lon: Longitude value to validate
        
    Raises:
        ValueError: If the longitude is outside the valid range
    """
    validate_in_range(lon, -180, 180, "Longitude")


# Error handling utilities
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default value if the denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value to return if denominator is zero
        
    Returns:
        Result of division or default value
    """
    return numerator / denominator if denominator != 0 else default


def safe_get(dictionary: Dict[str, T], key: str, default: Optional[T] = None) -> Optional[T]:
    """
    Safely get a value from a dictionary, returning a default value if the key is not present.
    
    Args:
        dictionary: Dictionary to get value from
        key: Key to look up
        default: Default value to return if key is not present
        
    Returns:
        Value from dictionary or default value
    """
    return dictionary.get(key, default)


def retry(func: Callable, max_attempts: int = 3, delay: float = 1.0) -> Any:
    """
    Retry a function call multiple times with exponential backoff.
    
    Args:
        func: Function to call
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts in seconds
        
    Returns:
        Result of the function call
        
    Raises:
        Exception: The last exception raised by the function
    """
    import time
    
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                time.sleep(delay * (2 ** attempt))
    
    if last_exception:
        raise last_exception
    return None


# Data conversion utilities
def dict_to_geojson_point(lat: float, lon: float, properties: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convert a latitude and longitude to a GeoJSON Point feature.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        properties: Properties to include in the feature
        
    Returns:
        GeoJSON Point feature
    """
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat]
        },
        "properties": properties or {}
    }


def dict_to_geojson_linestring(points: List[Tuple[float, float]], properties: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convert a list of points to a GeoJSON LineString feature.
    
    Args:
        points: List of (latitude, longitude) tuples
        properties: Properties to include in the feature
        
    Returns:
        GeoJSON LineString feature
    """
    # GeoJSON uses [longitude, latitude] order for coordinates
    coordinates = [[lon, lat] for lat, lon in points]
    
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates
        },
        "properties": properties or {}
    }


def dict_to_geojson_polygon(points: List[Tuple[float, float]], properties: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convert a list of points to a GeoJSON Polygon feature.
    
    Args:
        points: List of (latitude, longitude) tuples defining the polygon boundary
        properties: Properties to include in the feature
        
    Returns:
        GeoJSON Polygon feature
    """
    # GeoJSON uses [longitude, latitude] order for coordinates
    # For a valid polygon, the first and last points must be the same
    if points[0] != points[-1]:
        points = points + [points[0]]
    
    coordinates = [[lon, lat] for lat, lon in points]
    
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coordinates]  # Polygon requires an array of linear rings
        },
        "properties": properties or {}
    }


def create_geojson_feature_collection(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a GeoJSON FeatureCollection from a list of features.
    
    Args:
        features: List of GeoJSON features
        
    Returns:
        GeoJSON FeatureCollection
    """
    return {
        "type": "FeatureCollection",
        "features": features
    }
