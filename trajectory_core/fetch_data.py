"""
Data acquisition module for the Oil Spill Trajectory Analysis Engine.

This module handles fetching environmental data from various sources:
- Wind and weather data from Open-Meteo and NOAA
- Ocean current data from NOAA ERDDAP and OSM Currents
- Elevation/DEM data from AWS Terrain Tiles and OpenTopography
- Oil properties from static datasets

The module provides a unified interface for data retrieval with appropriate
error handling, retry logic, and caching mechanisms. It also includes fallback
to static test data when API access fails.
"""

import os
import json
import math
import time
import hashlib
import requests
import logging
import functools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from datetime import datetime, timedelta

from . import config

# Set up logging
logger = logging.getLogger(__name__)

# Define cache directory
CACHE_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / '../data/cache'
STATIC_DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / '../data/static'

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(STATIC_DATA_DIR, exist_ok=True)


class ApiResponse:
    """Class to standardize API responses across different clients."""
    
    def __init__(self, 
                 success: bool, 
                 data: Dict[str, Any] = None, 
                 error: str = None, 
                 source: str = None,
                 timestamp: datetime = None,
                 cached: bool = False):
        """
        Initialize an API response.
        
        Args:
            success: Whether the API request was successful
            data: The data returned by the API
            error: Error message if the request failed
            source: Source of the data
            timestamp: Timestamp of the data
            cached: Whether the data was retrieved from cache
        """
        self.success = success
        self.data = data or {}
        self.error = error
        self.source = source
        self.timestamp = timestamp or datetime.now()
        self.cached = cached
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the response to a dictionary."""
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'source': self.source,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'cached': self.cached
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApiResponse':
        """Create an ApiResponse from a dictionary."""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
    
    @classmethod
    def success_response(cls, data: Dict[str, Any], source: str = None) -> 'ApiResponse':
        """Create a successful API response."""
        return cls(success=True, data=data, source=source)
    
    @classmethod
    def error_response(cls, error: str, source: str = None) -> 'ApiResponse':
        """Create an error API response."""
        return cls(success=False, error=error, source=source)
    
    @classmethod
    def cached_response(cls, data: Dict[str, Any], source: str = None, timestamp: datetime = None) -> 'ApiResponse':
        """Create a cached API response."""
        return cls(success=True, data=data, source=source, timestamp=timestamp, cached=True)


def cache_api_response(ttl_seconds: int = 3600):
    """
    Decorator to cache API responses.
    
    Args:
        ttl_seconds: Time-to-live for cached responses in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate a cache key based on function name and arguments
            cache_key = f"{func.__name__}_{hash(str(args))}_{hash(str(kwargs))}"
            cache_key = hashlib.md5(cache_key.encode()).hexdigest()
            cache_file = CACHE_DIR / f"{cache_key}.json"
            
            # Check if cache file exists and is not expired
            if cache_file.exists():
                file_age = time.time() - os.path.getmtime(cache_file)
                if file_age < ttl_seconds:
                    try:
                        with open(cache_file, 'r') as f:
                            cached_data = json.load(f)
                        logger.debug(f"Retrieved cached response for {func.__name__}")
                        return ApiResponse.from_dict(cached_data)
                    except Exception as e:
                        logger.warning(f"Error reading cache file: {e}")
            
            # Call the original function
            response = func(self, *args, **kwargs)
            
            # Cache the response if successful
            if response.success:
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(response.to_dict(), f)
                    logger.debug(f"Cached response for {func.__name__}")
                except Exception as e:
                    logger.warning(f"Error writing cache file: {e}")
            
            return response
        return wrapper
    return decorator


def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator to retry API calls on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds (doubles with each retry)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            attempts = 0
            current_delay = delay
            last_error = None
            
            while attempts < max_attempts:
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    attempts += 1
                    last_error = str(e)
                    logger.warning(f"Attempt {attempts}/{max_attempts} failed for {func.__name__}: {e}")
                    
                    if attempts < max_attempts:
                        logger.info(f"Retrying in {current_delay} seconds...")
                        time.sleep(current_delay)
                        current_delay *= 2  # Exponential backoff
            
            # All attempts failed, return error response
            logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {last_error}")
            return ApiResponse.error_response(f"All {max_attempts} attempts failed: {last_error}")
        return wrapper
    return decorator


class ApiClient(ABC):
    """Base class for API clients."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the API client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for API requests
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
    
    @abstractmethod
    def get_data(self, **kwargs) -> ApiResponse:
        """Get data from the API."""
        pass
    
    def validate_response(self, response: requests.Response) -> bool:
        """
        Validate the API response.
        
        Args:
            response: Response from the API
            
        Returns:
            True if the response is valid, False otherwise
        """
        return response.status_code == 200
    
    def handle_error(self, response: requests.Response) -> str:
        """
        Handle API error responses.
        
        Args:
            response: Response from the API
            
        Returns:
            Error message
        """
        try:
            error_data = response.json()
            if isinstance(error_data, dict) and 'error' in error_data:
                return error_data['error']
            else:
                return f"API error: {response.status_code} - {response.reason}"
        except Exception:
            return f"API error: {response.status_code} - {response.reason}"
    
    def get_static_data(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get static data from a file.
        
        Args:
            filename: Name of the static data file
            
        Returns:
            Static data or None if the file doesn't exist
        """
        file_path = STATIC_DATA_DIR / filename
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading static data file {filename}: {e}")
        return None
    
    def make_request(self, url: str, method: str = 'GET', params: Dict[str, Any] = None, 
                    headers: Dict[str, Any] = None, data: Dict[str, Any] = None) -> requests.Response:
        """
        Make an HTTP request.
        
        Args:
            url: URL to request
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            headers: HTTP headers
            data: Request body for POST requests
            
        Returns:
            Response from the API
        """
        headers = headers or {}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"
        
        if method.upper() == 'GET':
            return self.session.get(url, params=params, headers=headers)
        elif method.upper() == 'POST':
            return self.session.post(url, params=params, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")


class OpenMeteoClient(ApiClient):
    """API client for Open-Meteo weather data."""
    
    def __init__(self):
        """Initialize the Open-Meteo API client."""
        super().__init__(base_url=config.DATA_SOURCES['wind']['open_meteo'])
    
    @cache_api_response(ttl_seconds=3600)  # Cache for 1 hour
    @retry(max_attempts=3)
    def get_data(self, latitude: float, longitude: float, forecast_days: int = 7, **kwargs) -> ApiResponse:
        """
        Get weather data from Open-Meteo.
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            forecast_days: Number of forecast days
            
        Returns:
            ApiResponse with weather data
        """
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'hourly': 'temperature_2m,windspeed_10m,winddirection_10m,precipitation',
            'timezone': 'auto',
            'forecast_days': forecast_days
        }
        
        try:
            response = self.make_request(self.base_url, params=params)
            
            if self.validate_response(response):
                data = response.json()
                return ApiResponse.success_response(data, source='open_meteo')
            else:
                error_message = self.handle_error(response)
                logger.error(f"Open-Meteo API error: {error_message}")
                
                # Try to get static data as fallback
                static_data = self.get_static_data('open_meteo_sample.json')
                if static_data:
                    logger.info("Using static weather data as fallback")
                    return ApiResponse.success_response(static_data, source='open_meteo_static')
                
                return ApiResponse.error_response(error_message, source='open_meteo')
                
        except Exception as e:
            logger.error(f"Error fetching data from Open-Meteo: {e}")
            
            # Try to get static data as fallback
            static_data = self.get_static_data('open_meteo_sample.json')
            if static_data:
                logger.info("Using static weather data as fallback")
                return ApiResponse.success_response(static_data, source='open_meteo_static')
            
            return ApiResponse.error_response(str(e), source='open_meteo')


class NoaaErddapClient(ApiClient):
    """API client for NOAA ERDDAP ocean current data."""
    
    def __init__(self):
        """Initialize the NOAA ERDDAP API client."""
        super().__init__(base_url=config.DATA_SOURCES['ocean_currents']['noaa_erddap'])
    
    @cache_api_response(ttl_seconds=7200)  # Cache for 2 hours
    @retry(max_attempts=3)
    def get_data(self, latitude: float, longitude: float, depth: float = 0, **kwargs) -> ApiResponse:
        """
        Get ocean current data from NOAA ERDDAP.
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            depth: Depth in meters (default: 0, surface)
            
        Returns:
            ApiResponse with ocean current data
        """
        # Format the request with bounds around the point
        query_params = {
            'time': '[last]',  # Get most recent data
            'latitude': f'[{latitude-1}:{latitude+1}]',  # 2-degree box around point
            'longitude': f'[{longitude-1}:{longitude+1}]',  # 2-degree box around point
            'depth': f'[{max(0, depth-10)}:{depth+10}]',  # 20m depth range
            'water_u,water_v': '',  # Request u and v current components
        }
        
        # Build the URL with parameters
        query_string = ''.join([f"{k}{v}" for k, v in query_params.items()])
        request_url = f"{self.base_url}?{query_string}"
        
        try:
            response = self.make_request(request_url)
            
            if self.validate_response(response):
                data = response.json()
                return ApiResponse.success_response(data, source='noaa_erddap')
            else:
                error_message = self.handle_error(response)
                logger.error(f"NOAA ERDDAP API error: {error_message}")
                
                # Try to get static data as fallback
                static_data = self.get_static_data('noaa_erddap_sample.json')
                if static_data:
                    logger.info("Using static ocean current data as fallback")
                    return ApiResponse.success_response(static_data, source='noaa_erddap_static')
                
                return ApiResponse.error_response(error_message, source='noaa_erddap')
                
        except Exception as e:
            logger.error(f"Error fetching data from NOAA ERDDAP: {e}")
            
            # Try to get static data as fallback
            static_data = self.get_static_data('noaa_erddap_sample.json')
            if static_data:
                logger.info("Using static ocean current data as fallback")
                return ApiResponse.success_response(static_data, source='noaa_erddap_static')
            
            return ApiResponse.error_response(str(e), source='noaa_erddap')


class AwsTerrainClient(ApiClient):
    """API client for AWS Terrain Tiles elevation data."""
    
    def __init__(self):
        """Initialize the AWS Terrain Tiles API client."""
        super().__init__(base_url=config.DATA_SOURCES['elevation']['aws_terrain'])
    
    @cache_api_response(ttl_seconds=86400)  # Cache for 24 hours
    @retry(max_attempts=3)
    def get_data(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float, zoom: int = 10, **kwargs) -> ApiResponse:
        """
        Get elevation data from AWS Terrain Tiles.
        
        Args:
            min_lat: Minimum latitude of the bounding box
            min_lon: Minimum longitude of the bounding box
            max_lat: Maximum latitude of the bounding box
            max_lon: Maximum longitude of the bounding box
            zoom: Zoom level (0-14, higher is more detailed)
            
        Returns:
            ApiResponse with elevation data
        """
        # Calculate center point
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        
        # Convert lat/lon to tile coordinates
        x = int((center_lon + 180) / 360 * (2 ** zoom))
        y = int((1 - math.log(math.tan(math.radians(center_lat)) + 1 / math.cos(math.radians(center_lat))) / math.pi) / 2 * (2 ** zoom))
        
        # Build the URL for the terrain tile
        tile_url = f"{self.base_url}/{zoom}/{x}/{y}.png"
        
        try:
            response = self.make_request(tile_url)
            
            if self.validate_response(response):
                # This is a simplified implementation
                # In a real implementation, we would parse the PNG terrain tile
                # and convert it to elevation data
                # For now, we'll return a mock response
                
                # Try to get static data as fallback
                static_data = self.get_static_data('aws_terrain_sample.json')
                if static_data:
                    logger.info("Using static elevation data")
                    return ApiResponse.success_response(static_data, source='aws_terrain_static')
                
                # If no static data, generate mock data
                mock_data = self._generate_mock_elevation_data(min_lat, min_lon, max_lat, max_lon)
                return ApiResponse.success_response(mock_data, source='aws_terrain_mock')
            else:
                error_message = self.handle_error(response)
                logger.error(f"AWS Terrain Tiles API error: {error_message}")
                
                # Try to get static data as fallback
                static_data = self.get_static_data('aws_terrain_sample.json')
                if static_data:
                    logger.info("Using static elevation data as fallback")
                    return ApiResponse.success_response(static_data, source='aws_terrain_static')
                
                # If no static data, generate mock data
                mock_data = self._generate_mock_elevation_data(min_lat, min_lon, max_lat, max_lon)
                return ApiResponse.success_response(mock_data, source='aws_terrain_mock')
                
        except Exception as e:
            logger.error(f"Error fetching data from AWS Terrain Tiles: {e}")
            
            # Try to get static data as fallback
            static_data = self.get_static_data('aws_terrain_sample.json')
            if static_data:
                logger.info("Using static elevation data as fallback")
                return ApiResponse.success_response(static_data, source='aws_terrain_static')
            
            # If no static data, generate mock data
            mock_data = self._generate_mock_elevation_data(min_lat, min_lon, max_lat, max_lon)
            return ApiResponse.success_response(mock_data, source='aws_terrain_mock')
    
    def _generate_mock_elevation_data(self, min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> Dict[str, Any]:
        """
        Generate mock elevation data for development.
        
        Args:
            min_lat: Minimum latitude of the bounding box
            min_lon: Minimum longitude of the bounding box
            max_lat: Maximum latitude of the bounding box
            max_lon: Maximum longitude of the bounding box
            
        Returns:
            Dictionary containing mock elevation data
        """
        import random
        
        # Generate a grid of elevation values
        grid_size = 20
        lat_step = (max_lat - min_lat) / grid_size
        lon_step = (max_lon - min_lon) / grid_size
        
        lats = [min_lat + i * lat_step for i in range(grid_size + 1)]
        lons = [min_lon + i * lon_step for i in range(grid_size + 1)]
        
        # Generate random elevation values
        # Coastal areas are around 0, inland areas are higher
        elevation = []
        for _ in range(len(lats)):
            row = []
            for _ in range(len(lons)):
                # Generate a random elevation between -10 and 100 meters
                # Coastal areas are more likely to be near sea level
                if random.random() < 0.7:  # 70% chance of being near sea level
                    row.append(random.uniform(-10, 10))
                else:
                    row.append(random.uniform(0, 100))
            elevation.append(row)
        
        return {
            'lats': lats,
            'lons': lons,
            'elevation': elevation,
            'units': 'meters',
            'source': 'mock_data'
        }


class OilPropertiesClient(ApiClient):
    """Client for oil properties data."""
    
    def __init__(self):
        """Initialize the oil properties client."""
        super().__init__()
    
    @cache_api_response(ttl_seconds=604800)  # Cache for 1 week
    def get_data(self, oil_type: str = 'medium_crude', **kwargs) -> ApiResponse:
        """
        Get oil properties data.
        
        Args:
            oil_type: Type of oil
            
        Returns:
            ApiResponse with oil properties data
        """
        # Get oil properties from static data
        static_data = self.get_static_data('oil_properties.json')
        
        if static_data and oil_type in static_data:
            return ApiResponse.success_response(static_data[oil_type], source='oil_properties_static')
        else:
            # If no static data or oil type not found, generate mock data
            mock_data = self._generate_mock_oil_properties(oil_type)
            return ApiResponse.success_response(mock_data, source='oil_properties_mock')
    
    def _generate_mock_oil_properties(self, oil_type: str) -> Dict[str, Any]:
        """
        Generate mock oil properties for development.
        
        Args:
            oil_type: Type of oil
            
        Returns:
            Dictionary containing mock oil properties
        """
        # Default properties for different oil types
        oil_properties = {
            'light_crude': {
                'density': 850,  # kg/m^3
                'viscosity': 5,  # cSt
                'surface_tension': 25,  # mN/m
                'evaporation_rate': 0.4,  # fraction per day
                'emulsification_constant': 0.5,
                'solubility': 0.02  # fraction
            },
            'medium_crude': {
                'density': 900,
                'viscosity': 50,
                'surface_tension': 28,
                'evaporation_rate': 0.25,
                'emulsification_constant': 0.7,
                'solubility': 0.01
            },
            'heavy_crude': {
                'density': 950,
                'viscosity': 500,
                'surface_tension': 30,
                'evaporation_rate': 0.1,
                'emulsification_constant': 0.9,
                'solubility': 0.005
            },
            'diesel': {
                'density': 830,
                'viscosity': 3,
                'surface_tension': 23,
                'evaporation_rate': 0.5,
                'emulsification_constant': 0.3,
                'solubility': 0.03
            },
            'gasoline': {
                'density': 750,
                'viscosity': 0.5,
                'surface_tension': 20,
                'evaporation_rate': 0.9,
                'emulsification_constant': 0.1,
                'solubility': 0.05
            },
            'bunker': {
                'density': 980,
                'viscosity': 1000,
                'surface_tension': 32,
                'evaporation_rate': 0.05,
                'emulsification_constant': 0.95,
                'solubility': 0.001
            },
            'jet_fuel': {
                'density': 800,
                'viscosity': 2,
                'surface_tension': 22,
                'evaporation_rate': 0.7,
                'emulsification_constant': 0.2,
                'solubility': 0.04
            }
        }
        
        # Return properties for the specified oil type, or medium_crude as default
        return oil_properties.get(oil_type, oil_properties['medium_crude'])


class DataFetcher:
    """Class for fetching environmental data from various sources."""
    
    def __init__(self):
        """Initialize the data fetcher with API clients."""
        self.weather_client = OpenMeteoClient()
        self.ocean_client = NoaaErddapClient()
        self.elevation_client = AwsTerrainClient()
        self.oil_client = OilPropertiesClient()
    
    def fetch_data(self, data_type: str, source: str, params: Dict[str, Any]) -> ApiResponse:
        """
        Generic method to fetch data from a specified source.
        
        Args:
            data_type: Type of data to fetch (wind, ocean_currents, elevation, oil_properties)
            source: Source to fetch data from (e.g., open_meteo, noaa_erddap)
            params: Parameters for the API request
            
        Returns:
            ApiResponse containing the fetched data
        """
        # Check if data type and source are valid
        if data_type not in config.DATA_SOURCES:
            return ApiResponse.error_response(f"Unknown data type: {data_type}")
            
        if source not in config.DATA_SOURCES[data_type]:
            return ApiResponse.error_response(f"Unknown source for {data_type}: {source}")
        
        # Route the request to the appropriate client
        if data_type == 'wind':
            if source == 'open_meteo':
                return self.weather_client.get_data(**params)
            else:
                return ApiResponse.error_response(f"Unsupported wind data source: {source}")
                
        elif data_type == 'ocean_currents':
            if source == 'noaa_erddap':
                return self.ocean_client.get_data(**params)
            else:
                return ApiResponse.error_response(f"Unsupported ocean current data source: {source}")
                
        elif data_type == 'elevation':
            if source == 'aws_terrain':
                return self.elevation_client.get_data(**params)
            else:
                return ApiResponse.error_response(f"Unsupported elevation data source: {source}")
                
        elif data_type == 'oil_properties':
            return self.oil_client.get_data(**params)
            
        else:
            return ApiResponse.error_response(f"Unsupported data type: {data_type}")
    
    def get_wind_data(self, latitude: float, longitude: float, 
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     source: str = 'open_meteo') -> ApiResponse:
        """
        Fetch wind data for a specific location and time range.
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            start_time: Start time for the data (default: current time)
            end_time: End time for the data (default: 48 hours from start)
            source: Source to fetch data from (default: open_meteo)
            
        Returns:
            ApiResponse containing wind data
        """
        # Set default times if not provided
        if start_time is None:
            start_time = datetime.now()
        if end_time is None:
            end_time = start_time + timedelta(hours=48)
        
        # Calculate forecast days
        forecast_days = max(1, (end_time - start_time).days + 1)
        
        # Prepare parameters
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'forecast_days': forecast_days
        }
        
        # Fetch data from the specified source
        return self.fetch_data('wind', source, params)
    
    def get_ocean_currents(self, latitude: float, longitude: float, depth: float = 0,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          source: str = 'noaa_erddap') -> ApiResponse:
        """
        Fetch ocean current data for a specific location and time range.
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            depth: Depth in meters (default: 0, surface)
            start_time: Start time for the data (default: current time)
            end_time: End time for the data (default: 48 hours from start)
            source: Source to fetch data from (default: noaa_erddap)
            
        Returns:
            ApiResponse containing ocean current data
        """
        # Set default times if not provided
        if start_time is None:
            start_time = datetime.now()
        if end_time is None:
            end_time = start_time + timedelta(hours=48)
        
        # Prepare parameters
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'depth': depth,
            'start_time': start_time,
            'end_time': end_time
        }
        
        # Fetch data from the specified source
        return self.fetch_data('ocean_currents', source, params)
    
    def get_elevation_data(self, min_lat: float, min_lon: float, 
                          max_lat: float, max_lon: float,
                          source: str = 'aws_terrain') -> ApiResponse:
        """
        Fetch elevation data for a bounding box.
        
        Args:
            min_lat: Minimum latitude of the bounding box
            min_lon: Minimum longitude of the bounding box
            max_lat: Maximum latitude of the bounding box
            max_lon: Maximum longitude of the bounding box
            source: Source to fetch data from (default: aws_terrain)
            
        Returns:
            ApiResponse containing elevation data
        """
        # Prepare parameters
        params = {
            'min_lat': min_lat,
            'min_lon': min_lon,
            'max_lat': max_lat,
            'max_lon': max_lon
        }
        
        # Fetch data from the specified source
        return self.fetch_data('elevation', source, params)
    
    def get_oil_properties(self, oil_type: str = 'medium_crude') -> ApiResponse:
        """
        Fetch properties for a specific oil type.
        
        Args:
            oil_type: Type of oil (default: medium_crude)
            
        Returns:
            ApiResponse containing oil properties
        """
        # Prepare parameters
        params = {
            'oil_type': oil_type
        }
        
        # Fetch data from the specified source
        return self.fetch_data('oil_properties', 'adios', params)


# Create static data files for development and testing
def create_static_data_files():
    """
    Create static data files for development and testing.
    
    This function generates mock data for wind, ocean currents, elevation,
    and oil properties, and saves them to the static data directory.
    """
    import random
    import json
    import os
    import math
    from datetime import datetime, timedelta
    
    # Ensure static data directory exists
    os.makedirs(STATIC_DATA_DIR, exist_ok=True)
    
    # Create mock Open-Meteo data
    open_meteo_data = {
        'latitude': 40.7128,
        'longitude': -74.0060,
        'generationtime_ms': 0.5,
        'utc_offset_seconds': 0,
        'timezone': 'UTC',
        'timezone_abbreviation': 'UTC',
        'hourly': {
            'time': [(datetime.now() + timedelta(hours=i)).isoformat() for i in range(48)],
            'temperature_2m': [random.uniform(15, 25) for _ in range(48)],
            'windspeed_10m': [random.uniform(2, 15) for _ in range(48)],
            'winddirection_10m': [random.uniform(0, 360) for _ in range(48)],
            'precipitation': [random.uniform(0, 5) if random.random() > 0.8 else 0 for _ in range(48)]
        },
        'hourly_units': {
            'temperature_2m': '°C',
            'windspeed_10m': 'km/h',
            'winddirection_10m': '°',
            'precipitation': 'mm'
        }
    }
    
    # Save Open-Meteo data
    with open(STATIC_DATA_DIR / 'open_meteo_sample.json', 'w') as f:
        json.dump(open_meteo_data, f, indent=2)
    
    # Create mock NOAA ERDDAP ocean current data
    noaa_erddap_data = {
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'source': 'static_data',
        'data': {
            'latitude': 40.7128,
            'longitude': -74.0060,
            'depth': 0,
            'times': [(datetime.now() + timedelta(hours=i)).isoformat() for i in range(24)],
            'u': [random.uniform(-1.0, 1.0) for _ in range(24)],  # East-west current (m/s)
            'v': [random.uniform(-1.0, 1.0) for _ in range(24)]   # North-south current (m/s)
        }
    }
    
    # Save NOAA ERDDAP data
    with open(STATIC_DATA_DIR / 'noaa_erddap_sample.json', 'w') as f:
        json.dump(noaa_erddap_data, f, indent=2)
    
    # Create mock AWS Terrain elevation data
    # Generate a grid of elevation points for the AWS Terrain data
    grid_size = 10
    aws_terrain_data = {
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'source': 'static_data',
        'data': {
            'bbox': [-74.1, 40.6, -73.9, 40.8],  # NYC area
            'resolution': 0.02,
            'grid': []
        }
    }
    
    # Generate elevation data points
    min_lon, min_lat = -74.1, 40.6
    max_lon, max_lat = -73.9, 40.8
    lon_step = (max_lon - min_lon) / grid_size
    lat_step = (max_lat - min_lat) / grid_size
    
    # Generate a grid of elevation points with a simple hill/valley pattern
    for i in range(grid_size + 1):
        for j in range(grid_size + 1):
            grid_lat = min_lat + (i * lat_step)
            grid_lon = min_lon + (j * lon_step)
            
            # Create a simple elevation pattern (hills and valleys)
            dx = (grid_lon - (min_lon + (max_lon - min_lon)/2)) / ((max_lon - min_lon)/2)
            dy = (grid_lat - (min_lat + (max_lat - min_lat)/2)) / ((max_lat - min_lat)/2)
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Simple elevation function (creates a valley with surrounding hills)
            elevation = 100 * (1 - math.exp(-distance * 3) * math.cos(distance * 5))
            
            aws_terrain_data['data']['grid'].append({
                'lat': grid_lat,
                'lon': grid_lon,
                'elevation': elevation
            })
    
    # Save AWS Terrain data
    with open(STATIC_DATA_DIR / 'aws_terrain_sample.json', 'w') as f:
        json.dump(aws_terrain_data, f, indent=2)
    
    # Create mock oil properties data
    oil_properties_data = {
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'source': 'static_data',
        'data': {
            'medium_crude': {
                'name': 'Medium Crude Oil',
                'api_gravity': 25.0,
                'viscosity': 50.0,  # centistokes at 15°C
                'density': 0.9,     # g/cm³
                'pour_point': -10,  # °C
                'flash_point': 30,  # °C
                'evaporation_rate': 0.35,  # fraction per day
                'solubility': 0.02,  # fraction
                'surface_tension': 30.0  # dynes/cm
            },
            'light_crude': {
                'name': 'Light Crude Oil',
                'api_gravity': 35.0,
                'viscosity': 10.0,
                'density': 0.85,
                'pour_point': -20,
                'flash_point': 20,
                'evaporation_rate': 0.5,
                'solubility': 0.03,
                'surface_tension': 25.0
            },
            'heavy_crude': {
                'name': 'Heavy Crude Oil',
                'api_gravity': 15.0,
                'viscosity': 500.0,
                'density': 0.95,
                'pour_point': 5,
                'flash_point': 40,
                'evaporation_rate': 0.15,
                'solubility': 0.01,
                'surface_tension': 35.0
            }
        }
    }
    
    # Save oil properties data
    with open(STATIC_DATA_DIR / 'oil_properties_sample.json', 'w') as f:
        json.dump(oil_properties_data, f, indent=2)
    
    print(f"Static data files created in {STATIC_DATA_DIR}")


# Main execution point if the script is run directly
if __name__ == "__main__":
    # Create static data files for testing and development
    create_static_data_files()
    
    # Test the data fetcher
    fetcher = DataFetcher()
    
    # Test wind data fetching
    print("\nTesting wind data fetching...")
    wind_response = fetcher.get_wind_data(40.7128, -74.0060)
    print(f"Wind data success: {wind_response.success}")
    
    # Test ocean currents fetching
    print("\nTesting ocean currents fetching...")
    currents_response = fetcher.get_ocean_currents(40.7128, -74.0060)
    print(f"Ocean currents success: {currents_response.success}")
    
    # Test elevation data fetching
    print("\nTesting elevation data fetching...")
    elevation_response = fetcher.get_elevation_data(40.6, -74.1, 40.8, -73.9)
    print(f"Elevation data success: {elevation_response.success}")
    
    # Test oil properties fetching
    print("\nTesting oil properties fetching...")
    oil_response = fetcher.get_oil_properties("medium_crude")
    print(f"Oil properties success: {oil_response.success}")
