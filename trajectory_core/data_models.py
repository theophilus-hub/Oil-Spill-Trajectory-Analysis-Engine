"""
Data models for the Oil Spill Trajectory Analysis Engine.

This module contains the data models used throughout the simulation:
- SpillConfig: Configuration parameters for an oil spill
- EnvironmentalData: Environmental data for the simulation
- Particle: Individual oil particles in the simulation
- SimulationResults: Results of a simulation run

These models provide validation, serialization, and utility methods
for working with simulation data.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import datetime
import math
import random


@dataclass
class SpillConfig:
    """
    Configuration parameters for an oil spill simulation.
    
    Attributes:
        location: Tuple of (latitude, longitude) coordinates
        volume: Volume of the spill in cubic meters
        oil_type: Type of oil (e.g., 'crude', 'diesel', 'bunker')
        duration: Duration of the simulation in hours
        timestep: Simulation timestep in seconds
        start_time: Start time of the simulation (default: current time)
        random_seed: Seed for random number generation (default: None)
        particle_count: Number of particles to simulate (default: 1000)
    """
    
    location: Tuple[float, float]
    volume: float
    oil_type: str
    duration: float
    timestep: float
    start_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    random_seed: Optional[int] = None
    particle_count: int = 1000
    
    # List of valid oil types
    VALID_OIL_TYPES = [
        'light_crude',
        'medium_crude',
        'heavy_crude',
        'diesel',
        'gasoline',
        'bunker',
        'jet_fuel'
    ]
    
    def __post_init__(self):
        """Validate the configuration parameters after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate the configuration parameters."""
        # Validate location
        lat, lon = self.location
        if not (-90 <= lat <= 90):
            raise ValueError(f"Latitude must be between -90 and 90, got {lat}")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Longitude must be between -180 and 180, got {lon}")
        
        # Validate volume
        if self.volume <= 0:
            raise ValueError(f"Volume must be positive, got {self.volume}")
        
        # Validate oil type
        if self.oil_type not in self.VALID_OIL_TYPES:
            raise ValueError(
                f"Oil type must be one of {self.VALID_OIL_TYPES}, got {self.oil_type}"
            )
        
        # Validate duration
        if self.duration <= 0:
            raise ValueError(f"Duration must be positive, got {self.duration}")
        
        # Validate timestep
        if self.timestep <= 0:
            raise ValueError(f"Timestep must be positive, got {self.timestep}")
        
        # Validate particle count
        if self.particle_count <= 0:
            raise ValueError(f"Particle count must be positive, got {self.particle_count}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        config_dict = asdict(self)
        # Convert datetime to ISO format string
        config_dict['start_time'] = self.start_time.isoformat()
        return config_dict
    
    def to_json(self) -> str:
        """Convert the configuration to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SpillConfig':
        """Create a SpillConfig from a dictionary."""
        # Handle datetime conversion
        if 'start_time' in config_dict and isinstance(config_dict['start_time'], str):
            config_dict['start_time'] = datetime.datetime.fromisoformat(
                config_dict['start_time']
            )
        
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SpillConfig':
        """Create a SpillConfig from a JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    def get_timesteps(self) -> int:
        """Calculate the number of timesteps in the simulation."""
        return int(self.duration * 3600 / self.timestep)
    
    def get_end_time(self) -> datetime.datetime:
        """Calculate the end time of the simulation."""
        return self.start_time + datetime.timedelta(hours=self.duration)
    
    def get_timestep_hours(self) -> float:
        """Convert the timestep from seconds to hours."""
        return self.timestep / 3600


@dataclass
class EnvironmentalData:
    """
    Environmental data for an oil spill simulation.
    
    This class stores and manages environmental data such as wind vectors,
    ocean currents, elevation data, and oil properties. It provides methods
    for data validation, interpolation, and retrieval.
    
    Attributes:
        wind_data: Dictionary containing wind vectors and metadata
        ocean_currents: Dictionary containing ocean current vectors and metadata
        elevation_data: Dictionary containing elevation data and metadata
        oil_properties: Dictionary containing oil physical properties
        domain: Tuple of (min_lat, max_lat, min_lon, max_lon) defining the spatial domain
        time_range: Tuple of (start_time, end_time) defining the temporal domain
    """
    
    wind_data: Dict[str, Any]
    ocean_currents: Dict[str, Any]
    elevation_data: Dict[str, Any]
    oil_properties: Dict[str, Any]
    domain: Tuple[float, float, float, float]
    time_range: Tuple[datetime.datetime, datetime.datetime]
    
    def __post_init__(self):
        """Validate the environmental data after initialization."""
        self._validate()
        self._prepare_interpolators()
    
    def _validate(self):
        """Validate the environmental data."""
        # Validate domain
        min_lat, max_lat, min_lon, max_lon = self.domain
        if not (-90 <= min_lat <= 90) or not (-90 <= max_lat <= 90):
            raise ValueError(f"Latitude must be between -90 and 90")
        if not (-180 <= min_lon <= 180) or not (-180 <= max_lon <= 180):
            raise ValueError(f"Longitude must be between -180 and 180")
        if min_lat >= max_lat:
            raise ValueError(f"min_lat must be less than max_lat")
        if min_lon >= max_lon:
            raise ValueError(f"min_lon must be less than max_lon")
        
        # Validate time range
        start_time, end_time = self.time_range
        if start_time >= end_time:
            raise ValueError(f"start_time must be before end_time")
        
        # Validate wind data structure
        required_wind_keys = ['times', 'lats', 'lons', 'u_component', 'v_component']
        for key in required_wind_keys:
            if key not in self.wind_data:
                raise ValueError(f"Wind data missing required key: {key}")
        
        # Validate ocean currents structure
        required_current_keys = ['times', 'lats', 'lons', 'depths', 'u_component', 'v_component']
        for key in required_current_keys:
            if key not in self.ocean_currents:
                raise ValueError(f"Ocean current data missing required key: {key}")
        
        # Validate elevation data structure
        required_elevation_keys = ['lats', 'lons', 'elevation']
        for key in required_elevation_keys:
            if key not in self.elevation_data:
                raise ValueError(f"Elevation data missing required key: {key}")
        
        # Validate oil properties
        required_oil_keys = ['density', 'viscosity', 'surface_tension', 'evaporation_rate']
        for key in required_oil_keys:
            if key not in self.oil_properties:
                raise ValueError(f"Oil properties missing required key: {key}")
    
    def _prepare_interpolators(self):
        """Prepare interpolation functions for the environmental data."""
        # This is just a placeholder - actual implementation would use scipy's interpolation
        # functions, but we'll need to import numpy and scipy first
        self._wind_interpolator = None
        self._current_interpolator = None
        self._elevation_interpolator = None
    
    def get_wind_at(self, lat: float, lon: float, time: datetime.datetime) -> Tuple[float, float]:
        """
        Get the wind vector at the specified location and time.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            time: Time for which to retrieve the wind data
            
        Returns:
            Tuple of (u_component, v_component) representing the wind vector
        """
        # Find the nearest grid points (simplified implementation)
        lat_idx = self._find_nearest_index(self.wind_data['lats'], lat)
        lon_idx = self._find_nearest_index(self.wind_data['lons'], lon)
        time_idx = self._find_nearest_time_index(self.wind_data['times'], time)
        
        # Get the wind components at the nearest point
        u = self.wind_data['u_component'][time_idx, lat_idx, lon_idx]
        v = self.wind_data['v_component'][time_idx, lat_idx, lon_idx]
        
        return u, v
    
    def get_current_at(self, lat: float, lon: float, depth: float, time: datetime.datetime) -> Tuple[float, float]:
        """
        Get the ocean current vector at the specified location, depth, and time.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            depth: Depth in meters (positive downward)
            time: Time for which to retrieve the current data
            
        Returns:
            Tuple of (u_component, v_component) representing the current vector
        """
        # Find the nearest grid points (simplified implementation)
        lat_idx = self._find_nearest_index(self.ocean_currents['lats'], lat)
        lon_idx = self._find_nearest_index(self.ocean_currents['lons'], lon)
        depth_idx = self._find_nearest_index(self.ocean_currents['depths'], depth)
        time_idx = self._find_nearest_time_index(self.ocean_currents['times'], time)
        
        # Get the current components at the nearest point
        u = self.ocean_currents['u_component'][time_idx, depth_idx, lat_idx, lon_idx]
        v = self.ocean_currents['v_component'][time_idx, depth_idx, lat_idx, lon_idx]
        
        return u, v
    
    def get_elevation_at(self, lat: float, lon: float) -> float:
        """
        Get the elevation at the specified location.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            
        Returns:
            Elevation in meters
        """
        # Find the nearest grid points (simplified implementation)
        lat_idx = self._find_nearest_index(self.elevation_data['lats'], lat)
        lon_idx = self._find_nearest_index(self.elevation_data['lons'], lon)
        
        # Get the elevation at the nearest point
        elevation = self.elevation_data['elevation'][lat_idx, lon_idx]
        
        return elevation
    
    def get_oil_property(self, property_name: str) -> Any:
        """
        Get the specified oil property.
        
        Args:
            property_name: Name of the oil property to retrieve
            
        Returns:
            Value of the oil property
        """
        if property_name not in self.oil_properties:
            raise ValueError(f"Unknown oil property: {property_name}")
        
        return self.oil_properties[property_name]
    
    def _find_nearest_index(self, array, value):
        """Find the index of the nearest value in the array."""
        return min(range(len(array)), key=lambda i: abs(array[i] - value))
    
    def _find_nearest_time_index(self, times, target_time):
        """Find the index of the nearest time in the array."""
        return min(range(len(times)), key=lambda i: abs((times[i] - target_time).total_seconds()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the environmental data to a dictionary."""
        env_dict = {
            'domain': self.domain,
            'time_range': (self.time_range[0].isoformat(), self.time_range[1].isoformat()),
            'oil_properties': self.oil_properties,
        }
        
        # We don't include the full data arrays in the dictionary
        # as they can be very large. Instead, we include metadata.
        env_dict['wind_data_info'] = {
            'shape': (
                len(self.wind_data['times']),
                len(self.wind_data['lats']),
                len(self.wind_data['lons'])
            ),
            'time_range': (
                self.wind_data['times'][0].isoformat(),
                self.wind_data['times'][-1].isoformat()
            ),
            'spatial_extent': [
                min(self.wind_data['lats']),
                max(self.wind_data['lats']),
                min(self.wind_data['lons']),
                max(self.wind_data['lons'])
            ]
        }
        
        env_dict['ocean_currents_info'] = {
            'shape': (
                len(self.ocean_currents['times']),
                len(self.ocean_currents['depths']),
                len(self.ocean_currents['lats']),
                len(self.ocean_currents['lons'])
            ),
            'time_range': (
                self.ocean_currents['times'][0].isoformat(),
                self.ocean_currents['times'][-1].isoformat()
            ),
            'depth_range': [
                min(self.ocean_currents['depths']),
                max(self.ocean_currents['depths'])
            ],
            'spatial_extent': [
                min(self.ocean_currents['lats']),
                max(self.ocean_currents['lats']),
                min(self.ocean_currents['lons']),
                max(self.ocean_currents['lons'])
            ]
        }
        
        env_dict['elevation_data_info'] = {
            'shape': (
                len(self.elevation_data['lats']),
                len(self.elevation_data['lons'])
            ),
            'spatial_extent': [
                min(self.elevation_data['lats']),
                max(self.elevation_data['lats']),
                min(self.elevation_data['lons']),
                max(self.elevation_data['lons'])
            ]
        }
        
        return env_dict
    
    @classmethod
    def from_data_sources(cls, wind_data_source: Dict[str, Any], 
                         ocean_data_source: Dict[str, Any],
                         elevation_data_source: Dict[str, Any],
                         oil_type: str) -> 'EnvironmentalData':
        """
        Create an EnvironmentalData instance from data source dictionaries.
        
        This is a factory method that creates an EnvironmentalData instance
        from the raw data sources, performing any necessary preprocessing.
        
        Args:
            wind_data_source: Dictionary containing wind data from a data source
            ocean_data_source: Dictionary containing ocean current data from a data source
            elevation_data_source: Dictionary containing elevation data from a data source
            oil_type: Type of oil to use for oil properties
            
        Returns:
            EnvironmentalData instance
        """
        # Process wind data
        wind_data = {
            'times': wind_data_source.get('times', []),
            'lats': wind_data_source.get('latitude', []),
            'lons': wind_data_source.get('longitude', []),
            'u_component': wind_data_source.get('u_component', []),
            'v_component': wind_data_source.get('v_component', [])
        }
        
        # Process ocean current data
        ocean_currents = {
            'times': ocean_data_source.get('times', []),
            'lats': ocean_data_source.get('latitude', []),
            'lons': ocean_data_source.get('longitude', []),
            'depths': ocean_data_source.get('depth', [0.0]),
            'u_component': ocean_data_source.get('u_component', []),
            'v_component': ocean_data_source.get('v_component', [])
        }
        
        # Process elevation data
        elevation_data = {
            'lats': elevation_data_source.get('latitude', []),
            'lons': elevation_data_source.get('longitude', []),
            'elevation': elevation_data_source.get('elevation', [])
        }
        
        # Get oil properties based on oil type
        oil_properties = cls._get_oil_properties(oil_type)
        
        # Determine domain from data
        min_lat = min(min(wind_data['lats']), min(ocean_currents['lats']), min(elevation_data['lats']))
        max_lat = max(max(wind_data['lats']), max(ocean_currents['lats']), max(elevation_data['lats']))
        min_lon = min(min(wind_data['lons']), min(ocean_currents['lons']), min(elevation_data['lons']))
        max_lon = max(max(wind_data['lons']), max(ocean_currents['lons']), max(elevation_data['lons']))
        
        # Determine time range from data
        start_time = min(wind_data['times'][0], ocean_currents['times'][0])
        end_time = max(wind_data['times'][-1], ocean_currents['times'][-1])
        
        return cls(
            wind_data=wind_data,
            ocean_currents=ocean_currents,
            elevation_data=elevation_data,
            oil_properties=oil_properties,
            domain=(min_lat, max_lat, min_lon, max_lon),
            time_range=(start_time, end_time)
        )
    
    @staticmethod
    def _get_oil_properties(oil_type: str) -> Dict[str, Any]:
        """
        Get the physical properties for the specified oil type.
        
        Args:
            oil_type: Type of oil
            
        Returns:
            Dictionary of oil properties
        """
        # Oil properties database (simplified)
        oil_properties_db = {
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
        
        if oil_type not in oil_properties_db:
            raise ValueError(f"Unknown oil type: {oil_type}")
        
        return oil_properties_db[oil_type]


@dataclass
class Particle:
    """
    Representation of an individual oil particle in the simulation.
    
    This class represents a single oil particle with its physical properties
    and state. It provides methods to update the particle's position, velocity,
    and mass based on environmental factors and weathering processes.
    
    Attributes:
        position: Tuple of (latitude, longitude, depth) coordinates
        velocity: Tuple of (u, v, w) velocity components in m/s
        mass: Mass of the particle in kg
        age: Age of the particle in seconds
        decay_factors: Dictionary of weathering parameters
        oil_type: Type of oil
        active: Whether the particle is active in the simulation
    """
    
    position: Tuple[float, float, float]  # (lat, lon, depth)
    velocity: Tuple[float, float, float]  # (u, v, w) in m/s
    mass: float  # kg
    age: float  # seconds
    decay_factors: Dict[str, float]
    oil_type: str
    active: bool = True
    
    def __post_init__(self):
        """Validate the particle parameters after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate the particle parameters."""
        # Validate position
        lat, lon, depth = self.position
        if not (-90 <= lat <= 90):
            raise ValueError(f"Latitude must be between -90 and 90, got {lat}")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Longitude must be between -180 and 180, got {lon}")
        if depth < 0 and self.oil_type not in ['gasoline', 'diesel', 'jet_fuel']:
            # Only light oils can have negative depth (above water)
            raise ValueError(f"Depth must be non-negative for {self.oil_type}, got {depth}")
        
        # Validate mass
        if self.mass <= 0:
            raise ValueError(f"Mass must be positive, got {self.mass}")
        
        # Validate age
        if self.age < 0:
            raise ValueError(f"Age must be non-negative, got {self.age}")
        
        # Validate decay factors
        required_decay_keys = ['evaporation', 'dissolution', 'biodegradation']
        for key in required_decay_keys:
            if key not in self.decay_factors:
                raise ValueError(f"Decay factors missing required key: {key}")
            if not (0 <= self.decay_factors[key] <= 1):
                raise ValueError(f"Decay factor {key} must be between 0 and 1, got {self.decay_factors[key]}")
    
    def update_position(self, timestep: float, environmental_data: 'EnvironmentalData') -> None:
        """
        Update the particle's position based on its velocity and environmental factors.
        
        Args:
            timestep: Time step in seconds
            environmental_data: Environmental data for the simulation
        """
        if not self.active:
            return
        
        # Extract current position
        lat, lon, depth = self.position
        u, v, w = self.velocity
        
        # Calculate position change based on velocity
        # Note: This is a simplified calculation that doesn't account for Earth's curvature
        # For a more accurate calculation, we would use the haversine formula
        # or a proper coordinate transformation
        lat_change = (v / 111000) * timestep  # 1 degree latitude is approximately 111 km
        lon_change = (u / (111000 * math.cos(math.radians(lat)))) * timestep
        depth_change = w * timestep
        
        # Update position
        new_lat = lat + lat_change
        new_lon = lon + lon_change
        new_depth = depth + depth_change
        
        # Ensure position is within valid bounds
        new_lat = max(-90, min(90, new_lat))
        new_lon = max(-180, min(180, new_lon))
        new_depth = max(0, new_depth)  # Depth cannot be negative
        
        # Check if particle has beached (hit land)
        elevation = environmental_data.get_elevation_at(new_lat, new_lon)
        if elevation > 0 and new_depth < elevation:
            # Particle has beached
            new_depth = 0
            self.velocity = (0, 0, 0)  # Stop the particle
        
        # Update position
        self.position = (new_lat, new_lon, new_depth)
    
    def update_velocity(self, timestep: float, environmental_data: 'EnvironmentalData') -> None:
        """
        Update the particle's velocity based on environmental factors.
        
        Args:
            timestep: Time step in seconds
            environmental_data: Environmental data for the simulation
        """
        if not self.active:
            return
        
        # Extract current position
        lat, lon, depth = self.position
        current_time = datetime.datetime.now()  # This should be the simulation time
        
        # Get wind and current vectors at the particle's position
        wind_u, wind_v = environmental_data.get_wind_at(lat, lon, current_time)
        current_u, current_v = environmental_data.get_current_at(lat, lon, depth, current_time)
        
        # Calculate new velocity components
        # Wind influence decreases with depth
        wind_factor = max(0, 1 - depth / 10)  # Wind influence up to 10m depth
        wind_influence = 0.03  # Wind influence factor (3% of wind speed)
        
        # Calculate new velocity
        new_u = current_u + wind_influence * wind_factor * wind_u
        new_v = current_v + wind_influence * wind_factor * wind_v
        
        # Vertical velocity component (simplified)
        # Buoyancy depends on oil density vs water density
        oil_density = environmental_data.get_oil_property('density')
        water_density = 1025  # kg/m^3 (average seawater density)
        
        # Buoyancy force (simplified)
        buoyancy = (water_density - oil_density) / water_density
        
        # Vertical velocity (positive is upward, so negative buoyancy means sinking)
        new_w = buoyancy * 0.1  # Scale factor for vertical velocity
        
        # Update velocity
        self.velocity = (new_u, new_v, new_w)
    
    def update_mass(self, timestep: float, environmental_data: 'EnvironmentalData') -> None:
        """
        Update the particle's mass based on weathering processes.
        
        Args:
            timestep: Time step in seconds
            environmental_data: Environmental data for the simulation
        """
        if not self.active:
            return
        
        # Extract current position and properties
        lat, lon, depth = self.position
        
        # Calculate mass loss due to different weathering processes
        # Convert timestep to days for evaporation calculation
        timestep_days = timestep / (24 * 3600)
        
        # Evaporation (only occurs at the surface)
        evaporation_loss = 0
        if depth <= 0.1:  # Surface layer
            evaporation_rate = environmental_data.get_oil_property('evaporation_rate')
            evaporation_loss = self.mass * evaporation_rate * timestep_days * self.decay_factors['evaporation']
        
        # Dissolution (occurs throughout the water column)
        dissolution_rate = 0.01  # 1% per day, simplified
        dissolution_loss = self.mass * dissolution_rate * timestep_days * self.decay_factors['dissolution']
        
        # Biodegradation (increases with time)
        biodegradation_rate = 0.005  # 0.5% per day, simplified
        biodegradation_factor = min(1, self.age / (30 * 24 * 3600))  # Increases over 30 days
        biodegradation_loss = self.mass * biodegradation_rate * biodegradation_factor * timestep_days * self.decay_factors['biodegradation']
        
        # Calculate total mass loss
        total_loss = evaporation_loss + dissolution_loss + biodegradation_loss
        
        # Update mass, ensuring it doesn't go below zero
        self.mass = max(0, self.mass - total_loss)
        
        # If mass is very small, deactivate the particle
        if self.mass < 1e-6:  # 1 microgram threshold
            self.active = False
    
    def update(self, timestep: float, environmental_data: 'EnvironmentalData') -> None:
        """
        Update the particle's state for the given timestep.
        
        This method updates the particle's position, velocity, mass, and age
        based on environmental factors and weathering processes.
        
        Args:
            timestep: Time step in seconds
            environmental_data: Environmental data for the simulation
        """
        if not self.active:
            return
        
        # Update velocity first, as it depends on the current position
        self.update_velocity(timestep, environmental_data)
        
        # Then update position based on the new velocity
        self.update_position(timestep, environmental_data)
        
        # Update mass based on weathering processes
        self.update_mass(timestep, environmental_data)
        
        # Update age
        self.age += timestep
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the particle to a dictionary."""
        return {
            'position': self.position,
            'velocity': self.velocity,
            'mass': self.mass,
            'age': self.age,
            'decay_factors': self.decay_factors,
            'oil_type': self.oil_type,
            'active': self.active
        }
    
    @classmethod
    def from_dict(cls, particle_dict: Dict[str, Any]) -> 'Particle':
        """Create a Particle from a dictionary."""
        return cls(**particle_dict)
    
    @classmethod
    def create_from_spill(cls, spill_config: SpillConfig, position: Tuple[float, float, float], 
                          mass: float, random_seed: Optional[int] = None) -> 'Particle':
        """
        Create a new particle from a spill configuration.
        
        Args:
            spill_config: Spill configuration
            position: Initial position (lat, lon, depth)
            mass: Initial mass in kg
            random_seed: Random seed for decay factors
            
        Returns:
            New Particle instance
        """
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
        
        # Generate random decay factors
        decay_factors = {
            'evaporation': random.uniform(0.8, 1.0),
            'dissolution': random.uniform(0.8, 1.0),
            'biodegradation': random.uniform(0.8, 1.0)
        }
        
        # Create and return the particle
        return cls(
            position=position,
            velocity=(0, 0, 0),  # Initial velocity is zero
            mass=mass,
            age=0,  # Initial age is zero
            decay_factors=decay_factors,
            oil_type=spill_config.oil_type
        )


@dataclass
class SimulationResults:
    """
    Results of an oil spill simulation.
    
    This class stores and manages the results of an oil spill simulation,
    including particle positions over time, concentration maps, affected areas,
    and summary statistics. It provides methods for analyzing and visualizing
    the simulation results.
    
    Attributes:
        config: Configuration parameters for the simulation
        particles: List of Particle objects at the final timestep
        particle_history: Dictionary mapping timesteps to lists of particle states
        concentration_maps: Dictionary mapping timesteps to concentration arrays
        affected_areas: GeoJSON-compatible dictionary of affected areas
        statistics: Dictionary of summary statistics
        start_time: Start time of the simulation
        end_time: End time of the simulation
    """
    
    config: SpillConfig
    particles: List[Particle]
    particle_history: Dict[float, List[Dict[str, Any]]]
    concentration_maps: Dict[float, Dict[str, Any]]
    affected_areas: Dict[str, Any]
    statistics: Dict[str, Any]
    start_time: datetime.datetime
    end_time: datetime.datetime
    
    def __post_init__(self):
        """Initialize derived attributes and validate the results."""
        self._validate()
        self._calculate_statistics()
    
    def _validate(self):
        """Validate the simulation results."""
        # Validate time range
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
        
        # Validate particle history
        if not self.particle_history:
            raise ValueError("Particle history cannot be empty")
        
        # Validate concentration maps
        if not self.concentration_maps:
            raise ValueError("Concentration maps cannot be empty")
        
        # Validate affected areas
        if 'type' not in self.affected_areas or self.affected_areas['type'] != 'FeatureCollection':
            raise ValueError("Affected areas must be a GeoJSON FeatureCollection")
        if 'features' not in self.affected_areas:
            raise ValueError("Affected areas must have a 'features' property")
    
    def _calculate_statistics(self):
        """Calculate summary statistics for the simulation results."""
        # Some statistics may already be provided, so we'll update the dictionary
        stats = self.statistics.copy()
        
        # Calculate total oil volume at the end of the simulation
        total_mass = sum(p.mass for p in self.particles if p.active)
        stats['total_mass_kg'] = total_mass
        stats['total_volume_m3'] = total_mass / 850  # Approximate conversion using average density
        
        # Calculate percentage of oil evaporated, dissolved, etc.
        initial_mass = self.statistics.get('initial_mass_kg', 0)
        if initial_mass > 0:
            stats['evaporated_percent'] = 100 * (1 - total_mass / initial_mass)
        
        # Calculate affected area in square kilometers
        affected_area_km2 = 0
        for feature in self.affected_areas['features']:
            if 'properties' in feature and 'area_km2' in feature['properties']:
                affected_area_km2 += feature['properties']['area_km2']
        stats['affected_area_km2'] = affected_area_km2
        
        # Calculate maximum concentration
        max_concentration = 0
        for timestep, conc_map in self.concentration_maps.items():
            if 'data' in conc_map:
                max_concentration = max(max_concentration, conc_map['data'].max())
        stats['max_concentration_kg_m3'] = max_concentration
        
        # Update the statistics dictionary
        self.statistics = stats
    
    def get_particle_positions_at_time(self, timestep: float) -> List[Tuple[float, float, float]]:
        """
        Get the positions of all particles at the specified timestep.
        
        Args:
            timestep: Timestep in seconds from the start of the simulation
            
        Returns:
            List of (lat, lon, depth) tuples for each particle
        """
        if timestep not in self.particle_history:
            raise ValueError(f"No data available for timestep {timestep}")
        
        return [tuple(p['position']) for p in self.particle_history[timestep]]
    
    def get_active_particle_count_at_time(self, timestep: float) -> int:
        """
        Get the number of active particles at the specified timestep.
        
        Args:
            timestep: Timestep in seconds from the start of the simulation
            
        Returns:
            Number of active particles
        """
        if timestep not in self.particle_history:
            raise ValueError(f"No data available for timestep {timestep}")
        
        return sum(1 for p in self.particle_history[timestep] if p.get('active', True))
    
    def get_concentration_at_time(self, timestep: float) -> Dict[str, Any]:
        """
        Get the concentration map at the specified timestep.
        
        Args:
            timestep: Timestep in seconds from the start of the simulation
            
        Returns:
            Dictionary containing the concentration map data and metadata
        """
        if timestep not in self.concentration_maps:
            raise ValueError(f"No concentration data available for timestep {timestep}")
        
        return self.concentration_maps[timestep]
    
    def get_affected_areas_at_time(self, timestep: float) -> Dict[str, Any]:
        """
        Get the affected areas at the specified timestep.
        
        Args:
            timestep: Timestep in seconds from the start of the simulation
            
        Returns:
            GeoJSON-compatible dictionary of affected areas
        """
        # This is a simplified implementation that returns the overall affected areas
        # A more complete implementation would filter features based on the timestep
        return self.affected_areas
    
    def get_timesteps(self) -> List[float]:
        """
        Get a list of all timesteps in the simulation.
        
        Returns:
            List of timesteps in seconds from the start of the simulation
        """
        return sorted(self.particle_history.keys())
    
    def get_total_simulation_time(self) -> float:
        """
        Get the total simulation time in seconds.
        
        Returns:
            Total simulation time in seconds
        """
        return (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the simulation results to a dictionary."""
        # Convert config to dictionary
        config_dict = self.config.to_dict()
        
        # Convert particles to dictionaries
        particles_dict = [p.to_dict() for p in self.particles]
        
        # Create the results dictionary
        results_dict = {
            'config': config_dict,
            'particles': particles_dict,
            'particle_history': self.particle_history,
            'affected_areas': self.affected_areas,
            'statistics': self.statistics,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat()
        }
        
        # We don't include the full concentration maps in the dictionary
        # as they can be very large. Instead, we include metadata.
        results_dict['concentration_maps_info'] = {
            'timesteps': list(self.concentration_maps.keys()),
            'shape': next(iter(self.concentration_maps.values()))['shape'] if self.concentration_maps else None,
            'max_concentration': self.statistics.get('max_concentration_kg_m3', 0)
        }
        
        return results_dict
    
    def to_json(self) -> str:
        """Convert the simulation results to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, results_dict: Dict[str, Any]) -> 'SimulationResults':
        """Create a SimulationResults instance from a dictionary."""
        # Convert config dictionary to SpillConfig
        config = SpillConfig.from_dict(results_dict['config'])
        
        # Convert particle dictionaries to Particle objects
        particles = [Particle.from_dict(p) for p in results_dict['particles']]
        
        # Convert time strings to datetime objects
        start_time = datetime.datetime.fromisoformat(results_dict['start_time'])
        end_time = datetime.datetime.fromisoformat(results_dict['end_time'])
        
        # Create and return the SimulationResults instance
        return cls(
            config=config,
            particles=particles,
            particle_history=results_dict['particle_history'],
            concentration_maps=results_dict.get('concentration_maps', {}),
            affected_areas=results_dict['affected_areas'],
            statistics=results_dict['statistics'],
            start_time=start_time,
            end_time=end_time
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SimulationResults':
        """Create a SimulationResults instance from a JSON string."""
        results_dict = json.loads(json_str)
        return cls.from_dict(results_dict)
    
    def export_to_geojson(self, filepath: str) -> None:
        """
        Export the affected areas to a GeoJSON file.
        
        Args:
            filepath: Path to the output file
        """
        with open(filepath, 'w') as f:
            json.dump(self.affected_areas, f, indent=2)
    
    def export_particle_tracks_to_csv(self, filepath: str) -> None:
        """
        Export the particle tracks to a CSV file.
        
        Args:
            filepath: Path to the output file
        """
        # This is a simplified implementation
        # A more complete implementation would use pandas or csv module
        with open(filepath, 'w') as f:
            # Write header
            f.write('timestep,particle_id,latitude,longitude,depth,mass,active\n')
            
            # Write data for each timestep and particle
            for timestep in sorted(self.particle_history.keys()):
                for i, particle in enumerate(self.particle_history[timestep]):
                    pos = particle['position']
                    f.write(f"{timestep},{i},{pos[0]},{pos[1]},{pos[2]},{particle.get('mass', 0)},{particle.get('active', True)}\n")
    
    def export_concentration_to_netcdf(self, filepath: str) -> None:
        """
        Export the concentration maps to a NetCDF file.
        
        Args:
            filepath: Path to the output file
        """
        # This is a placeholder - actual implementation would use netCDF4 or xarray
        # but we'll need to import those libraries first
        pass
