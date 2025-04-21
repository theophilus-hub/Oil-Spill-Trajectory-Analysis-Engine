"""
Modeling engine for the Oil Spill Trajectory Analysis Engine.

This module implements the core simulation algorithms:
- Water-based Lagrangian particle model
- Land-based downhill slope or cost-distance flow
- Diffusion, evaporation, and decay factors
- Time-stepped simulation with configurable parameters
"""

import numpy as np
import logging
import uuid
from typing import Dict, Any, List, Tuple, Optional, Callable, Union, Protocol
from datetime import datetime, timedelta
from dataclasses import dataclass

from . import config
from . import preprocess

logger = logging.getLogger(__name__)

class OilSpillModel:
    """Base class for oil spill trajectory modeling."""
    
    def __init__(self, simulation_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the oil spill model.
        
        Args:
            simulation_params: Dictionary of simulation parameters
                If None, default parameters from config will be used
        """
        # Use default params if none provided
        if simulation_params is None:
            self.params = config.DEFAULT_SIMULATION_PARAMS.copy()
        else:
            # Start with defaults and update with provided params
            self.params = config.DEFAULT_SIMULATION_PARAMS.copy()
            self.params.update(simulation_params)
        
        # Initialize state variables
        self.particles = []
        self.timestep = 0
        self.current_time = None
        self.start_time = None
        self.end_time = None
        self.wind_data = None
        self.current_data = None
        self.elevation_data = None
        self.preprocessor = preprocess.DataPreprocessor()
    
    def initialize(self, preprocessed_data: Dict[str, Any]) -> None:
        """
        Initialize the model with preprocessed data.
        
        Args:
            preprocessed_data: Output from preprocess.preprocess_all_data()
        """
        # Convert raw particle data to Particle objects
        raw_particles = preprocessed_data.get('raw', {}).get('particles', [])
        if not raw_particles and 'particles' in preprocessed_data:
            raw_particles = preprocessed_data['particles']
            
        self.particles = [Particle.from_dict(p) for p in raw_particles]
        
        # Set up environmental data
        if 'raw' in preprocessed_data:
            # Use raw data if available
            self.wind_data = preprocessed_data['raw']['wind']
            self.current_data = preprocessed_data['raw']['currents']
            self.elevation_data = preprocessed_data['raw']['elevation']
        else:
            # Fall back to direct data
            self.wind_data = preprocessed_data.get('wind', {})
            self.current_data = preprocessed_data.get('currents', {})
            self.elevation_data = preprocessed_data.get('elevation', {})
        
        # Set up normalized data for efficient computation
        self.normalized_data = preprocessed_data.get('normalized', {})
        
        # Set up simulation time
        if 'metadata' in preprocessed_data and 'start_time' in preprocessed_data['metadata']:
            # Use provided start time if available
            start_time_str = preprocessed_data['metadata']['start_time']
            try:
                self.start_time = datetime.fromisoformat(start_time_str)
            except ValueError:
                self.start_time = datetime.now()
        else:
            self.start_time = datetime.now()
            
        self.current_time = self.start_time
        self.end_time = self.start_time + timedelta(
            hours=self.params['duration_hours']
        )
        
        # Reset timestep counter
        self.timestep = 0
        
        # Set up additional model state
        self.shoreline_data = preprocessed_data.get('shoreline', None)
        self.bathymetry_data = preprocessed_data.get('bathymetry', None)
        
        logger.info(f"Lagrangian model initialized with {len(self.particles)} particles")
    
    def run_simulation(self, save_interval: int = 1, progress_interval: int = 10) -> Dict[str, Any]:
        """Run the simulation from start to end time.
        
        Args:
            save_interval: Save particle states every N timesteps
            progress_interval: Log progress every N timesteps
            
        Returns:
            Dictionary containing simulation results
        """
        # Initialize results dictionary
        results = {
            'metadata': {
                'model_type': self.__class__.__name__,
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'timestep_minutes': self.params['timestep_minutes'],
                'particle_count': len(self.particles),
                'parameters': self.params
            },
            'timesteps': [],
            'final_status': {},
            'runtime_seconds': 0,
            'steps_per_second': 0,
            'mass_balance': {}
        }
        
        # Record initial state
        initial_state = self._get_current_state()
        results['timesteps'].append(initial_state)
        
        # Record start time for performance measurement
        start_time = datetime.now()
        
        # Calculate total number of timesteps
        total_minutes = (self.end_time - self.start_time).total_seconds() / 60
        total_timesteps = int(total_minutes / self.params['timestep_minutes'])
        
        logger.info(f"Starting simulation with {len([p for p in self.particles if p.status == 'active'])} active particles")
        logger.info(f"Simulation period: {self.start_time.isoformat()} to {self.end_time.isoformat()}")
        logger.info(f"Timestep: {self.params['timestep_minutes']} minutes, Total timesteps: {total_timesteps}")
        
        # Run simulation
        while self.current_time < self.end_time:
            # Advance one timestep
            self.step()
            
            # Save state at specified intervals
            if self.timestep % save_interval == 0:
                state = self._get_current_state()
                results['timesteps'].append(state)
            
            # Log progress at specified intervals
            if self.timestep % progress_interval == 0:
                active_count = len([p for p in self.particles if p.status == 'active'])
                logger.info(f"Timestep {self.timestep}/{total_timesteps} - {active_count} active particles")
        
        # Record end time and calculate performance metrics
        end_time = datetime.now()
        runtime_seconds = (end_time - start_time).total_seconds()
        results['runtime_seconds'] = runtime_seconds
        results['steps_per_second'] = self.timestep / max(1, runtime_seconds)  # Avoid division by zero
        
        # Record final particle status counts
        status_counts = {}
        for particle in self.particles:
            status = particle.status
            if status not in status_counts:
                status_counts[status] = 0
            status_counts[status] += 1
        results['final_status'] = status_counts
        
        # Calculate mass balance
        initial_mass = sum(p.mass for p in self.particles)
        remaining_mass = sum(p.mass for p in self.particles if p.status == 'active')
        weathered_mass = initial_mass - remaining_mass
        weathered_percent = (weathered_mass / initial_mass) * 100 if initial_mass > 0 else 0
        
        results['mass_balance'] = {
            'initial_mass': initial_mass,
            'remaining_mass': remaining_mass,
            'weathered_mass': weathered_mass,
            'weathered_percent': weathered_percent
        }
        
        # Add model-specific summary metrics
        model_name = self.__class__.__name__
        if model_name == 'LandFlowModel' and hasattr(self, '_calculate_terrain_metrics'):
            # Add land-specific summary metrics
            results['land_summary'] = {
                'average_elevation': self._calculate_average_elevation(),
                'average_slope': self._calculate_average_slope(),
                'soil_distribution': self._calculate_soil_distribution(),
                'absorbed_count': len([p for p in self.particles if p.status == 'absorbed']),
                'average_absorption_rate': self._calculate_average_absorption_rate(),
                'terrain_metrics': self._calculate_terrain_metrics()
            }
        elif model_name == 'LagrangianModel' and hasattr(self, '_calculate_average_depth'):
            # Add water-specific summary metrics
            results['water_summary'] = {
                'beached_count': len([p for p in self.particles if p.status == 'beached']),
                'average_depth': self._calculate_average_depth(),
                'evaporated_percent': self._calculate_evaporated_percent() if hasattr(self, '_calculate_evaporated_percent') else 0.0,
                'dispersed_percent': self._calculate_dispersed_percent() if hasattr(self, '_calculate_dispersed_percent') else 0.0
            }
        elif model_name == 'HybridModel':
            # Add hybrid-specific summary metrics
            results['hybrid_summary'] = {
                'land_count': len([p for p in self.particles if hasattr(p, 'surface_type') and p.surface_type == 'land']),
                'water_count': len([p for p in self.particles if hasattr(p, 'surface_type') and p.surface_type == 'water']),
                'transition_count': len([p for p in self.particles if p.status in ['transition_to_water', 'transition_to_land']])
            }
        
        logger.info(f"Simulation complete in {runtime_seconds:.2f} seconds")
        logger.info(f"Performance: {results['steps_per_second']:.2f} timesteps/second")
        
        return results
    
    def step(self) -> None:
        """Advance the simulation by one timestep."""
        # This will be implemented by subclasses
        raise NotImplementedError("Subclasses must implement step()")
    
    def _get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of the simulation.
        
        Returns:
            Dictionary with current particle states and timestamp
        """
        return {
            'timestep': self.timestep,
            'time': self.current_time.isoformat(),
            'particles': [p.to_dict() for p in self.particles]
        }


@dataclass
class Particle:
    """Class representing an oil particle in the Lagrangian model."""
    id: str                      # Unique identifier for the particle
    latitude: float              # Current latitude position
    longitude: float             # Current longitude position
    depth: float                 # Depth in meters (0 = surface)
    mass: float                  # Mass of oil in kg
    status: str                  # Status: 'active', 'beached', 'evaporated', etc.
    age: float                   # Age in hours since release
    velocity: Tuple[float, float] = (0.0, 0.0)  # Current velocity vector (u, v) in m/s
    surface_type: str = 'water'  # Current surface type ('water' or 'land')
    oil_type: Optional[str] = None  # Type of oil (affects weathering)
    weathering_state: Dict[str, float] = None  # State of various weathering processes
    
    def __post_init__(self):
        """Initialize any fields that need post-processing."""
        if self.weathering_state is None:
            self.weathering_state = {
                'evaporated_fraction': 0.0,  # Fraction of original mass evaporated
                'dissolved_fraction': 0.0,   # Fraction of original mass dissolved
                'dispersed_fraction': 0.0,   # Fraction of original mass dispersed
                'emulsified_fraction': 0.0,  # Fraction of original mass emulsified
                'biodegraded_fraction': 0.0  # Fraction of original mass biodegraded
            }
    
    @property
    def position(self) -> Tuple[float, float]:
        """Get the current position as (latitude, longitude)."""
        return (self.latitude, self.longitude)
    
    @property
    def remaining_mass(self) -> float:
        """Calculate the remaining mass after weathering."""
        total_weathered = sum(self.weathering_state.values())
        return self.mass * (1.0 - min(1.0, total_weathered))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert particle to dictionary representation."""
        return {
            'id': self.id,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'depth': self.depth,
            'mass': self.mass,
            'remaining_mass': self.remaining_mass,
            'status': self.status,
            'age': self.age,
            'velocity': self.velocity,
            'surface_type': self.surface_type,
            'oil_type': self.oil_type,
            'weathering_state': self.weathering_state
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Particle':
        """Create a Particle from a dictionary representation."""
        # Extract required fields
        particle_id = data.get('id', str(uuid.uuid4()))
        latitude = data.get('latitude', 0.0)
        longitude = data.get('longitude', 0.0)
        depth = data.get('depth', 0.0)
        mass = data.get('mass', 1.0)
        status = data.get('status', 'active')
        age = data.get('age', 0.0)
        
        # Extract optional fields
        velocity = data.get('velocity', (0.0, 0.0))
        surface_type = data.get('surface_type', 'water')
        oil_type = data.get('oil_type')
        weathering_state = data.get('weathering_state')
        
        return cls(
            id=particle_id,
            latitude=latitude,
            longitude=longitude,
            depth=depth,
            mass=mass,
            status=status,
            age=age,
            velocity=velocity,
            surface_type=surface_type,
            oil_type=oil_type,
            weathering_state=weathering_state
        )


class LagrangianModel(OilSpillModel):
    """Lagrangian particle model for water-based oil spill simulation."""
    
    def __init__(self, simulation_params: Optional[Dict[str, Any]] = None):
        """Initialize the Lagrangian model."""
        super().__init__(simulation_params)
        
        # Additional parameters for Lagrangian model
        self.wind_influence_factor = self.params.get('wind_influence_factor', 0.03)
        self.diffusion_coefficient = self.params.get('diffusion_coefficient', 10.0)  # m²/s
        self.random_seed = self.params.get('random_seed', None)
        
        # Set random seed if provided
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
    
    def step(self) -> None:
        """Advance the water-based simulation by one timestep."""
        # Increment timestep counter
        self.timestep += 1
        
        # Update current time
        self.current_time += timedelta(minutes=self.params['timestep_minutes'])
        
        # Calculate timestep in seconds for physics calculations
        dt_seconds = self.params['timestep_minutes'] * 60
        
        # Process each particle
        for particle in self.particles:
            # Skip inactive particles
            if particle.status != 'active':
                continue
            
            # Apply advection (movement due to currents and wind)
            self._apply_advection(particle, dt_seconds)
            
            # Apply diffusion (random spreading)
            self._apply_diffusion(particle, dt_seconds)
            
            # Apply weathering processes
            self._apply_weathering(particle, dt_seconds)
            
            # Update particle age
            particle.age += self.params['timestep_minutes'] / 60  # Age in hours
            
            # Check for beaching or other boundary conditions
            self._check_boundaries(particle)
    
    def _apply_advection(self, particle: Particle, dt_seconds: float) -> None:
        """Apply advection from currents and wind to a particle.
        
        Args:
            particle: Particle to update
            dt_seconds: Timestep in seconds
        """
        # Get current location
        lat, lon = particle.latitude, particle.longitude
        
        # Get environmental forces at this location
        wind_vector = self.preprocessor.interpolate_wind_to_location(
            self.wind_data, (lat, lon), self.current_time
        )
        
        current_vector = self.preprocessor.interpolate_currents_to_location(
            self.current_data, (lat, lon), particle.depth, self.current_time
        )
        
        # Calculate wind influence based on depth
        # Wind influence decreases with depth following an exponential decay
        depth_factor = np.exp(-particle.depth / self.params.get('wind_depth_decay', 0.5))
        effective_wind_factor = self.wind_influence_factor * depth_factor
        
        # Calculate Stokes drift (wave-induced transport)
        # Typically about 1-3% of wind speed at the surface
        stokes_drift_factor = self.params.get('stokes_drift_factor', 0.02) * depth_factor
        stokes_u = wind_vector[0] * stokes_drift_factor
        stokes_v = wind_vector[1] * stokes_drift_factor
        
        # Combine forces (wind drift + currents + Stokes drift)
        u_velocity = (wind_vector[0] * effective_wind_factor) + current_vector[0] + stokes_u
        v_velocity = (wind_vector[1] * effective_wind_factor) + current_vector[1] + stokes_v
        
        # Apply inertia from previous velocity (momentum)
        inertia_factor = self.params.get('inertia_factor', 0.2)
        u_velocity = (particle.velocity[0] * inertia_factor) + (u_velocity * (1 - inertia_factor))
        v_velocity = (particle.velocity[1] * inertia_factor) + (v_velocity * (1 - inertia_factor))
        
        # Update particle velocity
        particle.velocity = (u_velocity, v_velocity)
        
        # Convert velocity to position change
        # Approximate conversion from m/s to degrees
        # 1 degree latitude ≈ 111 km, 1 degree longitude ≈ 111 km * cos(latitude)
        # For small timesteps and areas, this approximation is reasonable
        lat_change = (v_velocity * dt_seconds) / (111000)  # North-South
        lon_change = (u_velocity * dt_seconds) / (111000 * np.cos(np.radians(lat)))  # East-West
        
        # Update position
        particle.latitude += lat_change
        particle.longitude += lon_change
    
    def _apply_diffusion(self, particle: Particle, dt_seconds: float) -> None:
        """Apply diffusion (random spreading) to a particle.
        
        Args:
            particle: Particle to update
            dt_seconds: Timestep in seconds
        """
        # Get base diffusion coefficient (horizontal diffusion in m²/s)
        base_diffusion = self.diffusion_coefficient
        
        # Scale diffusion based on environmental conditions
        # Higher wind speeds increase turbulent diffusion
        wind_vector = self.preprocessor.interpolate_wind_to_location(
            self.wind_data, (particle.latitude, particle.longitude), self.current_time
        )
        wind_speed = np.sqrt(wind_vector[0]**2 + wind_vector[1]**2)
        
        # Richardson's 4/3 power law for scale-dependent diffusion
        # D(L) = ε^(1/3) * L^(4/3) where ε is energy dissipation rate
        # We approximate this by scaling diffusion with particle age
        # Older particles have spread more, leading to larger effective diffusion length scales
        scale_factor = max(1.0, (particle.age / 24.0) ** 0.25)  # Scale with fourth root of age in days
        
        # Wind-enhanced diffusion
        wind_enhancement = 1.0 + (wind_speed / 10.0)  # 10 m/s wind doubles diffusion
        
        # Combine factors
        effective_diffusion = base_diffusion * scale_factor * wind_enhancement
        
        # Apply horizontal diffusion using random walk model
        # Variance = 2 * D * dt for each dimension
        variance = 2.0 * effective_diffusion * dt_seconds
        std_dev = np.sqrt(variance)
        
        # Generate random displacements with correlation
        # Correlated random walk: next step depends partly on previous step
        # This creates more realistic particle paths than pure random walk
        memory_factor = self.params.get('random_walk_memory', 0.3)
        
        # Get previous random components (if stored)
        prev_random_u = getattr(particle, '_prev_random_u', 0.0)
        prev_random_v = getattr(particle, '_prev_random_v', 0.0)
        
        # Generate new random components
        new_random_u = np.random.normal(0, std_dev)
        new_random_v = np.random.normal(0, std_dev)
        
        # Combine previous and new components
        random_u = (memory_factor * prev_random_u) + ((1 - memory_factor) * new_random_u)
        random_v = (memory_factor * prev_random_v) + ((1 - memory_factor) * new_random_v)
        
        # Store for next step
        setattr(particle, '_prev_random_u', random_u)
        setattr(particle, '_prev_random_v', random_v)
        
        # Convert to degrees (same as in advection)
        lat, lon = particle.latitude, particle.longitude
        lat_change = random_v / 111000  # North-South
        lon_change = random_u / (111000 * np.cos(np.radians(lat)))  # East-West
        
        # Apply diffusion to position
        particle.latitude += lat_change
        particle.longitude += lon_change
        
        # Apply vertical diffusion (if particle is not at surface)
        if particle.depth > 0:
            # Vertical diffusion is typically smaller than horizontal
            vertical_diffusion = self.params.get('vertical_diffusion_coefficient', base_diffusion / 10.0)
            vertical_variance = 2.0 * vertical_diffusion * dt_seconds
            vertical_std_dev = np.sqrt(vertical_variance)
            
            # Generate random vertical displacement
            depth_change = np.random.normal(0, vertical_std_dev)
            
            # Update depth (prevent negative depths)
            particle.depth = max(0.0, particle.depth + depth_change)
    
    def _apply_weathering(self, particle: Particle, dt_seconds: float) -> None:
        """Apply weathering processes to a particle.
        
        Args:
            particle: Particle to update
            dt_seconds: Timestep in seconds
        """
        # Skip if particle is not active
        if particle.status != 'active':
            return
        
        # Convert timestep to hours for weathering calculations
        dt_hours = dt_seconds / 3600.0
        
        # Get environmental conditions at particle location
        location = (particle.latitude, particle.longitude)
        wind_vector = self.preprocessor.interpolate_wind_to_location(
            self.wind_data, location, self.current_time
        )
        wind_speed = np.sqrt(wind_vector[0]**2 + wind_vector[1]**2)
        
        # Get water temperature if available
        water_temp = 15.0  # Default to 15°C if not available
        if hasattr(self, 'water_temp_data') and self.water_temp_data:
            # This would use a similar interpolation as wind/currents
            pass
        
        # 1. Evaporation
        # Evaporation depends on oil type, temperature, wind speed, and surface area
        # We use a simplified first-order decay model: dM/dt = -k * M
        # where k is the evaporation rate constant
        
        # Base evaporation rate (fraction per hour)
        base_evap_rate = self.params.get('base_evaporation_rate', 0.05)
        
        # Adjust for temperature (doubles every 10°C increase)
        temp_factor = 2.0 ** ((water_temp - 15.0) / 10.0)
        
        # Adjust for wind (increases with wind speed)
        wind_factor = 1.0 + (wind_speed / 5.0)  # 5 m/s wind doubles evaporation
        
        # Adjust for oil type if available
        oil_factor = 1.0
        if particle.oil_type:
            # Different oils evaporate at different rates
            # Light oils evaporate faster than heavy oils
            if particle.oil_type.lower() in ['gasoline', 'light_crude']:
                oil_factor = 2.0
            elif particle.oil_type.lower() in ['heavy_crude', 'bunker']:
                oil_factor = 0.5
        
        # Calculate effective evaporation rate
        evap_rate = base_evap_rate * temp_factor * wind_factor * oil_factor
        
        # Apply evaporation (limit to remaining non-evaporated fraction)
        current_evap = particle.weathering_state['evaporated_fraction']
        max_evaporable = self.params.get('max_evaporable_fraction', 0.3)  # Maximum fraction that can evaporate
        
        if current_evap < max_evaporable:
            # Calculate new evaporation using exponential decay
            new_evap = current_evap + (max_evaporable - current_evap) * (1.0 - np.exp(-evap_rate * dt_hours))
            particle.weathering_state['evaporated_fraction'] = min(max_evaporable, new_evap)
        
        # 2. Dissolution
        # Water-soluble components dissolve into the water column
        base_dissolution_rate = self.params.get('base_dissolution_rate', 0.005)  # fraction per hour
        
        # Dissolution decreases with age as soluble components are depleted
        age_factor = np.exp(-particle.age / 24.0)  # Exponential decrease with age in days
        
        # Calculate effective dissolution rate
        dissolution_rate = base_dissolution_rate * age_factor
        
        # Apply dissolution
        current_dissolution = particle.weathering_state['dissolved_fraction']
        max_dissolvable = self.params.get('max_dissolvable_fraction', 0.2)
        
        if current_dissolution < max_dissolvable:
            new_dissolution = current_dissolution + (max_dissolvable - current_dissolution) * \
                             (1.0 - np.exp(-dissolution_rate * dt_hours))
            particle.weathering_state['dissolved_fraction'] = min(max_dissolvable, new_dissolution)
        
        # 3. Biodegradation
        # Microbes break down oil components over time
        base_biodeg_rate = self.params.get('base_biodegradation_rate', 0.001)  # fraction per hour
        
        # Biodegradation increases with time as microbial populations grow
        # But limited by available nutrients and oxygen
        if particle.age < 24.0:  # First day has minimal biodegradation
            biodeg_factor = particle.age / 24.0
        else:  # After first day, increases to a plateau
            biodeg_factor = 1.0 + np.log(particle.age / 24.0) / np.log(10.0)  # Log10 scaling
        
        # Temperature effect (doubles every 10°C, but minimal below 5°C)
        biodeg_temp_factor = max(0.1, 2.0 ** ((water_temp - 15.0) / 10.0))
        
        # Calculate effective biodegradation rate
        biodeg_rate = base_biodeg_rate * biodeg_factor * biodeg_temp_factor
        
        # Apply biodegradation
        current_biodeg = particle.weathering_state['biodegraded_fraction']
        max_biodegradable = self.params.get('max_biodegradable_fraction', 0.4)
        
        new_biodeg = current_biodeg + (max_biodegradable - current_biodeg) * \
                    (1.0 - np.exp(-biodeg_rate * dt_hours))
        particle.weathering_state['biodegraded_fraction'] = min(max_biodegradable, new_biodeg)
        
        # Check if particle is no longer significant (most mass lost to weathering)
        total_weathered = sum(particle.weathering_state.values())
        if total_weathered > self.params.get('weathering_threshold', 0.95):
            particle.status = 'weathered'
    
    def _check_boundaries(self, particle: Particle) -> None:
        """Check and handle boundary conditions for a particle.
        
        Args:
            particle: Particle to update
        """
        # Skip if particle is not active
        if particle.status != 'active':
            return
            
        # 1. Check for shoreline interaction (beaching)
        if self._is_on_shoreline(particle):
            self._handle_beaching(particle)
            return
            
        # 2. Check for domain boundaries
        if self._is_outside_domain(particle):
            self._handle_domain_boundary(particle)
            
        # 3. Check for bathymetry constraints (e.g., shallow water)
        self._check_bathymetry_constraints(particle)
    
    def _is_on_shoreline(self, particle: Particle) -> bool:
        """Determine if a particle is on or near a shoreline.
        
        Args:
            particle: Particle to check
            
        Returns:
            True if particle is on shoreline, False otherwise
        """
        # If we have shoreline data, use it
        if self.shoreline_data is not None:
            # Check if particle location intersects with shoreline
            # This would use spatial indexing for efficiency in a real implementation
            # For now, we'll use a simple placeholder
            return False
        
        # If we have elevation data, we can use it to determine land/water boundaries
        if self.elevation_data is not None:
            # Check if the elevation at this location is above sea level
            # This would require interpolating the elevation at the particle location
            # For now, we'll use a simple placeholder
            return False
        
        # If we don't have shoreline or elevation data, use domain boundaries as a fallback
        # This is just a placeholder for demonstration
        domain_bounds = self.params.get('domain_bounds', None)
        if domain_bounds is not None:
            # Check if particle is near domain boundary
            # This is a very simplified approach
            buffer = 0.01  # ~1km buffer
            lat, lon = particle.latitude, particle.longitude
            min_lat, min_lon, max_lat, max_lon = domain_bounds
            
            # Check if particle is near boundary
            near_lat_bound = (abs(lat - min_lat) < buffer) or (abs(lat - max_lat) < buffer)
            near_lon_bound = (abs(lon - min_lon) < buffer) or (abs(lon - max_lon) < buffer)
            
            return near_lat_bound or near_lon_bound
        
        # Default: assume not on shoreline
        return False
    
    def _handle_beaching(self, particle: Particle) -> None:
        """Handle a particle that has reached a shoreline.
        
        Args:
            particle: Particle to update
        """
        # Set particle status to beached
        particle.status = 'beached'
        
        # Stop particle movement
        particle.velocity = (0.0, 0.0)
        
        # Apply shoreline interaction effects
        # Different shoreline types (rocky, sandy, etc.) would have different effects
        # For now, we'll use a simple model
        
        # Increase evaporation rate for beached oil
        # Oil spread on shoreline has more surface area exposed to air
        particle.weathering_state['evaporated_fraction'] += 0.05
        
        # Add some biodegradation
        particle.weathering_state['biodegraded_fraction'] += 0.02
    
    def _is_outside_domain(self, particle: Particle) -> bool:
        """Check if a particle is outside the domain boundaries.
        
        Args:
            particle: Particle to check
            
        Returns:
            True if outside domain, False otherwise
        """
        # Get domain boundaries from parameters
        domain_bounds = self.params.get('domain_bounds', None)
        if domain_bounds is None:
            return False  # No domain boundaries, so particle is never outside
        
        min_lat, min_lon, max_lat, max_lon = domain_bounds
        
        # Check if particle is outside domain
        if (particle.latitude < min_lat or 
            particle.latitude > max_lat or 
            particle.longitude < min_lon or 
            particle.longitude > max_lon):
            return True
        
        return False
    
    def _handle_domain_boundary(self, particle: Particle) -> None:
        """Handle a particle that has reached the domain boundary.
        
        Args:
            particle: Particle to handle
        """
        # Get domain boundaries from parameters
        domain_bounds = self.params.get('domain_bounds', None)
        if domain_bounds is None:
            return
        
        min_lat, min_lon, max_lat, max_lon = domain_bounds
        boundary_method = self.params.get('boundary_method', 'reflect')
        
        if boundary_method == 'reflect':
            # Reflect particle back into domain
            if particle.latitude < min_lat:
                particle.latitude = 2 * min_lat - particle.latitude
                particle.velocity = (particle.velocity[0], -particle.velocity[1])  # Reverse v component
            elif particle.latitude > max_lat:
                particle.latitude = 2 * max_lat - particle.latitude
                particle.velocity = (particle.velocity[0], -particle.velocity[1])  # Reverse v component
                
            if particle.longitude < min_lon:
                particle.longitude = 2 * min_lon - particle.longitude
                particle.velocity = (-particle.velocity[0], particle.velocity[1])  # Reverse u component
            elif particle.longitude > max_lon:
                particle.longitude = 2 * max_lon - particle.longitude
                particle.velocity = (-particle.velocity[0], particle.velocity[1])  # Reverse u component
        
        elif boundary_method == 'absorb':
            # Mark particle as inactive when it leaves the domain
            particle.status = 'out_of_bounds'
        
        elif boundary_method == 'periodic':
            # Wrap around to opposite side (periodic boundary conditions)
            if particle.latitude < min_lat:
                particle.latitude = max_lat - (min_lat - particle.latitude)
            elif particle.latitude > max_lat:
                particle.latitude = min_lat + (particle.latitude - max_lat)
                
            if particle.longitude < min_lon:
                particle.longitude = max_lon - (min_lon - particle.longitude)
            elif particle.longitude > max_lon:
                particle.longitude = min_lon + (particle.longitude - max_lon)
    
    def _check_bathymetry_constraints(self, particle: Particle) -> None:
        """Check and handle bathymetry constraints for a particle.
        
        Args:
            particle: Particle to update
        """
        # If we have bathymetry data, use it
        if self.bathymetry_data is None:
            return
        
        # Get water depth at particle location
        # This would require interpolating the bathymetry at the particle location
        # For now, we'll use a simple placeholder
        water_depth = 100.0  # Placeholder value
        
        # Check if particle depth is greater than water depth
        if particle.depth > water_depth:
            # Constrain particle depth to water depth
            particle.depth = water_depth
            
            # If particle hits bottom, it may get trapped in sediment
    
    def _check_beaching(self, particle: Dict[str, Any]) -> None:
        """
        Check if a particle has reached land and handle beaching.
        
        Args:
            particle: Particle dictionary to update
        """
        # This is a placeholder - in a full implementation, this would
        # check against coastline data or use elevation data to determine if below sea level
        
        # For now, we'll just assume no beaching
        pass


class LandFlowModel(OilSpillModel):
    """Land-based flow model for oil spill simulation using downhill slope approach."""
    
    def __init__(self, simulation_params: Optional[Dict[str, Any]] = None):
        """Initialize the land-based flow model."""
        super().__init__(simulation_params)
        
        # Additional parameters for land-based flow model
        self.flow_resistance_factor = self.params.get('flow_resistance_factor', 0.2)  # Resistance to flow
        self.absorption_rate = self.params.get('absorption_rate', 0.01)  # Fraction absorbed per hour
        self.slope_threshold = self.params.get('slope_threshold', 1.0)  # Minimum slope in degrees for flow
        self.random_seed = self.params.get('random_seed', None)
        
        # Terrain-specific parameters
        self.terrain_roughness_factor = self.params.get('terrain_roughness_factor', 0.3)  # Higher values = more resistance
        self.soil_absorption_factors = self.params.get('soil_absorption_factors', {
            'sand': 0.02,       # High absorption rate
            'clay': 0.005,      # Low absorption rate
            'loam': 0.01,       # Medium absorption rate
            'rock': 0.001,      # Very low absorption rate
            'urban': 0.003,     # Low absorption for urban surfaces
            'vegetation': 0.015  # Medium-high for vegetated areas
        })
        self.vegetation_resistance = self.params.get('vegetation_resistance', 0.4)  # Resistance due to vegetation
        
        # Cost-distance model parameters
        self.flat_terrain_spread_rate = self.params.get('flat_terrain_spread_rate', 0.5)  # m/s on flat terrain
        self.cost_distance_directions = self.params.get('cost_distance_directions', 8)  # Number of directions to consider
        self.use_cost_distance = self.params.get('use_cost_distance', True)  # Whether to use cost-distance for flat areas
        
        # Adaptive time-stepping parameters
        self.use_adaptive_timestep = self.params.get('use_adaptive_timestep', True)  # Whether to use adaptive time steps
        self.max_position_change = self.params.get('max_position_change', 0.001)  # Maximum position change in degrees per substep
        self.max_substeps = self.params.get('max_substeps', 5)  # Maximum number of substeps per main timestep
        self.min_substep_fraction = self.params.get('min_substep_fraction', 0.2)  # Minimum substep size as fraction of main timestep
        self.adaptive_threshold_slope = self.params.get('adaptive_threshold_slope', 15.0)  # Only use adaptive timesteps for slopes > this value
        
        # Performance optimization parameters
        self.use_spatial_index = self.params.get('use_spatial_index', True)  # Whether to use spatial indexing
        self.spatial_index_resolution = self.params.get('spatial_index_resolution', 0.01)  # Grid cell size in degrees
        self.use_terrain_caching = self.params.get('use_terrain_caching', True)  # Whether to cache terrain properties
        self.terrain_cache_size = self.params.get('terrain_cache_size', 10000)  # Maximum number of terrain cache entries
        self.batch_size = self.params.get('batch_size', 1000)  # Number of particles to process in a batch
        
        # Default soil type if not specified
        self.default_soil_type = self.params.get('default_soil_type', 'loam')
        
        # Set random seed if provided
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Initialize spatial index and caches
        self.terrain_cache = {}  # Cache for terrain properties
        self.elevation_cache = {}  # Cache for elevation data
        self.spatial_index = {}  # Spatial index for particles
    
    def initialize(self, preprocessed_data: Dict[str, Any]) -> None:
        """Initialize the model with preprocessed data.
        
        Args:
            preprocessed_data: Output from preprocess.preprocess_all_data()
        """
        # Call parent initialization
        super().initialize(preprocessed_data)
        
        # Clear caches and spatial index
        self.terrain_cache = {}
        self.elevation_cache = {}
        self.spatial_index = {}
        
        # Pre-process elevation data for faster access if available
        if self.elevation_data and 'data' in self.elevation_data and 'grid' in self.elevation_data['data']:
            self._preprocess_elevation_data()
        
        # Build spatial index for initial particles
        if self.use_spatial_index:
            self._build_spatial_index()
    
    def step(self) -> None:
        """Advance the land-based simulation by one timestep."""
        # Increment timestep counter
        self.timestep += 1
        
        # Update current time
        self.current_time += timedelta(minutes=self.params['timestep_minutes'])
        
        # Calculate timestep in seconds for physics calculations
        dt_seconds = self.params['timestep_minutes'] * 60
        
        # Filter active land particles
        active_particles = [p for p in self.particles if p.status == 'active' and p.surface_type == 'land']
        
        # Process particles in batches for better performance
        if len(active_particles) > self.batch_size:
            # Process in batches
            for i in range(0, len(active_particles), self.batch_size):
                batch = active_particles[i:i+self.batch_size]
                self._process_particle_batch(batch, dt_seconds)
        else:
            # Process all particles at once
            self._process_particle_batch(active_particles, dt_seconds)
        
        # Update spatial index if needed
        if self.use_spatial_index:
            self._update_spatial_index()
    
    def _process_particle_batch(self, particles: List[Particle], dt_seconds: float) -> None:
        """Process a batch of particles for one timestep.
        
        Args:
            particles: List of particles to process
            dt_seconds: Timestep in seconds
        """
        # Process each particle in the batch
        for particle in particles:
            # Apply adaptive time-stepping if enabled
            if self.use_adaptive_timestep:
                self._step_with_adaptive_timestep(particle, dt_seconds)
            else:
                # Get elevation and slope at current location
                lat, lon = particle.latitude, particle.longitude
                
                # Get terrain properties at current location
                terrain_properties = self._get_terrain_properties(lat, lon)
                
                # Determine which model to use based on slope
                if terrain_properties['slope'] < self.slope_threshold and self.use_cost_distance:
                    # Use cost-distance model for flat terrain
                    self._apply_cost_distance_model(particle, dt_seconds)
                else:
                    # Use downhill flow for sloped terrain
                    self._apply_downhill_flow(particle, dt_seconds)
            
            # Apply absorption and other land-specific processes
            self._apply_land_processes(particle, dt_seconds)
            
            # Update particle age
            particle.age += self.params['timestep_minutes'] / 60  # Age in hours
            
            # Check for boundaries or transitions
            self._check_boundaries(particle)
    
    def _preprocess_elevation_data(self) -> None:
        """Pre-process elevation data for faster access."""
        # Check if we have valid elevation data
        if not self.elevation_data or 'data' not in self.elevation_data:
            logger.warning("No valid elevation data for preprocessing")
            return
        
        # Get grid data
        grid_data = self.elevation_data['data'].get('grid', [])
        if not grid_data:
            logger.warning("No grid data in elevation data")
            return
        
        # Get bounds
        bounds = self.elevation_data['data'].get('bounds', {})
        if not bounds:
            logger.warning("No bounds in elevation data")
            return
        
        # Create a 2D grid for faster elevation lookups
        try:
            # Extract bounds
            min_lat = bounds.get('min_lat', 0)
            max_lat = bounds.get('max_lat', 0)
            min_lon = bounds.get('min_lon', 0)
            max_lon = bounds.get('max_lon', 0)
            
            # Calculate grid dimensions
            resolution = self.elevation_data['data'].get('resolution', 0.001)  # Default to ~100m
            lat_cells = int((max_lat - min_lat) / resolution) + 1
            lon_cells = int((max_lon - min_lon) / resolution) + 1
            
            # Create empty grids for elevation, slope, and aspect
            elevation_grid = np.full((lat_cells, lon_cells), np.nan)
            slope_grid = np.full((lat_cells, lon_cells), np.nan)
            aspect_grid = np.full((lat_cells, lon_cells), np.nan)
            
            # Fill grids with data
            for point in grid_data:
                if 'latitude' in point and 'longitude' in point and 'elevation' in point:
                    lat, lon = point['latitude'], point['longitude']
                    
                    # Calculate grid indices
                    lat_idx = int((lat - min_lat) / resolution)
                    lon_idx = int((lon - min_lon) / resolution)
                    
                    # Ensure indices are within bounds
                    if 0 <= lat_idx < lat_cells and 0 <= lon_idx < lon_cells:
                        elevation_grid[lat_idx, lon_idx] = point['elevation']
                        
                        # Store slope and aspect if available
                        if 'slope' in point:
                            slope_grid[lat_idx, lon_idx] = point['slope']
                        if 'aspect' in point:
                            aspect_grid[lat_idx, lon_idx] = point['aspect']
            
            # Store preprocessed data
            self.elevation_data['preprocessed'] = {
                'elevation_grid': elevation_grid,
                'slope_grid': slope_grid,
                'aspect_grid': aspect_grid,
                'min_lat': min_lat,
                'max_lat': max_lat,
                'min_lon': min_lon,
                'max_lon': max_lon,
                'resolution': resolution,
                'lat_cells': lat_cells,
                'lon_cells': lon_cells
            }
            
            logger.info(f"Preprocessed elevation data: {lat_cells}x{lon_cells} grid")
        
        except Exception as e:
            logger.error(f"Error preprocessing elevation data: {str(e)}")
    
    def _get_cached_elevation_and_slope(self, latitude: float, longitude: float) -> Tuple[float, float, float]:
        """Get elevation, slope, and aspect with caching for better performance.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Tuple of (elevation, slope, aspect)
        """
        # Check if we have this location in cache
        cache_key = (round(latitude, 6), round(longitude, 6))  # Round to ~10cm precision
        
        if cache_key in self.elevation_cache:
            return self.elevation_cache[cache_key]
        
        # Check if we have preprocessed elevation data
        if 'preprocessed' in self.elevation_data:
            # Try to get from preprocessed grid
            try:
                # Extract preprocessed data
                pp_data = self.elevation_data['preprocessed']
                
                # Check if location is within grid bounds
                if (pp_data['min_lat'] <= latitude <= pp_data['max_lat'] and
                    pp_data['min_lon'] <= longitude <= pp_data['max_lon']):
                    
                    # Calculate grid indices
                    lat_idx = int((latitude - pp_data['min_lat']) / pp_data['resolution'])
                    lon_idx = int((longitude - pp_data['min_lon']) / pp_data['resolution'])
                    
                    # Ensure indices are within bounds
                    if (0 <= lat_idx < pp_data['lat_cells'] and 
                        0 <= lon_idx < pp_data['lon_cells']):
                        
                        # Get values from grid
                        elevation = pp_data['elevation_grid'][lat_idx, lon_idx]
                        slope = pp_data['slope_grid'][lat_idx, lon_idx]
                        aspect = pp_data['aspect_grid'][lat_idx, lon_idx]
                        
                        # If any values are NaN, fall back to regular method
                        if not (np.isnan(elevation) or np.isnan(slope) or np.isnan(aspect)):
                            # Cache and return values
                            result = (float(elevation), float(slope), float(aspect))
                            self.elevation_cache[cache_key] = result
                            return result
            except Exception as e:
                logger.debug(f"Error accessing preprocessed elevation data: {str(e)}")
        
        # Fall back to regular method
        result = self.preprocessor.get_elevation_and_slope(
            self.elevation_data, (latitude, longitude)
        )
        
        # Cache the result
        if self.use_terrain_caching:
            # Limit cache size
            if len(self.elevation_cache) >= self.terrain_cache_size:
                # Remove a random 10% of entries when cache is full
                keys_to_remove = np.random.choice(list(self.elevation_cache.keys()), 
                                                 size=int(self.terrain_cache_size * 0.1),
                                                 replace=False)
                for key in keys_to_remove:
                    self.elevation_cache.pop(key, None)
            
            # Add to cache
            self.elevation_cache[cache_key] = result
        
        return result
    
    def _get_terrain_properties(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Get terrain properties at a specific location with caching.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Dictionary of terrain properties including roughness, soil type, etc.
        """
        # Check if we have this location in cache
        cache_key = (round(latitude, 6), round(longitude, 6))  # Round to ~10cm precision
        
        if self.use_terrain_caching and cache_key in self.terrain_cache:
            return self.terrain_cache[cache_key]
        
        # Initialize with default values
        properties = {
            'roughness': 0.2,               # Default roughness coefficient
            'soil_type': self.default_soil_type,  # Default soil type
            'soil_moisture': 0.5,           # Default soil moisture (0-1)
            'vegetation_density': 0.3,      # Default vegetation density (0-1)
            'temperature': 20.0,            # Default temperature in Celsius
            'channeling': None,             # Terrain channeling effect (0-1)
            'channel_direction': 0.0        # Direction of terrain channels in degrees
        }
        
        # In a real implementation, these properties would be retrieved from
        # environmental data layers (land cover, soil maps, etc.)
        # For now, we'll use simple approximations based on elevation and slope
        
        try:
            # Get elevation and slope
            elevation, slope, aspect = self._get_cached_elevation_and_slope(latitude, longitude)
            
            # Approximate roughness based on slope
            # Steeper slopes often have rougher terrain
            properties['roughness'] = min(0.8, 0.1 + (slope / 45.0) * 0.4)
            
            # Approximate soil type based on elevation
            # This is a very simple heuristic and would be replaced with actual soil data
            if elevation < 0:
                properties['soil_type'] = 'sand'  # Coastal areas
            elif elevation < 100:
                properties['soil_type'] = 'loam'  # Lowlands
            elif elevation < 1000:
                properties['soil_type'] = 'clay'  # Mid elevations
            else:
                properties['soil_type'] = 'rock'  # Mountains
            
            # Approximate vegetation density based on elevation and slope
            # Again, this would be replaced with actual land cover data
            if elevation > 3000 or slope > 60:
                properties['vegetation_density'] = 0.1  # Sparse at high elevations or steep slopes
            elif elevation > 1500:
                properties['vegetation_density'] = 0.4  # Moderate at higher elevations
            elif elevation < 0:
                properties['vegetation_density'] = 0.2  # Sparse in coastal areas
            else:
                properties['vegetation_density'] = 0.7  # Dense at mid elevations
            
            # Check for potential channeling (valleys, drainage patterns)
            # In a real implementation, this would use flow accumulation data
            # For now, use a simple approximation based on slope and aspect
            if slope > 5 and slope < 30:
                # Moderate slopes often have defined drainage patterns
                properties['channeling'] = 0.3
                properties['channel_direction'] = aspect  # Flow follows aspect direction
            
        except Exception as e:
            logger.warning(f"Error retrieving terrain properties: {str(e)}")
        
        # Cache the result
        if self.use_terrain_caching:
            # Limit cache size
            if len(self.terrain_cache) >= self.terrain_cache_size:
                # Remove a random 10% of entries when cache is full
                keys_to_remove = np.random.choice(list(self.terrain_cache.keys()), 
                                                 size=int(self.terrain_cache_size * 0.1),
                                                 replace=False)
                for key in keys_to_remove:
                    self.terrain_cache.pop(key, None)
            
            # Add to cache
            self.terrain_cache[cache_key] = properties.copy()
        
        return properties
    
    def _build_spatial_index(self) -> None:
        """Build a spatial index for particles to optimize spatial queries."""
        if not self.use_spatial_index:
            return
        
        # Clear existing index
        self.spatial_index = {}
        
        # Get active particles
        active_particles = [p for p in self.particles if p.status == 'active']
        
        # Add each particle to the index
        for particle in active_particles:
            self._add_to_spatial_index(particle)
    
    def _update_spatial_index(self) -> None:
        """Update the spatial index after particles have moved."""
        if not self.use_spatial_index:
            return
        
        # Rebuild the index (more efficient than updating individual entries)
        self._build_spatial_index()
    
    def _add_to_spatial_index(self, particle: Particle) -> None:
        """Add a particle to the spatial index.
        
        Args:
            particle: Particle to add to the index
        """
        if not self.use_spatial_index:
            return
        
        # Calculate grid cell indices
        lat_idx = int(particle.latitude / self.spatial_index_resolution)
        lon_idx = int(particle.longitude / self.spatial_index_resolution)
        
        # Create cell key
        cell_key = (lat_idx, lon_idx)
        
        # Add particle to cell
        if cell_key not in self.spatial_index:
            self.spatial_index[cell_key] = []
        
        self.spatial_index[cell_key].append(particle.id)
    
    def _get_particles_in_radius(self, lat: float, lon: float, radius: float) -> List[Particle]:
        """Get all particles within a radius of a location using the spatial index.
        
        Args:
            lat: Latitude of center point
            lon: Longitude of center point
            radius: Radius in degrees
            
        Returns:
            List of particles within the radius
        """
        if not self.use_spatial_index:
            # Fall back to brute force search
            return [p for p in self.particles if p.status == 'active' and 
                   self._haversine_distance(lat, lon, p.latitude, p.longitude) <= radius]
        
        # Calculate grid cell range to search
        lat_radius_cells = int(radius / self.spatial_index_resolution) + 1
        lon_radius_cells = int(radius / self.spatial_index_resolution) + 1
        
        center_lat_idx = int(lat / self.spatial_index_resolution)
        center_lon_idx = int(lon / self.spatial_index_resolution)
        
        # Get all cells within range
        result_particles = []
        
        for lat_offset in range(-lat_radius_cells, lat_radius_cells + 1):
            for lon_offset in range(-lon_radius_cells, lon_radius_cells + 1):
                cell_key = (center_lat_idx + lat_offset, center_lon_idx + lon_offset)
                
                if cell_key in self.spatial_index:
                    # Get particles in this cell
                    particle_ids = self.spatial_index[cell_key]
                    
                    # Find the actual particles
                    for particle in self.particles:
                        if particle.id in particle_ids and particle.status == 'active':
                            # Check actual distance
                            if self._haversine_distance(lat, lon, particle.latitude, particle.longitude) <= radius:
                                result_particles.append(particle)
        
        return result_particles
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points in degrees.
        
        Args:
            lat1: Latitude of first point
            lon1: Longitude of first point
            lat2: Latitude of second point
            lon2: Longitude of second point
            
        Returns:
            Distance in degrees
        """
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Convert back to degrees
        return np.degrees(c)
    
    def _apply_downhill_flow(self, particle: Particle, dt_seconds: float) -> None:
        """Apply downhill flow to a particle based on terrain slope.
        
        Args:
            particle: Particle to update
            dt_seconds: Timestep in seconds
        """
        # Get current location
        lat, lon = particle.latitude, particle.longitude
        
        # Get elevation and slope at current location
        elevation, slope, aspect = self._get_cached_elevation_and_slope(lat, lon)
        
        # Get terrain properties at current location
        terrain_properties = self._get_terrain_properties(lat, lon)
        
        # If slope is below threshold, apply minimal random movement
        # This should not happen if we're using the cost-distance model for flat terrain
        if slope < self.slope_threshold:
            # Apply small random movement for flat areas
            random_direction = np.random.uniform(0, 2 * np.pi)
            random_distance = np.random.uniform(0, 0.5) * dt_seconds / 3600  # Small random distance
            
            # Convert to lat/lon changes (approximate)
            lat_change = random_distance * np.cos(random_direction) / 111000  # North-South
            lon_change = random_distance * np.sin(random_direction) / (111000 * np.cos(np.radians(lat)))  # East-West
            
            # Update position
            particle.latitude += lat_change
            particle.longitude += lon_change
            return
        
        # Calculate flow direction (opposite to aspect)
        # Aspect is in degrees: 0=North, 90=East, 180=South, 270=West
        flow_direction_rad = np.radians((aspect + 180) % 360)
        
        # Calculate flow velocity based on slope
        # Using a simplified version of Manning's equation: v = k * S^0.5
        # where S is slope in m/m and k is a constant based on surface roughness
        slope_rad = np.radians(slope)
        slope_factor = np.sin(slope_rad)  # Convert degrees to slope gradient
        
        # Apply terrain roughness to flow resistance
        # Manning's roughness coefficient varies by terrain type
        effective_resistance = self.flow_resistance_factor
        
        # Adjust resistance based on terrain type
        if terrain_properties['roughness'] is not None:
            effective_resistance += terrain_properties['roughness'] * self.terrain_roughness_factor
        
        # Adjust resistance based on vegetation
        if terrain_properties['vegetation_density'] is not None:
            effective_resistance += terrain_properties['vegetation_density'] * self.vegetation_resistance
        
        # Ensure resistance doesn't exceed 0.95 (would make flow too slow)
        effective_resistance = min(0.95, effective_resistance)
        
        # Base velocity in m/s, adjusted by resistance factor
        base_velocity = np.sqrt(9.81 * slope_factor) * (1.0 - effective_resistance)
        
        # Scale velocity based on particle mass and viscosity
        # Heavier and more viscous oil moves slower
        mass_factor = np.exp(-particle.mass / 10.0)  # Exponential decay with mass
        
        # Adjust for oil viscosity if available
        viscosity_factor = 1.0
        if hasattr(particle, 'oil_properties') and 'viscosity' in particle.oil_properties:
            # Higher viscosity = slower flow
            viscosity = particle.oil_properties['viscosity']
            viscosity_factor = np.exp(-viscosity / 100.0)  # Normalize to reasonable range
        
        velocity = base_velocity * mass_factor * viscosity_factor
        
        # Calculate distance traveled in this timestep
        distance = velocity * dt_seconds  # in meters
        
        # Apply terrain channeling effects (e.g., following drainage patterns)
        if terrain_properties['channeling'] is not None and terrain_properties['channeling'] > 0:
            # Adjust flow direction to follow terrain channels
            channel_direction_rad = np.radians(terrain_properties['channel_direction'])
            channeling_strength = terrain_properties['channeling']
            
            # Blend between slope-based direction and channel direction
            flow_direction_rad = (flow_direction_rad * (1 - channeling_strength) + 
                                 channel_direction_rad * channeling_strength)
        
        # Convert to lat/lon changes (approximate)
        lat_change = (distance * np.cos(flow_direction_rad)) / 111000  # North-South
        lon_change = (distance * np.sin(flow_direction_rad)) / (111000 * np.cos(np.radians(lat)))  # East-West
        
        # Update position
        particle.latitude += lat_change
        particle.longitude += lon_change
        
        # Update particle velocity for visualization
        u_velocity = velocity * np.sin(flow_direction_rad)
        v_velocity = velocity * np.cos(flow_direction_rad)
        particle.velocity = (u_velocity, v_velocity)
    
    def _apply_cost_distance_model(self, particle: Particle, dt_seconds: float) -> None:
        """Apply cost-distance model for flat terrain.
        
        This model simulates oil spreading on flat terrain by considering the cost
        of movement in different directions based on terrain properties.
        
        Args:
            particle: Particle to update
            dt_seconds: Timestep in seconds
        """
        # Get current location
        lat, lon = particle.latitude, particle.longitude
        
        # Get terrain properties at current location
        terrain_properties = self._get_terrain_properties(lat, lon)
        
        # Calculate base spread rate adjusted for terrain
        base_spread_rate = self.flat_terrain_spread_rate
        
        # Adjust for terrain roughness
        roughness = terrain_properties.get('roughness', 0.2)
        roughness_factor = 1.0 - (roughness * 0.8)  # Higher roughness = slower spread
        
        # Adjust for vegetation
        vegetation_density = terrain_properties.get('vegetation_density', 0.3)
        vegetation_factor = 1.0 - (vegetation_density * 0.7)  # Higher vegetation = slower spread
        
        # Adjust for soil type (permeability)
        soil_type = terrain_properties.get('soil_type', self.default_soil_type)
        soil_permeability = {
            'sand': 0.9,       # High permeability = less surface flow
            'clay': 0.3,       # Low permeability = more surface flow
            'loam': 0.6,       # Medium permeability
            'rock': 0.1,       # Very low permeability = more surface flow
            'urban': 0.2,      # Low permeability for urban surfaces
            'vegetation': 0.7   # Medium-high for vegetated areas
        }.get(soil_type, 0.5)
        
        soil_factor = 1.0 - (soil_permeability * 0.5)  # Higher permeability = slower surface spread
        
        # Calculate effective spread rate
        effective_spread_rate = base_spread_rate * roughness_factor * vegetation_factor * soil_factor
        
        # Calculate distance to spread in this timestep
        spread_distance = effective_spread_rate * dt_seconds  # in meters
        
        # Check for preferential flow directions (e.g., micro-topography)
        preferential_direction = None
        channeling = terrain_properties.get('channeling', None)
        
        if channeling is not None and channeling > 0:
            preferential_direction = np.radians(terrain_properties.get('channel_direction', 0))
        
        # Calculate costs for movement in different directions
        directions = []
        costs = []
        
        # Number of directions to consider (8 = cardinal + diagonal, 16 = finer resolution)
        num_directions = self.cost_distance_directions
        
        for i in range(num_directions):
            # Calculate direction angle
            angle = 2 * np.pi * i / num_directions
            
            # If there's a preferential direction, adjust costs
            if preferential_direction is not None:
                # Calculate angular difference (0 to π)
                angle_diff = abs((angle - preferential_direction + np.pi) % (2 * np.pi) - np.pi)
                
                # Cost is higher for directions away from preferential direction
                # and lower for directions close to preferential direction
                direction_cost = 1.0 + (angle_diff / np.pi) * channeling * 2.0
            else:
                direction_cost = 1.0
            
            # Store direction and cost
            directions.append(angle)
            costs.append(direction_cost)
        
        # Normalize costs so they sum to 1
        total_cost = sum(costs)
        probabilities = [1.0 - (cost / total_cost) for cost in costs]
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]
        
        # Choose a direction based on probabilities
        chosen_direction = np.random.choice(directions, p=probabilities)
        
        # Calculate position change
        lat_change = (spread_distance * np.cos(chosen_direction)) / 111000  # North-South
        lon_change = (spread_distance * np.sin(chosen_direction)) / (111000 * np.cos(np.radians(lat)))  # East-West
        
        # Update position
        particle.latitude += lat_change
        particle.longitude += lon_change
        
        # Update particle velocity for visualization
        u_velocity = effective_spread_rate * np.sin(chosen_direction)
        v_velocity = effective_spread_rate * np.cos(chosen_direction)
        particle.velocity = (u_velocity, v_velocity)
    
    def _apply_land_processes(self, particle: Particle, dt_seconds: float) -> None:
        """Apply land-specific processes like absorption and spreading.
        
        Args:
            particle: Particle to update
            dt_seconds: Timestep in seconds
        """
        # Calculate hours for this timestep
        dt_hours = dt_seconds / 3600
        
        # Get terrain properties at current location
        terrain_properties = self._get_terrain_properties(particle.latitude, particle.longitude)
        
        # Get soil type and determine absorption rate
        soil_type = terrain_properties.get('soil_type', self.default_soil_type)
        base_absorption_rate = self.soil_absorption_factors.get(soil_type, self.absorption_rate)
        
        # Adjust absorption rate based on soil moisture
        soil_moisture = terrain_properties.get('soil_moisture', 0.5)  # Default to medium moisture
        moisture_factor = 1.0 - (soil_moisture * 0.5)  # Drier soil absorbs more
        
        # Adjust absorption rate based on temperature
        temperature = terrain_properties.get('temperature', 20.0)  # Default to 20°C
        # Higher temperatures increase absorption rate (oil becomes less viscous)
        temp_factor = 1.0 + ((temperature - 20.0) / 100.0)  # 10°C change = 10% change
        
        # Calculate effective absorption rate
        effective_absorption_rate = base_absorption_rate * moisture_factor * temp_factor
        
        # Apply absorption into soil/land
        absorption_fraction = effective_absorption_rate * dt_hours
        
        # Update weathering state
        if 'absorbed_fraction' not in particle.weathering_state:
            particle.weathering_state['absorbed_fraction'] = 0.0
        
        particle.weathering_state['absorbed_fraction'] += absorption_fraction
        
        # Apply biodegradation (enhanced in vegetated areas)
        if 'biodegraded_fraction' in particle.weathering_state:
            vegetation_factor = terrain_properties.get('vegetation_density', 0.0)
            biodegradation_rate = self.params.get('base_biodegradation_rate', 0.001)
            
            # Vegetation enhances biodegradation
            effective_biodegradation = biodegradation_rate * (1.0 + vegetation_factor * 2.0)
            
            # Apply biodegradation
            particle.weathering_state['biodegraded_fraction'] += effective_biodegradation * dt_hours
        
        # Check if particle is fully weathered
        total_weathered = sum(particle.weathering_state.values())
        if total_weathered >= 0.99:  # 99% weathered
            particle.status = 'absorbed'
    
    def _check_boundaries(self, particle: Particle) -> None:
        """Check and handle boundary conditions for land-based particles.
        
        Args:
            particle: Particle to check
        """
        # Check if particle has moved to water
        surface_type = self._get_surface_type_at_location(particle.latitude, particle.longitude)
        
        if surface_type != particle.surface_type:
            # Surface type has changed
            particle.surface_type = surface_type
            
            # If moved to water, mark for transition to water model
            if surface_type == 'water':
                particle.status = 'transition_to_water'
        
        # Check if outside domain bounds
        if self._is_outside_domain(particle):
            self._handle_domain_boundary(particle)
    
    def _get_surface_type_at_location(self, latitude: float, longitude: float) -> str:
        """Determine the surface type (land or water) at a given location.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Surface type: 'land' or 'water'
        """
        # Get elevation at location
        try:
            # Get elevation and slope
            elevation, _, _ = self._get_cached_elevation_and_slope(latitude, longitude)
            
            # Simple check: if elevation > 0, it's land; otherwise water
            # This is a simplification - in a real implementation, you would use
            # a land/water mask or more sophisticated method
            return 'land' if elevation > 0 else 'water'
        except Exception as e:
            logger.warning(f"Error determining surface type: {str(e)}")
            return 'land'  # Default to land in case of error
    
    def _is_outside_domain(self, particle: Particle) -> bool:
        """Check if a particle is outside the simulation domain.
        
        Args:
            particle: Particle to check
            
        Returns:
            True if particle is outside domain, False otherwise
        """
        # Get domain boundaries from parameters
        domain_bounds = self.params.get('domain_bounds', None)
        if domain_bounds is None:
            return False
        
        min_lat, min_lon, max_lat, max_lon = domain_bounds
        
        # Check if particle is outside bounds
        if (particle.latitude < min_lat or 
            particle.latitude > max_lat or 
            particle.longitude < min_lon or 
            particle.longitude > max_lon):
            return True
        
        return False
    
    def _handle_domain_boundary(self, particle: Particle) -> None:
        """Handle particles that reach the domain boundary.
        
        Args:
            particle: Particle to handle
        """
        # Get domain boundaries from parameters
        domain_bounds = self.params.get('domain_bounds', None)
        if domain_bounds is None:
            return
        
        min_lat, min_lon, max_lat, max_lon = domain_bounds
        boundary_method = self.params.get('boundary_method', 'reflect')
        
        if boundary_method == 'absorb':
            # Mark particle as inactive
            particle.status = 'boundary_exit'
        
        elif boundary_method == 'reflect':
            # Reflect particle back into domain
            if particle.latitude < min_lat:
                particle.latitude = 2 * min_lat - particle.latitude
                particle.velocity = (particle.velocity[0], -particle.velocity[1])
            elif particle.latitude > max_lat:
                particle.latitude = 2 * max_lat - particle.latitude
                particle.velocity = (particle.velocity[0], -particle.velocity[1])
            
            if particle.longitude < min_lon:
                particle.longitude = 2 * min_lon - particle.longitude
                particle.velocity = (-particle.velocity[0], particle.velocity[1])
            elif particle.longitude > max_lon:
                particle.longitude = 2 * max_lon - particle.longitude
                particle.velocity = (-particle.velocity[0], particle.velocity[1])
        
        elif boundary_method == 'periodic':
            # Wrap around to opposite side (periodic boundary)
            if particle.latitude < min_lat:
                particle.latitude = max_lat - (min_lat - particle.latitude)
            elif particle.latitude > max_lat:
                particle.latitude = min_lat + (particle.latitude - max_lat)
            
            if particle.longitude < min_lon:
                particle.longitude = max_lon - (min_lon - particle.longitude)
            elif particle.longitude > max_lon:
                particle.longitude = min_lon + (particle.longitude - max_lon)
    
    def _step_with_adaptive_timestep(self, particle: Particle, dt_seconds: float) -> None:
        """Apply adaptive time-stepping to a particle.
        
        This method breaks down the main timestep into smaller substeps based on
        terrain conditions to ensure numerical stability and accuracy.
        
        Args:
            particle: Particle to update
            dt_seconds: Main timestep in seconds
        """
        # Get initial position
        initial_lat = particle.latitude
        initial_lon = particle.longitude
        
        # Get elevation and slope at current location
        elevation, slope, aspect = self._get_cached_elevation_and_slope(
            initial_lat, initial_lon
        )
        
        # Only use adaptive timesteps for steep slopes (optimization)
        if slope < self.adaptive_threshold_slope:
            # For gentle slopes, just use a single step with the appropriate model
            if slope < self.slope_threshold and self.use_cost_distance:
                self._apply_cost_distance_model(particle, dt_seconds)
            else:
                self._apply_downhill_flow(particle, dt_seconds)
            return
        
        # Determine which model to use based on slope
        use_cost_distance = slope < self.slope_threshold and self.use_cost_distance
        
        # Calculate adaptive timestep size based on slope and terrain
        substep_size, num_substeps = self._calculate_adaptive_timestep(particle, dt_seconds, slope)
        
        # Apply movement in substeps
        remaining_time = dt_seconds
        
        for i in range(num_substeps):
            # Calculate substep time
            substep_time = min(substep_size, remaining_time)
            
            # Skip if substep time is too small
            if substep_time < 0.1:  # Less than 0.1 seconds
                break
            
            # Apply appropriate model for this substep
            if use_cost_distance:
                self._apply_cost_distance_model(particle, substep_time)
            else:
                self._apply_downhill_flow(particle, substep_time)
            
            # Update remaining time
            remaining_time -= substep_time
            
            # Check if we've reached the end of the main timestep
            if remaining_time <= 0:
                break
            
            # Recalculate slope and aspect at new position for next substep
            elevation, slope, aspect = self._get_cached_elevation_and_slope(
                particle.latitude, particle.longitude
            )
            
            # Recalculate substep size based on new conditions
            substep_size, _ = self._calculate_adaptive_timestep(particle, remaining_time, slope)
            
            # Check for boundaries or transitions
            # If particle status changes, stop substeps
            old_status = particle.status
            old_surface_type = particle.surface_type
            
            self._check_boundaries(particle)
            
            if particle.status != old_status or particle.surface_type != old_surface_type:
                break
    
    def _calculate_adaptive_timestep(self, particle: Particle, dt_seconds: float, slope: float) -> Tuple[float, int]:
        """Calculate adaptive timestep size based on terrain conditions.
        
        Args:
            particle: Particle to update
            dt_seconds: Main timestep in seconds
            slope: Terrain slope in degrees
            
        Returns:
            Tuple of (substep_size_seconds, number_of_substeps)
        """
        # Base number of substeps on slope
        # Steeper slopes need smaller timesteps for stability
        if slope > 30:
            base_substeps = 10  # Very steep slopes
        elif slope > 15:
            base_substeps = 5   # Moderate slopes
        elif slope > 5:
            base_substeps = 3   # Gentle slopes
        else:
            base_substeps = 1   # Flat or nearly flat
        
        # Get terrain properties
        terrain_properties = self._get_terrain_properties(particle.latitude, particle.longitude)
        
        # Adjust based on terrain roughness
        roughness = terrain_properties.get('roughness', 0.2)
        if roughness > 0.5:
            base_substeps += 2  # Very rough terrain needs smaller steps
        
        # Adjust based on particle mass (heavier particles may need smaller steps)
        if particle.mass > 10:
            base_substeps += 1
        
        # Limit to maximum number of substeps
        num_substeps = min(base_substeps, self.max_substeps)
        
        # Calculate substep size
        substep_size = dt_seconds / num_substeps
        
        # Ensure substep isn't too small
        min_substep = dt_seconds * self.min_substep_fraction
        if substep_size < min_substep:
            substep_size = min_substep
            num_substeps = int(dt_seconds / substep_size)
        
        return substep_size, num_substeps


class HybridModel(OilSpillModel):
    """
    Hybrid model that combines water and land-based oil spill simulation.
    
    This model determines whether each particle is on land or water
    and applies the appropriate physics.
    """
    
    def __init__(self, simulation_params: Optional[Dict[str, Any]] = None):
        """Initialize the hybrid model."""
        super().__init__(simulation_params)
        
        # Create sub-models
        self.water_model = LagrangianModel(simulation_params)
        self.land_model = LandFlowModel(simulation_params)
        
        # Track which model each particle belongs to
        self.particle_models = {}  # Mapping of particle ID to model type ('water' or 'land')
        
        # Set up transition parameters
        self.transition_buffer_distance = self.params.get('transition_buffer_distance', 0.001)  # ~100m buffer
        self.coastal_mixing_factor = self.params.get('coastal_mixing_factor', 0.3)  # Mixing at coastline
    
    def initialize(self, preprocessed_data: Dict[str, Any]) -> None:
        """Initialize the hybrid model with preprocessed data.
        
        Args:
            preprocessed_data: Output from preprocess.preprocess_all_data()
        """
        # Initialize both sub-models
        self.water_model.initialize(preprocessed_data)
        self.land_model.initialize(preprocessed_data)
        
        # Copy particles from water model
        self.particles = self.water_model.particles.copy()
        
        # Initialize particle model mapping
        for particle in self.particles:
            # Determine initial model based on surface type
            if particle.surface_type == 'land':
                self.particle_models[particle.id] = 'land'
            else:
                self.particle_models[particle.id] = 'water'
        
        # Set up simulation time
        self.start_time = self.water_model.start_time
        self.current_time = self.start_time
        self.end_time = self.water_model.end_time
        
        # Reset timestep counter
        self.timestep = 0
        
        logger.info(f"Hybrid model initialized with {len(self.particles)} particles")
        logger.info(f"Initial distribution: {sum(1 for p in self.particles if self.particle_models[p.id] == 'water')} water, "
                  f"{sum(1 for p in self.particles if self.particle_models[p.id] == 'land')} land")
    
    def step(self) -> None:
        """Advance the hybrid simulation by one timestep."""
        # Increment timestep counter
        self.timestep += 1
        
        # Update current time
        self.current_time += timedelta(minutes=self.params['timestep_minutes'])
        
        # Update sub-model times
        self.water_model.current_time = self.current_time
        self.land_model.current_time = self.current_time
        self.water_model.timestep = self.timestep
        self.land_model.timestep = self.timestep
        
        # Separate particles by model
        water_particles = []
        land_particles = []
        
        for particle in self.particles:
            if particle.status == 'active':
                model_type = self.particle_models.get(particle.id, 'water')
                if model_type == 'water':
                    water_particles.append(particle)
                else:
                    land_particles.append(particle)
        
        # Update water model particles
        self.water_model.particles = water_particles
        self.land_model.particles = land_particles
        
        # Step each model
        if water_particles:
            self.water_model.step()
        
        if land_particles:
            self.land_model.step()
        
        # Check for transitions between models
        self._handle_transitions()
        
        # Recombine particles
        self.particles = water_particles + land_particles
    
    def _handle_transitions(self) -> None:
        """Handle transitions between water and land models."""
        # Check water particles for beaching
        for particle in self.water_model.particles:
            if particle.status == 'beached':
                # Transition to land model
                particle.status = 'active'
                particle.surface_type = 'land'
                self.particle_models[particle.id] = 'land'
                logger.debug(f"Particle {particle.id} transitioned from water to land (beaching)")
            
            # Check for shallow water near coastline
            elif self._is_near_coastline(particle.latitude, particle.longitude):
                # Get elevation to check if we're in shallow water
                elevation, _, _ = self.preprocessor.get_elevation_and_slope(
                    self.water_model.elevation_data, (particle.latitude, particle.longitude)
                )
                
                # If in very shallow water, chance of transitioning to land
                if -5 < elevation < 0 and np.random.random() < self.coastal_mixing_factor:
                    particle.status = 'active'
                    particle.surface_type = 'land'
                    self.particle_models[particle.id] = 'land'
                    logger.debug(f"Particle {particle.id} transitioned from water to land (shallow water)")
        
        # Check land particles for transition to water
        for particle in self.land_model.particles:
            if particle.status == 'transition_to_water':
                # Transition to water model
                particle.status = 'active'
                particle.surface_type = 'water'
                self.particle_models[particle.id] = 'water'
                logger.debug(f"Particle {particle.id} transitioned from land to water")
    
    def _is_near_coastline(self, latitude: float, longitude: float) -> bool:
        """Check if a location is near a coastline.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            True if near coastline, False otherwise
        """
        # In a real implementation, this would use a coastline dataset
        # For now, we'll use a simple check based on elevation
        try:
            # Get elevation at this location
            elevation, _, _ = self.preprocessor.get_elevation_and_slope(
                self.water_model.elevation_data, (latitude, longitude)
            )
            
            # Check nearby points to see if any are land
            buffer = self.transition_buffer_distance
            directions = [
                (buffer, 0),      # East
                (buffer, buffer), # Northeast
                (0, buffer),      # North
                (-buffer, buffer), # Northwest
                (-buffer, 0),     # West
                (-buffer, -buffer), # Southwest
                (0, -buffer),     # South
                (buffer, -buffer)  # Southeast
            ]
            
            for dlon, dlat in directions:
                check_lat = latitude + dlat
                check_lon = longitude + dlon
                
                # Get elevation at check point
                check_elevation, _, _ = self.preprocessor.get_elevation_and_slope(
                    self.water_model.elevation_data, (check_lat, check_lon)
                )
                
                # If current point is water and check point is land (or vice versa)
                if (elevation < 0 and check_elevation >= 0) or (elevation >= 0 and check_elevation < 0):
                    return True
            
            return False
        
        except Exception as e:
            logger.warning(f"Error checking coastline proximity: {str(e)}")
            return False


def create_model(model_type: str, simulation_params: Optional[Dict[str, Any]] = None) -> OilSpillModel:
    """
    Factory function to create the appropriate model.
    
    Args:
        model_type: Type of model to create ('water', 'land', or 'hybrid')
        simulation_params: Dictionary of simulation parameters
            
    Returns:
        Initialized model instance
    """
    if model_type == 'water':
        return LagrangianModel(simulation_params)
    elif model_type == 'land':
        return LandFlowModel(simulation_params)
    elif model_type == 'hybrid':
        return HybridModel(simulation_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_model(model_type: str, 
             preprocessed_data: Dict[str, Any],
             simulation_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run a complete simulation with the specified model type.
    
    Args:
        model_type: Type of model to use ('water', 'land', or 'hybrid')
        preprocessed_data: Output from preprocess.preprocess_all_data()
        simulation_params: Dictionary of simulation parameters
            
    Returns:
        Dictionary containing simulation results
    """
    # Create model
    model = create_model(model_type, simulation_params)
    
    # Initialize model
    model.initialize(preprocessed_data)
    
    # Run simulation
    results = model.run_simulation()
    
    return results

def LagrangianModel_get_current_state(self) -> Dict[str, Any]:
    """Get the current state of the simulation.
    
    Returns:
        Dictionary containing current state information
    """
    state = {
        'timestep': self.timestep,
        'time': self.current_time.isoformat(),
        'particles': []
    }
    
    # Add water-specific metrics
    beached_count = len([p for p in self.particles if p.status == 'beached'])
    state['water_metrics'] = {
        'beached_count': beached_count,
        'average_depth': self._calculate_average_depth()
    }
    
    # Add particle states
    for particle in self.particles:
        particle_state = {
            'id': particle.id,
            'latitude': particle.latitude,
            'longitude': particle.longitude,
            'mass': particle.mass,
            'status': particle.status,
            'age': particle.age,
            'depth': particle.depth if hasattr(particle, 'depth') else 0.0,
            'velocity': particle.velocity if hasattr(particle, 'velocity') else (0.0, 0.0),
            'evaporated_fraction': particle.weathering_state.get('evaporated_fraction', 0.0) if hasattr(particle, 'weathering_state') else 0.0,
            'dispersed_fraction': particle.weathering_state.get('dispersed_fraction', 0.0) if hasattr(particle, 'weathering_state') else 0.0
        }
        
        state['particles'].append(particle_state)
    
    return state

# Add the method to LagrangianModel
LagrangianModel._get_current_state = LagrangianModel_get_current_state

# Add helper method for LagrangianModel
def _calculate_average_depth(self) -> float:
    """Calculate the average depth of active particles.
    
    Returns:
        Average depth in meters
    """
    active_particles = [p for p in self.particles if p.status == 'active']
    if not active_particles:
        return 0.0
    
    total_depth = sum(p.depth for p in active_particles if hasattr(p, 'depth'))
    return total_depth / len(active_particles)

# Add the method to LagrangianModel
LagrangianModel._calculate_average_depth = _calculate_average_depth

def LandFlowModel_get_current_state(self) -> Dict[str, Any]:
    """Get the current state of the land flow simulation.
    
    Returns:
        Dictionary containing current state information with land-specific properties
    """
    state = {
        'timestep': self.timestep,
        'time': self.current_time.isoformat(),
        'particles': [],
        'model_type': 'land'
    }
    
    # Add land-specific metrics
    absorbed_count = len([p for p in self.particles if p.status == 'absorbed'])
    state['land_metrics'] = {
        'absorbed_count': absorbed_count,
        'average_elevation': self._calculate_average_elevation(),
        'average_slope': self._calculate_average_slope(),
        'soil_distribution': self._calculate_soil_distribution(),
        'terrain_metrics': self._calculate_terrain_metrics()
    }
    
    # Add particle states with land-specific properties
    for particle in self.particles:
        particle_state = {
            'id': particle.id,
            'latitude': particle.latitude,
            'longitude': particle.longitude,
            'mass': particle.mass,
            'status': particle.status,
            'age': particle.age,
            'surface_type': particle.surface_type if hasattr(particle, 'surface_type') else 'land'
        }
        
        # Add land-specific particle properties
        try:
            elevation, slope, aspect = self._get_cached_elevation_and_slope(
                particle.latitude, particle.longitude
            )
            terrain_props = self._get_terrain_properties(
                particle.latitude, particle.longitude
            )
            
            particle_state.update({
                'elevation': elevation,
                'slope': slope,
                'aspect': aspect,
                'terrain_roughness': terrain_props.get('roughness', 0.0),
                'soil_type': terrain_props.get('soil_type', self.default_soil_type),
                'vegetation_density': terrain_props.get('vegetation_density', 0.0),
                'soil_moisture': terrain_props.get('soil_moisture', 0.5),
                'absorbed_fraction': particle.weathering_state.get('absorbed_fraction', 0.0) if hasattr(particle, 'weathering_state') else 0.0,
                'biodegraded_fraction': particle.weathering_state.get('biodegraded_fraction', 0.0) if hasattr(particle, 'weathering_state') else 0.0
            })
        except Exception as e:
            logger.debug(f"Error adding land properties to particle {particle.id}: {str(e)}")
        
        state['particles'].append(particle_state)
    
    return state

# Add the method to LandFlowModel
LandFlowModel._get_current_state = LandFlowModel_get_current_state

# Add helper methods for LandFlowModel
def _calculate_average_elevation(self) -> float:
    """Calculate the average elevation of active particles.
    
    Returns:
        Average elevation in meters
    """
    active_particles = [p for p in self.particles if p.status == 'active']
    if not active_particles:
        return 0.0
    
    total_elevation = 0.0
    count = 0
    
    for particle in active_particles:
        try:
            elevation, _, _ = self._get_cached_elevation_and_slope(
                particle.latitude, particle.longitude
            )
            total_elevation += elevation
            count += 1
        except Exception:
            pass
    
    return total_elevation / max(1, count)  # Avoid division by zero

# Add the method to LandFlowModel
LandFlowModel._calculate_average_elevation = _calculate_average_elevation

def _calculate_average_slope(self) -> float:
    """Calculate the average slope of active particles.
    
    Returns:
        Average slope in degrees
    """
    active_particles = [p for p in self.particles if p.status == 'active']
    if not active_particles:
        return 0.0
    
    total_slope = 0.0
    count = 0
    
    for particle in active_particles:
        try:
            _, slope, _ = self._get_cached_elevation_and_slope(
                particle.latitude, particle.longitude
            )
            total_slope += slope
            count += 1
        except Exception:
            pass
    
    return total_slope / max(1, count)  # Avoid division by zero

# Add the method to LandFlowModel
LandFlowModel._calculate_average_slope = _calculate_average_slope

def _calculate_soil_distribution(self) -> Dict[str, int]:
    """Calculate the distribution of particles across soil types.
    
    Returns:
        Dictionary mapping soil types to particle counts
    """
    active_particles = [p for p in self.particles if p.status == 'active']
    soil_counts = {}
    
    for particle in active_particles:
        try:
            terrain_props = self._get_terrain_properties(
                particle.latitude, particle.longitude
            )
            soil_type = terrain_props.get('soil_type', self.default_soil_type)
            
            if soil_type not in soil_counts:
                soil_counts[soil_type] = 0
            soil_counts[soil_type] += 1
        except Exception:
            pass
    
    return soil_counts

# Add the method to LandFlowModel
LandFlowModel._calculate_soil_distribution = _calculate_soil_distribution

def _calculate_average_absorption_rate(self) -> float:
    """Calculate the average absorption rate for active particles.
    
    Returns:
        Average absorption rate (fraction per hour)
    """
    active_particles = [p for p in self.particles if p.status == 'active']
    if not active_particles:
        return 0.0
    
    total_rate = 0.0
    count = 0
    
    for particle in active_particles:
        try:
            terrain_props = self._get_terrain_properties(
                particle.latitude, particle.longitude
            )
            soil_type = terrain_props.get('soil_type', self.default_soil_type)
            absorption_rate = self.soil_absorption_factors.get(soil_type, self.absorption_rate)
            
            total_rate += absorption_rate
            count += 1
        except Exception:
            pass
    
    return total_rate / max(1, count)  # Avoid division by zero

# Add the method to LandFlowModel
LandFlowModel._calculate_average_absorption_rate = _calculate_average_absorption_rate

def _calculate_terrain_metrics(self) -> Dict[str, Any]:
    """Calculate various terrain metrics for the current particle distribution.
    
    Returns:
        Dictionary containing terrain metrics
    """
    active_particles = [p for p in self.particles if p.status == 'active']
    if not active_particles:
        return {}
    
    # Initialize metrics
    metrics = {
        'elevation_range': [0.0, 0.0],  # [min, max]
        'slope_range': [0.0, 0.0],       # [min, max]
        'roughness_average': 0.0,
        'vegetation_density_average': 0.0,
        'channeled_flow_percentage': 0.0  # Percentage of particles in channels
    }
    
    # Collect data
    elevations = []
    slopes = []
    roughness_values = []
    vegetation_values = []
    channeled_count = 0
    
    for particle in active_particles:
        try:
            # Get elevation and slope
            elevation, slope, _ = self._get_cached_elevation_and_slope(
                particle.latitude, particle.longitude
            )
            elevations.append(elevation)
            slopes.append(slope)
            
            # Get terrain properties
            terrain_props = self._get_terrain_properties(
                particle.latitude, particle.longitude
            )
            roughness_values.append(terrain_props.get('roughness', 0.0))
            vegetation_values.append(terrain_props.get('vegetation_density', 0.0))
            
            # Check if in a channel
            if terrain_props.get('channeling', None) is not None:
                channeled_count += 1
        except Exception:
            pass
    
    # Calculate metrics
    if elevations:
        metrics['elevation_range'] = [min(elevations), max(elevations)]
    if slopes:
        metrics['slope_range'] = [min(slopes), max(slopes)]
    if roughness_values:
        metrics['roughness_average'] = sum(roughness_values) / len(roughness_values)
    if vegetation_values:
        metrics['vegetation_density_average'] = sum(vegetation_values) / len(vegetation_values)
    if active_particles:
        metrics['channeled_flow_percentage'] = (channeled_count / len(active_particles)) * 100
    
    return metrics

# Add the method to LandFlowModel
LandFlowModel._calculate_terrain_metrics = _calculate_terrain_metrics

def _calculate_evaporated_percent(self) -> float:
    """Calculate the percentage of oil mass that has evaporated.
    
    Returns:
        Percentage of oil mass evaporated
    """
    total_mass = sum(p.mass for p in self.particles)
    if total_mass <= 0:
        return 0.0
    
    evaporated_mass = sum(
        p.mass * p.weathering_state.get('evaporated_fraction', 0.0) 
        for p in self.particles if hasattr(p, 'weathering_state')
    )
    
    return (evaporated_mass / total_mass) * 100

# Add the method to LagrangianModel
LagrangianModel._calculate_evaporated_percent = _calculate_evaporated_percent

def _calculate_dispersed_percent(self) -> float:
    """Calculate the percentage of oil mass that has dispersed.
    
    Returns:
        Percentage of oil mass dispersed
    """
    total_mass = sum(p.mass for p in self.particles)
    if total_mass <= 0:
        return 0.0
    
    dispersed_mass = sum(
        p.mass * p.weathering_state.get('dispersed_fraction', 0.0) 
        for p in self.particles if hasattr(p, 'weathering_state')
    )
    
    return (dispersed_mass / total_mass) * 100

# Add the method to LagrangianModel
LagrangianModel._calculate_dispersed_percent = _calculate_dispersed_percent
