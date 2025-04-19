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
    
    def run_simulation(self, 
                        callback: Optional[Callable[[Dict[str, Any], int, int], None]] = None,
                        save_interval: int = 1,
                        progress_interval: int = 10) -> Dict[str, Any]:
        """
        Run the full simulation from start to end time.
        
        Args:
            callback: Optional callback function called after each save_interval
                     with arguments (current_state, timestep, total_timesteps)
            save_interval: Save particle state every N timesteps (default: 1 = every timestep)
            progress_interval: Log progress every N timesteps (default: 10)
            
        Returns:
            Dictionary containing simulation results
        """
        if not self.particles:
            raise ValueError("Model not initialized. Call initialize() first.")
        
        # Calculate number of timesteps
        timestep_minutes = self.params['timestep_minutes']
        total_minutes = self.params['duration_hours'] * 60
        total_timesteps = total_minutes // timestep_minutes
        
        # Store particle positions at specified intervals
        particle_history = []
        
        # Add initial state
        initial_state = self._get_current_state()
        particle_history.append(initial_state)
        
        # Call callback with initial state if provided
        if callback is not None:
            callback(initial_state, 0, total_timesteps)
        
        # Track active particle count
        active_count = sum(1 for p in self.particles if p.status == 'active')
        logger.info(f"Starting simulation with {active_count} active particles")
        logger.info(f"Simulation period: {self.start_time.isoformat()} to {self.end_time.isoformat()}")
        logger.info(f"Timestep: {timestep_minutes} minutes, Total timesteps: {total_timesteps}")
        
        # Performance monitoring
        start_time = datetime.now()
        last_progress_time = start_time
        
        # Run simulation for each timestep
        for i in range(1, total_timesteps + 1):
            # Advance simulation by one timestep
            self.step()
            
            # Save state at specified intervals
            if i % save_interval == 0:
                current_state = self._get_current_state()
                particle_history.append(current_state)
                
                # Call callback if provided
                if callback is not None:
                    callback(current_state, i, total_timesteps)
            
            # Log progress at specified intervals
            if i % progress_interval == 0 or i == total_timesteps:
                # Calculate active particles
                active_count = sum(1 for p in self.particles if p.status == 'active')
                beached_count = sum(1 for p in self.particles if p.status == 'beached')
                weathered_count = sum(1 for p in self.particles if p.status == 'weathered')
                
                # Calculate progress percentage
                progress = (i / total_timesteps) * 100
                
                # Calculate elapsed and estimated time
                now = datetime.now()
                elapsed = (now - start_time).total_seconds()
                time_per_step = elapsed / i
                remaining_steps = total_timesteps - i
                estimated_remaining = remaining_steps * time_per_step
                
                # Log progress
                logger.info(f"Progress: {progress:.1f}% (Step {i}/{total_timesteps})")
                logger.info(f"Particles: {active_count} active, {beached_count} beached, {weathered_count} weathered")
                logger.info(f"Elapsed: {elapsed:.1f}s, Est. remaining: {estimated_remaining:.1f}s")
                
                # Reset progress timer
                last_progress_time = now
        
        # Calculate final statistics
        end_time = datetime.now()
        total_runtime = (end_time - start_time).total_seconds()
        steps_per_second = total_timesteps / total_runtime if total_runtime > 0 else 0
        
        # Count particles by status
        status_counts = {}
        for p in self.particles:
            status_counts[p.status] = status_counts.get(p.status, 0) + 1
        
        # Calculate mass balance
        initial_mass = sum(p.mass for p in self.particles)
        remaining_mass = sum(p.remaining_mass for p in self.particles)
        weathered_mass = initial_mass - remaining_mass
        weathered_percent = (weathered_mass / initial_mass) * 100 if initial_mass > 0 else 0
        
        # Log final statistics
        logger.info(f"Simulation completed in {total_runtime:.2f} seconds")
        logger.info(f"Performance: {steps_per_second:.2f} timesteps/second")
        logger.info(f"Final particle status: {status_counts}")
        logger.info(f"Mass balance: {weathered_percent:.1f}% weathered, {100-weathered_percent:.1f}% remaining")
        
        # Return results
        return {
            'particle_history': particle_history,
            'start_time': self.start_time.isoformat(),
            'end_time': self.current_time.isoformat(),
            'timestep_minutes': timestep_minutes,
            'total_timesteps': total_timesteps,
            'params': self.params,
            'runtime_seconds': total_runtime,
            'steps_per_second': steps_per_second,
            'final_status': status_counts,
            'mass_balance': {
                'initial_mass': initial_mass,
                'remaining_mass': remaining_mass,
                'weathered_mass': weathered_mass,
                'weathered_percent': weathered_percent
            }
        }
    
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
        """Check if a particle is outside the simulation domain.
        
        Args:
            particle: Particle to check
            
        Returns:
            True if particle is outside domain, False otherwise
        """
        # Get domain boundaries from parameters
        domain_bounds = self.params.get('domain_bounds', None)
        if domain_bounds is None:
            # If no domain bounds specified, use global bounds
            domain_bounds = (-90.0, -180.0, 90.0, 180.0)  # (min_lat, min_lon, max_lat, max_lon)
        
        min_lat, min_lon, max_lat, max_lon = domain_bounds
        lat, lon = particle.latitude, particle.longitude
        
        # Check if particle is outside bounds
        return (lat < min_lat) or (lat > max_lat) or (lon < min_lon) or (lon > max_lon)
    
    def _handle_domain_boundary(self, particle: Particle) -> None:
        """Handle a particle that has reached the domain boundary.
        
        Args:
            particle: Particle to update
        """
        # Get domain boundaries from parameters
        domain_bounds = self.params.get('domain_bounds', None)
        if domain_bounds is None:
            # If no domain bounds specified, use global bounds
            domain_bounds = (-90.0, -180.0, 90.0, 180.0)  # (min_lat, min_lon, max_lat, max_lon)
        
        min_lat, min_lon, max_lat, max_lon = domain_bounds
        
        # Get boundary handling method from parameters
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
            particle.velocity = (0.0, 0.0)
        
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
        # check against coastline data or elevation data
        
        # For now, we'll just assume no beaching
        pass


class LandFlowModel(OilSpillModel):
    """Model for land-based oil spill flow using slope and terrain."""
    
    def __init__(self, simulation_params: Optional[Dict[str, Any]] = None):
        """Initialize the land flow model."""
        super().__init__(simulation_params)
    
    def step(self) -> None:
        """Advance the land-based simulation by one timestep."""
        # Increment timestep counter
        self.timestep += 1
        
        # Update current time
        self.current_time += timedelta(minutes=self.params['timestep_minutes'])
        
        # Process each particle
        for particle in self.particles:
            # Skip inactive particles
            if particle['status'] != 'active':
                continue
            
            # Get current location
            lat, lon = particle['latitude'], particle['longitude']
            
            # Get elevation and slope at this location
            elevation, slope, aspect = self.preprocessor.get_elevation_and_slope(
                self.elevation_data, (lat, lon)
            )
            
            # Skip if no valid slope data
            if slope == 0:
                continue
            
            # Calculate flow direction from aspect
            # Aspect is in degrees clockwise from north
            flow_direction_rad = np.radians(90 - aspect)  # Convert to math convention
            
            # Calculate velocity based on slope
            # Simple model: velocity proportional to slope
            speed = slope * self.params.get('slope_velocity_factor', 0.1)
            
            # Calculate velocity components
            u_velocity = speed * np.cos(flow_direction_rad)
            v_velocity = speed * np.sin(flow_direction_rad)
            
            # Add some random diffusion
            diffusion_scale = self.params['diffusion_coefficient'] * 0.5  # Less diffusion on land
            random_u = np.random.normal(0, diffusion_scale)
            random_v = np.random.normal(0, diffusion_scale)
            
            # Combine deterministic flow with diffusion
            u_velocity += random_u
            v_velocity += random_v
            
            # Update particle velocity
            particle['velocity'] = (u_velocity, v_velocity)
            
            # Convert velocity to position change
            # Approximate conversion from m/s to degrees
            dt_seconds = self.params['timestep_minutes'] * 60
            
            # Convert velocity to position change
            lat_change = (v_velocity * dt_seconds) / (111000)  # North-South
            lon_change = (u_velocity * dt_seconds) / (111000 * np.cos(np.radians(lat)))  # East-West
            
            # Update position
            particle['latitude'] += lat_change
            particle['longitude'] += lon_change
            
            # Update particle age
            particle['age'] += self.params['timestep_minutes'] / 60  # Age in hours
            
            # Apply absorption and other land-specific effects
            self._apply_land_effects(particle)
    
    def _apply_land_effects(self, particle: Dict[str, Any]) -> None:
        """
        Apply land-specific effects to a particle.
        
        Args:
            particle: Particle dictionary to update
        """
        # Simple absorption/retention model
        # Oil is absorbed by soil at a rate depending on soil type
        # For now, use a simple constant rate
        
        # Get absorption rate from params (fraction per hour)
        absorption_rate = self.params.get('absorption_rate', 0.1)
        
        # Calculate mass loss for this timestep
        hours = self.params['timestep_minutes'] / 60
        mass_loss_fraction = 1 - np.exp(-absorption_rate * hours)
        
        # Update particle mass
        mass_loss = particle['mass'] * mass_loss_fraction
        particle['mass'] -= mass_loss
        
        # If mass is very small, mark as absorbed
        if particle['mass'] < 0.01 * self.params.get('initial_particle_mass', 1.0):
            particle['status'] = 'absorbed'


class HybridModel(OilSpillModel):
    """
    Hybrid model that combines water and land models.
    
    This model determines whether each particle is on land or water
    and applies the appropriate physics.
    """
    
    def __init__(self, simulation_params: Optional[Dict[str, Any]] = None):
        """Initialize the hybrid model."""
        super().__init__(simulation_params)
        
        # Create sub-models
        self.water_model = LagrangianModel(simulation_params)
        self.land_model = LandFlowModel(simulation_params)
    
    def initialize(self, preprocessed_data: Dict[str, Any]) -> None:
        """
        Initialize the model with preprocessed data.
        
        Args:
            preprocessed_data: Output from preprocess.preprocess_all_data()
        """
        # Initialize base model
        super().initialize(preprocessed_data)
        
        # Initialize sub-models
        self.water_model.initialize(preprocessed_data)
        self.land_model.initialize(preprocessed_data)
        
        # Determine initial surface type for each particle
        for particle in self.particles:
            self._update_surface_type(particle)
    
    def step(self) -> None:
        """Advance the hybrid simulation by one timestep."""
        # Increment timestep counter
        self.timestep += 1
        
        # Update current time
        self.current_time += timedelta(minutes=self.params['timestep_minutes'])
        
        # Process each particle
        for particle in self.particles:
            # Skip inactive particles
            if particle['status'] != 'active':
                continue
            
            # Determine if particle is on land or water
            self._update_surface_type(particle)
            
            # Apply appropriate model based on surface type
            if particle.get('surface_type') == 'land':
                # Use land model physics for this particle
                # We need to create a temporary list with just this particle
                self.land_model.particles = [particle]
                self.land_model.timestep = self.timestep
                self.land_model.current_time = self.current_time
                self.land_model.step()
            else:
                # Use water model physics for this particle
                # We need to create a temporary list with just this particle
                self.water_model.particles = [particle]
                self.water_model.timestep = self.timestep
                self.water_model.current_time = self.current_time
                self.water_model.step()
            
            # Check for land-water transitions
            self._check_transitions(particle)
    
    def _update_surface_type(self, particle: Dict[str, Any]) -> None:
        """
        Determine if a particle is on land or water.
        
        Args:
            particle: Particle dictionary to update
        """
        # Get current location
        lat, lon = particle['latitude'], particle['longitude']
        
        # In a full implementation, this would check against coastline data
        # or use elevation data to determine if below sea level
        
        # For now, we'll use a placeholder implementation
        # This would be replaced with actual geographic data
        
        # Placeholder: assume all particles start on water
        if 'surface_type' not in particle:
            particle['surface_type'] = 'water'
    
    def _check_transitions(self, particle: Dict[str, Any]) -> None:
        """
        Check and handle transitions between land and water.
        
        Args:
            particle: Particle dictionary to update
        """
        # Get current location
        lat, lon = particle['latitude'], particle['longitude']
        
        # Determine current surface type
        current_type = particle.get('surface_type', 'water')
        
        # In a full implementation, this would check against coastline data
        # For now, we'll use our placeholder implementation
        
        # If surface type has changed, update particle properties
        new_type = self._get_surface_type_at_location((lat, lon))
        
        if new_type != current_type:
            particle['surface_type'] = new_type
            
            # Handle transition effects
            if new_type == 'land':
                # Water to land transition
                # Reduce velocity due to friction
                u, v = particle['velocity']
                particle['velocity'] = (u * 0.5, v * 0.5)
            else:
                # Land to water transition
                # Nothing special to do here
                pass
    
    def _get_surface_type_at_location(self, location: Tuple[float, float]) -> str:
        """
        Determine surface type at a location.
        
        Args:
            location: (latitude, longitude) tuple
            
        Returns:
            'land' or 'water'
        """
        # Placeholder implementation
        # In a real implementation, this would use coastline data
        
        # For now, just return the existing type or default to water
        # This means we don't have transitions yet
        return 'water'


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
