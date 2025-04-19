"""
Validation utilities for the Oil Spill Trajectory Analysis Engine.

This module provides validation functions for checking the integrity and
correctness of input data, configuration parameters, and simulation results.
These functions help ensure that the simulation operates with valid data
and produces reliable results.
"""

from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import datetime
import math
import json
import os

# Import local modules
from trajectory_core.data_models import SpillConfig, EnvironmentalData, Particle, SimulationResults


def validate_config(config: SpillConfig) -> List[str]:
    """
    Validate a SpillConfig object and return a list of validation errors.
    
    Args:
        config: SpillConfig object to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Validate location
    try:
        lat, lon = config.location
        if not (-90 <= lat <= 90):
            errors.append(f"Latitude must be between -90 and 90, got {lat}")
        if not (-180 <= lon <= 180):
            errors.append(f"Longitude must be between -180 and 180, got {lon}")
    except (ValueError, TypeError):
        errors.append("Location must be a tuple of (latitude, longitude)")
    
    # Validate volume
    if config.volume <= 0:
        errors.append(f"Volume must be positive, got {config.volume}")
    
    # Validate oil type
    if config.oil_type not in SpillConfig.VALID_OIL_TYPES:
        errors.append(f"Oil type must be one of {SpillConfig.VALID_OIL_TYPES}, got {config.oil_type}")
    
    # Validate duration
    if config.duration <= 0:
        errors.append(f"Duration must be positive, got {config.duration}")
    
    # Validate timestep
    if config.timestep <= 0:
        errors.append(f"Timestep must be positive, got {config.timestep}")
    
    # Validate particle count
    if config.particle_count <= 0:
        errors.append(f"Particle count must be positive, got {config.particle_count}")
    
    return errors


def validate_environmental_data(env_data: EnvironmentalData) -> List[str]:
    """
    Validate an EnvironmentalData object and return a list of validation errors.
    
    Args:
        env_data: EnvironmentalData object to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Validate domain
    try:
        min_lat, max_lat, min_lon, max_lon = env_data.domain
        if not (-90 <= min_lat <= 90) or not (-90 <= max_lat <= 90):
            errors.append(f"Latitude must be between -90 and 90")
        if not (-180 <= min_lon <= 180) or not (-180 <= max_lon <= 180):
            errors.append(f"Longitude must be between -180 and 180")
        if min_lat >= max_lat:
            errors.append(f"min_lat must be less than max_lat")
        if min_lon >= max_lon:
            errors.append(f"min_lon must be less than max_lon")
    except (ValueError, TypeError):
        errors.append("Domain must be a tuple of (min_lat, max_lat, min_lon, max_lon)")
    
    # Validate time range
    try:
        start_time, end_time = env_data.time_range
        if start_time >= end_time:
            errors.append(f"start_time must be before end_time")
    except (ValueError, TypeError):
        errors.append("Time range must be a tuple of (start_time, end_time)")
    
    # Validate wind data structure
    required_wind_keys = ['times', 'lats', 'lons', 'u_component', 'v_component']
    for key in required_wind_keys:
        if key not in env_data.wind_data:
            errors.append(f"Wind data missing required key: {key}")
    
    # Validate ocean currents structure
    required_current_keys = ['times', 'lats', 'lons', 'depths', 'u_component', 'v_component']
    for key in required_current_keys:
        if key not in env_data.ocean_currents:
            errors.append(f"Ocean current data missing required key: {key}")
    
    # Validate elevation data structure
    required_elevation_keys = ['lats', 'lons', 'elevation']
    for key in required_elevation_keys:
        if key not in env_data.elevation_data:
            errors.append(f"Elevation data missing required key: {key}")
    
    # Validate oil properties
    required_oil_keys = ['density', 'viscosity', 'surface_tension', 'evaporation_rate']
    for key in required_oil_keys:
        if key not in env_data.oil_properties:
            errors.append(f"Oil properties missing required key: {key}")
    
    return errors


def validate_particle(particle: Particle) -> List[str]:
    """
    Validate a Particle object and return a list of validation errors.
    
    Args:
        particle: Particle object to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Validate position
    try:
        lat, lon, depth = particle.position
        if not (-90 <= lat <= 90):
            errors.append(f"Latitude must be between -90 and 90, got {lat}")
        if not (-180 <= lon <= 180):
            errors.append(f"Longitude must be between -180 and 180, got {lon}")
        if depth < 0 and particle.oil_type not in ['gasoline', 'diesel', 'jet_fuel']:
            # Only light oils can have negative depth (above water)
            errors.append(f"Depth must be non-negative for {particle.oil_type}, got {depth}")
    except (ValueError, TypeError):
        errors.append("Position must be a tuple of (latitude, longitude, depth)")
    
    # Validate mass
    if particle.mass <= 0:
        errors.append(f"Mass must be positive, got {particle.mass}")
    
    # Validate age
    if particle.age < 0:
        errors.append(f"Age must be non-negative, got {particle.age}")
    
    # Validate decay factors
    required_decay_keys = ['evaporation', 'dissolution', 'biodegradation']
    for key in required_decay_keys:
        if key not in particle.decay_factors:
            errors.append(f"Decay factors missing required key: {key}")
        elif not (0 <= particle.decay_factors[key] <= 1):
            errors.append(f"Decay factor {key} must be between 0 and 1, got {particle.decay_factors[key]}")
    
    return errors


def validate_simulation_results(results: SimulationResults) -> List[str]:
    """
    Validate a SimulationResults object and return a list of validation errors.
    
    Args:
        results: SimulationResults object to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Validate time range
    if results.start_time >= results.end_time:
        errors.append("Start time must be before end time")
    
    # Validate particle history
    if not results.particle_history:
        errors.append("Particle history cannot be empty")
    
    # Validate concentration maps
    if not results.concentration_maps:
        errors.append("Concentration maps cannot be empty")
    
    # Validate affected areas
    if 'type' not in results.affected_areas or results.affected_areas['type'] != 'FeatureCollection':
        errors.append("Affected areas must be a GeoJSON FeatureCollection")
    if 'features' not in results.affected_areas:
        errors.append("Affected areas must have a 'features' property")
    
    # Validate config
    config_errors = validate_config(results.config)
    if config_errors:
        errors.append(f"Invalid configuration: {', '.join(config_errors)}")
    
    # Validate particles
    for i, particle in enumerate(results.particles):
        particle_errors = validate_particle(particle)
        if particle_errors:
            errors.append(f"Invalid particle at index {i}: {', '.join(particle_errors)}")
            break  # Only report the first invalid particle
    
    return errors


def validate_json_file(filepath: str) -> Tuple[bool, List[str]]:
    """
    Validate that a file contains valid JSON.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check if file exists
    if not os.path.exists(filepath):
        errors.append(f"File does not exist: {filepath}")
        return False, errors
    
    # Try to parse the JSON
    try:
        with open(filepath, 'r') as f:
            json.load(f)
        return True, []
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {str(e)}")
        return False, errors


def validate_csv_file(filepath: str, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that a file contains valid CSV with the required columns.
    
    Args:
        filepath: Path to the CSV file
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    import csv
    
    errors = []
    
    # Check if file exists
    if not os.path.exists(filepath):
        errors.append(f"File does not exist: {filepath}")
        return False, errors
    
    # Try to parse the CSV
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # Check for required columns
            for column in required_columns:
                if column not in header:
                    errors.append(f"Missing required column: {column}")
            
            if errors:
                return False, errors
            
            # Check that all rows have the correct number of columns
            for i, row in enumerate(reader):
                if len(row) != len(header):
                    errors.append(f"Row {i+2} has {len(row)} columns, expected {len(header)}")
                    if len(errors) >= 5:  # Limit the number of errors reported
                        errors.append("Too many errors, stopping validation")
                        break
            
            return len(errors) == 0, errors
    except Exception as e:
        errors.append(f"Error parsing CSV: {str(e)}")
        return False, errors


def validate_data_consistency(config: SpillConfig, env_data: EnvironmentalData) -> List[str]:
    """
    Validate that the configuration and environmental data are consistent.
    
    Args:
        config: SpillConfig object
        env_data: EnvironmentalData object
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check that the oil type in the config matches the oil properties in the env_data
    if config.oil_type not in env_data.oil_properties.get('oil_type', config.oil_type):
        errors.append(f"Oil type in config ({config.oil_type}) does not match oil properties")
    
    # Check that the simulation time range is within the environmental data time range
    env_start, env_end = env_data.time_range
    sim_start = config.start_time
    sim_end = config.get_end_time()
    
    if sim_start < env_start:
        errors.append(f"Simulation start time ({sim_start}) is before environmental data start time ({env_start})")
    if sim_end > env_end:
        errors.append(f"Simulation end time ({sim_end}) is after environmental data end time ({env_end})")
    
    # Check that the spill location is within the environmental data domain
    lat, lon = config.location
    min_lat, max_lat, min_lon, max_lon = env_data.domain
    
    if not (min_lat <= lat <= max_lat):
        errors.append(f"Spill latitude ({lat}) is outside the environmental data domain ({min_lat}, {max_lat})")
    if not (min_lon <= lon <= max_lon):
        errors.append(f"Spill longitude ({lon}) is outside the environmental data domain ({min_lon}, {max_lon})")
    
    return errors
