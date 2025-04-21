#!/usr/bin/env python
"""
Test script for the Land-based Flow Model of the Oil Spill Trajectory Analysis Engine.

This script creates a test case with mock terrain data and runs the LandFlowModel,
saving the output to a JSON file for inspection.
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import the modules from our package
from trajectory_core import model, preprocess, config


def create_mock_terrain_data():
    """
    Create mock terrain data for testing the land-based flow model.
    
    Returns:
        Dictionary containing mock wind, elevation, and terrain data
    """
    # Create a simple grid centered at the spill location
    spill_lat, spill_lon = 40.0, -70.0  # Example location
    grid_size = 20
    grid_resolution = 0.01  # degrees (approximately 1 km)
    
    # Create lat/lon grid
    lats = np.linspace(spill_lat - grid_size/2 * grid_resolution, 
                      spill_lat + grid_size/2 * grid_resolution, 
                      grid_size)
    lons = np.linspace(spill_lon - grid_size/2 * grid_resolution, 
                      spill_lon + grid_size/2 * grid_resolution, 
                      grid_size)
    
    # Create mock elevation data (terrain with hills and valleys)
    elevation_grid = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            # Create a terrain with a central valley and surrounding hills
            dist_from_center = np.sqrt((i - grid_size/2)**2 + (j - grid_size/2)**2)
            # Central area is land (positive elevation)
            elevation = 50 - dist_from_center * 5
            
            # Add some random variation to make it more realistic
            elevation += np.random.normal(0, 5)
            
            # Create a river channel running through the middle
            if abs(i - grid_size/2) < 1:
                elevation = -5  # Water channel (negative elevation)
            
            elevation_grid[i, j] = elevation
    
    # Calculate slope and aspect for each point
    slope_grid = np.zeros((grid_size, grid_size))
    aspect_grid = np.zeros((grid_size, grid_size))
    
    # Calculate slope using central differences
    for i in range(1, grid_size-1):
        for j in range(1, grid_size-1):
            # Calculate slope using central differences
            dx = (elevation_grid[i, j+1] - elevation_grid[i, j-1]) / (2 * grid_resolution * 111000)  # in m/m
            dy = (elevation_grid[i+1, j] - elevation_grid[i-1, j]) / (2 * grid_resolution * 111000)  # in m/m
            
            # Calculate slope magnitude (in degrees)
            slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
            slope_grid[i, j] = slope
            
            # Calculate aspect (direction of steepest descent)
            aspect = np.degrees(np.arctan2(-dy, -dx)) % 360
            aspect_grid[i, j] = aspect
    
    # Create a grid of points for elevation data
    elevation_data_points = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            elevation_data_points.append({
                'lat': float(lat),
                'lon': float(lon),
                'elevation': float(elevation_grid[i, j]),
                'slope': float(slope_grid[i, j]) if 0 < i < grid_size-1 and 0 < j < grid_size-1 else 0.0,
                'aspect': float(aspect_grid[i, j]) if 0 < i < grid_size-1 and 0 < j < grid_size-1 else 0.0
            })
    
    # Create mock wind data (light wind from west)
    start_time = datetime.now()
    times = []
    wind_u = []
    wind_v = []
    
    # Create 48 hours of hourly wind data
    for hour in range(48):
        time = start_time + timedelta(hours=hour)
        times.append(time.isoformat())
        
        # Light wind from west (u = 2 m/s, v = 0 m/s)
        wind_u.append(2.0)
        wind_v.append(0.0)
    
    # Create mock current data (simple circular pattern)
    current_u = []
    current_v = []
    
    # Create 48 hours of hourly current data
    for hour in range(48):
        # Circular current pattern
        angle = hour * (2 * np.pi / 24)  # Complete cycle every 24 hours
        current_u.append(0.2 * np.cos(angle))  # 0.2 m/s maximum
        current_v.append(0.2 * np.sin(angle))  # 0.2 m/s maximum
    
    # Package the data in the format expected by our preprocessing module
    mock_data = {
        'wind_data': {
            'data': {
                'hourly': {
                    'time': times,
                    'windspeed_10m': [np.sqrt(u**2 + v**2) for u, v in zip(wind_u, wind_v)],
                    'winddirection_10m': [np.degrees(np.arctan2(-u, -v)) % 360 for u, v in zip(wind_u, wind_v)]
                },
                'latitude': spill_lat,
                'longitude': spill_lon
            }
        },
        'current_data': {
            'data': {
                'times': times,
                'u': current_u,
                'v': current_v,
                'latitude': spill_lat,
                'longitude': spill_lon,
                'depth': 0.0  # Surface currents
            }
        },
        'elevation_data': {
            'data': {
                'grid': elevation_data_points,
                'bbox': [float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))],
                'resolution': float(grid_resolution)
            }
        }
    }
    
    return mock_data, (spill_lat, spill_lon)


def run_land_model_test():
    """
    Run a test simulation with the land-based flow model and save the results to a JSON file.
    
    Returns:
        Path to the output JSON file
    """
    logger.info("Setting up test parameters...")
    
    # Simulation parameters
    simulation_params = {
        'timestep_minutes': 30,  # Increased from 15 to 30 minutes for faster simulation
        'duration_hours': 12,    # Reduced from 24 to 12 hours for faster testing
        'particle_count': 100,   # Reduced from 500 to 100 for faster testing
        'random_seed': 42,
        'domain_bounds': [39.0, -71.0, 41.0, -69.0],  # [min_lat, min_lon, max_lat, max_lon]
        'boundary_method': 'reflect',  # Reflect particles at domain boundaries
        
        # Land-specific parameters
        'flow_resistance_factor': 0.2,
        'absorption_rate': 0.01,
        'slope_threshold': 1.0,
        'terrain_roughness_factor': 0.3,
        'flat_terrain_spread_rate': 0.5,
        
        # Adaptive time-stepping parameters - adjusted for better performance
        'use_adaptive_timestep': True,
        'max_substeps': 5,
        'min_substep_fraction': 0.2,
        'adaptive_threshold_slope': 15.0,  # Only use adaptive timesteps for slopes > this value
        
        # Performance optimization parameters
        'use_spatial_index': True,
        'use_terrain_caching': True,
        'batch_size': 50
    }
    
    # Spill parameters
    spill_location = (40.0, -70.0)  # Latitude, Longitude
    spill_volume = 1000.0  # m³
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=simulation_params['duration_hours'])
    
    logger.info("Creating mock data...")
    mock_data, _ = create_mock_terrain_data()
    
    logger.info("Preprocessing data...")
    try:
        # Create preprocessor
        preprocessor = preprocess.DataPreprocessor()
        
        # Preprocess wind data
        processed_wind = preprocessor.preprocess_wind_data(mock_data['wind_data'])
        
        # Preprocess currents data
        processed_currents = preprocessor.preprocess_ocean_currents(mock_data['current_data'])
        
        # Preprocess elevation data
        processed_elevation = preprocessor.preprocess_elevation_data(mock_data['elevation_data'])
        
        # Initialize particles
        particles = preprocessor.initialize_particles(
            spill_location, 
            spill_volume, 
            particle_count=simulation_params['particle_count']
        )
        
        # Manually set all particles to be on land
        for particle in particles:
            particle['status'] = 'active'
            particle['surface_type'] = 'land'
        
        # Combine all preprocessed data
        preprocessed_data = {
            'wind': processed_wind,
            'currents': processed_currents,
            'elevation': processed_elevation,
            'particles': particles,
            'metadata': {
                'spill_location': spill_location,
                'spill_volume': spill_volume,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'time_step_hours': simulation_params['timestep_minutes'] / 60,
                'target_resolution': 30.0,
                'particle_count': simulation_params['particle_count'],
                'processing_timestamp': datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise
    
    logger.info("Creating and initializing Land Flow model...")
    # Create and initialize the Land Flow model
    land_model = model.LandFlowModel(simulation_params)
    land_model.initialize(preprocessed_data)
    
    logger.info("Running simulation...")
    # Run the simulation
    try:
        results = land_model.run_simulation(
            save_interval=4,  # Save every 4 timesteps (2 hours)
            progress_interval=5  # Log progress every 5 timesteps
        )
    except Exception as e:
        logger.error(f"Error during simulation: {str(e)}")
        raise
    
    # Create output directory if it doesn't exist
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to JSON file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"test_land_model_{timestamp}.json")
    
    logger.info(f"Saving results to {output_file}...")
    
    # Convert numpy arrays and other non-serializable objects to serializable types
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    # Serialize the results
    serializable_results = make_serializable(results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info("Simulation complete!")
    return output_file


def run_water_model_test():
    """
    Run a test simulation with the water-based Lagrangian model and save the results to a JSON file.
    
    Returns:
        Path to the output JSON file
    """
    logger.info("Setting up test parameters...")
    
    # Simulation parameters
    simulation_params = {
        'timestep_minutes': 30,  # Increased from 15 to 30 minutes for faster simulation
        'duration_hours': 12,    # Reduced from 24 to 12 hours for faster testing
        'particle_count': 100,   # Reduced from 500 to 100 for faster testing
        'random_seed': 42,
        'domain_bounds': [39.0, -71.0, 41.0, -69.0],  # [min_lat, min_lon, max_lat, max_lon]
        'boundary_method': 'reflect',  # Reflect particles at domain boundaries
        
        # Water-specific parameters
        'diffusion_coefficient': 10.0,  # m²/s
        'wind_drift_factor': 0.03,      # 3% of wind speed
        'evaporation_rate': 0.05,       # Fraction per day
        'dispersion_rate': 0.02         # Fraction per day
    }
    
    # Spill parameters
    spill_location = (40.0, -70.0)  # Latitude, Longitude
    spill_volume = 1000.0  # m³
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=simulation_params['duration_hours'])
    
    logger.info("Creating mock data...")
    mock_data, _ = create_mock_terrain_data()
    
    logger.info("Preprocessing data...")
    try:
        # Create preprocessor
        preprocessor = preprocess.DataPreprocessor()
        
        # Preprocess wind data
        processed_wind = preprocessor.preprocess_wind_data(mock_data['wind_data'])
        
        # Preprocess currents data
        processed_currents = preprocessor.preprocess_ocean_currents(mock_data['current_data'])
        
        # Preprocess elevation data
        processed_elevation = preprocessor.preprocess_elevation_data(mock_data['elevation_data'])
        
        # Initialize particles
        particles = preprocessor.initialize_particles(
            spill_location, 
            spill_volume, 
            particle_count=simulation_params['particle_count']
        )
        
        # Manually set all particles to be on water
        for particle in particles:
            particle['status'] = 'active'
            particle['surface_type'] = 'water'
        
        # Combine all preprocessed data
        preprocessed_data = {
            'wind': processed_wind,
            'currents': processed_currents,
            'elevation': processed_elevation,
            'particles': particles,
            'metadata': {
                'spill_location': spill_location,
                'spill_volume': spill_volume,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'time_step_hours': simulation_params['timestep_minutes'] / 60,
                'target_resolution': 30.0,
                'particle_count': simulation_params['particle_count'],
                'processing_timestamp': datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise
    
    logger.info("Creating and initializing Lagrangian model...")
    # Create and initialize the Lagrangian model
    water_model = model.LagrangianModel(simulation_params)
    water_model.initialize(preprocessed_data)
    
    logger.info("Running simulation...")
    # Run the simulation
    try:
        results = water_model.run_simulation(
            save_interval=4,  # Save every 4 timesteps (2 hours)
            progress_interval=5  # Log progress every 5 timesteps
        )
    except Exception as e:
        logger.error(f"Error during simulation: {str(e)}")
        raise
    
    # Create output directory if it doesn't exist
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to JSON file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"test_water_model_{timestamp}.json")
    
    logger.info(f"Saving results to {output_file}...")
    
    # Convert numpy arrays and other non-serializable objects to serializable types
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    # Serialize the results
    serializable_results = make_serializable(results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info("Simulation complete!")
    return output_file


if __name__ == "__main__":
    print("\nRunning Land Flow Model Test...")
    land_output_file = run_land_model_test()
    print(f"\nLand model simulation results saved to: {land_output_file}")
    
    print("\nRunning Water Model Test...")
    water_output_file = run_water_model_test()
    print(f"\nWater model simulation results saved to: {water_output_file}")
    
    # Print some basic statistics for the land model
    with open(land_output_file, 'r') as f:
        land_results = json.load(f)
    
    print(f"\nLand Model Simulation Statistics:")
    print(f"  Duration: {land_results['runtime_seconds']:.2f} seconds")
    print(f"  Performance: {land_results['steps_per_second']:.2f} timesteps/second")
    print(f"  Particle Status:")
    for status, count in land_results['final_status'].items():
        print(f"    - {status}: {count}")
    print(f"  Mass Balance:")
    print(f"    - Initial: {land_results['mass_balance']['initial_mass']:.2f} kg")
    print(f"    - Remaining: {land_results['mass_balance']['remaining_mass']:.2f} kg")
    print(f"    - Weathered: {land_results['mass_balance']['weathered_percent']:.1f}%")
    
    # Print land-specific metrics if available
    if 'land_summary' in land_results:
        print(f"  Land-Specific Metrics:")
        print(f"    - Average Elevation: {land_results['land_summary']['average_elevation']:.2f} m")
        print(f"    - Average Slope: {land_results['land_summary']['average_slope']:.2f} degrees")
        print(f"    - Absorbed Count: {land_results['land_summary']['absorbed_count']}")
        print(f"    - Soil Distribution: {land_results['land_summary']['soil_distribution']}")
    
    # Print some basic statistics for the water model
    with open(water_output_file, 'r') as f:
        water_results = json.load(f)
    
    print(f"\nWater Model Simulation Statistics:")
    print(f"  Duration: {water_results['runtime_seconds']:.2f} seconds")
    print(f"  Performance: {water_results['steps_per_second']:.2f} timesteps/second")
    print(f"  Particle Status:")
    for status, count in water_results['final_status'].items():
        print(f"    - {status}: {count}")
    print(f"  Mass Balance:")
    print(f"    - Initial: {water_results['mass_balance']['initial_mass']:.2f} kg")
    print(f"    - Remaining: {water_results['mass_balance']['remaining_mass']:.2f} kg")
    print(f"    - Weathered: {water_results['mass_balance']['weathered_percent']:.1f}%")
    
    # Print water-specific metrics if available
    if 'water_summary' in water_results:
        print(f"  Water-Specific Metrics:")
        print(f"    - Beached Count: {water_results['water_summary']['beached_count']}")
        print(f"    - Average Depth: {water_results['water_summary']['average_depth']:.2f} m")
        print(f"    - Evaporated: {water_results['water_summary']['evaporated_percent']:.1f}%")
        print(f"    - Dispersed: {water_results['water_summary']['dispersed_percent']:.1f}%")
