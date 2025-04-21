#!/usr/bin/env python
"""
Test script for the Oil Spill Trajectory Analysis Engine.

This script creates a simple test case with mock data and runs the Lagrangian model,
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


def create_mock_data():
    """
    Create mock environmental data for testing the model.
    
    Returns:
        Dictionary containing mock wind, current, and elevation data
    """
    # Create a simple grid centered at the spill location
    spill_lat, spill_lon = 40.0, -70.0  # Example location in the Atlantic
    grid_size = 10
    grid_resolution = 0.1  # degrees
    
    # Create lat/lon grid
    lats = np.linspace(spill_lat - grid_size/2 * grid_resolution, 
                      spill_lat + grid_size/2 * grid_resolution, 
                      grid_size)
    lons = np.linspace(spill_lon - grid_size/2 * grid_resolution, 
                      spill_lon + grid_size/2 * grid_resolution, 
                      grid_size)
    
    # Create mock elevation data (simple bathymetry)
    elevation_grid = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            # Create a simple slope from west to east
            elevation_grid[i, j] = -100 + j * 10  # Depth in meters
    
    # Create a grid of points for elevation data
    elevation_data_points = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            elevation_data_points.append({
                'latitude': float(lat),
                'longitude': float(lon),
                'elevation': float(elevation_grid[i, j])
            })
    
    # Create mock wind data (constant wind from west)
    start_time = datetime.now()
    times = []
    wind_u = []
    wind_v = []
    
    # Create 48 hours of hourly wind data
    for hour in range(48):
        time = start_time + timedelta(hours=hour)
        times.append(time.isoformat())
        
        # Wind from west (u = 5 m/s, v = 0 m/s)
        wind_u.append(5.0)
        wind_v.append(0.0)
    
    # Create mock current data (simple circular pattern)
    current_u = []
    current_v = []
    
    # Create 48 hours of hourly current data
    for hour in range(48):
        # Circular current pattern
        angle = hour * (2 * np.pi / 24)  # Complete cycle every 24 hours
        current_u.append(0.5 * np.cos(angle))  # 0.5 m/s maximum
        current_v.append(0.5 * np.sin(angle))  # 0.5 m/s maximum
    
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
                'bounds': {
                    'min_lat': float(min(lats)),
                    'max_lat': float(max(lats)),
                    'min_lon': float(min(lons)),
                    'max_lon': float(max(lons))
                },
                'resolution': float(grid_resolution)
            }
        }
    }
    
    return mock_data, (spill_lat, spill_lon)


def run_test_simulation():
    """
    Run a test simulation with mock data and save the results to a JSON file.
    
    Returns:
        Path to the output JSON file
    """
    logger.info("Creating mock environmental data...")
    mock_data, spill_location = create_mock_data()
    
    # Set simulation parameters
    simulation_params = {
        'duration_hours': 24,
        'timestep_minutes': 30,
        'particle_count': 100,
        'diffusion_coefficient': 10.0,  # m²/s
        'wind_influence_factor': 0.03,
        'random_seed': 42,  # For reproducibility
        'domain_bounds': (39.5, -70.5, 40.5, -69.5),  # (min_lat, min_lon, max_lat, max_lon)
        'boundary_method': 'reflect',  # 'reflect', 'absorb', or 'periodic'
        'max_evaporable_fraction': 0.3,
        'base_evaporation_rate': 0.05,  # fraction per hour
        'base_dissolution_rate': 0.005,  # fraction per hour
        'base_biodegradation_rate': 0.001  # fraction per hour
    }
    
    # Spill parameters
    spill_volume = 1000.0  # liters
    start_time = datetime.now()
    
    logger.info("Preprocessing data...")
    # Create a preprocessor instance
    preprocessor = preprocess.DataPreprocessor()
    
    # Preprocess individual data components
    try:
        # Preprocess wind data
        processed_wind = preprocessor.preprocess_wind_data(mock_data['wind_data'])
        
        # Preprocess current data
        processed_currents = preprocessor.preprocess_ocean_currents(mock_data['current_data'])
        
        # Preprocess elevation data
        processed_elevation = preprocessor.preprocess_elevation_data(mock_data['elevation_data'])
        
        # Initialize particles
        particles = preprocessor.initialize_particles(
            spill_location, 
            spill_volume, 
            particle_count=simulation_params['particle_count']
        )
        
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
                'end_time': (start_time + timedelta(hours=simulation_params['duration_hours'])).isoformat(),
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
    lagrangian_model = model.LagrangianModel(simulation_params)
    lagrangian_model.initialize(preprocessed_data)
    
    logger.info("Running simulation...")
    # Run the simulation
    try:
        results = lagrangian_model.run_simulation(
            save_interval=2,  # Save every 2 timesteps
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
    output_file = os.path.join(output_dir, f"test_simulation_{timestamp}.json")
    
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
    output_file = run_test_simulation()
    print(f"\nSimulation results saved to: {output_file}")
    
    # Print some basic statistics
    with open(output_file, 'r') as f:
        results = json.load(f)
    
    print(f"\nSimulation Statistics:")
    print(f"  Duration: {results['runtime_seconds']:.2f} seconds")
    print(f"  Performance: {results['steps_per_second']:.2f} timesteps/second")
    print(f"  Particle Status:")
    for status, count in results['final_status'].items():
        print(f"    - {status}: {count}")
    print(f"  Mass Balance:")
    print(f"    - Initial: {results['mass_balance']['initial_mass']:.2f} kg")
    print(f"    - Remaining: {results['mass_balance']['remaining_mass']:.2f} kg")
    print(f"    - Weathered: {results['mass_balance']['weathered_percent']:.1f}%")
