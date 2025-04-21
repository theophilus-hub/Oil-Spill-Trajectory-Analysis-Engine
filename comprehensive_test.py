import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from trajectory_core import model, preprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_terrain_data():
    """
    Create mock terrain data for testing purposes.
    
    Returns:
        Tuple of (mock_data, spill_location)
    """
    # Create a grid for elevation data
    grid_size = 100
    lat_min, lon_min = 39.0, -71.0
    lat_max, lon_max = 41.0, -69.0
    
    # Create latitude and longitude arrays
    lats = np.linspace(lat_min, lat_max, grid_size)
    lons = np.linspace(lon_min, lon_max, grid_size)
    
    # Create a 2D grid of elevation values (meters above sea level)
    # Simulate a coastline with higher elevation towards the north
    elevation = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            # Higher elevation in the northern part (higher latitude)
            if lats[i] > 40.0:
                # Create some hills and valleys
                elevation[i, j] = 20 * (lats[i] - 40.0) * 100 + 5 * np.sin(lons[j] * 10) + 3 * np.cos(lats[i] * 15)
            else:
                # Water area (negative elevation)
                elevation[i, j] = -10 - 5 * np.sin(lons[j] * 5)
    
    # Create mock wind data
    wind_data = {
        'time': [datetime.now() + timedelta(hours=i) for i in range(48)],
        'lat': lats,
        'lon': lons,
        'u': [np.random.normal(5, 2, (grid_size, grid_size)) for _ in range(48)],  # u component (m/s)
        'v': [np.random.normal(2, 1, (grid_size, grid_size)) for _ in range(48)]   # v component (m/s)
    }
    
    # Create mock ocean current data
    current_data = {
        'time': [datetime.now() + timedelta(hours=i) for i in range(48)],
        'lat': lats,
        'lon': lons,
        'u': [np.random.normal(0.5, 0.2, (grid_size, grid_size)) for _ in range(48)],  # u component (m/s)
        'v': [np.random.normal(0.2, 0.1, (grid_size, grid_size)) for _ in range(48)]   # v component (m/s)
    }
    
    # Create mock elevation data in the format expected by the preprocessor
    elevation_data = {
        'data': {
            'lat': lats,
            'lon': lons,
            'elevation': elevation,
            'bbox': [lon_min, lat_min, lon_max, lat_max]
        }
    }
    
    # Package all mock data
    mock_data = {
        'wind_data': wind_data,
        'current_data': current_data,
        'elevation_data': elevation_data
    }
    
    # Set spill location in the water area
    spill_location = (39.5, -70.0)  # Latitude, Longitude
    
    return mock_data, spill_location

def run_land_model_test():
    """
    Run a test simulation with the land-based flow model and save the results to a JSON file.
    
    Returns:
        Path to the output JSON file
    """
    logger.info("Setting up test parameters...")
    
    # Simulation parameters
    simulation_params = {
        'timestep_minutes': 30,
        'duration_hours': 24,    # Longer duration for comprehensive test
        'particle_count': 500,   # More particles for better statistics
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
        'batch_size': 100
    }
    
    # Spill parameters
    mock_data, _ = create_mock_terrain_data()
    spill_location = (40.2, -70.0)  # Latitude, Longitude - on land
    spill_volume = 1000.0  # m³
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=simulation_params['duration_hours'])
    
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
            save_interval=6,  # Save every 6 timesteps (3 hours)
            progress_interval=10  # Log progress every 10 timesteps
        )
    except Exception as e:
        logger.error(f"Error during simulation: {str(e)}")
        raise
    
    # Create output directory if it doesn't exist
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to JSON file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"comprehensive_land_model_{timestamp}.json")
    
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
        'timestep_minutes': 30,
        'duration_hours': 24,    # Longer duration for comprehensive test
        'particle_count': 500,   # More particles for better statistics
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
    mock_data, _ = create_mock_terrain_data()
    spill_location = (39.5, -70.0)  # Latitude, Longitude - on water
    spill_volume = 1000.0  # m³
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=simulation_params['duration_hours'])
    
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
            save_interval=6,  # Save every 6 timesteps (3 hours)
            progress_interval=10  # Log progress every 10 timesteps
        )
    except Exception as e:
        logger.error(f"Error during simulation: {str(e)}")
        raise
    
    # Create output directory if it doesn't exist
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to JSON file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"comprehensive_water_model_{timestamp}.json")
    
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

def analyze_and_visualize_results(land_file, water_file):
    """
    Analyze and visualize the results from both land and water models.
    
    Args:
        land_file: Path to the land model results JSON file
        water_file: Path to the water model results JSON file
    """
    # Load results
    with open(land_file, 'r') as f:
        land_results = json.load(f)
    
    with open(water_file, 'r') as f:
        water_results = json.load(f)
    
    # Create output directory for plots
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Performance comparison
    plt.figure(figsize=(10, 6))
    models = ['Land Model', 'Water Model']
    runtimes = [land_results['runtime_seconds'], water_results['runtime_seconds']]
    steps_per_second = [land_results['steps_per_second'], water_results['steps_per_second']]
    
    plt.subplot(1, 2, 1)
    plt.bar(models, runtimes)
    plt.title('Runtime Comparison')
    plt.ylabel('Runtime (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.bar(models, steps_per_second)
    plt.title('Performance Comparison')
    plt.ylabel('Steps per Second')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'performance_comparison.png'))
    
    # 2. Mass balance comparison
    plt.figure(figsize=(12, 6))
    
    # Land model mass balance
    land_initial = land_results['mass_balance']['initial_mass']
    land_remaining = land_results['mass_balance']['remaining_mass']
    land_weathered = land_initial - land_remaining
    
    # Water model mass balance
    water_initial = water_results['mass_balance']['initial_mass']
    water_remaining = water_results['mass_balance']['remaining_mass']
    water_weathered = water_initial - water_remaining
    
    # Calculate percentages for pie charts
    land_percentages = [land_remaining/land_initial*100, land_weathered/land_initial*100]
    water_percentages = [water_remaining/water_initial*100, water_weathered/water_initial*100]
    
    plt.subplot(1, 2, 1)
    plt.pie(land_percentages, labels=['Remaining', 'Weathered'], autopct='%1.1f%%', startangle=90)
    plt.title('Land Model Mass Balance')
    
    plt.subplot(1, 2, 2)
    plt.pie(water_percentages, labels=['Remaining', 'Weathered'], autopct='%1.1f%%', startangle=90)
    plt.title('Water Model Mass Balance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mass_balance_comparison.png'))
    
    # 3. Particle status distribution
    plt.figure(figsize=(12, 6))
    
    # Land model particle status
    land_statuses = list(land_results['final_status'].keys())
    land_counts = list(land_results['final_status'].values())
    
    # Water model particle status
    water_statuses = list(water_results['final_status'].keys())
    water_counts = list(water_results['final_status'].values())
    
    plt.subplot(1, 2, 1)
    plt.bar(land_statuses, land_counts)
    plt.title('Land Model Particle Status')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.bar(water_statuses, water_counts)
    plt.title('Water Model Particle Status')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'particle_status_comparison.png'))
    
    # 4. Water-specific metrics
    if 'water_summary' in water_results:
        plt.figure(figsize=(8, 6))
        metrics = ['Evaporated', 'Dispersed']
        values = [
            water_results['water_summary']['evaporated_percent'],
            water_results['water_summary']['dispersed_percent']
        ]
        
        plt.bar(metrics, values)
        plt.title('Water Model Weathering Processes')
        plt.ylabel('Percentage (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'water_weathering_processes.png'))
    
    # 5. Land-specific metrics
    if 'land_summary' in land_results and land_results['land_summary']['soil_distribution']:
        plt.figure(figsize=(8, 6))
        soil_types = list(land_results['land_summary']['soil_distribution'].keys())
        soil_values = list(land_results['land_summary']['soil_distribution'].values())
        
        plt.bar(soil_types, soil_values)
        plt.title('Land Model Soil Distribution')
        plt.ylabel('Count')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'land_soil_distribution.png'))
    
    # Print summary statistics
    print("\n===== COMPREHENSIVE TEST RESULTS =====\n")
    
    print("Land Model Statistics:")
    print(f"  Runtime: {land_results['runtime_seconds']:.2f} seconds")
    print(f"  Performance: {land_results['steps_per_second']:.2f} timesteps/second")
    print(f"  Particle Status:")
    for status, count in land_results['final_status'].items():
        print(f"    - {status}: {count}")
    print(f"  Mass Balance:")
    print(f"    - Initial: {land_results['mass_balance']['initial_mass']:.2f} kg")
    print(f"    - Remaining: {land_results['mass_balance']['remaining_mass']:.2f} kg")
    print(f"    - Weathered: {land_results['mass_balance']['weathered_percent']:.1f}%")
    
    if 'land_summary' in land_results:
        print(f"  Land-Specific Metrics:")
        print(f"    - Average Elevation: {land_results['land_summary']['average_elevation']:.2f} m")
        print(f"    - Average Slope: {land_results['land_summary']['average_slope']:.2f} degrees")
        print(f"    - Absorbed Count: {land_results['land_summary']['absorbed_count']}")
        if land_results['land_summary']['soil_distribution']:
            print(f"    - Soil Distribution: {land_results['land_summary']['soil_distribution']}")
        else:
            print(f"    - Soil Distribution: Empty (no soil type data available)")
    
    print("\nWater Model Statistics:")
    print(f"  Runtime: {water_results['runtime_seconds']:.2f} seconds")
    print(f"  Performance: {water_results['steps_per_second']:.2f} timesteps/second")
    print(f"  Particle Status:")
    for status, count in water_results['final_status'].items():
        print(f"    - {status}: {count}")
    print(f"  Mass Balance:")
    print(f"    - Initial: {water_results['mass_balance']['initial_mass']:.2f} kg")
    print(f"    - Remaining: {water_results['mass_balance']['remaining_mass']:.2f} kg")
    print(f"    - Weathered: {water_results['mass_balance']['weathered_percent']:.1f}%")
    
    if 'water_summary' in water_results:
        print(f"  Water-Specific Metrics:")
        print(f"    - Beached Count: {water_results['water_summary']['beached_count']}")
        print(f"    - Average Depth: {water_results['water_summary']['average_depth']:.2f} m")
        print(f"    - Evaporated: {water_results['water_summary']['evaporated_percent']:.1f}%")
        print(f"    - Dispersed: {water_results['water_summary']['dispersed_percent']:.1f}%")
    
    print("\nPlots have been saved to the 'plots' directory.")

if __name__ == "__main__":
    print("\nRunning Comprehensive Land Flow Model Test...")
    land_output_file = run_land_model_test()
    print(f"\nLand model simulation results saved to: {land_output_file}")
    
    print("\nRunning Comprehensive Water Model Test...")
    water_output_file = run_water_model_test()
    print(f"\nWater model simulation results saved to: {water_output_file}")
    
    # Analyze and visualize the results
    analyze_and_visualize_results(land_output_file, water_output_file)
