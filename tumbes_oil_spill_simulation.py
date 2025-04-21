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

def determine_surface_type(latitude, longitude, elevation_data=None):
    """
    Determine if the given coordinates are on land or water using elevation data.
    
    For Tumbes, Peru (latitude: -3.5669, longitude: -80.4515), we'll use elevation
    data to make a more accurate determination.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        elevation_data: Optional elevation data dictionary
        
    Returns:
        str: 'land' or 'water'
    """
    # If elevation data is provided, use it to determine surface type
    if elevation_data is not None:
        try:
            # Extract grid information
            lats = elevation_data['data']['lat']
            lons = elevation_data['data']['lon']
            elevation_grid = elevation_data['data']['elevation']
            
            # Find the closest grid point
            lat_idx = np.abs(lats - latitude).argmin()
            lon_idx = np.abs(lons - longitude).argmin()
            
            # Get the elevation at that point
            elevation = elevation_grid[lat_idx, lon_idx]
            
            logger.info(f"Elevation at ({latitude}, {longitude}): {elevation} meters")
            
            # If elevation is positive, it's land; if negative, it's water
            if elevation > 0:
                logger.info("Surface type determined as LAND based on positive elevation")
                return 'land'
            else:
                logger.info("Surface type determined as WATER based on negative or zero elevation")
                return 'water'
        except Exception as e:
            logger.warning(f"Error determining surface type from elevation data: {str(e)}")
            # Fall back to simplified method if elevation lookup fails
    
    # Fallback method: Use approximate coastline
    # For Tumbes, Peru, the coastline is roughly at longitude -80.45
    # This is a simplified approximation
    logger.warning("Using simplified coastline approximation for surface type determination")
    
    # Check specific regions around Tumbes
    # Tumbes city center is inland
    if -80.46 < longitude < -80.44 and -3.58 < latitude < -3.55:
        logger.info("Surface type determined as LAND (Tumbes city area)")
        return 'land'
    # Mangrove sanctuary area
    elif -80.47 < longitude < -80.45 and -3.42 < latitude < -3.38:
        logger.info("Surface type determined as LAND (Mangrove sanctuary area)")
        return 'land'
    # General coastline check
    elif longitude < -80.45:  # West of coastline
        logger.info("Surface type determined as WATER (west of coastline)")
        return 'water'
    else:  # East of coastline
        logger.info("Surface type determined as LAND (east of coastline)")
        return 'land'

def barrels_to_cubic_meters(barrels):
    """
    Convert oil barrels to cubic meters.
    1 barrel of oil = approximately 0.159 cubic meters
    
    Args:
        barrels: Number of oil barrels
        
    Returns:
        float: Volume in cubic meters
    """
    return barrels * 0.159

def create_mock_environmental_data(latitude, longitude, start_time):
    """
    Create mock environmental data for the Tumbes region.
    
    Args:
        latitude: Latitude of the spill location
        longitude: Longitude of the spill location
        start_time: Start time of the simulation
        
    Returns:
        dict: Mock environmental data
    """
    # Create a grid centered around the spill location
    grid_size = 100
    lat_range = 1.0  # Degrees
    lon_range = 1.0  # Degrees
    
    lat_min = latitude - lat_range/2
    lat_max = latitude + lat_range/2
    lon_min = longitude - lon_range/2
    lon_max = longitude + lon_range/2
    
    # Create latitude and longitude arrays
    lats = np.linspace(lat_min, lat_max, grid_size)
    lons = np.linspace(lon_min, lon_max, grid_size)
    
    # Create a 2D grid of elevation values (meters above sea level)
    # For Tumbes, we'll simulate a coastline with mangroves
    elevation = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate distance from coastline (simplified)
            lon_dist = lons[j] - (-80.45)  # Approximate coastline longitude
            
            if lon_dist < 0:  # Ocean
                # Deeper as we move away from coast
                elevation[i, j] = -10 - 5 * abs(lon_dist) * 100
            elif lon_dist < 0.05:  # Mangrove zone (within ~5km of coastline)
                # Mangroves are at or slightly above sea level
                elevation[i, j] = 0.5 + 2 * np.sin(lats[i] * 10) * lon_dist * 100
            else:  # Inland
                # Gradually increasing elevation inland
                elevation[i, j] = 5 + 20 * lon_dist * 100
    
    # Add Tumbes city center with higher elevation
    for i in range(grid_size):
        for j in range(grid_size):
            # Check if this point is near Tumbes city center
            if -80.46 < lons[j] < -80.44 and -3.58 < lats[i] < -3.55:
                # City center is higher than surroundings
                elevation[i, j] = 10 + 5 * np.sin(lats[i] * 20) * np.cos(lons[j] * 20)
    
    # Create time array for 48 hours (to ensure we cover the simulation period)
    times = [start_time + timedelta(hours=i) for i in range(48)]
    
    # Create mock wind data for Tumbes region
    # Tumbes typically has winds from the south/southwest
    wind_data = {
        'time': times,
        'lat': lats,
        'lon': lons,
        'u': [],  # East-west component (m/s)
        'v': []   # North-south component (m/s)
    }
    
    # Generate wind data with daily variation
    for t in range(48):
        hour = (start_time.hour + t) % 24
        
        # Wind is stronger during the day, lighter at night
        # Direction is predominantly from south/southwest
        if 8 <= hour <= 18:  # Daytime
            u_base = -2.0  # From west
            v_base = 4.0   # From south
            variation = 1.5
        else:  # Nighttime
            u_base = -1.0
            v_base = 2.0
            variation = 0.8
        
        u_field = np.ones((grid_size, grid_size)) * u_base + np.random.normal(0, variation, (grid_size, grid_size))
        v_field = np.ones((grid_size, grid_size)) * v_base + np.random.normal(0, variation, (grid_size, grid_size))
        
        wind_data['u'].append(u_field)
        wind_data['v'].append(v_field)
    
    # Create mock ocean current data
    # The Peru/Humboldt Current flows northward along the coast
    current_data = {
        'time': times,
        'lat': lats,
        'lon': lons,
        'u': [],  # East-west component (m/s)
        'v': []   # North-south component (m/s)
    }
    
    # Generate current data
    for t in range(48):
        # Currents are stronger offshore, weaker near coast
        u_field = np.zeros((grid_size, grid_size))
        v_field = np.zeros((grid_size, grid_size))
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Distance from coast (negative is offshore)
                lon_dist = lons[j] - (-80.45)
                
                if lon_dist < 0:  # Offshore
                    # Northward current (Peru Current)
                    # Stronger further offshore
                    u_field[i, j] = 0.1 * np.random.normal(0, 0.05)
                    v_field[i, j] = 0.3 + 0.2 * abs(lon_dist) + np.random.normal(0, 0.1)
                else:  # Coastal/inland
                    # Weak or no currents
                    u_field[i, j] = 0.05 * np.random.normal(0, 0.02)
                    v_field[i, j] = 0.05 * np.random.normal(0, 0.02)
        
        current_data['u'].append(u_field)
        current_data['v'].append(v_field)
    
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
    
    return mock_data

def run_tumbes_oil_spill_simulation():
    """
    Run an oil spill simulation for Tumbes, Peru.
    
    Returns:
        Path to the output JSON file
    """
    # Spill parameters
    spill_location = (-3.5669, -80.4515)  # Tumbes, Peru (latitude, longitude)
    spill_volume_barrels = 20  # Barrels of crude oil
    spill_volume_cubic_meters = barrels_to_cubic_meters(spill_volume_barrels)
    
    # Time parameters
    spill_time = datetime(2025, 4, 21, 5, 25)  # 5:25am on April 21, 2025
    simulation_duration_hours = 24
    end_time = spill_time + timedelta(hours=simulation_duration_hours)
    
    logger.info(f"Simulating oil spill at Tumbes, Peru ({spill_location[0]}, {spill_location[1]})")
    logger.info(f"Spill volume: {spill_volume_barrels} barrels ({spill_volume_cubic_meters:.2f} m³)")
    logger.info(f"Spill time: {spill_time}")
    logger.info(f"Simulation duration: {simulation_duration_hours} hours")
    
    # Create mock environmental data
    mock_data = create_mock_environmental_data(spill_location[0], spill_location[1], spill_time)
    
    # Determine if the spill is on land or water using elevation data
    surface_type = determine_surface_type(spill_location[0], spill_location[1], mock_data['elevation_data'])
    logger.info(f"Determined surface type: {surface_type}")
    
    # Simulation parameters
    simulation_params = {
        'timestep_minutes': 30,
        'duration_hours': simulation_duration_hours,
        'particle_count': 500,  # More particles for better resolution
        'random_seed': 42,
        'domain_bounds': [spill_location[0] - 0.5, spill_location[1] - 0.5, 
                         spill_location[0] + 0.5, spill_location[1] + 0.5],
        'boundary_method': 'reflect',  # Reflect particles at domain boundaries
    }
    
    # Add model-specific parameters based on surface type
    if surface_type == 'land':
        # Land-specific parameters
        simulation_params.update({
            'flow_resistance_factor': 0.2,
            'absorption_rate': 0.01,
            'slope_threshold': 1.0,
            'terrain_roughness_factor': 0.3,
            'flat_terrain_spread_rate': 0.5,
            'use_adaptive_timestep': True,
            'max_substeps': 5,
            'min_substep_fraction': 0.2,
            'adaptive_threshold_slope': 15.0,
            'use_spatial_index': True,
            'use_terrain_caching': True,
            'batch_size': 100
        })
    else:  # water
        # Water-specific parameters
        simulation_params.update({
            'diffusion_coefficient': 10.0,  # m²/s
            'wind_drift_factor': 0.03,      # 3% of wind speed
            'evaporation_rate': 0.05,       # Fraction per day
            'dispersion_rate': 0.02         # Fraction per day
        })
    
    logger.info("Preprocessing environmental data...")
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
            spill_volume_cubic_meters, 
            particle_count=simulation_params['particle_count']
        )
        
        # Set the surface type for all particles
        for particle in particles:
            particle['status'] = 'active'
            particle['surface_type'] = surface_type
        
        # Combine all preprocessed data
        preprocessed_data = {
            'wind': processed_wind,
            'currents': processed_currents,
            'elevation': processed_elevation,
            'particles': particles,
            'metadata': {
                'spill_location': spill_location,
                'spill_volume': spill_volume_cubic_meters,
                'start_time': spill_time.isoformat(),
                'end_time': end_time.isoformat(),
                'time_step_hours': simulation_params['timestep_minutes'] / 60,
                'target_resolution': 30.0,
                'particle_count': simulation_params['particle_count'],
                'processing_timestamp': datetime.now().isoformat(),
                'spill_type': 'crude_oil',
                'location_name': 'Tumbes, Peru'
            }
        }
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise
    
    # Initialize the appropriate model
    if surface_type == 'land':
        logger.info("Creating and initializing Land Flow model...")
        simulation_model = model.LandFlowModel(simulation_params)
    else:  # water
        logger.info("Creating and initializing Lagrangian model...")
        simulation_model = model.LagrangianModel(simulation_params)
    
    simulation_model.initialize(preprocessed_data)
    
    logger.info("Running simulation...")
    try:
        results = simulation_model.run_simulation(
            save_interval=4,  # Save every 4 timesteps (2 hours)
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
    output_file = os.path.join(output_dir, f"tumbes_oil_spill_{timestamp}.json")
    
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
    return output_file, surface_type

def visualize_tumbes_simulation(output_file, surface_type):
    """
    Visualize the results of the Tumbes oil spill simulation.
    
    Args:
        output_file: Path to the simulation results JSON file
        surface_type: 'land' or 'water'
    """
    # Load results
    with open(output_file, 'r') as f:
        results = json.load(f)
    
    # Create output directory for plots
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract metadata
    metadata = results.get('metadata', {})
    
    # Define the spill location directly since it might not be in metadata
    spill_location = (-3.5669, -80.4515)  # Tumbes, Peru coordinates
    spill_volume = 3.18  # 20 barrels in cubic meters
    
    # Get start and end times from metadata or timesteps
    if 'start_time' in metadata and 'end_time' in metadata:
        start_time = datetime.fromisoformat(metadata['start_time'])
        end_time = datetime.fromisoformat(metadata['end_time'])
    else:
        # Try to get times from timesteps
        if results['timesteps'] and len(results['timesteps']) > 0:
            start_time = datetime.fromisoformat(results['timesteps'][0]['time'])
            end_time = datetime.fromisoformat(results['timesteps'][-1]['time'])
        else:
            # Default to current time if all else fails
            start_time = datetime.now()
            end_time = start_time + timedelta(hours=24)
    
    # Extract timesteps
    timesteps = results['timesteps']
    
    # 1. Detailed Particle trajectory plot
    plt.figure(figsize=(14, 12))
    
    # Create a more detailed coastline
    # For Tumbes, the coastline is irregular with mangroves
    coastline_lon_base = -80.45
    coastline_lat = np.linspace(spill_location[0] - 0.5, spill_location[0] + 0.5, 200)
    coastline_lon = np.ones_like(coastline_lat) * coastline_lon_base
    
    # Add some irregularity to the coastline to represent mangroves and inlets
    for i in range(len(coastline_lat)):
        # Add small variations to make it look more natural
        if -3.42 < coastline_lat[i] < -3.38:  # Mangrove sanctuary area
            # More irregular in mangrove areas
            coastline_lon[i] += 0.02 * np.sin(coastline_lat[i] * 100) + 0.01
        else:
            coastline_lon[i] += 0.01 * np.sin(coastline_lat[i] * 50)
    
    plt.plot(coastline_lon, coastline_lat, 'k-', linewidth=2, label='Coastline')
    
    # Add city marker for Tumbes
    plt.scatter(-80.455, -3.57, color='gray', s=120, marker='s', label='Tumbes City')
    
    # Color map for time progression
    colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps)))
    
    # Plot initial spill location with larger marker and different style
    plt.scatter(spill_location[1], spill_location[0], color='red', s=200, marker='*', 
               edgecolor='black', linewidth=1.5, label='Spill Origin')
    
    # Add a circle around the spill location for emphasis
    spill_circle = plt.Circle((spill_location[1], spill_location[0]), 0.01, fill=False, 
                             color='red', linestyle='--', linewidth=1.5)
    plt.gca().add_patch(spill_circle)
    
    # Plot particle positions at each timestep with better visibility
    for i, timestep in enumerate(timesteps):
        # Skip some timesteps for clarity if there are many
        if i % 2 != 0 and i != len(timesteps) - 1:
            continue
            
        particles = timestep['particles']
        lats = [p['latitude'] for p in particles if p['status'] == 'active']
        lons = [p['longitude'] for p in particles if p['status'] == 'active']
        
        if lats and lons:  # Only plot if there are active particles
            time_str = datetime.fromisoformat(timestep['time']).strftime('%Y-%m-%d %H:%M')
            plt.scatter(lons, lats, color=colors[i], s=15, alpha=0.7, 
                       edgecolor='none', label=f'T+{i*2}h ({time_str})')
    
    # Add land/water shading with more detail
    x = np.linspace(spill_location[1] - 0.5, spill_location[1] + 0.5, 200)
    y = np.linspace(spill_location[0] - 0.5, spill_location[0] + 0.5, 200)
    X, Y = np.meshgrid(x, y)
    
    # Create a more detailed land mask
    land_mask = np.zeros_like(X, dtype=bool)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Check against the irregular coastline
            closest_idx = np.abs(coastline_lat - Y[i, j]).argmin()
            land_mask[i, j] = X[i, j] >= coastline_lon[closest_idx]
    
    # Use better colors for land/water
    plt.contourf(X, Y, land_mask, colors=['#b3d9ff', '#90EE90'], alpha=0.3, levels=[0, 0.5, 1])
    
    # Add compass rose
    compass_x = spill_location[1] - 0.45
    compass_y = spill_location[0] - 0.45
    compass_size = 0.05
    
    plt.arrow(compass_x, compass_y, 0, compass_size, head_width=compass_size/5, 
             head_length=compass_size/5, fc='k', ec='k')
    plt.arrow(compass_x, compass_y, compass_size, 0, head_width=compass_size/5, 
             head_length=compass_size/5, fc='k', ec='k')
    
    plt.text(compass_x, compass_y + compass_size*1.2, 'N', ha='center', fontsize=12)
    plt.text(compass_x + compass_size*1.2, compass_y, 'E', va='center', fontsize=12)
    
    # Add scale bar
    scale_lon = spill_location[1] + 0.4
    scale_lat = spill_location[0] - 0.45
    scale_length = 0.1  # approximately 10 km at this latitude
    
    plt.plot([scale_lon, scale_lon + scale_length], [scale_lat, scale_lat], 'k-', linewidth=2)
    plt.text(scale_lon + scale_length/2, scale_lat - 0.02, '~10 km', ha='center')
    
    # Add labels and title with more information
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Tumbes, Peru Oil Spill Trajectory Simulation\n'
              f'Location: {spill_location[0]:.4f}°, {spill_location[1]:.4f}° ({"Land" if surface_type == "land" else "Water"})\n'
              f'Time: {start_time.strftime("%Y-%m-%d %H:%M")} to {end_time.strftime("%Y-%m-%d %H:%M")}\n'
              f'Volume: {spill_volume:.2f} m³ (20 barrels) of Crude Oil', 
              fontsize=14)
    
    # Add gridlines for better coordinate reference
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Improve axis limits to focus on the relevant area
    plt.xlim(spill_location[1] - 0.5, spill_location[1] + 0.5)
    plt.ylim(spill_location[0] - 0.5, spill_location[0] + 0.5)
    
    # Add legend but keep it manageable
    handles, labels = plt.gca().get_legend_handles_labels()
    # Select a subset of timesteps for the legend to avoid overcrowding
    if len(handles) > 10:
        step = len(handles) // 8
        handles = handles[:2] + handles[2::step] + handles[-1:]
        labels = labels[:2] + labels[2::step] + labels[-1:]
    plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'tumbes_trajectory_detailed.png'), bbox_inches='tight', dpi=300)
    
    # 2. Mass balance over time
    plt.figure(figsize=(10, 6))
    
    times = []
    remaining_mass = []
    weathered_mass = []
    
    initial_mass = results['mass_balance']['initial_mass']
    
    for i, timestep in enumerate(timesteps):
        if 'mass_balance' in timestep:
            times.append(i * 2)  # Hours since start
            remaining_mass.append(timestep['mass_balance']['remaining_mass'])
            weathered_mass.append(initial_mass - timestep['mass_balance']['remaining_mass'])
    
    plt.stackplot(times, [remaining_mass, weathered_mass], 
                 labels=['Remaining', 'Weathered'],
                 colors=['royalblue', 'darkred'], alpha=0.7)
    
    plt.xlabel('Hours Since Spill')
    plt.ylabel('Mass (kg)')
    plt.title('Oil Mass Balance Over Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'tumbes_mass_balance.png'))
    
    # 3. Model-specific metrics
    if surface_type == 'water' and 'water_summary' in results:
        plt.figure(figsize=(10, 6))
        
        metrics = ['Evaporated', 'Dispersed']
        values = [
            results['water_summary']['evaporated_percent'],
            results['water_summary']['dispersed_percent']
        ]
        
        plt.bar(metrics, values, color=['orange', 'purple'])
        plt.title('Oil Weathering Processes')
        plt.ylabel('Percentage (%)')
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'tumbes_weathering.png'))
    
    elif surface_type == 'land' and 'land_summary' in results:
        if results['land_summary']['soil_distribution']:
            plt.figure(figsize=(10, 6))
            
            soil_types = list(results['land_summary']['soil_distribution'].keys())
            soil_values = list(results['land_summary']['soil_distribution'].values())
            
            plt.bar(soil_types, soil_values, color='sienna')
            plt.title('Oil Distribution by Soil Type')
            plt.ylabel('Count')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'tumbes_soil_distribution.png'))
    
    # Print summary statistics
    print("\n===== TUMBES OIL SPILL SIMULATION RESULTS =====\n")
    
    print(f"Location: Tumbes, Peru ({spill_location[0]}, {spill_location[1]})")
    print(f"Surface Type: {surface_type.capitalize()}")
    print(f"Spill Volume: {spill_volume:.2f} m³ of Crude Oil")
    print(f"Simulation Period: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"Runtime: {results['runtime_seconds']:.2f} seconds")
    print(f"Performance: {results['steps_per_second']:.2f} timesteps/second")
    
    print(f"\nParticle Status at End of Simulation:")
    for status, count in results['final_status'].items():
        print(f"  - {status}: {count}")
    
    print(f"\nMass Balance:")
    print(f"  - Initial: {results['mass_balance']['initial_mass']:.2f} kg")
    print(f"  - Remaining: {results['mass_balance']['remaining_mass']:.2f} kg")
    print(f"  - Weathered: {results['mass_balance']['weathered_percent']:.1f}%")
    
    if surface_type == 'water' and 'water_summary' in results:
        print(f"\nWater-Specific Metrics:")
        print(f"  - Beached Count: {results['water_summary']['beached_count']}")
        print(f"  - Average Depth: {results['water_summary']['average_depth']:.2f} m")
        print(f"  - Evaporated: {results['water_summary']['evaporated_percent']:.1f}%")
        print(f"  - Dispersed: {results['water_summary']['dispersed_percent']:.1f}%")
    
    elif surface_type == 'land' and 'land_summary' in results:
        print(f"\nLand-Specific Metrics:")
        print(f"  - Average Elevation: {results['land_summary']['average_elevation']:.2f} m")
        print(f"  - Average Slope: {results['land_summary']['average_slope']:.2f} degrees")
        print(f"  - Absorbed Count: {results['land_summary']['absorbed_count']}")
        
        if results['land_summary']['soil_distribution']:
            print(f"  - Soil Distribution: {results['land_summary']['soil_distribution']}")
        else:
            print(f"  - Soil Distribution: Empty (no soil type data available)")
    
    print("\nPlots have been saved to the 'plots' directory.")

if __name__ == "__main__":
    print("\nRunning Tumbes, Peru Oil Spill Simulation...")
    output_file, surface_type = run_tumbes_oil_spill_simulation()
    print(f"\nSimulation results saved to: {output_file}")
    
    print("\nGenerating visualization...")
    visualize_tumbes_simulation(output_file, surface_type)
