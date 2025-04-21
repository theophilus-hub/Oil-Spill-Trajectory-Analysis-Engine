#!/usr/bin/env python

import os
import json
import argparse
import logging
import numpy as np
from datetime import datetime, timedelta
from trajectory_core import model, preprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OilSpillSimulator:
    """
    A flexible command-line interface for running oil spill simulations with various parameters.
    This class handles parameter parsing, simulation setup, and execution.
    """
    
    def __init__(self):
        """
        Initialize the simulator with default values.
        """
        # Default simulation parameters
        self.default_params = {
            'timestep_minutes': 30,
            'duration_hours': 24,
            'particle_count': 500,
            'random_seed': 42,
            'boundary_method': 'reflect',
            
            # Land-specific parameters
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
            'batch_size': 100,
            
            # Water-specific parameters
            'diffusion_coefficient': 10.0,  # m²/s
            'wind_drift_factor': 0.03,      # 3% of wind speed
            'evaporation_rate': 0.05,       # Fraction per day
            'dispersion_rate': 0.02         # Fraction per day
        }
        
        # Output directory
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Plots directory
        self.plots_dir = "plots"
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def parse_arguments(self):
        """
        Parse command-line arguments for the simulation.
        
        Returns:
            argparse.Namespace: Parsed arguments
        """
        parser = argparse.ArgumentParser(description='Oil Spill Trajectory Simulation Tool')
        
        # Required arguments
        parser.add_argument('--latitude', type=float, required=True,
                            help='Latitude of the spill location')
        parser.add_argument('--longitude', type=float, required=True,
                            help='Longitude of the spill location')
        parser.add_argument('--volume', type=float, required=True,
                            help='Volume of oil spilled in barrels')
        
        # Optional arguments with defaults
        parser.add_argument('--oil-type', type=str, default='medium_crude',
                            help='Type of oil spilled (e.g., light_crude, medium_crude, heavy_crude, bunker)')
        parser.add_argument('--start-time', type=str, default=None,
                            help='Start time of the spill in ISO format (YYYY-MM-DDTHH:MM:SS). Defaults to current time.')
        parser.add_argument('--duration', type=int, default=24,
                            help='Duration of the simulation in hours')
        parser.add_argument('--timestep', type=int, default=30,
                            help='Simulation timestep in minutes')
        parser.add_argument('--particles', type=int, default=500,
                            help='Number of particles to simulate')
        parser.add_argument('--surface-type', type=str, default='auto',
                            choices=['auto', 'land', 'water'],
                            help='Surface type at spill location. "auto" will determine based on elevation.')
        parser.add_argument('--output-prefix', type=str, default='',
                            help='Prefix for output files')
        parser.add_argument('--save-interval', type=int, default=4,
                            help='Save results every N timesteps')
        parser.add_argument('--progress-interval', type=int, default=10,
                            help='Log progress every N timesteps')
        parser.add_argument('--domain-size', type=float, default=1.0,
                            help='Size of the simulation domain in degrees (creates a square domain)')
        parser.add_argument('--verbose', action='store_true',
                            help='Enable verbose logging')
        
        # Advanced parameters
        advanced_group = parser.add_argument_group('Advanced Parameters')
        advanced_group.add_argument('--random-seed', type=int, default=42,
                                  help='Random seed for reproducibility')
        advanced_group.add_argument('--config-file', type=str, default=None,
                                  help='Path to JSON configuration file with additional parameters')
        
        # Export arguments
        export_group = parser.add_argument_group('Export Options')
        export_group.add_argument(
            '--export-format',
            choices=['json', 'geojson', 'csv', 'all', 'mapping'],
            default='all',
            help='Format(s) to export results in. "mapping" includes GeoJSON and time series CSV.'
        )
        export_group.add_argument(
            '--export-dir',
            type=str,
            help='Directory to save exported files (default: ./results)'
        )
        export_group.add_argument(
            '--export-filename',
            type=str,
            help='Base filename for exports (default: auto-generated with timestamp)'
        )
        export_group.add_argument(
            '--no-timestamp',
            action='store_true',
            help='Do not include timestamp in export filenames'
        )
        
        # Parse arguments
        args = parser.parse_args()
        
        # Set logging level based on verbose flag
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        return args
    
    def barrels_to_cubic_meters(self, barrels):
        """
        Convert oil barrels to cubic meters.
        1 barrel of oil = approximately 0.159 cubic meters
        
        Args:
            barrels: Number of oil barrels
            
        Returns:
            float: Volume in cubic meters
        """
        return barrels * 0.159
    
    def determine_surface_type(self, latitude, longitude, elevation_data=None):
        """
        Determine if the given coordinates are on land or water using elevation data.
        
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
        
        # Fallback method: Use approximate coastline lookup
        logger.warning("Using simplified coastline approximation for surface type determination")
        
        # This is a very simplified approach - in a real application, this would use a GIS database
        # For now, we'll use a simple heuristic based on known coastlines for common areas
        
        # Check if it's in an ocean (very approximate)
        if -180 <= longitude <= -30 and -60 <= latitude <= 60:  # Atlantic
            if longitude < -70 and latitude > 25:  # North American East Coast
                return 'water'
            elif longitude < -80 and latitude < 25:  # Caribbean/Gulf of Mexico
                return 'water'
        elif -30 <= longitude <= 40 and -60 <= latitude <= 60:  # Mediterranean/North Sea
            if 30 <= longitude <= 40 and 30 <= latitude <= 40:  # Mediterranean
                return 'water'
        elif 40 <= longitude <= 180 and -60 <= latitude <= 60:  # Indian/Pacific
            if 100 <= longitude <= 150 and -50 <= latitude <= 10:  # Australia/Indonesia
                return 'water'
            elif 120 <= longitude <= 150 and 20 <= latitude <= 45:  # East Asia
                return 'water'
        
        # Default to land if we can't determine
        logger.warning("Could not determine surface type with confidence. Defaulting to land.")
        return 'land'

    def create_mock_environmental_data(self, latitude, longitude, start_time):
        """
        Create mock environmental data for the specified region.
        
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
        # This is a simplified model that creates a coastline near the center
        elevation = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate distance from center
                lat_dist = lats[i] - latitude
                lon_dist = lons[j] - longitude
                
                # Create a simple elevation model with a coastline
                # This is highly simplified and would be replaced with real data in production
                if lon_dist < -0.1:  # Ocean (west of spill)
                    elevation[i, j] = -10 - 5 * abs(lon_dist) * 100
                elif lon_dist < 0.1:  # Coastal zone
                    elevation[i, j] = -5 + 10 * lon_dist * 100 + 2 * np.sin(lats[i] * 10)
                else:  # Inland (east of spill)
                    elevation[i, j] = 5 + 20 * lon_dist * 100
        
        # Create time array for 48 hours (to ensure we cover the simulation period)
        times = [start_time + timedelta(hours=i) for i in range(48)]
        
        # Create mock wind data
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
            if 8 <= hour <= 18:  # Daytime
                u_base = 3.0  # From east
                v_base = 2.0  # From south
                variation = 1.5
            else:  # Nighttime
                u_base = 1.5
                v_base = 1.0
                variation = 0.8
            
            u_field = np.ones((grid_size, grid_size)) * u_base + np.random.normal(0, variation, (grid_size, grid_size))
            v_field = np.ones((grid_size, grid_size)) * v_base + np.random.normal(0, variation, (grid_size, grid_size))
            
            wind_data['u'].append(u_field)
            wind_data['v'].append(v_field)
        
        # Create mock ocean current data
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
                    lon_dist = lons[j] - longitude
                    
                    if lon_dist < -0.1:  # Offshore
                        # Stronger currents offshore
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
    
    def run_simulation(self, args):
        """
        Run an oil spill simulation with the specified parameters.
        
        Args:
            args: Command-line arguments
            
        Returns:
            tuple: (output_file_path, surface_type)
        """
        # Spill parameters
        spill_location = (args.latitude, args.longitude)
        spill_volume_barrels = args.volume
        spill_volume_cubic_meters = self.barrels_to_cubic_meters(spill_volume_barrels)
        
        # Time parameters
        if args.start_time:
            try:
                spill_time = datetime.fromisoformat(args.start_time)
            except ValueError:
                logger.warning(f"Invalid start time format: {args.start_time}. Using current time.")
                spill_time = datetime.now()
        else:
            spill_time = datetime.now()
            
        simulation_duration_hours = args.duration
        end_time = spill_time + timedelta(hours=simulation_duration_hours)
        
        logger.info(f"Simulating oil spill at ({spill_location[0]}, {spill_location[1]})")
        logger.info(f"Spill volume: {spill_volume_barrels} barrels ({spill_volume_cubic_meters:.2f} m³)")
        logger.info(f"Spill time: {spill_time}")
        logger.info(f"Simulation duration: {simulation_duration_hours} hours")
        
        # Create mock environmental data
        mock_data = self.create_mock_environmental_data(spill_location[0], spill_location[1], spill_time)
        
        # Determine surface type
        if args.surface_type == 'auto':
            surface_type = self.determine_surface_type(spill_location[0], spill_location[1], mock_data['elevation_data'])
        else:
            surface_type = args.surface_type
            
        logger.info(f"Surface type: {surface_type}")
        
        # Load additional parameters from config file if provided
        simulation_params = self.default_params.copy()
        if args.config_file and os.path.exists(args.config_file):
            try:
                with open(args.config_file, 'r') as f:
                    config_params = json.load(f)
                    simulation_params.update(config_params)
                    logger.info(f"Loaded parameters from config file: {args.config_file}")
            except Exception as e:
                logger.error(f"Error loading config file: {str(e)}")
        
        # Update parameters from command line arguments
        simulation_params.update({
            'timestep_minutes': args.timestep,
            'duration_hours': args.duration,
            'particle_count': args.particles,
            'random_seed': args.random_seed,
            'domain_bounds': [
                spill_location[0] - args.domain_size/2, 
                spill_location[1] - args.domain_size/2,
                spill_location[0] + args.domain_size/2, 
                spill_location[1] + args.domain_size/2
            ]
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
                    'spill_type': args.oil_type,
                    'location_name': f"Lat: {spill_location[0]}, Lon: {spill_location[1]}"
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
                save_interval=args.save_interval,
                progress_interval=args.progress_interval
            )
        except Exception as e:
            logger.error(f"Error during simulation: {str(e)}")
            raise
        
        # Save results to JSON file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = args.output_prefix + "_" if args.output_prefix else ""
        output_file = os.path.join(self.output_dir, f"{prefix}oil_spill_{timestamp}.json")
        
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

    def export_simulation_results(self, results, args):
        """
        Export simulation results according to command line arguments.
        
        Args:
            results: Simulation results dictionary
            args: Command line arguments
        
        Returns:
            Dictionary mapping export formats to output filenames
        """
        from trajectory_core import export
        
        # Determine output directory
        output_dir = args.export_dir or './results'
        
        # Determine base filename
        if args.export_filename:
            filename_base = args.export_filename
        else:
            # Auto-generate filename based on location and time
            lat = args.latitude
            lon = args.longitude
            location_str = f"{abs(lat):.2f}{'N' if lat >= 0 else 'S'}_{abs(lon):.2f}{'E' if lon >= 0 else 'W'}"
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename_base = f"oil_spill_{location_str}_{timestamp}"
        
        # Export in requested format(s)
        if args.export_format == 'json':
            filepath = export.export_to_json(results, output_dir, f"{filename_base}.json")
            return {'json': filepath}
        
        elif args.export_format == 'geojson':
            filepath = export.export_to_geojson(results, output_dir, f"{filename_base}.geojson")
            return {'geojson': filepath}
        
        elif args.export_format == 'csv':
            filepath = export.export_to_csv(results, output_dir, f"{filename_base}.csv")
            time_series = export.export_to_time_series(results, output_dir, f"{filename_base}_time_series.csv")
            return {'csv': filepath, 'time_series': time_series}
        
        elif args.export_format == 'mapping':
            return export.export_for_mapping(results, output_dir, filename_base)
        
        else:  # 'all' or any other value
            return export.export_all_formats(results, output_dir, filename_base)

    def visualize_results(self, output_file, surface_type):
        """
        Visualize the results of the oil spill simulation.
        
        Args:
            output_file: Path to the simulation results JSON file
            surface_type: 'land' or 'water'
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib is required for visualization. Please install it with 'pip install matplotlib'.")
            return
            
        # Load results
        with open(output_file, 'r') as f:
            results = json.load(f)
        
        # Extract metadata
        metadata = results.get('metadata', {})
        
        # Get spill location
        if 'spill_location' in metadata:
            spill_location = metadata['spill_location']
        else:
            # Default to center of domain bounds if not specified
            domain_bounds = results.get('domain_bounds', [-1, -1, 1, 1])
            spill_location = [
                (domain_bounds[0] + domain_bounds[2]) / 2,
                (domain_bounds[1] + domain_bounds[3]) / 2
            ]
            
        # Get spill volume
        spill_volume = metadata.get('spill_volume', 0)
        
        # Get start and end times
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
        
        # 1. Trajectory plot
        plt.figure(figsize=(14, 12))
        
        # Create a coastline approximation
        coastline_lon_base = spill_location[1] - 0.1  # Simplified coastline
        coastline_lat = np.linspace(spill_location[0] - 0.5, spill_location[0] + 0.5, 200)
        coastline_lon = np.ones_like(coastline_lat) * coastline_lon_base
        
        # Add some irregularity to the coastline
        for i in range(len(coastline_lat)):
            # Add small variations to make it look more natural
            coastline_lon[i] += 0.02 * np.sin(coastline_lat[i] * 50)
        
        plt.plot(coastline_lon, coastline_lat, 'k-', linewidth=2, label='Coastline')
        
        # Color map for time progression
        colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps)))
        
        # Plot initial spill location with larger marker
        plt.scatter(spill_location[1], spill_location[0], color='red', s=200, marker='*', 
                   edgecolor='black', linewidth=1.5, label='Spill Origin')
        
        # Add a circle around the spill location for emphasis
        spill_circle = plt.Circle((spill_location[1], spill_location[0]), 0.01, fill=False, 
                                 color='red', linestyle='--', linewidth=1.5)
        plt.gca().add_patch(spill_circle)
        
        # Plot particle positions at each timestep
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
        
        # Add land/water shading
        x = np.linspace(spill_location[1] - 0.5, spill_location[1] + 0.5, 200)
        y = np.linspace(spill_location[0] - 0.5, spill_location[0] + 0.5, 200)
        X, Y = np.meshgrid(x, y)
        
        # Create a land mask based on the coastline
        land_mask = np.zeros_like(X, dtype=bool)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Check against the coastline
                closest_idx = np.abs(coastline_lat - Y[i, j]).argmin()
                land_mask[i, j] = X[i, j] >= coastline_lon[closest_idx]
        
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
        
        # Add labels and title
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Oil Spill Trajectory Simulation\n'
                  f'Location: {spill_location[0]:.4f}°, {spill_location[1]:.4f}° ({"Land" if surface_type == "land" else "Water"})\n'
                  f'Time: {start_time.strftime("%Y-%m-%d %H:%M")} to {end_time.strftime("%Y-%m-%d %H:%M")}\n'
                  f'Volume: {spill_volume:.2f} m³ of {metadata.get("spill_type", "Oil")}', 
                  fontsize=14)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(spill_location[1] - 0.5, spill_location[1] + 0.5)
        plt.ylim(spill_location[0] - 0.5, spill_location[0] + 0.5)
        
        # Add legend but keep it manageable
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles) > 10:
            step = len(handles) // 8
            handles = handles[:2] + handles[2::step] + handles[-1:]
            labels = labels[:2] + labels[2::step] + labels[-1:]
        plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        
        plt.tight_layout()
        
        # Generate filename from the input file
        base_name = os.path.basename(output_file)
        base_name = os.path.splitext(base_name)[0]
        trajectory_file = os.path.join(self.plots_dir, f"{base_name}_trajectory.png")
        plt.savefig(trajectory_file, bbox_inches='tight', dpi=300)
        logger.info(f"Saved trajectory plot to {trajectory_file}")
        
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
        mass_balance_file = os.path.join(self.plots_dir, f"{base_name}_mass_balance.png")
        plt.savefig(mass_balance_file)
        logger.info(f"Saved mass balance plot to {mass_balance_file}")
        
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
            weathering_file = os.path.join(self.plots_dir, f"{base_name}_weathering.png")
            plt.savefig(weathering_file)
            logger.info(f"Saved weathering processes plot to {weathering_file}")
        
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
                soil_file = os.path.join(self.plots_dir, f"{base_name}_soil_distribution.png")
                plt.savefig(soil_file)
                logger.info(f"Saved soil distribution plot to {soil_file}")
        
        # Print summary statistics
        print("\n===== OIL SPILL SIMULATION RESULTS =====\n")
        
        print(f"Location: ({spill_location[0]:.4f}, {spill_location[1]:.4f})")
        print(f"Surface Type: {surface_type.capitalize()}")
        print(f"Spill Volume: {spill_volume:.2f} m³ of {metadata.get('spill_type', 'Oil')}")
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

def main():
    """
    Main entry point for the oil spill simulation CLI.
    """
    # Parse command line arguments
    args = OilSpillSimulator().parse_arguments()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create simulator
    simulator = OilSpillSimulator()
    
    # Run the simulation
    output_file, surface_type = simulator.run_simulation(args)
    
    # Load the results from the output file
    with open(output_file, 'r') as f:
        results = json.load(f)
    
    # Export results in requested formats
    export_files = simulator.export_simulation_results(results, args)
    
    # Visualize the results
    simulator.visualize_results(output_file, surface_type)
    
    # Print summary
    print("\n===== OIL SPILL SIMULATION RESULTS =====\n")
    
    print(f"Location: ({args.latitude}, {args.longitude})")
    print(f"Surface Type: {surface_type.capitalize()}")
    print(f"Spill Volume: {args.volume:.2f} barrels ({args.volume * 0.159:.2f} m³)")
    
    # Format start and end times
    start_time = results.get('metadata', {}).get('start_time', '')
    if start_time:
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = start_dt + timedelta(hours=args.duration)
        print(f"Simulation Period: {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')}")
    else:
        print(f"Simulation Duration: {args.duration} hours")
    
    # Print export information
    print("\nResults exported to:")
    for format_type, filepath in export_files.items():
        print(f"  - {format_type}: {filepath}")
    
    print("\nSimulation completed successfully!")
    print("Run with --help for more options.")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
