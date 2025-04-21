"""
Main orchestration module for the Oil Spill Trajectory Analysis Engine.

This module provides the main entry point and orchestration for the simulation:
- Coordinates the data acquisition, preprocessing, modeling, and export steps
- Provides a simple CLI interface for running simulations
- Handles configuration and parameter management
- Implements progress reporting and error handling
"""

import os
import sys
import json
import logging
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Callable
from pathlib import Path

# Import project modules
from trajectory_core import fetch_data, preprocess, model, export, config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Completely remove signal handling to avoid the "signal only works in main thread" error
# We'll handle graceful shutdown at the application level instead


class SimulationManager:
    """Main class for orchestrating the oil spill trajectory simulation."""
    
    def __init__(self, simulation_params: Optional[Dict[str, Any]] = None):
        """Initialize the simulation manager.
        
        Args:
            simulation_params: Optional dictionary of simulation parameters
        """
        # Initialize simulation parameters
        self.params = simulation_params or {}
        
        # Initialize simulation state
        self.simulation_state = {
            'progress': 0.0,
            'current_stage': 'initialized',
            'status': 'initialized'
        }
        
        # Create a temporary directory for intermediate files
        self.temp_dir = tempfile.mkdtemp(prefix="oil_spill_sim_")
        logger.debug(f"Created temporary directory: {self.temp_dir}")
        
        # Initialize progress callback
        self.progress_callback = None
    
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        try:
            # Remove temporary directory and its contents
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.debug(f"Removed temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def set_progress_callback(self, callback):
        """
        Set a callback function for progress reporting.
        
        Args:
            callback: Function that takes progress (float) and stage (str) as arguments
        """
        self.progress_callback = callback
    
    def _update_progress(self, progress, stage=None):
        """
        Update progress and call the progress callback if set.
        
        Args:
            progress: Progress value (0-100)
            stage: Current stage of the simulation
        """
        self.simulation_state['progress'] = progress
        if stage:
            self.simulation_state['current_stage'] = stage
        
        # Call progress callback if set
        if hasattr(self, 'progress_callback') and self.progress_callback:
            try:
                self.progress_callback(progress, stage)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    def run_simulation(self, 
                      spill_location: Tuple[float, float],
                      spill_volume: float,
                      oil_type: str,
                      model_type: str = 'hybrid',
                      duration_hours: int = 72,
                      timestep_minutes: int = 30,
                      output_formats: List[str] = None,
                      output_directory: str = None) -> Dict[str, Any]:
        """Run the simulation with the given parameters.
        
        Args:
            spill_location: Tuple of (latitude, longitude)
            spill_volume: Volume of the spill in cubic meters
            oil_type: Type of oil (e.g., 'light_crude', 'medium_crude', 'heavy_crude')
            model_type: Type of model to use (e.g., 'simple', 'hybrid', 'advanced')
            duration_hours: Duration of the simulation in hours
            timestep_minutes: Timestep of the simulation in minutes
            output_formats: List of output formats (e.g., ['geojson', 'json', 'csv'])
            output_directory: Directory to save output files
            
        Returns:
            Dictionary containing simulation results
        """
        # Set default output formats if not provided
        if output_formats is None:
            output_formats = ['geojson', 'json', 'csv']
        
        # Set default output directory if not provided
        if output_directory is None:
            output_directory = config.OUTPUT_CONFIG['output_directory']
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Update simulation state
        self.simulation_state['status'] = 'running'
        self.simulation_state['start_time'] = datetime.now()
        self.simulation_state['progress'] = 0.0
        self.simulation_state['current_stage'] = 'starting'
        
        try:
            # Validate inputs
            self._validate_inputs(spill_location, spill_volume, oil_type, model_type)
            
            logger.info(f"Starting simulation for spill at {spill_location}")
            
            # Step 1: Fetch environmental data (20% of progress)
            self.simulation_state['current_stage'] = 'data_acquisition'
            logger.info("Fetching environmental data...")
            
            lat, lon = spill_location
            
            # Get wind data
            logger.debug("Fetching wind data...")
            try:
                wind_data = fetch_data.get_wind_data(lat, lon)
                self._update_progress(5.0, 'data_acquisition')
                logger.debug("Wind data fetched successfully")
            except Exception as e:
                logger.error(f"Error fetching wind data: {e}")
                raise RuntimeError(f"Failed to fetch wind data: {e}")
            
            # Get ocean current data
            logger.debug("Fetching ocean current data...")
            try:
                current_data = fetch_data.get_ocean_currents(lat, lon)
                self._update_progress(10.0, 'data_acquisition')
                logger.debug("Ocean current data fetched successfully")
            except Exception as e:
                logger.error(f"Error fetching ocean current data: {e}")
                raise RuntimeError(f"Failed to fetch ocean current data: {e}")
            
            # Get elevation data for a bounding box around the spill
            # Approximately 10km in each direction
            buffer = 0.1  # Roughly 10km in decimal degrees
            logger.debug("Fetching elevation data...")
            try:
                elevation_data = fetch_data.get_elevation_data(
                    lat - buffer, lon - buffer,
                    lat + buffer, lon + buffer
                )
                self._update_progress(15.0, 'data_acquisition')
                logger.debug("Elevation data fetched successfully")
            except Exception as e:
                logger.error(f"Error fetching elevation data: {e}")
                raise RuntimeError(f"Failed to fetch elevation data: {e}")
            
            # Get oil properties
            logger.debug(f"Fetching properties for oil type: {oil_type}")
            try:
                oil_properties = fetch_data.get_oil_properties(oil_type)
                self._update_progress(20.0, 'data_acquisition')
                logger.debug("Oil properties fetched successfully")
            except Exception as e:
                logger.error(f"Error fetching oil properties: {e}")
                raise RuntimeError(f"Failed to fetch oil properties: {e}")
            
            # Step 2: Preprocess data (20% of progress)
            self.simulation_state['current_stage'] = 'preprocessing'
            logger.info("Preprocessing data...")
            
            try:
                preprocessed_data = preprocess.preprocess_all_data(
                    wind_data, current_data, elevation_data,
                    spill_location, spill_volume,
                    self.params.get('particle_count', 1000)
                )
                self._update_progress(40.0, 'preprocessing')
                logger.debug("Data preprocessing completed successfully")
            except Exception as e:
                logger.error(f"Error preprocessing data: {e}")
                raise RuntimeError(f"Failed to preprocess data: {e}")
            
            # Step 3: Run the model (40% of progress)
            self.simulation_state['current_stage'] = 'modeling'
            logger.info(f"Running {model_type} model...")
            
            try:
                simulation_results = model.run_model(
                    model_type, preprocessed_data, self.params
                )
                self._update_progress(80.0, 'modeling')
                logger.debug("Model execution completed successfully")
            except Exception as e:
                logger.error(f"Error running model: {e}")
                raise RuntimeError(f"Failed to run model: {e}")
            
            # Step 4: Export results (20% of progress)
            self.simulation_state['current_stage'] = 'exporting'
            logger.info("Exporting results...")
            
            if output_formats is None:
                output_formats = ['geojson', 'json', 'csv']
            
            # Generate timestamp for filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create a more descriptive filename base
            lat_str = f"{abs(lat):.2f}{'S' if lat < 0 else 'N'}"
            lon_str = f"{abs(lon):.2f}{'W' if lon < 0 else 'E'}"
            filename_base = f"oil_spill_{lat_str}_{lon_str}_{timestamp}"
            
            # Export results
            output_files = {}
            export_progress_increment = 20.0 / len(output_formats)
            current_export_progress = 80.0
            
            try:
                if 'geojson' in output_formats:
                    logger.debug("Exporting to GeoJSON format...")
                    geojson_file = export.export_to_geojson(
                        simulation_results, filename=f"{filename_base}.geojson"
                    )
                    output_files['geojson'] = geojson_file
                    current_export_progress += export_progress_increment
                    self._update_progress(current_export_progress, 'exporting')
                    logger.debug(f"GeoJSON export completed: {geojson_file}")
                
                if 'json' in output_formats:
                    logger.debug("Exporting to JSON format...")
                    json_file = export.export_to_json(
                        simulation_results, filename=f"{filename_base}.json"
                    )
                    output_files['json'] = json_file
                    current_export_progress += export_progress_increment
                    self._update_progress(current_export_progress, 'exporting')
                    logger.debug(f"JSON export completed: {json_file}")
                
                if 'csv' in output_formats:
                    logger.debug("Exporting to CSV format...")
                    csv_file = export.export_to_csv(
                        simulation_results, filename=f"{filename_base}.csv"
                    )
                    output_files['csv'] = csv_file
                    current_export_progress += export_progress_increment
                    self._update_progress(current_export_progress, 'exporting')
                    logger.debug(f"CSV export completed: {csv_file}")
                    
                    # Also export time series data if available
                    if 'particle_history' in simulation_results:
                        logger.debug("Exporting time series data to CSV...")
                        time_series_file = export.export_to_time_series_csv(
                            simulation_results, filename=f"{filename_base}_time_series.csv"
                        )
                        output_files['time_series_csv'] = time_series_file
                        logger.debug(f"Time series CSV export completed: {time_series_file}")
            except Exception as e:
                logger.error(f"Error exporting results: {e}")
                raise RuntimeError(f"Failed to export results: {e}")
            
            # Simulation completed successfully
            self._update_progress(100.0, 'completed')
            self.simulation_state['status'] = 'completed'
            self.simulation_state['end_time'] = datetime.now()
            self.simulation_state['current_stage'] = 'completed'
            
            # Calculate execution time
            execution_time = (self.simulation_state['end_time'] - self.simulation_state['start_time']).total_seconds()
            logger.info(f"Simulation completed in {execution_time:.2f} seconds")
            
            # Return results and output files
            return {
                'results': simulation_results,
                'output_files': output_files,
                'parameters': {
                    'spill_location': spill_location,
                    'spill_volume': spill_volume,
                    'oil_type': oil_type,
                    'model_type': model_type,
                    'simulation_params': self.params
                },
                'execution_time': execution_time,
                'status': 'success'
            }
        
        except Exception as e:
            # Update simulation state to failed
            self.simulation_state['status'] = 'failed'
            self.simulation_state['end_time'] = datetime.now()
            
            # Log the error
            logger.error(f"Simulation failed: {e}")
            
            # Clean up any temporary resources
            self._cleanup()
            
            # Re-raise the exception
            raise
        
        finally:
            # Clean up temporary resources
            self._cleanup()
    
    def _validate_inputs(self, spill_location: Tuple[float, float], 
                         spill_volume: float, 
                         oil_type: str, 
                         model_type: str) -> None:
        """
        Validate simulation input parameters.
        
        Args:
            spill_location: (latitude, longitude) of the spill center
            spill_volume: Volume of the spill in liters
            oil_type: Type of oil
            model_type: Type of model to use
            
        Raises:
            ValueError: If any input parameter is invalid
        """
        # Validate latitude and longitude
        lat, lon = spill_location
        if not (-90 <= lat <= 90):
            raise ValueError(f"Invalid latitude: {lat}. Must be between -90 and 90.")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Invalid longitude: {lon}. Must be between -180 and 180.")
        
        # Validate spill volume
        if spill_volume <= 0:
            raise ValueError(f"Invalid spill volume: {spill_volume}. Must be positive.")
        
        # Validate oil type
        valid_oil_types = ['light_crude', 'medium_crude', 'heavy_crude', 'diesel', 'bunker_fuel']
        if oil_type not in valid_oil_types:
            raise ValueError(f"Invalid oil type: {oil_type}. Must be one of {valid_oil_types}.")
        
        # Validate model type
        valid_model_types = ['water', 'land', 'hybrid']
        if model_type not in valid_model_types:
            raise ValueError(f"Invalid model type: {model_type}. Must be one of {valid_model_types}.")
        
        # Validate simulation parameters
        if self.params.get('duration_hours', 0) <= 0:
            raise ValueError(f"Invalid duration: {self.params.get('duration_hours')}. Must be positive.")
        if self.params.get('timestep_minutes', 0) <= 0:
            raise ValueError(f"Invalid timestep: {self.params.get('timestep_minutes')}. Must be positive.")
        if self.params.get('particle_count', 0) <= 0:
            raise ValueError(f"Invalid particle count: {self.params.get('particle_count')}. Must be positive.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Oil Spill Trajectory Analysis Engine',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    required_group = parser.add_argument_group('Required Arguments')
    required_group.add_argument('--lat', type=float, required=True,
                        help='Latitude of the spill location')
    required_group.add_argument('--lon', type=float, required=True,
                        help='Longitude of the spill location')
    required_group.add_argument('--volume', type=float, required=True,
                        help='Volume of the spill in liters')
    
    # Simulation parameters
    sim_group = parser.add_argument_group('Simulation Parameters')
    sim_group.add_argument('--oil-type', type=str, default='medium_crude',
                        choices=['light_crude', 'medium_crude', 'heavy_crude', 'diesel', 'bunker_fuel'],
                        help='Type of oil')
    sim_group.add_argument('--model-type', type=str, default='hybrid',
                        choices=['water', 'land', 'hybrid'],
                        help='Type of model to use')
    sim_group.add_argument('--duration', type=int, default=48,
                        help='Simulation duration in hours')
    sim_group.add_argument('--timestep', type=int, default=30,
                        help='Simulation timestep in minutes')
    sim_group.add_argument('--particles', type=int, default=1000,
                        help='Number of particles to simulate')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output-formats', type=str, nargs='+',
                        choices=['geojson', 'json', 'csv', 'time_series_csv'], 
                        default=['geojson', 'json', 'csv'],
                        help='Output formats')
    output_group.add_argument('--output-dir', type=str,
                        default=config.OUTPUT_CONFIG['output_directory'],
                        help='Output directory')
    output_group.add_argument('--output-prefix', type=str,
                        help='Prefix for output filenames')
    
    # Configuration options
    config_group = parser.add_argument_group('Configuration Options')
    config_group.add_argument('--config-file', type=str,
                        help='Configuration file (JSON, INI, CFG, or CONF format)')
    config_group.add_argument('--save-config', type=str,
                        help='Save current configuration to file')
    
    # Logging options
    logging_group = parser.add_argument_group('Logging Options')
    logging_group.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    logging_group.add_argument('--quiet', action='store_true',
                        help='Suppress all output except errors')
    logging_group.add_argument('--log-file', type=str,
                        help='Log file path')
    
    # Predefined scenarios
    scenario_group = parser.add_argument_group('Predefined Scenarios')
    scenario_group.add_argument('--scenario', type=str,
                        choices=['tumbes', 'gulf', 'north_sea'],
                        help='Run a predefined scenario')
    
    return parser.parse_args()


def get_scenario_params(scenario_name):
    """Get parameters for a predefined scenario."""
    scenarios = {
        'tumbes': {
            'lat': -3.57,
            'lon': -80.45,
            'volume': 5000,
            'oil_type': 'medium_crude',
            'model_type': 'hybrid',
            'duration': 72,
            'timestep': 30,
            'particles': 2000
        },
        'gulf': {
            'lat': 28.74,
            'lon': -88.37,
            'volume': 10000,
            'oil_type': 'light_crude',
            'model_type': 'water',
            'duration': 120,
            'timestep': 60,
            'particles': 5000
        },
        'north_sea': {
            'lat': 58.5,
            'lon': 1.0,
            'volume': 7500,
            'oil_type': 'heavy_crude',
            'model_type': 'hybrid',
            'duration': 96,
            'timestep': 45,
            'particles': 3000
        }
    }
    
    if scenario_name not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    return scenarios[scenario_name]


def save_configuration(args, filename):
    """Save the current configuration to a file."""
    # Convert args namespace to dictionary
    config_dict = vars(args).copy()
    
    # Remove None values
    config_dict = {k: v for k, v in config_dict.items() if v is not None}
    
    # Determine file format based on extension
    file_ext = os.path.splitext(filename)[1].lower()
    
    try:
        if file_ext == '.json':
            # Save as JSON
            with open(filename, 'w') as f:
                json.dump(config_dict, f, indent=4)
        
        elif file_ext in ['.ini', '.cfg', '.conf']:
            # Save as INI
            config_parser = configparser.ConfigParser()
            config_parser['DEFAULT'] = {}
            
            # Convert values to strings
            for key, value in config_dict.items():
                if isinstance(value, list):
                    config_parser['DEFAULT'][key] = ','.join(value)
                else:
                    config_parser['DEFAULT'][key] = str(value)
            
            with open(filename, 'w') as f:
                config_parser.write(f)
        
        else:
            raise ValueError(f"Unsupported configuration file format: {file_ext}")
        
        logger.info(f"Configuration saved to {filename}")
    
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise


def display_progress(simulation_manager):
    """Display progress information for the simulation."""
    try:
        from tqdm import tqdm
        import time
        
        # Create progress bar
        with tqdm(total=100, desc="Simulation Progress", unit="%") as pbar:
            last_progress = 0
            
            # Update progress until simulation completes or fails
            while simulation_manager.simulation_state['status'] == 'running':
                # Get current progress
                current_progress = int(simulation_manager.simulation_state['progress'])
                current_stage = simulation_manager.simulation_state['current_stage']
                
                # Update progress bar if progress has changed
                if current_progress > last_progress:
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress
                
                # Update description with current stage
                if current_stage:
                    pbar.set_description(f"Stage: {current_stage.replace('_', ' ').title()}")
                
                # Sleep briefly to avoid consuming too much CPU
                time.sleep(0.1)
            
            # Ensure progress bar reaches 100% if simulation completed successfully
            if simulation_manager.simulation_state['status'] == 'completed':
                pbar.update(100 - last_progress)
    
    except ImportError:
        # Fall back to simple progress reporting if tqdm is not available
        logger.info("Progress reporting requires tqdm package. Install with 'pip install tqdm'")
        logger.info("Simulation running...")


def main():
    """Main entry point for the command line interface."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Configure logging based on arguments
        if args.quiet:
            logging.getLogger().setLevel(logging.ERROR)
        elif args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        if args.log_file:
            file_handler = logging.FileHandler(args.log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logging.getLogger().addHandler(file_handler)
        
        # Update output directory in config
        config.OUTPUT_CONFIG['output_directory'] = args.output_dir
        
        # Check if we're using a predefined scenario
        if args.scenario:
            scenario_params = get_scenario_params(args.scenario)
            
            # Override command-line arguments with scenario parameters
            # but only if they weren't explicitly provided
            if not args.lat:
                args.lat = scenario_params['lat']
            if not args.lon:
                args.lon = scenario_params['lon']
            if not args.volume:
                args.volume = scenario_params['volume']
            
            # Set other parameters if not explicitly provided
            for param, value in scenario_params.items():
                if param not in ['lat', 'lon', 'volume'] and getattr(args, param, None) is None:
                    setattr(args, param, value)
            
            logger.info(f"Using predefined scenario: {args.scenario}")
        
        # Set up simulation parameters
        simulation_params = {
            'duration_hours': args.duration,
            'timestep_minutes': args.timestep,
            'particle_count': args.particles
        }
        
        # If config file provided, load and merge parameters
        if args.config_file:
            try:
                # Config file loading is now handled by SimulationManager
                logger.info(f"Loading configuration from {args.config_file}")
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
                sys.exit(1)
        
        # Save configuration if requested
        if args.save_config:
            try:
                save_configuration(args, args.save_config)
            except Exception as e:
                logger.error(f"Error saving configuration: {e}")
                sys.exit(1)
        
        # Create simulation manager
        manager = SimulationManager(
            simulation_params=simulation_params, 
            verbose=args.verbose, 
            output_dir=args.output_dir,
            config_file=args.config_file
        )
        
        # Start simulation in a separate thread for progress reporting
        import threading
        
        def run_simulation():
            nonlocal results
            try:
                results = manager.run_simulation(
                    spill_location=(args.lat, args.lon),
                    spill_volume=args.volume,
                    oil_type=args.oil_type,
                    model_type=args.model_type,
                    output_formats=args.output_formats
                )
            except Exception as e:
                # Exception will be re-raised in the main thread
                pass
        
        # Initialize results
        results = None
        
        # Create and start simulation thread
        sim_thread = threading.Thread(target=run_simulation)
        sim_thread.start()
        
        # Display progress while simulation runs
        if not args.quiet:
            display_progress(manager)
        
        # Wait for simulation to complete
        sim_thread.join()
        
        # Check if simulation was successful
        if manager.simulation_state['status'] == 'failed':
            logger.error("Simulation failed")
            sys.exit(1)
        
        # Print output file paths
        if not args.quiet and results:
            print("\nSimulation completed successfully!")
            print(f"Execution time: {results.get('execution_time', 0):.2f} seconds")
            print("Output files:")
            for format_type, filepath in results['output_files'].items():
                print(f"  {format_type}: {filepath}")
        
        return 0
    
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
