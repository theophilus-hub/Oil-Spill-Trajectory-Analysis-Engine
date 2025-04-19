"""
Main orchestration module for the Oil Spill Trajectory Analysis Engine.

This module provides the main entry point and orchestration for the simulation:
- Coordinates the data acquisition, preprocessing, modeling, and export steps
- Provides a simple CLI interface for running simulations
- Handles configuration and parameter management
"""

import argparse
import logging
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from . import config
from . import fetch_data
from . import preprocess
from . import model
from . import export

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trajectory_simulation.log')
    ]
)

logger = logging.getLogger(__name__)


class SimulationManager:
    """Main class for orchestrating the oil spill trajectory simulation."""
    
    def __init__(self, simulation_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the simulation manager.
        
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
        
        # Create output directory if it doesn't exist
        os.makedirs(config.OUTPUT_CONFIG['output_directory'], exist_ok=True)
    
    def run_simulation(self, 
                      spill_location: Tuple[float, float],
                      spill_volume: float,
                      oil_type: str = 'medium_crude',
                      model_type: str = 'hybrid',
                      output_formats: Optional[list] = None) -> Dict[str, Any]:
        """
        Run a complete oil spill trajectory simulation.
        
        Args:
            spill_location: (latitude, longitude) of the spill center
            spill_volume: Volume of the spill in liters
            oil_type: Type of oil (default: medium_crude)
            model_type: Type of model to use ('water', 'land', or 'hybrid')
            output_formats: List of output formats ('geojson', 'json', 'csv')
                If None, uses all formats
                
        Returns:
            Dictionary containing simulation results and output file paths
        """
        logger.info(f"Starting simulation for spill at {spill_location}")
        
        # Step 1: Fetch environmental data
        logger.info("Fetching environmental data...")
        
        lat, lon = spill_location
        
        # Get wind data
        wind_data = fetch_data.get_wind_data(lat, lon)
        
        # Get ocean current data
        current_data = fetch_data.get_ocean_currents(lat, lon)
        
        # Get elevation data for a bounding box around the spill
        # Approximately 10km in each direction
        buffer = 0.1  # Roughly 10km in decimal degrees
        elevation_data = fetch_data.get_elevation_data(
            lat - buffer, lon - buffer,
            lat + buffer, lon + buffer
        )
        
        # Get oil properties
        oil_properties = fetch_data.get_oil_properties(oil_type)
        
        # Step 2: Preprocess data
        logger.info("Preprocessing data...")
        
        preprocessed_data = preprocess.preprocess_all_data(
            wind_data, current_data, elevation_data,
            spill_location, spill_volume,
            self.params.get('particle_count', 1000)
        )
        
        # Step 3: Run the model
        logger.info(f"Running {model_type} model...")
        
        simulation_results = model.run_model(
            model_type, preprocessed_data, self.params
        )
        
        # Step 4: Export results
        logger.info("Exporting results...")
        
        if output_formats is None:
            output_formats = ['geojson', 'json', 'csv']
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_base = f"oil_spill_{timestamp}"
        
        # Export results
        output_files = {}
        
        if 'geojson' in output_formats:
            geojson_file = export.export_to_geojson(
                simulation_results, filename=f"{filename_base}.geojson"
            )
            output_files['geojson'] = geojson_file
        
        if 'json' in output_formats:
            json_file = export.export_to_json(
                simulation_results, filename=f"{filename_base}.json"
            )
            output_files['json'] = json_file
        
        if 'csv' in output_formats:
            csv_file = export.export_to_csv(
                simulation_results, filename=f"{filename_base}.csv"
            )
            output_files['csv'] = csv_file
        
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
            }
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Oil Spill Trajectory Analysis Engine'
    )
    
    # Required arguments
    parser.add_argument('--lat', type=float, required=True,
                        help='Latitude of the spill location')
    parser.add_argument('--lon', type=float, required=True,
                        help='Longitude of the spill location')
    parser.add_argument('--volume', type=float, required=True,
                        help='Volume of the spill in liters')
    
    # Optional arguments
    parser.add_argument('--oil-type', type=str, default='medium_crude',
                        help='Type of oil (default: medium_crude)')
    parser.add_argument('--model-type', type=str, default='hybrid',
                        choices=['water', 'land', 'hybrid'],
                        help='Type of model to use (default: hybrid)')
    parser.add_argument('--duration', type=int, default=48,
                        help='Simulation duration in hours (default: 48)')
    parser.add_argument('--timestep', type=int, default=30,
                        help='Simulation timestep in minutes (default: 30)')
    parser.add_argument('--particles', type=int, default=1000,
                        help='Number of particles to simulate (default: 1000)')
    parser.add_argument('--output-formats', type=str, nargs='+',
                        choices=['geojson', 'json', 'csv'], default=['geojson', 'json', 'csv'],
                        help='Output formats (default: all)')
    parser.add_argument('--output-dir', type=str,
                        default=config.OUTPUT_CONFIG['output_directory'],
                        help=f'Output directory (default: {config.OUTPUT_CONFIG["output_directory"]})')
    parser.add_argument('--config-file', type=str,
                        help='JSON configuration file')
    
    return parser.parse_args()


def main():
    """Main entry point for the command line interface."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Update output directory in config
    config.OUTPUT_CONFIG['output_directory'] = args.output_dir
    
    # Set up simulation parameters
    simulation_params = {
        'duration_hours': args.duration,
        'timestep_minutes': args.timestep,
        'particle_count': args.particles
    }
    
    # If config file provided, load and merge parameters
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                file_params = json.load(f)
                simulation_params.update(file_params)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            sys.exit(1)
    
    # Create simulation manager
    manager = SimulationManager(simulation_params)
    
    # Run simulation
    results = manager.run_simulation(
        spill_location=(args.lat, args.lon),
        spill_volume=args.volume,
        oil_type=args.oil_type,
        model_type=args.model_type,
        output_formats=args.output_formats
    )
    
    # Print output file paths
    print("\nSimulation completed successfully!")
    print("Output files:")
    for format_type, filepath in results['output_files'].items():
        print(f"  {format_type}: {filepath}")


if __name__ == '__main__':
    main()
