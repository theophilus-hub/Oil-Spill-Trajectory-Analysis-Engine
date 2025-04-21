#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive demonstration script for the Oil Spill Trajectory Analysis Engine.

This script demonstrates the full functionality of the main orchestration module
by running a complete oil spill simulation with various options and configurations.

Usage:
    python run_simulation.py --scenario=tumbes
    python run_simulation.py --lat=-3.57 --lon=-80.45 --volume=5000 --duration=24
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path to import trajectory_core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from trajectory_core import main
from trajectory_core import config


def run_demo_simulation():
    """
    Run a demonstration simulation using the main orchestration module.
    
    This function demonstrates various ways to use the main orchestration module:
    1. Running a predefined scenario
    2. Running a custom simulation with specific parameters
    3. Saving and loading configuration
    4. Using different output formats
    """
    print("\n" + "=" * 80)
    print("Oil Spill Trajectory Analysis Engine - Demonstration")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), 'demo_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Example 1: Run a predefined scenario
    print("\nExample 1: Running predefined 'tumbes' scenario...")
    scenario_params = main.get_scenario_params('tumbes')
    
    # Adjust parameters for faster demo
    scenario_params['duration'] = 12  # 12 hours instead of 72
    scenario_params['timestep'] = 60  # 60 minutes instead of 30
    scenario_params['particles'] = 500  # 500 particles instead of 2000
    
    # Create simulation manager
    manager = main.SimulationManager(
        simulation_params={
            'duration_hours': scenario_params['duration'],
            'timestep_minutes': scenario_params['timestep'],
            'particle_count': scenario_params['particles']
        },
        verbose=True,
        output_dir=output_dir
    )
    
    # Run simulation
    start_time = time.time()
    results = manager.run_simulation(
        spill_location=(scenario_params['lat'], scenario_params['lon']),
        spill_volume=scenario_params['volume'],
        oil_type=scenario_params['oil_type'],
        model_type=scenario_params['model_type'],
        output_formats=['geojson', 'json', 'csv']
    )
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\nSimulation completed in {elapsed_time:.2f} seconds")
    print("Output files:")
    for format_type, filepath in results['output_files'].items():
        print(f"  {format_type}: {filepath}")
    
    # Example 2: Save and load configuration
    print("\nExample 2: Saving and loading configuration...")
    
    # Create a configuration file
    config_file = os.path.join(output_dir, 'demo_config.json')
    
    # Create mock args object for save_configuration
    class MockArgs:
        def __init__(self):
            self.lat = 28.74
            self.lon = -88.37
            self.volume = 10000
            self.oil_type = 'light_crude'
            self.model_type = 'water'
            self.duration = 24
            self.timestep = 60
            self.particles = 1000
            self.output_formats = ['geojson', 'json', 'csv']
            self.output_dir = output_dir
            self.verbose = True
    
    args = MockArgs()
    
    # Save configuration
    main.save_configuration(args, config_file)
    print(f"Configuration saved to {config_file}")
    
    # Load configuration and run simulation
    print("\nRunning simulation with loaded configuration...")
    
    # Create simulation manager with loaded config
    manager = main.SimulationManager(
        config_file=config_file,
        output_dir=output_dir
    )
    
    # Run simulation
    start_time = time.time()
    results = manager.run_simulation(
        spill_location=(args.lat, args.lon),
        spill_volume=args.volume,
        oil_type=args.oil_type,
        model_type=args.model_type,
        output_formats=['geojson']
    )
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\nSimulation completed in {elapsed_time:.2f} seconds")
    print("Output files:")
    for format_type, filepath in results['output_files'].items():
        print(f"  {format_type}: {filepath}")
    
    # Example 3: Run simulation with custom parameters and progress reporting
    print("\nExample 3: Running simulation with custom parameters and progress reporting...")
    
    # Create simulation manager
    manager = main.SimulationManager(
        simulation_params={
            'duration_hours': 6,  # Short duration for demo
            'timestep_minutes': 30,
            'particle_count': 200,  # Few particles for demo
            'random_seed': 42,  # For reproducibility
            'evaporation_rate': 0.03,  # Custom evaporation rate
            'diffusion_coefficient': 0.8  # Custom diffusion coefficient
        },
        output_dir=output_dir
    )
    
    # Create a separate thread for progress reporting
    import threading
    
    def report_progress():
        last_progress = -1
        last_stage = None
        
        while manager.simulation_state['status'] == 'running':
            progress = int(manager.simulation_state['progress'])
            stage = manager.simulation_state['current_stage']
            
            if progress != last_progress or stage != last_stage:
                timestamp = datetime.now().strftime('%H:%M:%S')
                if stage:
                    stage_name = stage.replace('_', ' ').title()
                    print(f"[{timestamp}] Progress: {progress}% - Stage: {stage_name}")
                else:
                    print(f"[{timestamp}] Progress: {progress}%")
                
                last_progress = progress
                last_stage = stage
            
            time.sleep(0.5)
    
    # Start progress reporting thread
    progress_thread = threading.Thread(target=report_progress)
    progress_thread.daemon = True
    progress_thread.start()
    
    # Run simulation
    try:
        results = manager.run_simulation(
            spill_location=(58.5, 1.0),  # North Sea
            spill_volume=7500,
            oil_type='heavy_crude',
            model_type='hybrid',
            output_formats=['geojson', 'json', 'csv']
        )
        
        # Wait for progress thread to finish
        time.sleep(1)
        
        # Print results
        print(f"\nSimulation completed successfully!")
        print(f"Execution time: {results.get('execution_time', 0):.2f} seconds")
        print("Output files:")
        for format_type, filepath in results['output_files'].items():
            print(f"  {format_type}: {filepath}")
    
    except Exception as e:
        print(f"\nSimulation failed: {e}")
    
    print("\n" + "=" * 80)
    print("Demonstration completed!")
    print("=" * 80)


if __name__ == '__main__':
    run_demo_simulation()
