#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example client for the Oil Spill Trajectory Analysis Engine API.

This script demonstrates how to use the API to:
1. Start a simulation
2. Check simulation status
3. Retrieve simulation results
4. Download result files

Usage:
    python api_client_example.py [--host HOST] [--port PORT] [--api-key API_KEY]
"""

import os
import sys
import time
import json
import argparse
import requests
from pathlib import Path
from typing import Dict, Any, Optional


class OilSpillAPIClient:
    """Client for the Oil Spill Trajectory Analysis Engine API."""
    
    def __init__(self, host: str = 'localhost', port: int = 5000):
        """Initialize the API client.
        
        Args:
            host: API server host
            port: API server port
        """
        self.base_url = f"http://{host}:{port}/api/v1"
        self.headers = {}
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the API server.
        
        Returns:
            Dict containing health status information
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def start_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new simulation.
        
        Args:
            params: Simulation parameters
        
        Returns:
            Dict containing simulation ID and status
        """
        response = requests.post(
            f"{self.base_url}/simulate",
            json=params,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_simulation_status(self, simulation_id: str) -> Dict[str, Any]:
        """Get the status of a simulation.
        
        Args:
            simulation_id: ID of the simulation
        
        Returns:
            Dict containing simulation status information
        """
        response = requests.get(
            f"{self.base_url}/status/{simulation_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_simulation_results(self, simulation_id: str) -> Dict[str, Any]:
        """Get the results of a simulation.
        
        Args:
            simulation_id: ID of the simulation
        
        Returns:
            Dict containing simulation results
        """
        response = requests.get(
            f"{self.base_url}/results/{simulation_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def download_result_file(self, simulation_id: str, file_format: str, output_path: Optional[str] = None) -> str:
        """Download a result file.
        
        Args:
            simulation_id: ID of the simulation
            file_format: Format of the file to download (geojson, json, csv, time_series_csv)
            output_path: Path to save the file (default: current directory)
        
        Returns:
            Path to the downloaded file
        """
        response = requests.get(
            f"{self.base_url}/download/{simulation_id}/{file_format}",
            headers=self.headers
        )
        response.raise_for_status()
        
        # Determine filename from Content-Disposition header if available
        content_disposition = response.headers.get('Content-Disposition')
        if content_disposition and 'filename=' in content_disposition:
            filename = content_disposition.split('filename=')[1].strip('"')
        else:
            filename = f"{simulation_id}.{file_format}"
        
        # Determine output directory
        if output_path:
            output_dir = Path(output_path)
        else:
            output_dir = Path.cwd() / 'downloads'
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save file
        file_path = output_dir / filename
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        return str(file_path)
    
    def list_simulations(self, status: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """List all simulations.
        
        Args:
            status: Filter by status (e.g., queued, running, completed, error)
            limit: Maximum number of simulations to return
        
        Returns:
            Dict containing list of simulations
        """
        params = {}
        if status:
            params['status'] = status
        if limit:
            params['limit'] = limit
        
        response = requests.get(
            f"{self.base_url}/simulations",
            params=params,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def delete_simulation(self, simulation_id: str) -> Dict[str, Any]:
        """Delete a simulation.
        
        Args:
            simulation_id: ID of the simulation
        
        Returns:
            Dict containing success message
        """
        response = requests.delete(
            f"{self.base_url}/simulation/{simulation_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, simulation_id: str, polling_interval: int = 5, timeout: int = 3600) -> Dict[str, Any]:
        """Wait for a simulation to complete.
        
        Args:
            simulation_id: ID of the simulation
            polling_interval: Time between status checks in seconds
            timeout: Maximum time to wait in seconds
        
        Returns:
            Dict containing simulation results
        
        Raises:
            TimeoutError: If the simulation doesn't complete within the timeout
            RuntimeError: If the simulation fails
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_simulation_status(simulation_id)
            
            if status['status'] == 'completed':
                return self.get_simulation_results(simulation_id)
            
            if status['status'] == 'error':
                raise RuntimeError(f"Simulation failed: {status.get('error', 'Unknown error')}")
            
            # Print progress
            progress = status.get('progress', 0)
            stage = status.get('current_stage', 'unknown')
            print(f"Progress: {progress:.1f}% (Stage: {stage})")
            
            time.sleep(polling_interval)
        
        raise TimeoutError(f"Simulation did not complete within {timeout} seconds")
    
    def start_batch_simulation(self, simulations: list, common_params: Optional[Dict[str, Any]] = None, batch_name: Optional[str] = None) -> Dict[str, Any]:
        """Start multiple simulations in batch.
        
        Args:
            simulations: List of simulation parameters
            common_params: Common parameters to apply to all simulations
            batch_name: Optional name for the batch
        
        Returns:
            Dict containing batch ID and simulation IDs
        """
        payload = {
            'simulations': simulations
        }
        
        if common_params:
            payload['common_params'] = common_params
        
        if batch_name:
            payload['batch_name'] = batch_name
        
        response = requests.post(
            f"{self.base_url}/batch-simulate",
            json=payload,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get the status of a batch simulation.
        
        Args:
            batch_id: ID of the batch
        
        Returns:
            Dict containing batch status information
        """
        response = requests.get(
            f"{self.base_url}/batch-status/{batch_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_batch_completion(self, batch_id: str, polling_interval: int = 5, timeout: int = 3600) -> Dict[str, Any]:
        """Wait for a batch simulation to complete.
        
        Args:
            batch_id: ID of the batch
            polling_interval: Time between status checks in seconds
            timeout: Maximum time to wait in seconds
        
        Returns:
            Dict containing batch status
        
        Raises:
            TimeoutError: If the batch doesn't complete within the timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_batch_status(batch_id)
            
            # Check if all simulations are completed or have errors
            all_done = True
            for sim in status['simulations']:
                if sim['status'] not in ['completed', 'error']:
                    all_done = False
                    break
            
            if all_done:
                return status
            
            # Print progress
            print(f"Overall progress: {status['overall_progress']:.1f}% ({status['completed_count']}/{status['total_count']} completed)")
            
            time.sleep(polling_interval)
        
        raise TimeoutError(f"Batch did not complete within {timeout} seconds")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Example client for the Oil Spill Trajectory Analysis Engine API.'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='API server host (default: localhost)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='API server port (default: 5000)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create API client
    client = OilSpillAPIClient(
        host=args.host,
        port=args.port
    )
    
    try:
        # Check API health
        print("Checking API health...")
        health = client.health_check()
        print(f"API health: {health['status']}")
        print(f"API version: {health['version']}")
        print(f"Active simulations: {health['active_simulations']}")
        print(f"Total simulations: {health['total_simulations']}")
        print()
        
        # Choose between single simulation or batch simulation
        example_type = input("Run a single simulation (1) or batch simulation (2)? [1/2]: ").strip()
        
        if example_type == "2":
            # Batch simulation example
            print("\nStarting a batch simulation...")
            
            # Define multiple simulations with different parameters
            simulations = [
                {
                    'latitude': -3.57,
                    'longitude': -80.45,
                    'volume': 5000,
                    'oil_type': 'medium_crude'
                },
                {
                    'latitude': -3.60,
                    'longitude': -80.50,
                    'volume': 3000,
                    'oil_type': 'light_crude'
                }
            ]
            
            # Define common parameters
            common_params = {
                'model_type': 'hybrid',
                'duration_hours': 24,  # Shorter duration for example
                'timestep_minutes': 60,  # Larger timestep for example
                'particle_count': 100,  # Fewer particles for example
                'output_formats': ['geojson', 'json', 'csv']
            }
            
            # Start batch simulation
            batch = client.start_batch_simulation(
                simulations=simulations,
                common_params=common_params,
                batch_name="Example Batch"
            )
            
            batch_id = batch['batch_id']
            print(f"Batch started with ID: {batch_id}")
            print(f"Simulation IDs: {batch['simulation_ids']}")
            print()
            
            # Wait for batch to complete
            print("Waiting for batch to complete...")
            try:
                batch_status = client.wait_for_batch_completion(batch_id, polling_interval=2, timeout=300)
                print("\nBatch completed!")
                print(f"Overall progress: {batch_status['overall_progress']:.1f}%")
                print(f"Completed: {batch_status['completed_count']}/{batch_status['total_count']}")
                print()
                
                # Get results for each simulation in the batch
                print("Getting results for each simulation...")
                for sim in batch_status['simulations']:
                    sim_id = sim['id']
                    if sim['status'] == 'completed':
                        results = client.get_simulation_results(sim_id)
                        print(f"\nResults for simulation {sim_id}:")
                        print(f"Status: {results['status']}")
                        
                        # Download result files
                        print(f"Downloading result files for simulation {sim_id}...")
                        for file_format in ['geojson', 'json', 'csv']:
                            file_path = client.download_result_file(sim_id, file_format)
                            print(f"Downloaded {file_format} file to: {file_path}")
                    else:
                        print(f"\nSimulation {sim_id} did not complete successfully.")
                        print(f"Status: {sim['status']}")
                print()
                
            except (TimeoutError, RuntimeError) as e:
                print(f"Error: {e}")
        else:
            # Single simulation example
            print("\nStarting a new simulation...")
            simulation_params = {
                'latitude': -3.57,
                'longitude': -80.45,
                'volume': 5000,
                'oil_type': 'medium_crude',
                'model_type': 'hybrid',
                'duration_hours': 24,  # Shorter duration for example
                'timestep_minutes': 60,  # Larger timestep for example
                'particle_count': 100,  # Fewer particles for example
                'output_formats': ['geojson', 'json', 'csv']
            }
            
            simulation = client.start_simulation(simulation_params)
            simulation_id = simulation['id']
            print(f"Simulation started with ID: {simulation_id}")
            print(f"Status: {simulation['status']}")
            print()
            
            # Wait for simulation to complete
            print("Waiting for simulation to complete...")
            try:
                results = client.wait_for_completion(simulation_id, polling_interval=2, timeout=300)
                print("\nSimulation completed successfully!")
                print(f"Results: {json.dumps(results, indent=2)}")
                print()
                
                # Download result files
                print("Downloading result files...")
                for file_format in ['geojson', 'json', 'csv']:
                    file_path = client.download_result_file(simulation_id, file_format)
                    print(f"Downloaded {file_format} file to: {file_path}")
                print()
                
            except (TimeoutError, RuntimeError) as e:
                print(f"Error: {e}")
        
        # List all simulations
        print("Listing all simulations...")
        simulations = client.list_simulations()
        print(f"Total simulations: {simulations['total']}")
        for sim in simulations['simulations']:
            print(f"ID: {sim['id']}, Status: {sim['status']}, Progress: {sim.get('progress', 0)}%")
        print()
        
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == '__main__':
    main()
