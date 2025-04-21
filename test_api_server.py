#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the Flask API server of the Oil Spill Trajectory Analysis Engine.

This script tests the functionality of the API server, including:
- Starting a simulation
- Checking simulation status
- Retrieving simulation results
- Downloading result files
"""

import os
import sys
import json
import unittest
import tempfile
import time
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path to import trajectory_core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from trajectory_core import server
from trajectory_core import config


class TestAPIServer(unittest.TestCase):
    """Test cases for the Flask API server."""
    
    def setUp(self):
        """Set up test environment."""
        # Configure Flask app for testing
        server.app.config['TESTING'] = True
        self.client = server.app.test_client()
        
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp(prefix="oil_spill_api_test_")
        
        # Set output directory in config
        config.OUTPUT_CONFIG['output_directory'] = self.test_dir
        
        # Disable authentication for testing
        config.FLASK_CONFIG['debug'] = True
        config.FLASK_CONFIG['disable_auth'] = True
        
        # Sample simulation parameters
        self.sample_params = {
            'latitude': -3.57,
            'longitude': -80.45,
            'volume': 5000,
            'oil_type': 'medium_crude',
            'model_type': 'hybrid',
            'duration_hours': 24,  # Shorter duration for testing
            'timestep_minutes': 60,  # Larger timestep for testing
            'particle_count': 100  # Fewer particles for testing
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory and its contents
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
        # Clear simulations dictionary
        server.simulations.clear()
    
    @patch('trajectory_core.main.SimulationManager.run_simulation')
    def test_start_simulation(self, mock_run_simulation):
        """Test starting a simulation."""
        # Set up mock
        mock_run_simulation.return_value = {
            'status': 'success',
            'output_files': {
                'geojson': os.path.join(self.test_dir, 'test.geojson'),
                'json': os.path.join(self.test_dir, 'test.json'),
                'csv': os.path.join(self.test_dir, 'test.csv')
            }
        }
        
        # Send request to start simulation
        response = self.client.post(
            '/api/v1/simulate',
            json=self.sample_params,
            headers={'X-API-Key': 'test_key'}
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('id', data)
        self.assertEqual(data['status'], 'queued')
        
        # Verify simulation was added to simulations dictionary
        simulation_id = data['id']
        self.assertIn(simulation_id, server.simulations)
        
        # Wait for simulation to complete (since it's running in a background thread)
        max_wait = 5  # seconds
        start_time = time.time()
        while server.simulations[simulation_id]['status'] == 'queued' and time.time() - start_time < max_wait:
            time.sleep(0.1)
        
        # Verify simulation was started
        self.assertNotEqual(server.simulations[simulation_id]['status'], 'queued')
    
    def test_invalid_parameters(self):
        """Test validation of simulation parameters."""
        # Test missing required parameter
        invalid_params = self.sample_params.copy()
        del invalid_params['latitude']
        
        response = self.client.post(
            '/api/v1/simulate',
            json=invalid_params,
            headers={'X-API-Key': 'test_key'}
        )
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
        # Test invalid latitude
        invalid_params = self.sample_params.copy()
        invalid_params['latitude'] = 100  # Out of range
        
        response = self.client.post(
            '/api/v1/simulate',
            json=invalid_params,
            headers={'X-API-Key': 'test_key'}
        )
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
        # Test invalid oil type
        invalid_params = self.sample_params.copy()
        invalid_params['oil_type'] = 'invalid_oil'  # Not in valid options
        
        response = self.client.post(
            '/api/v1/simulate',
            json=invalid_params,
            headers={'X-API-Key': 'test_key'}
        )
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    @patch('trajectory_core.main.SimulationManager.run_simulation')
    def test_get_simulation_status(self, mock_run_simulation):
        """Test getting simulation status."""
        # Set up mock
        mock_run_simulation.return_value = {
            'status': 'success',
            'output_files': {
                'geojson': os.path.join(self.test_dir, 'test.geojson'),
                'json': os.path.join(self.test_dir, 'test.json'),
                'csv': os.path.join(self.test_dir, 'test.csv')
            }
        }
        
        # Start a simulation
        response = self.client.post(
            '/api/v1/simulate',
            json=self.sample_params,
            headers={'X-API-Key': 'test_key'}
        )
        
        data = json.loads(response.data)
        simulation_id = data['id']
        
        # Wait for simulation to complete
        max_wait = 5  # seconds
        start_time = time.time()
        while server.simulations[simulation_id]['status'] == 'queued' and time.time() - start_time < max_wait:
            time.sleep(0.1)
        
        # Get simulation status
        response = self.client.get(
            f'/api/v1/status/{simulation_id}',
            headers={'X-API-Key': 'test_key'}
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['id'], simulation_id)
        self.assertIn('status', data)
        self.assertIn('progress', data)
    
    def test_get_nonexistent_simulation(self):
        """Test getting status of a nonexistent simulation."""
        response = self.client.get(
            '/api/v1/status/nonexistent-id',
            headers={'X-API-Key': 'test_key'}
        )
        
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    @patch('trajectory_core.main.SimulationManager.run_simulation')
    def test_list_simulations(self, mock_run_simulation):
        """Test listing all simulations."""
        # Set up mock
        mock_run_simulation.return_value = {
            'status': 'success',
            'output_files': {
                'geojson': os.path.join(self.test_dir, 'test.geojson'),
                'json': os.path.join(self.test_dir, 'test.json'),
                'csv': os.path.join(self.test_dir, 'test.csv')
            }
        }
        
        # Start a few simulations
        for i in range(3):
            params = self.sample_params.copy()
            params['latitude'] += i * 0.1  # Make each simulation slightly different
            
            self.client.post(
                '/api/v1/simulate',
                json=params,
                headers={'X-API-Key': 'test_key'}
            )
        
        # List all simulations
        response = self.client.get(
            '/api/v1/simulations',
            headers={'X-API-Key': 'test_key'}
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('simulations', data)
        self.assertIn('total', data)
        self.assertEqual(data['total'], 3)
        self.assertEqual(len(data['simulations']), 3)
    
    @patch('trajectory_core.main.SimulationManager.run_simulation')
    def test_health_check(self, mock_run_simulation):
        """Test health check endpoint."""
        response = self.client.get('/api/v1/health')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')
        self.assertIn('version', data)
        self.assertIn('active_simulations', data)
        self.assertIn('total_simulations', data)


if __name__ == '__main__':
    unittest.main()
