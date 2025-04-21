#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the main orchestration module of the Oil Spill Trajectory Analysis Engine.

This script tests the functionality of the main.py module, including:
- Command-line interface
- Configuration loading and saving
- Simulation execution
- Progress reporting
- Error handling
"""

import os
import sys
import unittest
import tempfile
import json
import configparser
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path to import trajectory_core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from trajectory_core import main
from trajectory_core import config


class TestMainOrchestration(unittest.TestCase):
    """Test cases for the main orchestration module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp(prefix="oil_spill_test_")
        
        # Sample simulation parameters
        self.sample_params = {
            'lat': -3.57,
            'lon': -80.45,
            'volume': 5000,
            'oil_type': 'medium_crude',
            'model_type': 'hybrid',
            'duration': 24,  # Shorter duration for testing
            'timestep': 60,  # Larger timestep for testing
            'particles': 100  # Fewer particles for testing
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory and its contents
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_simulation_manager_init(self):
        """Test SimulationManager initialization."""
        # Test with default parameters
        manager = main.SimulationManager()
        self.assertEqual(manager.params, config.DEFAULT_SIMULATION_PARAMS)
        self.assertEqual(manager.simulation_state['status'], 'initialized')
        
        # Test with custom parameters
        custom_params = {'duration_hours': 24, 'timestep_minutes': 60}
        manager = main.SimulationManager(simulation_params=custom_params)
        self.assertEqual(manager.params['duration_hours'], 24)
        self.assertEqual(manager.params['timestep_minutes'], 60)
    
    def test_load_config_file(self):
        """Test loading configuration from different file formats."""
        # Create a temporary JSON config file
        json_config = {
            'duration_hours': 36,
            'timestep_minutes': 45,
            'particle_count': 500
        }
        
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w+', delete=False) as f:
            json_path = f.name
            json.dump(json_config, f)
        
        # Create a temporary INI config file
        ini_config = configparser.ConfigParser()
        ini_config['DEFAULT'] = {
            'duration_hours': '48',
            'timestep_minutes': '30',
            'particle_count': '1000'
        }
        
        with tempfile.NamedTemporaryFile(suffix='.ini', mode='w+', delete=False) as f:
            ini_path = f.name
            ini_config.write(f)
        
        try:
            # Test loading JSON config
            manager = main.SimulationManager(config_file=json_path)
            self.assertEqual(manager.params['duration_hours'], 36)
            self.assertEqual(manager.params['timestep_minutes'], 45)
            self.assertEqual(manager.params['particle_count'], 500)
            
            # Test loading INI config
            manager = main.SimulationManager(config_file=ini_path)
            self.assertEqual(manager.params['duration_hours'], 48)
            self.assertEqual(manager.params['timestep_minutes'], 30)
            self.assertEqual(manager.params['particle_count'], 1000)
            
            # Test nonexistent config file
            with self.assertRaises(FileNotFoundError):
                main.SimulationManager(config_file="nonexistent.json")
        
        finally:
            # Clean up temporary files
            if os.path.exists(json_path):
                os.unlink(json_path)
            if os.path.exists(ini_path):
                os.unlink(ini_path)
    
    def test_validate_inputs(self):
        """Test input validation."""
        manager = main.SimulationManager()
        
        # Test valid inputs
        try:
            manager._validate_inputs(
                spill_location=(-3.57, -80.45),
                spill_volume=5000,
                oil_type='medium_crude',
                model_type='hybrid'
            )
        except ValueError:
            self.fail("_validate_inputs() raised ValueError unexpectedly!")
        
        # Test invalid latitude
        with self.assertRaises(ValueError):
            manager._validate_inputs(
                spill_location=(-91, -80.45),  # Invalid latitude
                spill_volume=5000,
                oil_type='medium_crude',
                model_type='hybrid'
            )
        
        # Test invalid longitude
        with self.assertRaises(ValueError):
            manager._validate_inputs(
                spill_location=(-3.57, -181),  # Invalid longitude
                spill_volume=5000,
                oil_type='medium_crude',
                model_type='hybrid'
            )
        
        # Test invalid spill volume
        with self.assertRaises(ValueError):
            manager._validate_inputs(
                spill_location=(-3.57, -80.45),
                spill_volume=-100,  # Invalid volume
                oil_type='medium_crude',
                model_type='hybrid'
            )
        
        # Test invalid oil type
        with self.assertRaises(ValueError):
            manager._validate_inputs(
                spill_location=(-3.57, -80.45),
                spill_volume=5000,
                oil_type='invalid_oil',  # Invalid oil type
                model_type='hybrid'
            )
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            manager._validate_inputs(
                spill_location=(-3.57, -80.45),
                spill_volume=5000,
                oil_type='medium_crude',
                model_type='invalid_model'  # Invalid model type
            )
    
    @patch('trajectory_core.main.fetch_data')
    @patch('trajectory_core.main.preprocess')
    @patch('trajectory_core.main.model')
    @patch('trajectory_core.main.export')
    def test_run_simulation(self, mock_export, mock_model, mock_preprocess, mock_fetch_data):
        """Test the run_simulation method with mocked dependencies."""
        # Set up mocks
        mock_fetch_data.get_wind_data.return_value = {'data': 'wind_data'}
        mock_fetch_data.get_ocean_currents.return_value = {'data': 'current_data'}
        mock_fetch_data.get_elevation_data.return_value = {'data': 'elevation_data'}
        mock_fetch_data.get_oil_properties.return_value = {'data': 'oil_properties'}
        
        mock_preprocess.preprocess_all_data.return_value = {'data': 'preprocessed_data'}
        
        mock_model.run_model.return_value = {
            'metadata': {'start_time': '2025-04-21T12:00:00', 'end_time': '2025-04-22T12:00:00'},
            'params': {'duration_hours': 24, 'timestep_minutes': 60},
            'particles': [{'id': 1, 'lat': -3.58, 'lon': -80.46, 'status': 'active'}],
            'particle_history': [{'timestep': 0, 'active': 100, 'beached': 0, 'evaporated': 0}]
        }
        
        mock_export.export_to_geojson.return_value = os.path.join(self.test_dir, 'test.geojson')
        mock_export.export_to_json.return_value = os.path.join(self.test_dir, 'test.json')
        mock_export.export_to_csv.return_value = os.path.join(self.test_dir, 'test.csv')
        mock_export.export_to_time_series_csv.return_value = os.path.join(self.test_dir, 'test_time_series.csv')
        
        # Create simulation manager
        manager = main.SimulationManager()
        
        # Run simulation
        results = manager.run_simulation(
            spill_location=(-3.57, -80.45),
            spill_volume=5000,
            oil_type='medium_crude',
            model_type='hybrid',
            output_formats=['geojson', 'json', 'csv']
        )
        
        # Verify all mocks were called
        mock_fetch_data.get_wind_data.assert_called_once()
        mock_fetch_data.get_ocean_currents.assert_called_once()
        mock_fetch_data.get_elevation_data.assert_called_once()
        mock_fetch_data.get_oil_properties.assert_called_once()
        mock_preprocess.preprocess_all_data.assert_called_once()
        mock_model.run_model.assert_called_once()
        mock_export.export_to_geojson.assert_called_once()
        mock_export.export_to_json.assert_called_once()
        mock_export.export_to_csv.assert_called_once()
        mock_export.export_to_time_series_csv.assert_called_once()
        
        # Verify simulation state
        self.assertEqual(manager.simulation_state['status'], 'completed')
        self.assertEqual(manager.simulation_state['progress'], 100.0)
        
        # Verify results
        self.assertEqual(results['status'], 'success')
        self.assertIn('execution_time', results)
        self.assertIn('output_files', results)
        self.assertEqual(len(results['output_files']), 4)  # geojson, json, csv, time_series_csv
    
    @patch('trajectory_core.main.parse_arguments')
    @patch('trajectory_core.main.SimulationManager')
    @patch('trajectory_core.main.display_progress')
    def test_main_function(self, mock_display_progress, mock_manager_class, mock_parse_args):
        """Test the main function."""
        # Set up mock arguments
        mock_args = MagicMock()
        mock_args.lat = -3.57
        mock_args.lon = -80.45
        mock_args.volume = 5000
        mock_args.oil_type = 'medium_crude'
        mock_args.model_type = 'hybrid'
        mock_args.duration = 24
        mock_args.timestep = 60
        mock_args.particles = 100
        mock_args.output_formats = ['geojson', 'json', 'csv']
        mock_args.output_dir = self.test_dir
        mock_args.config_file = None
        mock_args.save_config = None
        mock_args.verbose = False
        mock_args.quiet = False
        mock_args.log_file = None
        mock_args.scenario = None
        mock_args.output_prefix = None
        mock_parse_args.return_value = mock_args
        
        # Set up mock manager
        mock_manager = MagicMock()
        mock_manager.simulation_state = {'status': 'completed'}
        mock_manager.run_simulation.return_value = {
            'status': 'success',
            'execution_time': 10.5,
            'output_files': {
                'geojson': os.path.join(self.test_dir, 'test.geojson'),
                'json': os.path.join(self.test_dir, 'test.json'),
                'csv': os.path.join(self.test_dir, 'test.csv')
            }
        }
        mock_manager_class.return_value = mock_manager
        
        # Run main function
        with patch('sys.stdout'):
            exit_code = main.main()
        
        # Verify exit code
        self.assertEqual(exit_code, 0)
        
        # Verify manager was created with correct parameters
        mock_manager_class.assert_called_once()
        
        # Verify run_simulation was called with correct parameters
        mock_manager.run_simulation.assert_called_once_with(
            spill_location=(-3.57, -80.45),
            spill_volume=5000,
            oil_type='medium_crude',
            model_type='hybrid',
            output_formats=['geojson', 'json', 'csv']
        )
    
    def test_get_scenario_params(self):
        """Test the get_scenario_params function."""
        # Test valid scenario
        tumbes_params = main.get_scenario_params('tumbes')
        self.assertEqual(tumbes_params['lat'], -3.57)
        self.assertEqual(tumbes_params['lon'], -80.45)
        self.assertEqual(tumbes_params['volume'], 5000)
        
        # Test invalid scenario
        with self.assertRaises(ValueError):
            main.get_scenario_params('invalid_scenario')
    
    def test_save_configuration(self):
        """Test the save_configuration function."""
        # Create mock args
        class MockArgs:
            def __init__(self):
                self.lat = -3.57
                self.lon = -80.45
                self.volume = 5000
                self.oil_type = 'medium_crude'
                self.model_type = 'hybrid'
                self.duration = 24
                self.timestep = 60
                self.particles = 100
                self.output_formats = ['geojson', 'json']
                self.output_dir = '/tmp'
                self.verbose = True
        
        args = MockArgs()
        
        # Test saving to JSON
        json_path = os.path.join(self.test_dir, 'test_config.json')
        main.save_configuration(args, json_path)
        
        # Verify JSON file was created and contains correct data
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, 'r') as f:
            saved_config = json.load(f)
        
        self.assertEqual(saved_config['lat'], -3.57)
        self.assertEqual(saved_config['lon'], -80.45)
        self.assertEqual(saved_config['volume'], 5000)
        
        # Test saving to INI
        ini_path = os.path.join(self.test_dir, 'test_config.ini')
        main.save_configuration(args, ini_path)
        
        # Verify INI file was created and contains correct data
        self.assertTrue(os.path.exists(ini_path))
        config_parser = configparser.ConfigParser()
        config_parser.read(ini_path)
        
        self.assertEqual(float(config_parser['DEFAULT']['lat']), -3.57)
        self.assertEqual(float(config_parser['DEFAULT']['lon']), -80.45)
        self.assertEqual(float(config_parser['DEFAULT']['volume']), 5000)
        
        # Test invalid file format
        with self.assertRaises(ValueError):
            main.save_configuration(args, os.path.join(self.test_dir, 'test_config.xyz'))


if __name__ == '__main__':
    unittest.main()
