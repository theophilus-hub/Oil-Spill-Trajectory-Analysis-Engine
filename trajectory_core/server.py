"""
Flask API server for the Oil Spill Trajectory Analysis Engine.

This module provides a REST API for the simulation:
- POST /simulate -> triggers analysis with input payload
- GET /status/:id -> returns result or progress
- Output files served as downloads or inline JSON
"""

import os
import uuid
import json
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS

from . import config
from . import main

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trajectory_api.log')
    ]
)

logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Dictionary to store simulation status and results
simulations = {}


def run_simulation_task(simulation_id: str, params: Dict[str, Any]) -> None:
    """
    Run a simulation in a background thread.
    
    Args:
        simulation_id: Unique ID for the simulation
        params: Simulation parameters
    """
    try:
        # Update simulation status
        simulations[simulation_id]['status'] = 'running'
        
        # Extract parameters
        spill_location = (
            params.get('latitude', 0),
            params.get('longitude', 0)
        )
        spill_volume = params.get('volume', 1000)
        oil_type = params.get('oil_type', 'medium_crude')
        model_type = params.get('model_type', 'hybrid')
        
        # Extract simulation parameters
        simulation_params = {
            'duration_hours': params.get('duration_hours', 48),
            'timestep_minutes': params.get('timestep_minutes', 30),
            'particle_count': params.get('particle_count', 1000)
        }
        
        # Create simulation manager
        manager = main.SimulationManager(simulation_params)
        
        # Run simulation
        results = manager.run_simulation(
            spill_location=spill_location,
            spill_volume=spill_volume,
            oil_type=oil_type,
            model_type=model_type
        )
        
        # Update simulation status and results
        simulations[simulation_id]['status'] = 'completed'
        simulations[simulation_id]['results'] = results
        simulations[simulation_id]['completed_at'] = datetime.now().isoformat()
        
    except Exception as e:
        # Update simulation status with error
        simulations[simulation_id]['status'] = 'error'
        simulations[simulation_id]['error'] = str(e)
        logger.error(f"Error in simulation {simulation_id}: {e}")


@app.route('/simulate', methods=['POST'])
def start_simulation():
    """Start a new simulation with the provided parameters."""
    try:
        # Parse request data
        data = request.json
        
        if not data:
            return jsonify({
                'error': 'No data provided'
            }), 400
        
        # Validate required parameters
        required_params = ['latitude', 'longitude', 'volume']
        for param in required_params:
            if param not in data:
                return jsonify({
                    'error': f'Missing required parameter: {param}'
                }), 400
        
        # Generate unique ID for this simulation
        simulation_id = str(uuid.uuid4())
        
        # Store simulation info
        simulations[simulation_id] = {
            'id': simulation_id,
            'status': 'queued',
            'params': data,
            'created_at': datetime.now().isoformat(),
            'results': None,
            'error': None
        }
        
        # Start simulation in background thread
        thread = threading.Thread(
            target=run_simulation_task,
            args=(simulation_id, data)
        )
        thread.daemon = True
        thread.start()
        
        # Return simulation ID
        return jsonify({
            'id': simulation_id,
            'status': 'queued',
            'message': 'Simulation queued successfully'
        })
        
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/status/<simulation_id>', methods=['GET'])
def get_simulation_status(simulation_id):
    """Get the status of a simulation."""
    if simulation_id not in simulations:
        return jsonify({
            'error': 'Simulation not found'
        }), 404
    
    # Get simulation info
    simulation = simulations[simulation_id]
    
    # Return status and basic info
    response = {
        'id': simulation_id,
        'status': simulation['status'],
        'created_at': simulation['created_at']
    }
    
    # Add completion time if available
    if 'completed_at' in simulation:
        response['completed_at'] = simulation['completed_at']
    
    # Add error if available
    if simulation['error']:
        response['error'] = simulation['error']
    
    # Add result summary if completed
    if simulation['status'] == 'completed' and simulation['results']:
        # Include output file paths
        response['output_files'] = {
            format_type: os.path.basename(filepath)
            for format_type, filepath in simulation['results']['output_files'].items()
        }
        
        # Add download URLs
        base_url = request.url_root.rstrip('/')
        response['download_urls'] = {
            format_type: f"{base_url}/download/{simulation_id}/{format_type}"
            for format_type in simulation['results']['output_files'].keys()
        }
    
    return jsonify(response)


@app.route('/results/<simulation_id>', methods=['GET'])
def get_simulation_results(simulation_id):
    """Get the full results of a completed simulation."""
    if simulation_id not in simulations:
        return jsonify({
            'error': 'Simulation not found'
        }), 404
    
    simulation = simulations[simulation_id]
    
    if simulation['status'] != 'completed':
        return jsonify({
            'error': 'Simulation not completed',
            'status': simulation['status']
        }), 400
    
    if not simulation['results']:
        return jsonify({
            'error': 'No results available'
        }), 500
    
    # Return the full results
    return jsonify(simulation['results'])


@app.route('/download/<simulation_id>/<format_type>', methods=['GET'])
def download_result_file(simulation_id, format_type):
    """Download a result file."""
    if simulation_id not in simulations:
        return jsonify({
            'error': 'Simulation not found'
        }), 404
    
    simulation = simulations[simulation_id]
    
    if simulation['status'] != 'completed':
        return jsonify({
            'error': 'Simulation not completed',
            'status': simulation['status']
        }), 400
    
    if not simulation['results'] or 'output_files' not in simulation['results']:
        return jsonify({
            'error': 'No output files available'
        }), 500
    
    if format_type not in simulation['results']['output_files']:
        return jsonify({
            'error': f'No {format_type} file available'
        }), 404
    
    # Get file path
    file_path = simulation['results']['output_files'][format_type]
    
    # Check if file exists
    if not os.path.exists(file_path):
        return jsonify({
            'error': 'File not found on server'
        }), 404
    
    # Determine content type
    content_types = {
        'geojson': 'application/geo+json',
        'json': 'application/json',
        'csv': 'text/csv'
    }
    
    # Send file
    return send_file(
        file_path,
        mimetype=content_types.get(format_type, 'application/octet-stream'),
        as_attachment=True,
        download_name=os.path.basename(file_path)
    )


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'ok',
        'version': '0.1.0',
        'active_simulations': len([s for s in simulations.values() if s['status'] == 'running']),
        'total_simulations': len(simulations)
    })


def run_server(host=None, port=None, debug=None):
    """
    Run the Flask server.
    
    Args:
        host: Host to bind to (default from config)
        port: Port to bind to (default from config)
        debug: Whether to run in debug mode (default from config)
    """
    # Use config values if not provided
    if host is None:
        host = config.FLASK_CONFIG['host']
    
    if port is None:
        port = config.FLASK_CONFIG['port']
    
    if debug is None:
        debug = config.FLASK_CONFIG['debug']
    
    # Run the Flask app
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server()
