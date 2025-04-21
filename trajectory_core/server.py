"""
Flask API server for the Oil Spill Trajectory Analysis Engine.

This module provides a REST API for the simulation:
- POST /api/v1/simulate -> triggers analysis with input payload
- GET /api/v1/status/:id -> returns result or progress
- GET /api/v1/results/:id -> returns full simulation results
- GET /api/v1/download/:id/:format -> downloads result files
"""

import os
import uuid
import json
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Import necessary modules
try:
    from flask import Flask, request, jsonify, send_file, Response
    from flask_cors import CORS
    from werkzeug.exceptions import HTTPException
except ImportError:
    print("Flask and related packages not found. Please install them with:")
    print("pip install flask flask-cors werkzeug")
    raise

# Import project modules
from trajectory_core import config, main, export

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Dictionary to store simulation status and results
simulations = {}

def run_simulation_task(simulation_id: str, params: Dict[str, Any]) -> None:
    """Run a simulation in a background thread."""
    try:
        # Mark simulation as running
        simulations[simulation_id]['status'] = 'running'
        
        # Create simulation manager
        manager = main.SimulationManager(params)
        
        # Set progress callback
        def progress_callback(progress: float, stage: Optional[str] = None):
            """Update simulation progress in the simulations dictionary."""
            simulations[simulation_id]['progress'] = progress
            if stage:
                simulations[simulation_id]['current_stage'] = stage
            logger.debug(f"Simulation {simulation_id} progress: {progress:.1f}% ({stage})")
        
        manager.set_progress_callback(progress_callback)
        
        # Run simulation
        logger.info(f"Starting simulation {simulation_id}")
        try:
            results = manager.run_simulation(
                spill_location=(params['latitude'], params['longitude']),
                spill_volume=params['volume'],
                oil_type=params['oil_type'],
                model_type=params.get('model_type', 'hybrid'),
                duration_hours=params.get('duration_hours', 72),
                timestep_minutes=params.get('timestep_minutes', 15),
                output_formats=['geojson'],  # Only generate GeoJSON for simplicity
                output_directory=params.get('output_directory', config.OUTPUT_CONFIG['output_directory'])
            )
            
            # Store results
            simulations[simulation_id]['status'] = 'completed'
            simulations[simulation_id]['progress'] = 100.0
            simulations[simulation_id]['results'] = results
            simulations[simulation_id]['completed_at'] = datetime.now().isoformat()
            
            logger.info(f"Simulation {simulation_id} completed successfully")
        except Exception as inner_e:
            # Handle simulation-specific errors
            simulations[simulation_id]['status'] = 'error'
            simulations[simulation_id]['error'] = str(inner_e)
            logger.error(f"Simulation {simulation_id} failed: {inner_e}")
            logger.exception(inner_e)
    
    except Exception as e:
        # Handle thread-level errors
        simulations[simulation_id]['status'] = 'error'
        simulations[simulation_id]['error'] = str(e)
        logger.error(f"Thread error for simulation {simulation_id}: {e}")
        logger.exception(e)


@app.route('/api/v1/simulate', methods=['POST'])
def start_simulation():
    """Start a new simulation."""
    try:
        # Parse request data
        data = request.json
        
        # Validate required parameters
        required_params = ['latitude', 'longitude', 'volume', 'oil_type']
        for param in required_params:
            if param not in data:
                return jsonify({
                    'error': f'Missing required parameter: {param}'
                }), 400
        
        # Generate simulation ID
        simulation_id = str(uuid.uuid4())
        
        # Store simulation in memory
        simulations[simulation_id] = {
            'id': simulation_id,
            'params': data,
            'status': 'queued',
            'progress': 0.0,
            'created_at': datetime.now().isoformat(),
            'results': None
        }
        
        # Start simulation in a background thread
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
            'created_at': simulations[simulation_id]['created_at']
        })
        
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/v1/simulation/<simulation_id>', methods=['GET'])
def get_simulation(simulation_id):
    """Get simulation results or status."""
    # Check if simulation exists
    if simulation_id not in simulations:
        return jsonify({
            'error': 'Simulation not found'
        }), 404
    
    # Get simulation data
    simulation = simulations[simulation_id]
    
    # Check if simulation is completed
    if simulation['status'] == 'completed':
        # Return GeoJSON results directly
        if simulation['results'] and 'output_files' in simulation['results'] and 'geojson' in simulation['results']['output_files']:
            geojson_file = simulation['results']['output_files']['geojson']
            try:
                with open(geojson_file, 'r') as f:
                    geojson_data = json.load(f)
                return jsonify(geojson_data)
            except Exception as e:
                logger.error(f"Error reading GeoJSON file: {e}")
                return jsonify({
                    'error': f'Error reading GeoJSON file: {e}',
                    'status': 'error'
                }), 500
        else:
            return jsonify({
                'error': 'No GeoJSON results available',
                'status': 'error'
            }), 500
    
    # If simulation has error, return error message
    elif simulation['status'] == 'error':
        return jsonify({
            'error': simulation.get('error', 'Unknown error'),
            'status': 'error'
        }), 500
    
    # Otherwise, return status
    else:
        return jsonify({
            'id': simulation_id,
            'status': simulation['status'],
            'progress': simulation.get('progress', 0.0),
            'current_stage': simulation.get('current_stage', ''),
            'created_at': simulation['created_at']
        })


@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'ok',
        'version': '1.0.0',
        'active_simulations': len([s for s in simulations.values() if s['status'] in ['queued', 'running']]),
        'total_simulations': len(simulations)
    })


# Error handling
@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': 'Bad request',
        'message': str(error)
    }), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not found',
        'message': 'The requested resource was not found'
    }), 404


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500


@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors
    if isinstance(e, HTTPException):
        return e
    
    # Log the error
    logger.error(f"Unhandled exception: {e}")
    
    # Return a generic error response
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


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
    app.run(host=host, port=port, debug=debug, threaded=config.FLASK_CONFIG.get('threaded', True))


if __name__ == '__main__':
    run_server()
