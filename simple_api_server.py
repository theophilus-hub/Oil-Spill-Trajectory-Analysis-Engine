#!/usr/bin/env python
"""
Simplified Oil Spill Trajectory Analysis API server.
"""

import os
import uuid
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Import Flask and related modules
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    from flasgger import Swagger, swag_from
except ImportError:
    print("Required packages not found. Please install them with:")
    print("pip install flask flask-cors flasgger")
    raise

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Swagger
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Oil Spill Trajectory Analysis API",
        "description": "REST API for oil spill trajectory simulation",
        "version": "1.0.0",
        "contact": {
            "name": "Oil Spill Trajectory Analysis Team"
        }
    },
    "basePath": "/api/v1",
    "schemes": ["http", "https"],
    "securityDefinitions": {},
    "security": []
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

# Dictionary to store simulation status and results
simulations = {}

# Mock simulation function (replace with actual simulation logic later)
def run_mock_simulation(simulation_id: str, params: Dict[str, Any]) -> None:
    """Run a mock simulation in a background thread."""
    try:
        # Mark simulation as running
        simulations[simulation_id]['status'] = 'running'
        simulations[simulation_id]['progress'] = 0
        
        # Simulate progress
        for progress in range(0, 101, 10):
            # Update progress
            simulations[simulation_id]['progress'] = progress
            logger.info(f"Simulation {simulation_id} progress: {progress}%")
            
            # Sleep to simulate work
            time.sleep(1)
            
            # Check if we should simulate an error (for testing)
            if 'simulate_error' in params and params['simulate_error'] and progress > 50:
                simulations[simulation_id]['status'] = 'error'
                simulations[simulation_id]['error'] = "Simulated error for testing"
                return
        
        # Create output directory if it doesn't exist
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        # Generate a mock GeoJSON file
        geojson_file = output_dir / f"{simulation_id}.geojson"
        
        # Create a simple GeoJSON with the spill location and a few points
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "name": "Spill Origin",
                        "time": datetime.now().isoformat()
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [params['longitude'], params['latitude']]
                    }
                }
            ],
            "properties": {
                "simulation_id": simulation_id,
                "oil_type": params.get('oil_type', 'medium_crude'),
                "volume": params.get('volume', 1000),
                "duration_hours": params.get('duration_hours', 24),
                "created_at": simulations[simulation_id]['created_at']
            }
        }
        
        # Add some mock trajectory points
        for i in range(10):
            # Create points that drift slightly from the origin
            lon_offset = (i + 1) * 0.01
            lat_offset = (i + 1) * 0.005
            
            # Adjust direction based on hemisphere
            lon_direction = 1 if params['longitude'] > 0 else -1
            lat_direction = 1 if params['latitude'] > 0 else -1
            
            geojson_data["features"].append({
                "type": "Feature",
                "properties": {
                    "name": f"Trajectory Point {i+1}",
                    "time": (datetime.now().timestamp() + i * 3600),
                    "hour": i + 1
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        params['longitude'] + (lon_offset * lon_direction),
                        params['latitude'] + (lat_offset * lat_direction)
                    ]
                }
            })
        
        # Write the GeoJSON file
        with open(geojson_file, 'w') as f:
            json.dump(geojson_data, f, indent=2)
        
        # Mark simulation as completed
        simulations[simulation_id]['status'] = 'completed'
        simulations[simulation_id]['progress'] = 100
        simulations[simulation_id]['completed_at'] = datetime.now().isoformat()
        simulations[simulation_id]['results'] = {
            'geojson_file': str(geojson_file)
        }
        
        logger.info(f"Mock simulation {simulation_id} completed successfully")
        
    except Exception as e:
        # Handle errors
        simulations[simulation_id]['status'] = 'error'
        simulations[simulation_id]['error'] = str(e)
        logger.error(f"Error in mock simulation {simulation_id}: {e}")


@app.route('/api/v1/simulate', methods=['POST'])
@swag_from({
    'tags': ['Simulation'],
    'summary': 'Start a new simulation',
    'description': 'Starts a new oil spill simulation with the provided parameters',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'latitude': {
                        'type': 'number',
                        'description': 'Latitude of the spill location (decimal degrees)'
                    },
                    'longitude': {
                        'type': 'number',
                        'description': 'Longitude of the spill location (decimal degrees)'
                    },
                    'volume': {
                        'type': 'number',
                        'description': 'Volume of the spill in cubic meters (m³)'
                    },
                    'oil_type': {
                        'type': 'string',
                        'description': 'Type of oil spilled',
                        'enum': ['light_crude', 'medium_crude', 'heavy_crude', 'bunker_fuel']
                    },
                    'duration_hours': {
                        'type': 'integer',
                        'description': 'Duration of the simulation in hours',
                        'default': 24
                    },
                    'timestep_minutes': {
                        'type': 'integer',
                        'description': 'Timestep of the simulation in minutes',
                        'default': 60
                    },
                    'particle_count': {
                        'type': 'integer',
                        'description': 'Number of particles to use in the simulation',
                        'default': 1000
                    }
                },
                'required': ['latitude', 'longitude', 'volume', 'oil_type']
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Simulation started successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'id': {
                        'type': 'string',
                        'description': 'Unique ID for the simulation'
                    },
                    'status': {
                        'type': 'string',
                        'description': 'Current status of the simulation'
                    },
                    'created_at': {
                        'type': 'string',
                        'description': 'Timestamp when the simulation was created'
                    }
                }
            }
        },
        '400': {
            'description': 'Bad request - Invalid parameters',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {
                        'type': 'string',
                        'description': 'Error message'
                    }
                }
            }
        }
    }
})
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
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'results': None
        }
        
        # Start simulation in a background thread
        thread = threading.Thread(
            target=run_mock_simulation,
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
@swag_from({
    'tags': ['Simulation'],
    'summary': 'Get simulation status or results',
    'description': 'Returns the current status of a simulation or the complete GeoJSON results if the simulation is completed',
    'parameters': [
        {
            'name': 'simulation_id',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Unique ID for the simulation'
        }
    ],
    'responses': {
        '200': {
            'description': 'Simulation status or results retrieved successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'id': {
                        'type': 'string',
                        'description': 'Unique ID for the simulation'
                    },
                    'status': {
                        'type': 'string',
                        'description': 'Current status of the simulation (queued, running, completed, error)'
                    },
                    'progress': {
                        'type': 'number',
                        'description': 'Progress of the simulation (0-100)'
                    },
                    'created_at': {
                        'type': 'string',
                        'description': 'Timestamp when the simulation was created'
                    }
                }
            }
        },
        '404': {
            'description': 'Simulation not found',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {
                        'type': 'string',
                        'description': 'Error message'
                    }
                }
            }
        },
        '500': {
            'description': 'Internal server error',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {
                        'type': 'string',
                        'description': 'Error message'
                    },
                    'status': {
                        'type': 'string',
                        'description': 'Error status'
                    }
                }
            }
        }
    }
})
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
        if simulation['results'] and 'geojson_file' in simulation['results']:
            geojson_file = simulation['results']['geojson_file']
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
            'progress': simulation.get('progress', 0),
            'created_at': simulation['created_at']
        })


@app.route('/api/v1/health', methods=['GET'])
@swag_from({
    'tags': ['System'],
    'summary': 'API health check',
    'description': 'Returns the health status of the API and basic statistics',
    'responses': {
        '200': {
            'description': 'Health status',
            'schema': {
                'type': 'object',
                'properties': {
                    'status': {
                        'type': 'string',
                        'description': 'Health status of the API'
                    },
                    'version': {
                        'type': 'string',
                        'description': 'API version'
                    },
                    'active_simulations': {
                        'type': 'integer',
                        'description': 'Number of active simulations'
                    },
                    'total_simulations': {
                        'type': 'integer',
                        'description': 'Total number of simulations'
                    }
                }
            }
        }
    }
})
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'ok',
        'version': '1.0.0',
        'active_simulations': len([s for s in simulations.values() if s['status'] in ['queued', 'running']]),
        'total_simulations': len(simulations)
    })


@app.route('/api/v1/docs', methods=['GET'])
def redirect_to_docs():
    """Redirect to API documentation."""
    from flask import redirect
    return redirect('/docs/')


@app.route('/', methods=['GET'])
def index():
    """Root endpoint that provides basic API information and links."""
    return jsonify({
        'name': 'Oil Spill Trajectory Analysis API',
        'version': '1.0.0',
        'description': 'API for simulating oil spill trajectories',
        'endpoints': {
            'simulate': {
                'url': '/api/v1/simulate',
                'method': 'POST',
                'description': 'Start a new simulation',
                'parameters': {
                    'latitude': 'Latitude of the spill location (decimal degrees)',
                    'longitude': 'Longitude of the spill location (decimal degrees)',
                    'volume': 'Volume of the spill in cubic meters (m³)',
                    'oil_type': 'Type of oil (e.g., light_crude, medium_crude, heavy_crude)',
                    'duration_hours': 'Duration of the simulation in hours (optional, default: 24)',
                    'timestep_minutes': 'Timestep of the simulation in minutes (optional, default: 60)'
                }
            },
            'simulation_status': {
                'url': '/api/v1/simulation/:id',
                'method': 'GET',
                'description': 'Get simulation status or results',
                'parameters': {
                    'id': 'Simulation ID returned from the simulate endpoint'
                }
            },
            'health': {
                'url': '/api/v1/health',
                'method': 'GET',
                'description': 'Check API health status'
            }
        }
    })


def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask server."""
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the simplified Oil Spill API server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Log server settings
    logger.info(f"Starting simplified API server on {args.host}:{args.port}")
    logger.info(f"Debug mode: {args.debug}")
    
    # Run the server
    run_server(host=args.host, port=args.port, debug=args.debug)
