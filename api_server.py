#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone Flask API server for the Oil Spill Trajectory Analysis Engine.

This script provides a REST API for the oil spill simulation:
- POST /api/v1/simulate -> triggers analysis with input payload
- GET /api/v1/status/:id -> returns result or progress
- GET /api/v1/results/:id -> returns full simulation results
- GET /api/v1/download/:id/:format -> downloads result files

Usage:
    python api_server.py [--host HOST] [--port PORT] [--debug]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path to import trajectory_core modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from trajectory_core import server
from trajectory_core import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Oil Spill Trajectory Analysis API Server'
    )
    
    parser.add_argument('--host', type=str, default=config.FLASK_CONFIG['host'],
                        help=f'Host to bind to (default: {config.FLASK_CONFIG["host"]})')
    parser.add_argument('--port', type=int, default=config.FLASK_CONFIG['port'],
                        help=f'Port to bind to (default: {config.FLASK_CONFIG["port"]})')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    parser.add_argument('--log-file', type=str, default='api_server.log',
                        help='Log file path (default: api_server.log)')
    
    return parser.parse_args()


def setup_logging(log_file):
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    # Set Flask and Werkzeug logging to WARNING level to reduce noise
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('flask').setLevel(logging.WARNING)


def main():
    """Main entry point for the API server."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args.log_file)
    
    # Get logger
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    output_dir = config.OUTPUT_CONFIG['output_directory']
    os.makedirs(output_dir, exist_ok=True)
    
    # Log startup information
    logger.info(f"Starting Oil Spill Trajectory Analysis API Server")
    logger.info(f"Host: {args.host}, Port: {args.port}, Debug: {args.debug}")
    
    # Run the server
    try:
        server.run_server(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
