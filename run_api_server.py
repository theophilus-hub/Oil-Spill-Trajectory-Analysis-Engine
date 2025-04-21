#!/usr/bin/env python
"""
Run the Oil Spill Trajectory Analysis API server.
"""

import os
import sys
import argparse
import logging
from trajectory_core import server, config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the API server."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the Oil Spill Trajectory Analysis API server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory for output files')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set output directory in config
    config.OUTPUT_CONFIG['output_directory'] = args.output_dir
    
    # Log server settings
    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Run the server
    server.run_server(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
