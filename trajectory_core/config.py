"""
Configuration settings for the Oil Spill Trajectory Analysis Engine.

This module contains default configuration parameters and settings
for the simulation, data sources, and output formats.
"""

# API Keys and endpoints (to be configured by the user)
API_KEYS = {
    # All services used are free and don't require API keys
}

# Default simulation parameters
DEFAULT_SIMULATION_PARAMS = {
    'timestep_minutes': 30,
    'duration_hours': 48,
    'particle_count': 1000,
    'random_seed': 42,
    'diffusion_coefficient': 0.1,
    'evaporation_rate': 0.05,
}

# Data source URLs
DATA_SOURCES = {
    'wind': {
        'open_meteo': 'https://api.open-meteo.com/v1/forecast',
    },
    'ocean_currents': {
        'noaa_erddap': 'https://coastwatch.pfeg.noaa.gov/erddap/griddap/hycom_gom.json',
        'osm_currents': 'https://gis.ices.dk/geoserver/OSPAR/wms',
    },
    'elevation': {
        'aws_terrain': 'https://s3.amazonaws.com/elevation-tiles-prod',
        'open_topo': 'https://portal.opentopography.org/API/globaldem',
    },
    'oil_properties': {
        'adios': 'static/oil_properties.json',  # Local file path
    }
}

# Output configuration
OUTPUT_CONFIG = {
    'default_format': 'geojson',
    'available_formats': ['geojson', 'json', 'csv'],
    'output_directory': './output',
}

# Flask API configuration
FLASK_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'threaded': True,
}
