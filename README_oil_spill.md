# Oil Spill Trajectory Analysis Engine

A modular Python system for simulating oil spill trajectories on both land and water surfaces.

## Overview

The Oil Spill Trajectory Analysis Engine is designed to be integrated as a core feature in a larger desktop application built with Rust and TypeScript. It leverages scientific models and real environmental data to provide accurate predictions of oil spill movement over time.

The system:
- Acquires environmental data (wind, weather, ocean currents, elevation)
- Preprocesses this data for modeling
- Simulates spill movement using advanced physics models
- Outputs results in standard formats (JSON, GeoJSON, CSV)

## Features

### Data Acquisition
- Fetches wind and weather data from Open-Meteo and NOAA
- Retrieves ocean current data from CMEMS and HYCOM
- Downloads elevation/DEM data from USGS and Copernicus
- Loads oil properties from static datasets

### Modeling Engine
- Water-based Lagrangian particle model for ocean/lake spills
- Land-based downhill slope or cost-distance flow modeling
- Accounts for diffusion, evaporation, and decay factors
- Runs time-stepped simulation with configurable parameters

### Export Module
- Exports results to GeoJSON for mapping visualization
- Provides JSON output for raw structured results
- Offers CSV export for summary statistics

### Optional Flask API
- Exposes functionality via REST endpoints
- Provides POST /simulate endpoint to trigger analysis
- Offers GET /status/:id endpoint to check progress

## Installation

### Requirements
- Python 3.8+
- Required packages listed in requirements.txt

### Setup
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Usage

### Command Line Interface
```bash
# Run a simulation
oil-trajectory --lat 37.7749 --lon -122.4194 --volume 10000 --model-type hybrid

# Start the Flask API server
oil-trajectory-server
```

### Python API
```python
from trajectory_core import main

# Create simulation manager
manager = main.SimulationManager()

# Run simulation
results = manager.run_simulation(
    spill_location=(37.7749, -122.4194),
    spill_volume=10000,
    oil_type='medium_crude',
    model_type='hybrid'
)

# Access results
print(results['output_files'])
```

### Flask API
```bash
# Start the server
oil-trajectory-server

# Use the API
curl -X POST http://localhost:5000/simulate \
  -H "Content-Type: application/json" \
  -d '{"latitude": 37.7749, "longitude": -122.4194, "volume": 10000}'
```

## Project Structure
```
trajectory_core/
├── __init__.py        # Package initialization
├── config.py          # Configuration settings
├── fetch_data.py      # Data acquisition module
├── preprocess.py      # Data preprocessing module
├── model.py           # Simulation models
├── export.py          # Result export functionality
├── main.py            # Main orchestration module
└── server.py          # Flask API server
```

## License
MIT

## Author
SKAGE.dev
