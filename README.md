# Oil Spill Trajectory Analysis Engine

A modular Python system for simulating oil spill trajectories on both land and water surfaces.

## Project Overview

The Oil Spill Trajectory Analysis Engine is designed to be integrated as a core feature in a larger desktop application built with Rust and TypeScript. It leverages scientific models and real environmental data to provide accurate predictions of oil spill movement over time.

The system:
- Acquires environmental data (wind, weather, ocean currents, elevation)
- Preprocesses this data for modeling
- Simulates spill movement using advanced physics models
- Outputs results in standard formats (JSON, GeoJSON, CSV)

## Installation

### Requirements
- Python 3.8+
- Required packages listed in requirements.txt

### Setup
```bash
# Clone the repository
git clone https://github.com/theophilus-hub/Oil-Spill-Trajectory-Analysis-Engine.git
cd Oil-Spill-Trajectory-Analysis-Engine

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

# Check simulation status
curl http://localhost:5000/status/<simulation_id>

# Download results
curl http://localhost:5000/download/<simulation_id>/geojson -o result.geojson
```

## Module Overview

### trajectory_core
The main package containing all modules for the oil spill trajectory simulation.

#### config.py
Configuration settings for the simulation, data sources, and output formats. Contains default parameters and API endpoints.

#### fetch_data.py
Handles acquisition of environmental data from various sources:
- Wind and weather data from Open-Meteo
- Ocean current data from NOAA ERDDAP and OSM Currents
- Elevation data from AWS Terrain Tiles and OpenTopography
- Oil properties from static datasets

#### preprocess.py
Prepares and normalizes input data for modeling:
- DEM resampling and slope calculation
- Wind/current data interpolation
- Particle position initialization for Lagrangian model

#### model.py
Implements the core simulation algorithms:
- Water-based Lagrangian particle model
- Land-based downhill slope flow model
- Hybrid model combining both approaches
- Physics models for diffusion, evaporation, and decay

#### export.py
Formats and exports simulation results:
- GeoJSON export for mapping visualization
- JSON export for raw structured results
- CSV export for summary statistics

#### main.py
Provides the main entry point and orchestration for the simulation:
- Coordinates the data acquisition, preprocessing, modeling, and export steps
- Provides a CLI interface for running simulations
- Handles configuration and parameter management

#### server.py
Implements a Flask REST API for the simulation:
- POST /simulate endpoint to trigger analysis
- GET /status/:id endpoint to check progress
- Output files served as downloads or inline JSON

## Data Sources

The system uses free, publicly available data sources that don't require API keys:

- **Wind Data**: Open-Meteo API
- **Ocean Currents**: NOAA ERDDAP and OSM Currents
- **Elevation Data**: AWS Terrain Tiles and OpenTopography
- **Oil Properties**: Static dataset included in the package

## Development

### Project Structure
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

### Adding New Features
To extend the system:
1. Update the appropriate module based on the feature type
2. Add any new configuration parameters to config.py
3. Update tests to cover the new functionality
4. Document the feature in this README

## License
MIT

## Author
SKAGE.dev
