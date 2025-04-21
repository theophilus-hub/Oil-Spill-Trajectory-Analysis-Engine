# Oil Spill Trajectory Analysis Engine API

This API provides RESTful access to the Oil Spill Trajectory Analysis Engine, allowing users to trigger simulations, check their status, retrieve results, and download output files.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the API server: `python run_api_server.py`

### Running the API Server

```bash
python run_api_server.py [--host HOST] [--port PORT] [--debug]
```

Options:
- `--host HOST`: Host to bind to (default: 127.0.0.1)
- `--port PORT`: Port to bind to (default: 5000)
- `--debug`: Run in debug mode (default: False)

## API Documentation

The API documentation is available at `/docs/` when the server is running. You can also access it at `/api/v1/docs`.

### Authentication

Authentication has been removed from the API for simplicity. All endpoints are now accessible without any API key.

### Endpoints

#### Start a Simulation

```
POST /api/v1/simulate
```

Starts a new oil spill simulation with the provided parameters.

**Request Body:**

```json
{
  "latitude": -3.57,
  "longitude": -80.45,
  "volume": 5000,
  "oil_type": "medium_crude",
  "duration_hours": 24,
  "timestep_minutes": 60,
  "particle_count": 1000
}
```

**Parameters:**

- `latitude` (required): Latitude of the spill location (decimal degrees)
- `longitude` (required): Longitude of the spill location (decimal degrees)
- `volume` (required): Volume of the spill in cubic meters (m³)
- `oil_type` (required): Type of oil (e.g., light_crude, medium_crude, heavy_crude)
- `duration_hours` (optional): Duration of the simulation in hours (default: 24)
- `timestep_minutes` (optional): Timestep of the simulation in minutes (default: 60)
- `particle_count` (optional): Number of particles to use in the simulation (default: 1000)

#### Start Multiple Simulations in Batch

```
POST /api/v1/batch-simulate
```

Starts multiple oil spill simulations with different parameters in a single request.

**Request Body:**

```json
{
  "simulations": [
    {
      "latitude": -3.57,
      "longitude": -80.45,
      "volume": 5000,
      "oil_type": "medium_crude"
    },
    {
      "latitude": -3.60,
      "longitude": -80.50,
      "volume": 3000,
      "oil_type": "light_crude"
    }
  ],
  "common_params": {
    "model_type": "hybrid",
    "duration_hours": 48,
    "timestep_minutes": 30,
    "particle_count": 1000,
    "output_formats": ["geojson", "json", "csv"]
  },
  "batch_name": "Tumbes coastal simulations"
}
```

**Response:**

```json
{
  "batch_id": "batch-12345678",
  "simulation_ids": ["sim-abcdef12", "sim-98765432"],
  "batch_name": "Tumbes coastal simulations",
  "created_at": "2023-07-01T12:00:00"
}
```

#### Check Simulation Status

```
GET /api/v1/status/:id
```

Returns the status of a simulation.

**Response:**

```json
{
  "id": "simulation-id",
  "status": "running",
  "progress": 45.5,
  "current_stage": "modeling",
  "created_at": "2023-07-01T12:00:00"
}
```

#### Get Batch Status

```
GET /api/v1/batch-status/:batch_id
```

Returns the status of all simulations in a batch.

**Response:**

```json
{
  "batch_id": "batch-12345678",
  "batch_name": "Tumbes coastal simulations",
  "created_at": "2023-07-01T12:00:00",
  "simulations": [
    {
      "id": "sim-abcdef12",
      "status": "completed",
      "progress": 100.0,
      "current_stage": "completed"
    },
    {
      "id": "sim-98765432",
      "status": "running",
      "progress": 45.5,
      "current_stage": "modeling"
    }
  ],
  "overall_progress": 72.75,
  "completed_count": 1,
  "total_count": 2
}
```

#### Get Simulation Results

```
GET /api/v1/results/:id
```

Returns the results of a completed simulation.

**Response:**

```json
{
  "id": "simulation-id",
  "status": "completed",
  "results": {
    "summary": {
      "total_particles": 1000,
      "evaporated_percentage": 15.2,
      "beached_percentage": 5.8,
      "maximum_distance": 12.5
    },
    "output_files": {
      "geojson": "/path/to/file.geojson",
      "json": "/path/to/file.json",
      "csv": "/path/to/file.csv"
    }
  },
  "created_at": "2023-07-01T12:00:00",
  "completed_at": "2023-07-01T13:00:00"
}
```

#### Download Result File

```
GET /api/v1/download/:id/:format
```

Downloads a result file in the specified format.

**Parameters:**
- `id`: Simulation ID
- `format`: File format (geojson, json, csv, time_series_csv)

**Response:**
The file content with appropriate content type header.

#### List All Simulations

```
GET /api/v1/simulations
```

Returns a list of all simulations.

**Query Parameters:**
- `status`: Filter by status (e.g., queued, running, completed, error)
- `limit`: Maximum number of simulations to return

**Response:**

```json
{
  "simulations": [
    {
      "id": "simulation-id-1",
      "status": "completed",
      "created_at": "2023-07-01T12:00:00",
      "progress": 100.0,
      "current_stage": "completed"
    },
    {
      "id": "simulation-id-2",
      "status": "running",
      "created_at": "2023-07-01T13:00:00",
      "progress": 45.5,
      "current_stage": "modeling"
    }
  ],
  "total": 2
}
```

#### Delete a Simulation

```
DELETE /api/v1/simulation/:id
```

Deletes a simulation and its results. Requires admin role.

**Response:**

```json
{
  "message": "Simulation simulation-id deleted successfully"
}
```

#### Health Check

```
GET /api/v1/health
```

Returns the health status of the API server.

**Response:**

```json
{
  "status": "ok",
  "version": "1.0.0",
  "active_simulations": 2,
  "total_simulations": 10
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages in case of errors.

**Example Error Response:**

```json
{
  "error": "Bad request",
  "message": "Missing required parameter: latitude"
}
```

## Testing

To run the API tests:

```bash
python test_api_server.py
```

## Export Formats

The API supports the following export formats:

- **GeoJSON**: For mapping visualization, includes spill origin point, trajectory lines with timestamps, and concentration heatmap polygons
- **JSON**: Raw structured results
- **CSV**: Summary statistics
- **Time Series CSV**: Temporal analysis with detailed metrics over time
