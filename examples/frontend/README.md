# Oil Spill Trajectory Analysis API - Frontend Example

This is a simple web-based frontend example that demonstrates how to interact with the Oil Spill Trajectory Analysis API. It provides a user interface for starting simulations, monitoring their progress, and visualizing the results on a map.

## Features

- Form for entering simulation parameters
- Real-time simulation status monitoring
- Map visualization of simulation results using Leaflet
- Download simulation results in various formats
- List of all simulations with their status

## How to Use

1. Make sure the API server is running (using `python run_api_server.py`)
2. Open the `index.html` file in a web browser
3. Fill in the simulation parameters in the form
4. Click "Start Simulation" to begin a new simulation
5. Monitor the progress in the Simulations list
6. Once completed, click "View Results" to see detailed results
7. Click "Show on Map" to visualize the simulation on the map

## API Integration

This example demonstrates how to integrate with the Oil Spill Trajectory Analysis API using JavaScript. Key integration points include:

- Starting a simulation with POST `/api/v1/simulate`
- Checking simulation status with GET `/api/v1/status/:id`
- Retrieving results with GET `/api/v1/results/:id`
- Downloading output files with GET `/api/v1/download/:id/:format`
- Listing all simulations with GET `/api/v1/simulations`

## Customization

To use this example with a different API server:

1. Open `script.js`
2. Change the `API_BASE_URL` constant at the top of the file to point to your API server

## Technologies Used

- HTML5/CSS3
- JavaScript (ES6+)
- Bootstrap 5 for UI components
- Leaflet.js for map visualization

## Notes

This is a simple example for demonstration purposes. In a production environment, you would want to add:

- Error handling and retry logic
- User authentication
- Input validation
- Responsive design improvements
- More sophisticated map visualization options
