// API Base URL - Change this to match your API server
const API_BASE_URL = 'http://localhost:5000/api/v1';

// Initialize map
let map;
let markers = [];
let simulationLayer = null;
let activeSimulationId = null;
let pollingInterval = null;

// Initialize the map when the page loads
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    setupEventListeners();
    fetchSimulations();
});

// Initialize Leaflet map
function initMap() {
    map = L.map('map').setView([-3.57, -80.45], 8);
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);
}

// Set up event listeners
function setupEventListeners() {
    // Form submission
    document.getElementById('simulationForm').addEventListener('submit', (e) => {
        e.preventDefault();
        startSimulation();
    });
    
    // Refresh button
    document.getElementById('refreshButton').addEventListener('click', fetchSimulations);
}

// Start a new simulation
async function startSimulation() {
    try {
        // Get form values
        const latitude = parseFloat(document.getElementById('latitude').value);
        const longitude = parseFloat(document.getElementById('longitude').value);
        const volume = parseFloat(document.getElementById('volume').value);
        const oilType = document.getElementById('oilType').value;
        const modelType = document.getElementById('modelType').value;
        const durationHours = parseInt(document.getElementById('durationHours').value);
        const timestepMinutes = parseInt(document.getElementById('timestepMinutes').value);
        const particleCount = parseInt(document.getElementById('particleCount').value);
        
        // Get selected output formats
        const outputFormats = [];
        if (document.getElementById('formatGeoJSON').checked) outputFormats.push('geojson');
        if (document.getElementById('formatJSON').checked) outputFormats.push('json');
        if (document.getElementById('formatCSV').checked) outputFormats.push('csv');
        
        // Create simulation parameters
        const params = {
            latitude,
            longitude,
            volume,
            oil_type: oilType,
            model_type: modelType,
            duration_hours: durationHours,
            timestep_minutes: timestepMinutes,
            particle_count: particleCount,
            output_formats: outputFormats
        };
        
        // Add marker to map
        addMarker(latitude, longitude, `Spill Location: ${volume} m³ of ${oilType}`);
        
        // Send API request
        const response = await fetch(`${API_BASE_URL}/simulate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to start simulation');
        }
        
        const data = await response.json();
        console.log('Simulation started:', data);
        
        // Show success message
        alert(`Simulation started successfully! ID: ${data.id}`);
        
        // Refresh simulations list
        fetchSimulations();
        
        // Start polling for this simulation
        pollSimulationStatus(data.id);
        
    } catch (error) {
        console.error('Error starting simulation:', error);
        alert(`Error: ${error.message}`);
    }
}

// Fetch all simulations
async function fetchSimulations() {
    try {
        const response = await fetch(`${API_BASE_URL}/simulations`);
        
        if (!response.ok) {
            throw new Error('Failed to fetch simulations');
        }
        
        const data = await response.json();
        console.log('Simulations:', data);
        
        // Update simulations list
        updateSimulationsList(data.simulations);
        
    } catch (error) {
        console.error('Error fetching simulations:', error);
    }
}

// Update the simulations list in the UI
function updateSimulationsList(simulations) {
    const simulationsList = document.getElementById('simulationsList');
    
    if (!simulations || simulations.length === 0) {
        simulationsList.innerHTML = '<div class="text-center text-muted py-3">No simulations found</div>';
        return;
    }
    
    simulationsList.innerHTML = '';
    
    simulations.forEach(simulation => {
        const card = document.createElement('div');
        card.className = 'card simulation-card mb-3';
        
        // Set card color based on status
        let statusBadgeClass = 'bg-secondary';
        if (simulation.status === 'completed') statusBadgeClass = 'bg-success';
        if (simulation.status === 'error') statusBadgeClass = 'bg-danger';
        if (simulation.status === 'running') statusBadgeClass = 'bg-primary';
        
        let progressHtml = '';
        if (simulation.status === 'running' || simulation.status === 'queued') {
            progressHtml = `
                <div class="progress mt-2">
                    <div class="progress-bar progress-bar-striped progress-bar-animated" 
                         role="progressbar" 
                         style="width: ${simulation.progress || 0}%" 
                         aria-valuenow="${simulation.progress || 0}" 
                         aria-valuemin="0" 
                         aria-valuemax="100">
                        ${Math.round(simulation.progress || 0)}%
                    </div>
                </div>
                <div class="text-muted small mt-1">${simulation.current_stage || 'Initializing'}</div>
            `;
        }
        
        let actionsHtml = '';
        if (simulation.status === 'completed') {
            actionsHtml = `
                <div class="mt-2">
                    <a href="#" class="btn btn-sm btn-primary view-results" data-id="${simulation.id}">View Results</a>
                    <a href="#" class="btn btn-sm btn-success view-map" data-id="${simulation.id}">Show on Map</a>
                </div>
            `;
        }
        
        card.innerHTML = `
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-start">
                    <h6 class="card-title mb-1">Simulation ${simulation.id}</h6>
                    <span class="badge ${statusBadgeClass}">${simulation.status}</span>
                </div>
                <div class="text-muted small mb-2">
                    Created: ${new Date(simulation.created_at).toLocaleString()}
                </div>
                ${progressHtml}
                ${actionsHtml}
            </div>
        `;
        
        simulationsList.appendChild(card);
    });
    
    // Add event listeners to buttons
    document.querySelectorAll('.view-results').forEach(button => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            const simId = e.target.getAttribute('data-id');
            viewSimulationResults(simId);
        });
    });
    
    document.querySelectorAll('.view-map').forEach(button => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            const simId = e.target.getAttribute('data-id');
            showSimulationOnMap(simId);
        });
    });
}

// Poll for simulation status updates
function pollSimulationStatus(simulationId) {
    // Clear any existing polling
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }
    
    // Set active simulation ID
    activeSimulationId = simulationId;
    
    // Start polling
    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/status/${simulationId}`);
            
            if (!response.ok) {
                throw new Error('Failed to fetch simulation status');
            }
            
            const data = await response.json();
            console.log('Simulation status:', data);
            
            // Update UI with status
            updateSimulationStatus(data);
            
            // If simulation is completed or has error, stop polling
            if (data.status === 'completed' || data.status === 'error') {
                clearInterval(pollingInterval);
                pollingInterval = null;
                
                // Refresh simulations list
                fetchSimulations();
                
                // If completed, show on map
                if (data.status === 'completed') {
                    showSimulationOnMap(simulationId);
                }
            }
            
        } catch (error) {
            console.error('Error polling simulation status:', error);
        }
    }, 2000); // Poll every 2 seconds
}

// Update simulation status in the UI
function updateSimulationStatus(simulation) {
    // This will be updated when we refresh the simulations list
    // But we could update the specific simulation card here for real-time updates
}

// View simulation results
async function viewSimulationResults(simulationId) {
    try {
        const response = await fetch(`${API_BASE_URL}/results/${simulationId}`);
        
        if (!response.ok) {
            throw new Error('Failed to fetch simulation results');
        }
        
        const data = await response.json();
        console.log('Simulation results:', data);
        
        // Display results in modal
        const modal = new bootstrap.Modal(document.getElementById('simulationModal'));
        const modalBody = document.getElementById('simulationDetails');
        
        // Format the results
        let resultsHtml = `
            <h5>Simulation ${data.id}</h5>
            <p><strong>Status:</strong> ${data.status}</p>
            <p><strong>Created:</strong> ${new Date(data.created_at).toLocaleString()}</p>
        `;
        
        if (data.results && data.results.summary) {
            resultsHtml += `
                <h6 class="mt-3">Summary</h6>
                <ul>
                    <li><strong>Total Particles:</strong> ${data.results.summary.total_particles || 'N/A'}</li>
                    <li><strong>Evaporated:</strong> ${data.results.summary.evaporated_percentage ? data.results.summary.evaporated_percentage.toFixed(2) + '%' : 'N/A'}</li>
                    <li><strong>Beached:</strong> ${data.results.summary.beached_percentage ? data.results.summary.beached_percentage.toFixed(2) + '%' : 'N/A'}</li>
                    <li><strong>Maximum Distance:</strong> ${data.results.summary.maximum_distance ? data.results.summary.maximum_distance.toFixed(2) + ' km' : 'N/A'}</li>
                </ul>
            `;
        }
        
        if (data.results && data.results.output_files) {
            resultsHtml += `
                <h6 class="mt-3">Download Results</h6>
                <div class="d-flex flex-wrap">
            `;
            
            for (const [format, path] of Object.entries(data.results.output_files)) {
                resultsHtml += `
                    <a href="${API_BASE_URL}/download/${data.id}/${format}" 
                       class="btn btn-sm btn-outline-primary result-link" 
                       target="_blank">
                        Download ${format.toUpperCase()}
                    </a>
                `;
            }
            
            resultsHtml += `</div>`;
        }
        
        modalBody.innerHTML = resultsHtml;
        modal.show();
        
    } catch (error) {
        console.error('Error fetching simulation results:', error);
        alert(`Error: ${error.message}`);
    }
}

// Show simulation on map
async function showSimulationOnMap(simulationId) {
    try {
        // First get the simulation results to get the output files
        const response = await fetch(`${API_BASE_URL}/results/${simulationId}`);
        
        if (!response.ok) {
            throw new Error('Failed to fetch simulation results');
        }
        
        const data = await response.json();
        console.log('Simulation results for map:', data);
        
        if (!data.results || !data.results.output_files || !data.results.output_files.geojson) {
            throw new Error('No GeoJSON data available for this simulation');
        }
        
        // Now fetch the GeoJSON data
        const geojsonResponse = await fetch(`${API_BASE_URL}/download/${simulationId}/geojson`);
        
        if (!geojsonResponse.ok) {
            throw new Error('Failed to fetch GeoJSON data');
        }
        
        const geojsonData = await geojsonResponse.json();
        console.log('GeoJSON data:', geojsonData);
        
        // Remove existing simulation layer if any
        if (simulationLayer) {
            map.removeLayer(simulationLayer);
        }
        
        // Add GeoJSON to map
        simulationLayer = L.geoJSON(geojsonData, {
            style: function(feature) {
                // Style for polygons (concentration areas)
                if (feature.geometry.type === 'Polygon' || feature.geometry.type === 'MultiPolygon') {
                    return {
                        color: '#ff7800',
                        weight: 1,
                        opacity: 0.8,
                        fillOpacity: 0.4
                    };
                }
                
                // Style for lines (trajectories)
                if (feature.geometry.type === 'LineString' || feature.geometry.type === 'MultiLineString') {
                    return {
                        color: '#0000ff',
                        weight: 2,
                        opacity: 0.7
                    };
                }
            },
            pointToLayer: function(feature, latlng) {
                // Style for points (particles)
                return L.circleMarker(latlng, {
                    radius: 3,
                    fillColor: '#ff0000',
                    color: '#000',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.8
                });
            },
            onEachFeature: function(feature, layer) {
                // Add popups with properties
                if (feature.properties) {
                    let popupContent = '<div>';
                    for (const [key, value] of Object.entries(feature.properties)) {
                        if (key !== 'style') { // Skip style property
                            popupContent += `<strong>${key}:</strong> ${value}<br>`;
                        }
                    }
                    popupContent += '</div>';
                    layer.bindPopup(popupContent);
                }
            }
        }).addTo(map);
        
        // Fit map to the GeoJSON bounds
        const bounds = simulationLayer.getBounds();
        if (bounds.isValid()) {
            map.fitBounds(bounds);
        }
        
    } catch (error) {
        console.error('Error showing simulation on map:', error);
        alert(`Error: ${error.message}`);
    }
}

// Add a marker to the map
function addMarker(lat, lng, popupText) {
    // Remove existing markers
    markers.forEach(marker => map.removeLayer(marker));
    markers = [];
    
    // Add new marker
    const marker = L.marker([lat, lng]).addTo(map);
    if (popupText) {
        marker.bindPopup(popupText).openPopup();
    }
    
    markers.push(marker);
    
    // Center map on marker
    map.setView([lat, lng], 8);
}
