"""
Export module for the Oil Spill Trajectory Analysis Engine.

This module handles formatting and exporting simulation results:
- Export to GeoJSON (for mapping)
- Export to JSON (raw structured results)
- Export to CSV (summary/statistics)
"""

import os
import json
import csv
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import geojson
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

class ResultExporter:
    """Class for exporting simulation results in various formats."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the result exporter.
        
        Args:
            output_dir: Directory to save output files
                If None, uses the default from config
        """
        if output_dir is None:
            self.output_dir = config.OUTPUT_CONFIG['output_directory']
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def export_results(self, 
                      results: Dict[str, Any], 
                      format_type: str = 'all',
                      filename_base: Optional[str] = None) -> Dict[str, str]:
        """
        Export simulation results in the specified format.
        
        Args:
            results: Simulation results from model.run_simulation()
            format_type: Format to export ('geojson', 'json', 'csv', or 'all')
            filename_base: Base filename without extension
                If None, generates a timestamped filename
                
        Returns:
            Dictionary mapping format types to output filenames
        """
        # Generate default filename if not provided
        if filename_base is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename_base = f"oil_spill_simulation_{timestamp}"
        
        # Dictionary to store output filenames
        output_files = {}
        
        # Export in requested format(s)
        if format_type in ['geojson', 'all']:
            geojson_file = self._export_geojson(results, f"{filename_base}.geojson")
            output_files['geojson'] = geojson_file
            
        if format_type in ['json', 'all']:
            json_file = self._export_json(results, f"{filename_base}.json")
            output_files['json'] = json_file
            
        if format_type in ['csv', 'all']:
            csv_file = self._export_csv(results, f"{filename_base}.csv")
            output_files['csv'] = csv_file
        
        return output_files
    
    def _export_geojson(self, results: Dict[str, Any], filename: str) -> str:
        """
        Export results as GeoJSON for mapping.
        
        Args:
            results: Simulation results
            filename: Output filename
            
        Returns:
            Full path to the output file
        """
        # Create GeoJSON structure
        feature_collection = {
            "type": "FeatureCollection",
            "features": [],
            "properties": {
                "metadata": {
                    "simulation_start": results.get('metadata', {}).get('start_time', ''),
                    "simulation_end": results.get('metadata', {}).get('end_time', ''),
                    "timestep_minutes": results.get('metadata', {}).get('timestep_minutes', 0),
                    "duration_hours": results.get('metadata', {}).get('parameters', {}).get('duration_hours', 0),
                    "particle_count": results.get('metadata', {}).get('particle_count', 0),
                    "oil_type": results.get('metadata', {}).get('parameters', {}).get('oil_type', 'unknown'),
                    "spill_volume": results.get('metadata', {}).get('parameters', {}).get('spill_volume', 0),
                }
            }
        }
        
        # Add spill origin point
        spill_location = None
        
        # Try to find spill location in different possible locations in the results
        if 'spill_origin' in results and 'latitude' in results['spill_origin'] and 'longitude' in results['spill_origin']:
            spill_location = results['spill_origin']
        elif 'metadata' in results and 'spill_location' in results['metadata']:
            spill_location = results['metadata']['spill_location']
        elif 'timesteps' in results and results['timesteps'] and 'particles' in results['timesteps'][0]:
            # Use the first particle's initial position as an approximation
            first_particle = results['timesteps'][0]['particles'][0]
            if 'latitude' in first_particle and 'longitude' in first_particle:
                spill_location = {
                    'latitude': first_particle['latitude'],
                    'longitude': first_particle['longitude']
                }
        
        if spill_location:
            origin_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [spill_location.get('longitude', 0), spill_location.get('latitude', 0)]
                },
                "properties": {
                    "type": "origin",
                    "description": "Spill origin point",
                    "time": results.get('metadata', {}).get('start_time', ''),
                    "volume": results.get('metadata', {}).get('parameters', {}).get('spill_volume', 0),
                    "oil_type": results.get('metadata', {}).get('parameters', {}).get('oil_type', 'unknown'),
                }
            }
            feature_collection["features"].append(origin_feature)
        
        # Process particle data
        timesteps = results.get('timesteps', [])
        if not timesteps and 'particle_history' in results:
            timesteps = results['particle_history']
            
        # Get the last timestep for final positions
        if timesteps:
            final_state = timesteps[-1]
            
            # Create a feature for each active particle
            for particle in final_state.get('particles', []):
                # Skip inactive particles
                if 'status' in particle and particle.get('status') != 'active':
                    continue
                    
                # Create GeoJSON feature
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            particle.get('longitude', 0),
                            particle.get('latitude', 0)
                        ]
                    },
                    "properties": {
                        "id": particle.get('id', 0),
                        "mass": particle.get('mass', 0),
                        "age": particle.get('age', 0),
                        "status": particle.get('status', 'active'),
                        "depth": particle.get('depth', 0),
                        "time": final_state.get('time', ''),
                        "marker-color": "#ff0000",  # Red for final positions
                        "marker-size": "medium"
                    }
                }
                
                feature_collection["features"].append(feature)
            
            # Create LineString features for particle trajectories
            self._add_trajectory_features(feature_collection, timesteps)
            
            # Add concentration heatmap data if available
            self._add_concentration_features(feature_collection, results)
        
        # Write GeoJSON to file
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(feature_collection, f, indent=2)
        
        logger.info(f"Exported GeoJSON to {output_path}")
        return output_path
    
    def _add_trajectory_features(self, feature_collection: Dict[str, Any], timesteps: List[Dict[str, Any]]) -> None:
        """
        Add trajectory LineString features to the GeoJSON feature collection.
        
        Args:
            feature_collection: GeoJSON feature collection to add to
            timesteps: List of particle states at each timestep
        """
        # Group particles by ID
        particle_trajectories = {}
        particle_timestamps = {}
        
        for state in timesteps:
            time = state.get('time', '')
            for particle in state.get('particles', []):
                # Skip particles without position data
                if 'latitude' not in particle or 'longitude' not in particle:
                    continue
                    
                # Get particle ID or generate one if not present
                particle_id = particle.get('id', None)
                if particle_id is None:
                    # If no ID, use index in the particles list as a fallback
                    particle_index = state.get('particles', []).index(particle)
                    particle_id = f"particle_{particle_index}"
                
                if particle_id not in particle_trajectories:
                    particle_trajectories[particle_id] = []
                    particle_timestamps[particle_id] = []
                
                # Add point to trajectory if it's a new position
                new_point = [
                    particle.get('longitude', 0),
                    particle.get('latitude', 0)
                ]
                
                # Only add if it's a new position (avoid duplicates)
                if not particle_trajectories[particle_id] or new_point != particle_trajectories[particle_id][-1]:
                    particle_trajectories[particle_id].append(new_point)
                    particle_timestamps[particle_id].append(time)
        
        # Create a LineString feature for each trajectory
        for particle_id, coordinates in particle_trajectories.items():
            # Skip if less than 2 points
            if len(coordinates) < 2:
                continue
                
            # Create LineString feature
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                },
                "properties": {
                    "id": particle_id,
                    "type": "trajectory",
                    "timestamps": particle_timestamps[particle_id],
                    "stroke": "#3388ff",  # Blue for trajectories
                    "stroke-width": 2,
                    "stroke-opacity": 0.7
                }
            }
            
            feature_collection["features"].append(feature)
    
    def _add_concentration_features(self, feature_collection: Dict[str, Any], results: Dict[str, Any]) -> None:
        """
        Add oil concentration features to the GeoJSON feature collection.
        
        Args:
            feature_collection: GeoJSON feature collection to add to
            results: Simulation results
        """
        # Check if concentration data is available
        concentration_grid = results.get('concentration_grid', {})
        if not concentration_grid or 'data' not in concentration_grid:
            return
        
        # Get grid data and metadata
        grid_data = concentration_grid.get('data', [])
        grid_bounds = concentration_grid.get('bounds', {})
        grid_resolution = concentration_grid.get('resolution', {})
        
        if not grid_data or not grid_bounds or not grid_resolution:
            return
        
        # Extract bounds
        min_lat = grid_bounds.get('min_lat', 0)
        max_lat = grid_bounds.get('max_lat', 0)
        min_lon = grid_bounds.get('min_lon', 0)
        max_lon = grid_bounds.get('max_lon', 0)
        
        # Extract resolution
        lat_res = grid_resolution.get('lat', 0.01)
        lon_res = grid_resolution.get('lon', 0.01)
        
        # Create a polygon feature for each grid cell with non-zero concentration
        for i, row in enumerate(grid_data):
            for j, concentration in enumerate(row):
                if concentration <= 0:
                    continue
                
                # Calculate cell bounds
                cell_min_lat = min_lat + i * lat_res
                cell_max_lat = cell_min_lat + lat_res
                cell_min_lon = min_lon + j * lon_res
                cell_max_lon = cell_min_lon + lon_res
                
                # Create polygon coordinates (5 points, closing the loop)
                coordinates = [
                    [cell_min_lon, cell_min_lat],
                    [cell_max_lon, cell_min_lat],
                    [cell_max_lon, cell_max_lat],
                    [cell_min_lon, cell_max_lat],
                    [cell_min_lon, cell_min_lat]  # Close the loop
                ]
                
                # Normalize concentration to 0-1 range for color mapping
                max_concentration = concentration_grid.get('max_value', 1)
                normalized_concentration = min(concentration / max_concentration, 1)
                
                # Create color based on concentration (red with varying opacity)
                opacity = 0.2 + (normalized_concentration * 0.6)  # 0.2 to 0.8 opacity range
                
                # Create polygon feature
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coordinates]
                    },
                    "properties": {
                        "type": "concentration",
                        "concentration": concentration,
                        "fill": "#ff0000",  # Red fill
                        "fill-opacity": opacity,
                        "stroke": "#ff0000",
                        "stroke-width": 0.5,
                        "stroke-opacity": 0.2
                    }
                }
                
                feature_collection["features"].append(feature)

    def _export_json(self, results: Dict[str, Any], filename: str) -> str:
        """
        Export raw results as JSON.
        
        Args:
            results: Simulation results
            filename: Output filename
            
        Returns:
            Full path to the output file
        """
        # Write JSON to file
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Exported JSON to {output_path}")
        return output_path
    
    def _export_csv(self, results: Dict[str, Any], filename: str) -> str:
        """
        Export summary statistics as CSV.
        
        Args:
            results: Simulation results
            filename: Output filename
            
        Returns:
            Full path to the output file
        """
        # Create summary statistics
        summary = self._calculate_summary_statistics(results)
        
        # Write CSV to file
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Statistic', 'Value', 'Unit'])
            
            # Write statistics
            for key, value in summary.items():
                # Add units based on the statistic type
                unit = self._get_unit_for_statistic(key)
                writer.writerow([key, value, unit])
        
        # Also export time series data if available
        if results.get('particle_history'):
            time_series_path = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}_time_series.csv")
            self._export_time_series_csv(results, time_series_path)
        
        logger.info(f"Exported CSV to {output_path}")
        return output_path
    
    def _export_time_series_csv(self, results: Dict[str, Any], filename: str) -> str:
        """
        Export time series data from simulation results to a CSV file.
        
        Args:
            results: Dictionary containing simulation results
            filename: Name of the output file
            
        Returns:
            Path to the exported CSV file
        
        Raises:
            ValueError: If results don't contain time series data
        """
        logger = logging.getLogger(__name__)
        logger.info("Exporting time series data to CSV...")
        
        # Use specified output directory or default from config
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"oil_spill_time_series_{timestamp}.csv"
        
        # Full path to output file
        output_path = os.path.join(self.output_dir, filename)
        
        # Extract time series data from results
        particle_history = results.get('particle_history', [])
        
        if not particle_history:
            # Try to extract from nested structure
            particle_history = results.get('results', {}).get('particle_history', [])
        
        if not particle_history:
            logger.warning("No time series data found in simulation results")
            # Create an empty CSV file with headers
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestep', 'time', 'active_particles', 'beached_particles', 
                               'evaporated_particles', 'max_distance_km', 'affected_area_km2'])
            return output_path
        
        # Calculate timestep duration in minutes
        timestep_minutes = results.get('params', {}).get('timestep_minutes', 30)
        if not timestep_minutes:
            # Try to extract from nested structure
            timestep_minutes = results.get('parameters', {}).get('simulation_params', {}).get('timestep_minutes', 30)
        
        # Extract start time if available
        start_time_str = results.get('metadata', {}).get('start_time', '')
        if not start_time_str:
            # Try to extract from nested structure
            start_time_str = results.get('parameters', {}).get('start_time', '')
        
        try:
            start_time = datetime.fromisoformat(start_time_str)
        except (ValueError, TypeError):
            # If start time is not available or invalid, use None
            start_time = None
        
        # Write time series data to CSV
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header row
            writer.writerow(['timestep', 'time', 'active_particles', 'beached_particles', 
                           'evaporated_particles', 'max_distance_km', 'affected_area_km2'])
            
            # Write data rows
            for entry in particle_history:
                timestep = entry.get('timestep', 0)
                
                # Calculate time for this timestep
                if start_time:
                    from datetime import timedelta
                    time_str = (start_time + timedelta(minutes=timestep * timestep_minutes)).isoformat()
                else:
                    time_str = f"T+{timestep * timestep_minutes} minutes"
                
                # Extract particle counts
                active = entry.get('active', 0)
                beached = entry.get('beached', 0)
                evaporated = entry.get('evaporated', 0)
                
                # Extract or calculate additional metrics
                max_distance = entry.get('max_distance_km', 0)
                affected_area = entry.get('affected_area_km2', 0)
                
                # Write row
                writer.writerow([timestep, time_str, active, beached, evaporated, max_distance, affected_area])
        
        logger.info(f"Time series data exported to {output_path}")
        return output_path
    
    def _get_unit_for_statistic(self, stat_name: str) -> str:
        """
        Return the appropriate unit for a given statistic.
        
        Args:
            stat_name: Name of the statistic
            
        Returns:
            Unit string
        """
        # Map statistic names to units
        unit_map = {
            'affected_area_km2': 'km²',
            'bounding_box_km2': 'km²',
            'timestep_minutes': 'minutes',
            'duration_hours': 'hours',
            'initial_particle_count': 'particles',
            'final_active_count': 'particles',
            'final_beached_count': 'particles',
            'final_active_percent': '%',
            'final_beached_percent': '%',
        }
        
        # Return the unit if found, otherwise empty string
        return unit_map.get(stat_name, '')
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees).
        
        Args:
            lat1, lon1: Coordinates of point 1
            lat2, lon2: Coordinates of point 2
            
        Returns:
            Distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r
    
    def _calculate_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate summary statistics from simulation results.
        
        Args:
            results: Simulation results
            
        Returns:
            Dictionary of summary statistics
        """
        summary = {}
        
        # Basic simulation parameters
        summary['start_time'] = results.get('metadata', {}).get('start_time', '')
        summary['end_time'] = results.get('metadata', {}).get('end_time', '')
        summary['timestep_minutes'] = results.get('metadata', {}).get('timestep_minutes', 0)
        summary['duration_hours'] = results.get('metadata', {}).get('parameters', {}).get('duration_hours', 0)
        
        # Process particle data
        timesteps = results.get('timesteps', [])
        if not timesteps and 'particle_history' in results:
            timesteps = results['particle_history']
        
        if timesteps:
            # Get initial and final states
            initial_state = timesteps[0]
            final_state = timesteps[-1]
            
            # Count particles by status
            initial_count = len(initial_state.get('particles', []))
            
            # Count final particles by status
            status_counts = {}
            for particle in final_state.get('particles', []):
                status = particle.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Add counts to summary
            summary['initial_particle_count'] = initial_count
            for status, count in status_counts.items():
                summary[f'final_{status}_count'] = count
                summary[f'final_{status}_percent'] = round(100 * count / initial_count, 1) if initial_count > 0 else 0
            
            # Calculate affected area (convex hull area)
            # This would require additional geometric calculations
            # For now, we'll just estimate based on the bounding box
            if final_state.get('particles'):
                lats = [p.get('latitude', 0) for p in final_state.get('particles', []) 
                       if p.get('status', '') == 'active']
                lons = [p.get('longitude', 0) for p in final_state.get('particles', [])
                       if p.get('status', '') == 'active']
                
                if lats and lons:
                    # Calculate bounding box
                    min_lat, max_lat = min(lats), max(lats)
                    min_lon, max_lon = min(lons), max(lons)
                    
                    # Approximate area in square kilometers
                    # 1 degree latitude ≈ 111 km
                    # 1 degree longitude ≈ 111 km * cos(latitude)
                    avg_lat = (min_lat + max_lat) / 2
                    lat_distance = (max_lat - min_lat) * 111
                    lon_distance = (max_lon - min_lon) * 111 * abs(np.cos(np.radians(avg_lat)))
                    
                    # Area of bounding box
                    area_km2 = lat_distance * lon_distance
                    
                    # Adjust for actual coverage (particles don't fill the whole box)
                    # This is a very rough approximation
                    coverage_factor = 0.5
                    adjusted_area = area_km2 * coverage_factor
                    
                    summary['affected_area_km2'] = round(adjusted_area, 2)
                    summary['bounding_box_km2'] = round(area_km2, 2)
        
        return summary


def export_to_geojson(results: Dict[str, Any], 
                     output_dir: Optional[str] = None,
                     filename: Optional[str] = None) -> str:
    """
    Export simulation results to GeoJSON.
    
    Args:
        results: Simulation results from model.run_simulation()
        output_dir: Directory to save output file (default from config if None)
        filename: Output filename (timestamped default if None)
        
    Returns:
        Path to the output file
    """
    exporter = ResultExporter(output_dir)
    
    if filename is None:
        output_files = exporter.export_results(results, 'geojson')
        return output_files['geojson']
    else:
        return exporter._export_geojson(results, filename)


def export_to_json(results: Dict[str, Any],
                  output_dir: Optional[str] = None,
                  filename: Optional[str] = None) -> str:
    """
    Export simulation results to JSON.
    
    Args:
        results: Simulation results from model.run_simulation()
        output_dir: Directory to save output file (default from config if None)
        filename: Output filename (timestamped default if None)
        
    Returns:
        Path to the output file
    """
    exporter = ResultExporter(output_dir)
    
    if filename is None:
        output_files = exporter.export_results(results, 'json')
        return output_files['json']
    else:
        return exporter._export_json(results, filename)


def export_to_csv(results: Dict[str, Any],
                 output_dir: Optional[str] = None,
                 filename: Optional[str] = None) -> str:
    """
    Export simulation results to CSV.
    
    Args:
        results: Simulation results from model.run_simulation()
        output_dir: Directory to save output file (default from config if None)
        filename: Output filename (timestamped default if None)
        
    Returns:
        Path to the output file
    """
    exporter = ResultExporter(output_dir)
    
    if filename is None:
        output_files = exporter.export_results(results, 'csv')
        return output_files['csv']
    else:
        return exporter._export_csv(results, filename)


def export_all_formats(results: Dict[str, Any],
                      output_dir: Optional[str] = None,
                      filename_base: Optional[str] = None) -> Dict[str, str]:
    """
    Export simulation results to all supported formats.
    
    Args:
        results: Simulation results from model.run_simulation()
        output_dir: Directory to save output files (default from config if None)
        filename_base: Base filename without extension (timestamped default if None)
        
    Returns:
        Dictionary mapping format types to output filenames
    """
    exporter = ResultExporter(output_dir)
    return exporter.export_results(results, 'all', filename_base)


def export_to_time_series(results: Dict[str, Any],
                      output_dir: Optional[str] = None,
                      filename: Optional[str] = None) -> str:
    """
    Export simulation results to a time series CSV file.
    
    Args:
        results: Simulation results from model.run_simulation()
        output_dir: Directory to save output file (default from config if None)
        filename: Output filename (timestamped default if None)
        
    Returns:
        Path to the output file
    """
    exporter = ResultExporter(output_dir)
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"oil_spill_time_series_{timestamp}.csv"
        
    return exporter._export_time_series_csv(results, filename)


def export_for_mapping(results: Dict[str, Any],
                     output_dir: Optional[str] = None,
                     filename_base: Optional[str] = None,
                     include_time_series: bool = True) -> Dict[str, str]:
    """
    Export simulation results in formats optimized for mapping visualization.
    
    This function exports both GeoJSON for map visualization and supporting
    CSV files for time series data that can be used in interactive maps.
    
    Args:
        results: Simulation results from model.run_simulation()
        output_dir: Directory to save output files (default from config if None)
        filename_base: Base filename without extension (timestamped default if None)
        include_time_series: Whether to include time series CSV export
        
    Returns:
        Dictionary mapping format types to output filenames
    """
    exporter = ResultExporter(output_dir)
    
    # Generate default filename if not provided
    if filename_base is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_base = f"oil_spill_mapping_{timestamp}"
    
    # Dictionary to store output filenames
    output_files = {}
    
    # Export GeoJSON for mapping
    geojson_file = exporter._export_geojson(results, f"{filename_base}.geojson")
    output_files['geojson'] = geojson_file
    
    # Export time series data if requested
    if include_time_series:
        time_series_file = exporter._export_time_series_csv(results, f"{filename_base}_time_series.csv")
        output_files['time_series'] = time_series_file
    
    return output_files


def get_export_filename(base_name: str, 
                       format_type: str,
                       timestamp: bool = True,
                       output_dir: Optional[str] = None) -> str:
    """
    Generate a standardized filename for export files.
    
    Args:
        base_name: Base name for the file (e.g., 'oil_spill')
        format_type: Type of export format ('json', 'geojson', 'csv')
        timestamp: Whether to include a timestamp in the filename
        output_dir: Output directory (uses default from config if None)
        
    Returns:
        Full path to the output file
    """
    # Use default output directory if not specified
    if output_dir is None:
        output_dir = config.OUTPUT_CONFIG['output_directory']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add timestamp if requested
    if timestamp:
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{base_name}_{time_str}.{format_type}"
    else:
        filename = f"{base_name}.{format_type}"
    
    return os.path.join(output_dir, filename)


def export_to_time_series_csv(results: Dict[str, Any], filename: Optional[str] = None, output_dir: Optional[str] = None) -> str:
    """
    Export time series data from simulation results to a CSV file.
    
    Args:
        results: Dictionary containing simulation results
        filename: Name of the output file (default: auto-generated)
        output_dir: Directory to save the file (default: from config)
        
    Returns:
        Path to the exported CSV file
    
    Raises:
        ValueError: If results don't contain time series data
    """
    logger = logging.getLogger(__name__)
    logger.info("Exporting time series data to CSV...")
    
    # Use specified output directory or default from config
    if output_dir is None:
        output_dir = config.OUTPUT_CONFIG['output_directory']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"oil_spill_time_series_{timestamp}.csv"
    
    # Full path to output file
    output_path = os.path.join(output_dir, filename)
    
    # Extract time series data from results
    particle_history = results.get('particle_history', [])
    
    if not particle_history:
        # Try to extract from nested structure
        particle_history = results.get('results', {}).get('particle_history', [])
    
    if not particle_history:
        logger.warning("No time series data found in simulation results")
        # Create an empty CSV file with headers
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestep', 'time', 'active_particles', 'beached_particles', 
                           'evaporated_particles', 'max_distance_km', 'affected_area_km2'])
        return output_path
    
    # Calculate timestep duration in minutes
    timestep_minutes = results.get('params', {}).get('timestep_minutes', 30)
    if not timestep_minutes:
        # Try to extract from nested structure
        timestep_minutes = results.get('parameters', {}).get('simulation_params', {}).get('timestep_minutes', 30)
    
    # Extract start time if available
    start_time_str = results.get('metadata', {}).get('start_time', '')
    if not start_time_str:
        # Try to extract from nested structure
        start_time_str = results.get('parameters', {}).get('start_time', '')
    
    try:
        start_time = datetime.fromisoformat(start_time_str)
    except (ValueError, TypeError):
        # If start time is not available or invalid, use None
        start_time = None
    
    # Write time series data to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row
        writer.writerow(['timestep', 'time', 'active_particles', 'beached_particles', 
                       'evaporated_particles', 'max_distance_km', 'affected_area_km2'])
        
        # Write data rows
        for entry in particle_history:
            timestep = entry.get('timestep', 0)
            
            # Calculate time for this timestep
            if start_time:
                from datetime import timedelta
                time_str = (start_time + timedelta(minutes=timestep * timestep_minutes)).isoformat()
            else:
                time_str = f"T+{timestep * timestep_minutes} minutes"
            
            # Extract particle counts
            active = entry.get('active', 0)
            beached = entry.get('beached', 0)
            evaporated = entry.get('evaporated', 0)
            
            # Extract or calculate additional metrics
            max_distance = entry.get('max_distance_km', 0)
            affected_area = entry.get('affected_area_km2', 0)
            
            # Write row
            writer.writerow([timestep, time_str, active, beached, evaporated, max_distance, affected_area])
    
    logger.info(f"Time series data exported to {output_path}")
    return output_path
