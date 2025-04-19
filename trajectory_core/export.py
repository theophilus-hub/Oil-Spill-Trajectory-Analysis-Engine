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
from typing import Dict, Any, List, Optional
from datetime import datetime

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
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        
        # Process particle history
        particle_history = results.get('particle_history', [])
        
        # Get the last timestep for final positions
        if particle_history:
            final_state = particle_history[-1]
            
            # Create a feature for each active particle
            for particle in final_state.get('particles', []):
                # Skip inactive particles
                if particle.get('status') != 'active':
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
                        "id": particle.get('id'),
                        "mass": particle.get('mass'),
                        "age": particle.get('age'),
                        "status": particle.get('status')
                    }
                }
                
                geojson["features"].append(feature)
            
            # Also create a LineString feature for each particle's trajectory
            # Group particles by ID
            particle_trajectories = {}
            
            for state in particle_history:
                for particle in state.get('particles', []):
                    particle_id = particle.get('id')
                    
                    if particle_id not in particle_trajectories:
                        particle_trajectories[particle_id] = []
                    
                    # Add point to trajectory
                    particle_trajectories[particle_id].append([
                        particle.get('longitude', 0),
                        particle.get('latitude', 0)
                    ])
            
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
                        "type": "trajectory"
                    }
                }
                
                geojson["features"].append(feature)
        
        # Write GeoJSON to file
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        logger.info(f"Exported GeoJSON to {output_path}")
        return output_path
    
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
            writer.writerow(['Statistic', 'Value'])
            
            # Write statistics
            for key, value in summary.items():
                writer.writerow([key, value])
        
        logger.info(f"Exported CSV to {output_path}")
        return output_path
    
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
        summary['start_time'] = results.get('start_time', '')
        summary['end_time'] = results.get('end_time', '')
        summary['timestep_minutes'] = results.get('params', {}).get('timestep_minutes', 0)
        summary['duration_hours'] = results.get('params', {}).get('duration_hours', 0)
        
        # Process particle history
        particle_history = results.get('particle_history', [])
        
        if particle_history:
            # Get initial and final states
            initial_state = particle_history[0]
            final_state = particle_history[-1]
            
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
                       if p.get('status') == 'active']
                lons = [p.get('longitude', 0) for p in final_state.get('particles', [])
                       if p.get('status') == 'active']
                
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
