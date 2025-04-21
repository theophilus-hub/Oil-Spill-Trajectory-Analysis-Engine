#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for export functionality.

This script tests the export functionality of the Oil Spill Trajectory Analysis Engine
by loading a sample simulation result and exporting it in various formats.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

from trajectory_core import export

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
OUTPUT_DIR = "./export_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_sample_results(filename):
    """Load sample simulation results from a JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading sample results: {e}")
        return None

def test_all_export_formats(results, base_name="test_export"):
    """Test all export formats with the given results."""
    logger.info("Testing all export formats...")
    
    # Export all formats
    output_files = export.export_all_formats(
        results,
        output_dir=OUTPUT_DIR,
        filename_base=base_name
    )
    
    # Log the output files
    for format_type, filepath in output_files.items():
        file_size = Path(filepath).stat().st_size / 1024  # Size in KB
        logger.info(f"Exported {format_type}: {filepath} ({file_size:.2f} KB)")
    
    return output_files

def test_mapping_export(results, base_name="test_mapping"):
    """Test mapping-specific export formats."""
    logger.info("Testing mapping export formats...")
    
    # Export mapping formats
    output_files = export.export_for_mapping(
        results,
        output_dir=OUTPUT_DIR,
        filename_base=base_name
    )
    
    # Log the output files
    for format_type, filepath in output_files.items():
        file_size = Path(filepath).stat().st_size / 1024  # Size in KB
        logger.info(f"Exported {format_type}: {filepath} ({file_size:.2f} KB)")
    
    return output_files

def test_time_series_export(results, base_name="test_time_series"):
    """Test time series export."""
    logger.info("Testing time series export...")
    
    # Export time series
    filename = f"{base_name}.csv"
    filepath = export.export_to_time_series(
        results,
        output_dir=OUTPUT_DIR,
        filename=filename
    )
    
    if filepath:
        file_size = Path(filepath).stat().st_size / 1024  # Size in KB
        logger.info(f"Exported time series: {filepath} ({file_size:.2f} KB)")
    else:
        logger.warning("Time series export failed or returned empty path")
    
    return filepath

def validate_exports(output_files):
    """Validate that all exported files exist and have content."""
    logger.info("Validating exported files...")
    
    all_valid = True
    for format_type, filepath in output_files.items():
        path = Path(filepath)
        
        if not path.exists():
            logger.error(f"{format_type} export file does not exist: {filepath}")
            all_valid = False
            continue
            
        if path.stat().st_size == 0:
            logger.error(f"{format_type} export file is empty: {filepath}")
            all_valid = False
            continue
            
        logger.info(f"{format_type} export is valid: {filepath}")
    
    return all_valid

def main():
    """Main function to run the export tests."""
    logger.info("Starting export functionality test")
    
    # Find the most recent simulation result file
    result_files = list(Path("./results").glob("*.json"))
    if not result_files:
        logger.error("No simulation result files found in ./results directory")
        return False
    
    # Sort by modification time (newest first)
    result_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_result_file = result_files[0]
    
    logger.info(f"Using latest simulation result: {latest_result_file}")
    
    # Load the sample results
    results = load_sample_results(latest_result_file)
    if not results:
        logger.error("Failed to load sample results")
        return False
    
    # Test timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Run the export tests
    all_exports = test_all_export_formats(results, f"all_formats_{timestamp}")
    mapping_exports = test_mapping_export(results, f"mapping_{timestamp}")
    time_series_file = test_time_series_export(results, f"time_series_{timestamp}")
    
    # Validate all exports
    all_valid = validate_exports(all_exports)
    mapping_valid = validate_exports(mapping_exports)
    time_series_valid = time_series_file and Path(time_series_file).exists() and Path(time_series_file).stat().st_size > 0
    
    # Overall test result
    if all_valid and mapping_valid and time_series_valid:
        logger.info("All export tests passed successfully!")
        return True
    else:
        logger.error("Some export tests failed. Check the logs for details.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
