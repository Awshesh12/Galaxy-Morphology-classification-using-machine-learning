#!/usr/bin/env python3
"""
Galaxy Morphology Classification Package

A comprehensive machine learning package for astronomical data analysis.

Author: Awshesh
Version: 1.0.0
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from galaxy_ml import (
    FITSProcessor, ImageStacker, RedshiftPredictor, GalaxyClassifier,
    DataUtils, VisualizationUtils, PerformanceUtils
)

def main():
    """Main entry point for the galaxy_ml package."""
    print("Galaxy Morphology Classification Package")
    print("Version 1.0.0")
    print("Author: Awshesh")
    print()
    print("Available modules:")
    print("  - FITSProcessor: FITS file processing")
    print("  - ImageStacker: Image stacking operations")
    print("  - RedshiftPredictor: Galaxy redshift prediction")
    print("  - GalaxyClassifier: Galaxy morphology classification")
    print("  - DataUtils: Data processing utilities")
    print("  - VisualizationUtils: Visualization utilities")
    print("  - PerformanceUtils: Performance analysis utilities")
    print()
    print("For examples, see the examples/ directory")
    print("For documentation, see README.md")

if __name__ == "__main__":
    main()
