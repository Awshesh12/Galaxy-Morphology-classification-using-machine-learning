"""
Galaxy Morphology Classification Package

A comprehensive machine learning package for astronomical data analysis,
including galaxy morphology classification, redshift prediction, and
FITS image processing.

Author: Awshesh
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Awshesh"

from .data_processing import FITSProcessor, ImageStacker
from .redshift_prediction import RedshiftPredictor
from .classification import GalaxyClassifier
from .utils import DataUtils, VisualizationUtils

__all__ = [
    "FITSProcessor",
    "ImageStacker", 
    "RedshiftPredictor",
    "GalaxyClassifier",
    "DataUtils",
    "VisualizationUtils"
]
