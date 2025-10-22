# Galaxy Morphology Classification Configuration

# Data paths
DATA_DIR = "data"
FITS_DIR = "data/fits"
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"

# Model parameters
DEFAULT_MAX_DEPTH = 19
DEFAULT_N_ESTIMATORS = 50
DEFAULT_RANDOM_STATE = 42

# Cross-validation
DEFAULT_CV_FOLDS = 10
DEFAULT_TRAIN_FRACTION = 0.7

# Visualization settings
DEFAULT_FIGURE_SIZE = (10, 8)
DEFAULT_DPI = 300
DEFAULT_COLORMAP = 'viridis'

# Performance settings
DEFAULT_N_TRIALS = 10
DEFAULT_BENCHMARK_SIZE = 100000

# File extensions
FITS_EXTENSION = '.fits'
NUMPY_EXTENSION = '.npy'
PLOT_EXTENSIONS = ['.png', '.jpg', '.pdf']

# Feature names for galaxy classification
GALAXY_FEATURE_NAMES = [
    'u-g', 'g-r', 'r-i', 'i-z', 'ecc',
    'm4_u', 'm4_g', 'm4_r', 'm4_i', 'm4_z',
    'conc_u', 'conc_r', 'conc_z'
]

# Color names for redshift prediction
REDSHIFT_COLOR_NAMES = ['u-g', 'g-r', 'r-i', 'i-z']

# Galaxy class labels
GALAXY_CLASSES = ['elliptical', 'spiral', 'merger']

# SDSS filter names
SDSS_FILTERS = ['u', 'g', 'r', 'i', 'z']

# Performance thresholds
MIN_ACCURACY_THRESHOLD = 0.7
MAX_ERROR_THRESHOLD = 0.1
