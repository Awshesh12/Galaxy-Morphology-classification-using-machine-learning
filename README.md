# Galaxy Morphology Classification

A comprehensive machine learning package for astronomical data analysis, including galaxy morphology classification, redshift prediction, and FITS image processing.

## Overview

This package provides tools for analyzing astronomical data using machine learning techniques. It includes functionality for:

- **FITS Image Processing**: Loading, stacking, and processing astronomical images
- **Redshift Prediction**: Using decision tree regression to predict galaxy redshifts from photometric colors
- **Galaxy Morphology Classification**: Classifying galaxies into different morphological types using random forest classifiers
- **Data Visualization**: Creating publication-ready plots and visualizations

## Features

### 🔭 Astronomical Data Processing
- FITS file loading and processing
- Image stacking (mean and median) with memory optimization
- Bright source detection and analysis
- Performance benchmarking tools

### 🌌 Galaxy Redshift Prediction
- Decision tree regression for redshift estimation
- Cross-validation and model optimization
- Feature extraction from SDSS photometric data
- Visualization of color-redshift relationships

### 🌀 Galaxy Morphology Classification
- Random forest classification for galaxy types
- Support for elliptical, spiral, and merger classifications
- Feature importance analysis
- Confusion matrix visualization

### 📊 Data Analysis Tools
- Comprehensive utility functions
- Performance analysis and benchmarking
- Data validation and preprocessing
- Statistical analysis tools

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/awshesh/galaxy-morphology-classification.git
cd galaxy-morphology-classification

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

### Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or install manually
pip install pytest black flake8 mypy
```

## Quick Start

### 1. FITS Image Processing

```python
from galaxy_ml import FITSProcessor, ImageStacker

# Load and process FITS files
processor = FITSProcessor()
data = processor.load_fits('path/to/image.fits')

# Find brightest pixel
brightest = processor.find_brightest_pixel(data)

# Stack multiple images
stacker = ImageStacker()
stacked_image, exec_time, memory = stacker.median_stack(['image1.fits', 'image2.fits'])
```

### 2. Redshift Prediction

```python
from galaxy_ml import RedshiftPredictor
import numpy as np

# Load SDSS data
data = np.load('sdss_galaxy_colors.npy')

# Create predictor
predictor = RedshiftPredictor(max_depth=19)

# Extract features and targets
features, targets = predictor.extract_features_targets(data)

# Train the model
predictor.train(features, targets)

# Make predictions
predictions = predictor.predict(features)

# Evaluate with cross-validation
errors = predictor.cross_validate(features, targets, k_folds=10)
print(f"Mean error: {np.mean(errors):.4f}")
```

### 3. Galaxy Classification

```python
from galaxy_ml import GalaxyClassifier

# Load galaxy catalog data
data = np.load('galaxy_catalogue.npy')

# Create classifier
classifier = GalaxyClassifier(model_type='random_forest', n_estimators=50)

# Evaluate model
results = classifier.evaluate_model(data, cv_folds=10)
print(f"Accuracy: {results['accuracy']:.4f}")

# Plot confusion matrix
classifier.plot_confusion_matrix(
    results['confusion_matrix'], 
    results['class_labels']
)
```

## Project Structure

```
galaxy-morphology-classification/
├── src/
│   └── galaxy_ml/
│       ├── __init__.py
│       ├── config.py
│       ├── data_processing/
│       │   └── __init__.py
│       ├── redshift_prediction/
│       │   └── __init__.py
│       ├── classification/
│       │   └── __init__.py
│       └── utils/
│           └── __init__.py
├── tests/
├── examples/
├── docs/
├── data/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## API Documentation

### Data Processing Module

#### `FITSProcessor`
- `load_fits(file_path)`: Load FITS file data
- `find_brightest_pixel(data)`: Find coordinates of brightest pixel
- `visualize_fits(data, title, cmap, save_path)`: Visualize FITS data

#### `ImageStacker`
- `mean_stack(file_paths)`: Calculate mean stack of images
- `median_stack(file_paths)`: Calculate median stack with timing info
- `batch_process_directory(directory_path, method)`: Process all FITS files in directory

### Redshift Prediction Module

#### `RedshiftPredictor`
- `extract_features_targets(data)`: Extract color features from SDSS data
- `train(features, targets)`: Train the prediction model
- `predict(features)`: Predict redshifts
- `cross_validate(features, targets, k_folds)`: Perform k-fold cross-validation
- `optimize_depth(features, targets, max_depths)`: Optimize tree depth

### Classification Module

#### `GalaxyClassifier`
- `extract_features_targets(data)`: Extract morphological features
- `train(features, targets)`: Train the classification model
- `predict(features)`: Predict galaxy classes
- `evaluate_model(data, cv_folds)`: Evaluate model performance
- `plot_confusion_matrix(cm, labels)`: Visualize confusion matrix
- `plot_feature_importance(feature_names)`: Show feature importance

### Utility Modules

#### `DataUtils`
- `load_numpy_data(file_path)`: Load numpy arrays
- `remove_outliers(data, method)`: Remove outliers from data
- `normalize_data(data, method)`: Normalize data

#### `VisualizationUtils`
- `create_histogram(data, bins, title)`: Create histogram plots
- `create_scatter_plot(x, y, c, title)`: Create scatter plots
- `set_plot_style(style)`: Set matplotlib style

#### `PerformanceUtils`
- `time_function(func, *args)`: Time function execution
- `benchmark_functions(functions, data)`: Benchmark multiple functions

## Examples

### Complete Workflow Example

```python
import numpy as np
from galaxy_ml import RedshiftPredictor, GalaxyClassifier, FITSProcessor

# 1. Process FITS images
processor = FITSProcessor()
data = processor.load_fits('galaxy_image.fits')
processor.visualize_fits(data, title="Galaxy Image")

# 2. Predict redshifts
predictor = RedshiftPredictor(max_depth=19)
sdss_data = np.load('sdss_galaxy_colors.npy')
features, targets = predictor.extract_features_targets(sdss_data)

# Cross-validate the model
errors = predictor.cross_validate(features, targets)
print(f"Redshift prediction error: {np.mean(errors):.4f}")

# 3. Classify galaxy morphologies
classifier = GalaxyClassifier(model_type='random_forest')
galaxy_data = np.load('galaxy_catalogue.npy')
results = classifier.evaluate_model(galaxy_data)
print(f"Classification accuracy: {results['accuracy']:.4f}")
```

### Performance Analysis

```python
from galaxy_ml.utils import PerformanceUtils, VisualizationUtils
import numpy as np

# Benchmark different statistical functions
data = np.random.rand(100000)
functions = [np.mean, np.median, np.std]

results = PerformanceUtils.benchmark_functions(functions, data)
PerformanceUtils.print_performance_summary(results)

# Create visualizations
VisualizationUtils.create_histogram(data, bins=50, title="Data Distribution")
```

### Running the Examples

```bash
# Run individual examples
python examples/example_1_fits_processing.py
python examples/example_2_redshift_prediction.py
python examples/example_3_galaxy_classification.py
python examples/example_4_complete_workflow.py
python examples/example_5_advanced_analysis.py

# Or run all examples
for example in examples/example_*.py; do
    echo "Running $example..."
    python "$example"
done
```

## Configuration

The package includes a configuration file (`config.py`) with default parameters:

- Model parameters (max_depth, n_estimators, random_state)
- Cross-validation settings (cv_folds, train_fraction)
- Visualization settings (figure_size, dpi, colormap)
- File paths and extensions

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/galaxy_ml

# Run specific test file
pytest tests/test_data_processing.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Sloan Digital Sky Survey (SDSS) for providing astronomical data
- Scikit-learn team for machine learning tools
- Astropy project for astronomical data processing
- Galaxy Zoo project for citizen science inspiration

## Contact

- **Author**: Awshesh
- **Email**: awshesh@example.com
- **GitHub**: [@awshesh](https://github.com/awshesh)

## Changelog

### Version 1.0.0
- Initial release
- FITS image processing capabilities
- Redshift prediction using decision trees
- Galaxy morphology classification using random forests
- Comprehensive utility functions
- Full documentation and examples

## Roadmap

- [ ] Support for additional astronomical surveys
- [ ] Deep learning models for galaxy classification
- [ ] Interactive visualization tools
- [ ] Cloud deployment support
- [ ] Additional feature extraction methods
- [ ] Real-time data processing capabilities
