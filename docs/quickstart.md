# Galaxy Morphology Classification

A comprehensive machine learning package for astronomical data analysis, including galaxy morphology classification, redshift prediction, and FITS image processing.

## Quick Start

```python
from galaxy_ml import RedshiftPredictor, GalaxyClassifier

# Predict galaxy redshifts
predictor = RedshiftPredictor()
features, targets = predictor.extract_features_targets(sdss_data)
predictor.train(features, targets)
predictions = predictor.predict(features)

# Classify galaxy morphologies
classifier = GalaxyClassifier()
results = classifier.evaluate_model(galaxy_data)
print(f"Accuracy: {results['accuracy']:.4f}")
```

## Installation

```bash
pip install -e .
```

## Examples

See the `examples/` directory for comprehensive usage examples.

## Documentation

See `README.md` for full documentation.
