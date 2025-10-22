#!/usr/bin/env python3
"""
Example 2: Galaxy Redshift Prediction

This example demonstrates how to use the galaxy_ml package for predicting
galaxy redshifts using decision tree regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from galaxy_ml import RedshiftPredictor, DataUtils, VisualizationUtils


def create_sample_sdss_data(n_samples=1000):
    """
    Create sample SDSS-like data for demonstration.
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        np.ndarray: Structured array with SDSS-like data
    """
    # Create structured array with SDSS-like fields
    dtype = [
        ('u', 'f4'), ('g', 'f4'), ('r', 'f4'), ('i', 'f4'), ('z', 'f4'),
        ('redshift', 'f4'), ('spec_class', 'S10')
    ]
    
    data = np.zeros(n_samples, dtype=dtype)
    
    # Generate realistic magnitudes
    data['u'] = np.random.normal(20.0, 2.0, n_samples)
    data['g'] = data['u'] - np.random.normal(1.0, 0.3, n_samples)
    data['r'] = data['g'] - np.random.normal(0.5, 0.2, n_samples)
    data['i'] = data['r'] - np.random.normal(0.3, 0.1, n_samples)
    data['z'] = data['i'] - np.random.normal(0.2, 0.1, n_samples)
    
    # Generate redshifts (higher redshifts for bluer galaxies)
    u_g_color = data['u'] - data['g']
    data['redshift'] = np.random.exponential(0.1) + u_g_color * 0.05
    
    # Assign spectral classes
    galaxy_mask = np.random.random(n_samples) < 0.8
    data['spec_class'][galaxy_mask] = b'GALAXY'
    data['spec_class'][~galaxy_mask] = b'QSO'
    
    return data


def main():
    """Main function demonstrating redshift prediction capabilities."""
    
    print("=== Galaxy ML - Redshift Prediction Example ===\n")
    
    # Create sample data
    print("1. Creating Sample SDSS Data")
    print("-" * 30)
    
    sdss_data = create_sample_sdss_data(1000)
    print(f"Created {len(sdss_data)} sample galaxies")
    print(f"Data shape: {sdss_data.shape}")
    print(f"Data fields: {sdss_data.dtype.names}")
    
    # Initialize predictor
    predictor = RedshiftPredictor(max_depth=15, random_state=42)
    
    # Extract features and targets
    print("\n2. Feature Extraction")
    print("-" * 30)
    
    features, targets = predictor.extract_features_targets(sdss_data)
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Feature names: ['u-g', 'g-r', 'r-i', 'i-z']")
    print(f"Target range: {targets.min():.3f} - {targets.max():.3f}")
    
    # Train the model
    print("\n3. Model Training")
    print("-" * 30)
    
    predictor.train(features, targets)
    print("Model trained successfully!")
    
    # Make predictions
    print("\n4. Making Predictions")
    print("-" * 30)
    
    predictions = predictor.predict(features)
    error = predictor.median_absolute_error(predictions, targets)
    print(f"Median absolute error: {error:.4f}")
    
    # Cross-validation
    print("\n5. Cross-Validation")
    print("-" * 30)
    
    cv_errors = predictor.cross_validate(features, targets, k_folds=5)
    print(f"Cross-validation errors: {[f'{e:.4f}' for e in cv_errors]}")
    print(f"Mean CV error: {np.mean(cv_errors):.4f}")
    print(f"Std CV error: {np.std(cv_errors):.4f}")
    
    # Model optimization
    print("\n6. Model Optimization")
    print("-" * 30)
    
    max_depths = [5, 10, 15, 20, 25]
    opt_results = predictor.optimize_depth(features, targets, max_depths)
    
    print(f"Optimal depth: {opt_results['optimal_depth']}")
    print(f"Minimum error: {opt_results['min_error']:.4f}")
    
    # Plot optimization results
    plt.figure(figsize=(10, 6))
    plt.plot(opt_results['depths'], opt_results['train_errors'], 'o-', label='Training Error')
    plt.plot(opt_results['depths'], opt_results['test_errors'], 's-', label='Validation Error')
    plt.xlabel('Maximum Tree Depth')
    plt.ylabel('Median Absolute Error')
    plt.title('Model Optimization: Tree Depth vs Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Cross-validated predictions
    print("\n7. Cross-Validated Predictions")
    print("-" * 30)
    
    cv_predictions = predictor.cross_validate_predictions(features, targets, k_folds=5)
    cv_error = predictor.median_absolute_error(cv_predictions, targets)
    print(f"Cross-validated median error: {cv_error:.4f}")
    
    # Plot predictions vs actual
    predictor.plot_predictions(targets, cv_predictions)
    
    # Color-redshift relationship
    print("\n8. Color-Redshift Analysis")
    print("-" * 30)
    
    predictor.plot_color_redshift(sdss_data)
    
    # Galaxy vs QSO comparison
    print("\n9. Galaxy vs QSO Comparison")
    print("-" * 30)
    
    from galaxy_ml.redshift_prediction import compare_galaxy_qso_performance
    
    comparison_results = compare_galaxy_qso_performance(sdss_data)
    print(f"Galaxy error: {comparison_results['galaxy_error']:.4f}")
    print(f"QSO error: {comparison_results['qso_error']:.4f}")
    print(f"Galaxy count: {comparison_results['galaxy_count']}")
    print(f"QSO count: {comparison_results['qso_count']}")
    
    # Feature importance analysis
    print("\n10. Feature Analysis")
    print("-" * 30)
    
    # Analyze color distributions
    u_g = features[:, 0]
    g_r = features[:, 1]
    r_i = features[:, 2]
    i_z = features[:, 3]
    
    print("Color statistics:")
    print(f"u-g: mean={np.mean(u_g):.3f}, std={np.std(u_g):.3f}")
    print(f"g-r: mean={np.mean(g_r):.3f}, std={np.std(g_r):.3f}")
    print(f"r-i: mean={np.mean(r_i):.3f}, std={np.std(r_i):.3f}")
    print(f"i-z: mean={np.mean(i_z):.3f}, std={np.std(i_z):.3f}")
    
    # Create feature correlation plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].scatter(u_g, targets, alpha=0.6, s=1)
    axes[0, 0].set_xlabel('u-g Color')
    axes[0, 0].set_ylabel('Redshift')
    axes[0, 0].set_title('u-g vs Redshift')
    
    axes[0, 1].scatter(g_r, targets, alpha=0.6, s=1)
    axes[0, 1].set_xlabel('g-r Color')
    axes[0, 1].set_ylabel('Redshift')
    axes[0, 1].set_title('g-r vs Redshift')
    
    axes[1, 0].scatter(r_i, targets, alpha=0.6, s=1)
    axes[1, 0].set_xlabel('r-i Color')
    axes[1, 0].set_ylabel('Redshift')
    axes[1, 0].set_title('r-i vs Redshift')
    
    axes[1, 1].scatter(i_z, targets, alpha=0.6, s=1)
    axes[1, 1].set_xlabel('i-z Color')
    axes[1, 1].set_ylabel('Redshift')
    axes[1, 1].set_title('i-z vs Redshift')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
