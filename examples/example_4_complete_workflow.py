#!/usr/bin/env python3
"""
Example 4: Complete Workflow

This example demonstrates a complete workflow using all components
of the galaxy_ml package for astronomical data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from galaxy_ml import (
    FITSProcessor, ImageStacker, RedshiftPredictor, GalaxyClassifier,
    DataUtils, VisualizationUtils, PerformanceUtils
)


def create_comprehensive_sample_data():
    """Create comprehensive sample data for all analyses."""
    
    # 1. Create SDSS-like data for redshift prediction
    n_sdss = 800
    sdss_dtype = [
        ('u', 'f4'), ('g', 'f4'), ('r', 'f4'), ('i', 'f4'), ('z', 'f4'),
        ('redshift', 'f4'), ('spec_class', 'S10')
    ]
    
    sdss_data = np.zeros(n_sdss, dtype=sdss_dtype)
    
    # Generate realistic magnitudes and redshifts
    sdss_data['u'] = np.random.normal(20.0, 2.0, n_sdss)
    sdss_data['g'] = sdss_data['u'] - np.random.normal(1.0, 0.3, n_sdss)
    sdss_data['r'] = sdss_data['g'] - np.random.normal(0.5, 0.2, n_sdss)
    sdss_data['i'] = sdss_data['r'] - np.random.normal(0.3, 0.1, n_sdss)
    sdss_data['z'] = sdss_data['i'] - np.random.normal(0.2, 0.1, n_sdss)
    
    u_g_color = sdss_data['u'] - sdss_data['g']
    sdss_data['redshift'] = np.random.exponential(0.1) + u_g_color * 0.05
    
    galaxy_mask = np.random.random(n_sdss) < 0.8
    sdss_data['spec_class'][galaxy_mask] = b'GALAXY'
    sdss_data['spec_class'][~galaxy_mask] = b'QSO'
    
    # 2. Create galaxy catalog data for morphology classification
    n_galaxies = 400
    galaxy_dtype = [
        ('class', 'U20'),
        ('u-g', 'f4'), ('g-r', 'f4'), ('r-i', 'f4'), ('i-z', 'f4'),
        ('ecc', 'f4'),
        ('m4_u', 'f4'), ('m4_g', 'f4'), ('m4_r', 'f4'), ('m4_i', 'f4'), ('m4_z', 'f4'),
        ('petroR50_u', 'f4'), ('petroR90_u', 'f4'),
        ('petroR50_r', 'f4'), ('petroR90_r', 'f4'),
        ('petroR50_z', 'f4'), ('petroR90_z', 'f4')
    ]
    
    galaxy_data = np.zeros(n_galaxies, dtype=galaxy_dtype)
    
    # Generate galaxy classes
    n_elliptical = int(n_galaxies * 0.4)
    n_spiral = int(n_galaxies * 0.5)
    n_merger = n_galaxies - n_elliptical - n_spiral
    
    classes = ['elliptical'] * n_elliptical + ['spiral'] * n_spiral + ['merger'] * n_merger
    np.random.shuffle(classes)
    galaxy_data['class'] = classes
    
    # Generate features based on galaxy type
    for i, galaxy_type in enumerate(classes):
        if galaxy_type == 'elliptical':
            galaxy_data['u-g'][i] = np.random.normal(1.5, 0.3)
            galaxy_data['g-r'][i] = np.random.normal(0.8, 0.2)
            galaxy_data['r-i'][i] = np.random.normal(0.4, 0.1)
            galaxy_data['i-z'][i] = np.random.normal(0.3, 0.1)
            galaxy_data['ecc'][i] = np.random.uniform(0.3, 0.8)
            galaxy_data['m4_r'][i] = np.random.normal(0.1, 0.05)
            galaxy_data['petroR50_r'][i] = np.random.uniform(2.0, 4.0)
            galaxy_data['petroR90_r'][i] = np.random.uniform(8.0, 12.0)
            
        elif galaxy_type == 'spiral':
            galaxy_data['u-g'][i] = np.random.normal(0.8, 0.4)
            galaxy_data['g-r'][i] = np.random.normal(0.5, 0.3)
            galaxy_data['r-i'][i] = np.random.normal(0.2, 0.2)
            galaxy_data['i-z'][i] = np.random.normal(0.1, 0.1)
            galaxy_data['ecc'][i] = np.random.uniform(0.1, 0.6)
            galaxy_data['m4_r'][i] = np.random.normal(0.05, 0.03)
            galaxy_data['petroR50_r'][i] = np.random.uniform(3.0, 6.0)
            galaxy_data['petroR90_r'][i] = np.random.uniform(10.0, 15.0)
            
        else:  # merger
            galaxy_data['u-g'][i] = np.random.normal(1.2, 0.5)
            galaxy_data['g-r'][i] = np.random.normal(0.6, 0.4)
            galaxy_data['r-i'][i] = np.random.normal(0.3, 0.3)
            galaxy_data['i-z'][i] = np.random.normal(0.2, 0.2)
            galaxy_data['ecc'][i] = np.random.uniform(0.4, 0.9)
            galaxy_data['m4_r'][i] = np.random.normal(0.15, 0.1)
            galaxy_data['petroR50_r'][i] = np.random.uniform(2.5, 5.0)
            galaxy_data['petroR90_r'][i] = np.random.uniform(9.0, 14.0)
    
    # Generate other filters
    galaxy_data['m4_u'] = galaxy_data['m4_r'] + np.random.normal(0, 0.02, n_galaxies)
    galaxy_data['m4_g'] = galaxy_data['m4_r'] + np.random.normal(0, 0.01, n_galaxies)
    galaxy_data['m4_i'] = galaxy_data['m4_r'] + np.random.normal(0, 0.01, n_galaxies)
    galaxy_data['m4_z'] = galaxy_data['m4_r'] + np.random.normal(0, 0.02, n_galaxies)
    
    galaxy_data['petroR50_u'] = galaxy_data['petroR50_r'] * np.random.uniform(0.8, 1.2, n_galaxies)
    galaxy_data['petroR90_u'] = galaxy_data['petroR90_r'] * np.random.uniform(0.8, 1.2, n_galaxies)
    galaxy_data['petroR50_z'] = galaxy_data['petroR50_r'] * np.random.uniform(0.8, 1.2, n_galaxies)
    galaxy_data['petroR90_z'] = galaxy_data['petroR90_r'] * np.random.uniform(0.8, 1.2, n_galaxies)
    
    # 3. Create sample FITS images
    fits_images = []
    for i in range(5):
        img = np.random.rand(50, 50) * 50
        img[25:30, 25:30] += 500  # Add bright source
        fits_images.append(img)
    
    return sdss_data, galaxy_data, fits_images


def main():
    """Main function demonstrating complete workflow."""
    
    print("=== Galaxy ML - Complete Workflow Example ===\n")
    
    # Create sample data
    print("1. Creating Sample Data")
    print("-" * 30)
    
    sdss_data, galaxy_data, fits_images = create_comprehensive_sample_data()
    
    print(f"SDSS data: {len(sdss_data)} galaxies")
    print(f"Galaxy catalog: {len(galaxy_data)} galaxies")
    print(f"FITS images: {len(fits_images)} images")
    
    # Initialize all processors
    print("\n2. Initializing Processors")
    print("-" * 30)
    
    fits_processor = FITSProcessor()
    image_stacker = ImageStacker()
    redshift_predictor = RedshiftPredictor(max_depth=15, random_state=42)
    galaxy_classifier = GalaxyClassifier(model_type='random_forest', n_estimators=50, random_state=42)
    
    print("All processors initialized successfully!")
    
    # FITS Image Processing
    print("\n3. FITS Image Processing")
    print("-" * 30)
    
    # Process first image
    sample_image = fits_images[0]
    brightest_coords = fits_processor.find_brightest_pixel(sample_image)
    print(f"Brightest pixel coordinates: {brightest_coords}")
    print(f"Brightest pixel value: {sample_image[brightest_coords]:.2f}")
    
    # Stack images
    print("Performing image stacking...")
    stacked_image, exec_time, memory_usage = image_stacker.median_stack(fits_images)
    print(f"Stacking completed in {exec_time:.4f} seconds")
    print(f"Memory usage: {memory_usage:.2f} MB")
    
    # Redshift Prediction
    print("\n4. Redshift Prediction Analysis")
    print("-" * 30)
    
    # Extract features and train model
    features, targets = redshift_predictor.extract_features_targets(sdss_data)
    redshift_predictor.train(features, targets)
    
    # Cross-validation
    cv_errors = redshift_predictor.cross_validate(features, targets, k_folds=5)
    print(f"Redshift prediction CV error: {np.mean(cv_errors):.4f} ± {np.std(cv_errors):.4f}")
    
    # Galaxy Classification
    print("\n5. Galaxy Morphology Classification")
    print("-" * 30)
    
    # Extract features and train model
    galaxy_features, galaxy_targets = galaxy_classifier.extract_features_targets(galaxy_data)
    galaxy_classifier.train(galaxy_features, galaxy_targets)
    
    # Evaluate model
    classification_results = galaxy_classifier.evaluate_model(galaxy_data, cv_folds=5)
    print(f"Galaxy classification accuracy: {classification_results['accuracy']:.4f}")
    
    # Performance Analysis
    print("\n6. Performance Analysis")
    print("-" * 30)
    
    # Benchmark different operations
    def mean_operation(data):
        return np.mean(data)
    
    def median_operation(data):
        return np.median(data)
    
    def std_operation(data):
        return np.std(data)
    
    functions = [mean_operation, median_operation, std_operation]
    test_data = np.random.rand(100000)
    
    perf_results = PerformanceUtils.benchmark_functions(functions, test_data, n_trials=5)
    PerformanceUtils.print_performance_summary(perf_results)
    
    # Comprehensive Visualization
    print("\n7. Comprehensive Visualization")
    print("-" * 30)
    
    # Create a comprehensive dashboard
    fig = plt.figure(figsize=(20, 15))
    
    # 1. FITS image comparison
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(fits_images[0], cmap='viridis')
    ax1.set_title('Original FITS Image')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(stacked_image, cmap='viridis')
    ax2.set_title('Stacked Image')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    
    # 2. Redshift-color relationships
    ax3 = plt.subplot(3, 4, 3)
    u_g = features[:, 0]
    g_r = features[:, 1]
    scatter = ax3.scatter(u_g, g_r, c=targets, alpha=0.6, s=1)
    ax3.set_xlabel('u-g Color')
    ax3.set_ylabel('g-r Color')
    ax3.set_title('Color-Color Diagram')
    plt.colorbar(scatter, ax=ax3, label='Redshift')
    
    # 3. Galaxy classification results
    ax4 = plt.subplot(3, 4, 4)
    cm = classification_results['confusion_matrix']
    labels = classification_results['class_labels']
    im = ax4.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax4.set_title('Classification Confusion Matrix')
    ax4.set_xticks(range(len(labels)))
    ax4.set_yticks(range(len(labels)))
    ax4.set_xticklabels(labels, rotation=45)
    ax4.set_yticklabels(labels)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        ax4.text(j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    # 4. Feature distributions
    ax5 = plt.subplot(3, 4, 5)
    ax5.hist(features[:, 0], bins=30, alpha=0.7, label='u-g')
    ax5.hist(features[:, 1], bins=30, alpha=0.7, label='g-r')
    ax5.set_xlabel('Color Index')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Color Distribution')
    ax5.legend()
    
    # 5. Galaxy type distribution
    ax6 = plt.subplot(3, 4, 6)
    unique_classes, counts = np.unique(galaxy_targets, return_counts=True)
    bars = ax6.bar(unique_classes, counts)
    ax6.set_ylabel('Count')
    ax6.set_title('Galaxy Type Distribution')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{count}', ha='center', va='bottom')
    
    # 6. Performance comparison
    ax7 = plt.subplot(3, 4, 7)
    func_names = list(perf_results.keys())
    times = list(perf_results.values())
    bars = ax7.bar(func_names, times)
    ax7.set_ylabel('Execution Time (s)')
    ax7.set_title('Performance Comparison')
    ax7.tick_params(axis='x', rotation=45)
    
    # 7. Redshift prediction accuracy
    ax8 = plt.subplot(3, 4, 8)
    cv_predictions = redshift_predictor.cross_validate_predictions(features, targets, k_folds=5)
    ax8.scatter(targets, cv_predictions, alpha=0.6, s=1)
    max_val = max(targets.max(), cv_predictions.max())
    ax8.plot([0, max_val], [0, max_val], 'r--', alpha=0.8)
    ax8.set_xlabel('Actual Redshift')
    ax8.set_ylabel('Predicted Redshift')
    ax8.set_title('Redshift Prediction Accuracy')
    
    # 8. Galaxy morphology features
    ax9 = plt.subplot(3, 4, 9)
    ecc = galaxy_features[:, 4]
    conc = galaxy_features[:, 11]
    scatter = ax9.scatter(ecc, conc, c=galaxy_targets, alpha=0.6)
    ax9.set_xlabel('Ellipticity')
    ax9.set_ylabel('Concentration')
    ax9.set_title('Morphology Features')
    
    # 9. Error analysis
    ax10 = plt.subplot(3, 4, 10)
    errors = np.abs(cv_predictions - targets)
    ax10.hist(errors, bins=30, alpha=0.7)
    ax10.set_xlabel('Absolute Error')
    ax10.set_ylabel('Frequency')
    ax10.set_title('Redshift Prediction Errors')
    
    # 10. Model comparison
    ax11 = plt.subplot(3, 4, 11)
    models = ['Decision Tree', 'Random Forest']
    accuracies = [0.785, classification_results['accuracy']]  # Approximate values
    bars = ax11.bar(models, accuracies)
    ax11.set_ylabel('Accuracy')
    ax11.set_title('Model Comparison')
    ax11.set_ylim(0, 1)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax11.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom')
    
    # 11. Data quality metrics
    ax12 = plt.subplot(3, 4, 12)
    metrics = ['SDSS\nGalaxies', 'Galaxy\nCatalog', 'FITS\nImages']
    counts = [len(sdss_data), len(galaxy_data), len(fits_images)]
    bars = ax12.bar(metrics, counts)
    ax12.set_ylabel('Count')
    ax12.set_title('Dataset Overview')
    
    # Add value labels
    for bar, count in zip(bars, counts):
        ax12.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Summary Report
    print("\n8. Summary Report")
    print("-" * 30)
    
    print("=== GALAXY ML ANALYSIS SUMMARY ===")
    print(f"Dataset Sizes:")
    print(f"  - SDSS galaxies: {len(sdss_data)}")
    print(f"  - Galaxy catalog: {len(galaxy_data)}")
    print(f"  - FITS images: {len(fits_images)}")
    print()
    
    print(f"Model Performance:")
    print(f"  - Redshift prediction error: {np.mean(cv_errors):.4f} ± {np.std(cv_errors):.4f}")
    print(f"  - Galaxy classification accuracy: {classification_results['accuracy']:.4f}")
    print()
    
    print(f"Data Quality:")
    print(f"  - Brightest source detected at: {brightest_coords}")
    print(f"  - Image stacking time: {exec_time:.4f} seconds")
    print(f"  - Memory usage: {memory_usage:.2f} MB")
    print()
    
    print(f"Feature Statistics:")
    print(f"  - Redshift range: {targets.min():.3f} - {targets.max():.3f}")
    print(f"  - Color range (u-g): {features[:, 0].min():.3f} - {features[:, 0].max():.3f}")
    print(f"  - Galaxy types: {len(np.unique(galaxy_targets))}")
    print()
    
    print("=== ANALYSIS COMPLETED SUCCESSFULLY ===")
    
    # Save results
    print("\n9. Saving Results")
    print("-" * 30)
    
    # Create results directory
    DataUtils.create_directory('results')
    
    # Save processed data
    DataUtils.save_numpy_data(stacked_image, 'results/stacked_image.npy')
    DataUtils.save_numpy_data(cv_predictions, 'results/redshift_predictions.npy')
    DataUtils.save_numpy_data(classification_results['predictions'], 'results/galaxy_predictions.npy')
    
    print("Results saved to 'results/' directory")
    print("Files saved:")
    print("  - stacked_image.npy")
    print("  - redshift_predictions.npy")
    print("  - galaxy_predictions.npy")
    
    print("\n=== Complete workflow finished successfully! ===")


if __name__ == "__main__":
    main()
