#!/usr/bin/env python3
"""
Example 3: Galaxy Morphology Classification

This example demonstrates how to use the galaxy_ml package for classifying
galaxy morphologies using random forest classifiers.
"""

import numpy as np
import matplotlib.pyplot as plt
from galaxy_ml import GalaxyClassifier, DataUtils, VisualizationUtils


def create_sample_galaxy_data(n_samples=500):
    """
    Create sample galaxy catalog data for demonstration.
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        np.ndarray: Structured array with galaxy data
    """
    # Create structured array with galaxy catalog fields
    dtype = [
        ('class', 'U20'),
        ('u-g', 'f4'), ('g-r', 'f4'), ('r-i', 'f4'), ('i-z', 'f4'),
        ('ecc', 'f4'),
        ('m4_u', 'f4'), ('m4_g', 'f4'), ('m4_r', 'f4'), ('m4_i', 'f4'), ('m4_z', 'f4'),
        ('petroR50_u', 'f4'), ('petroR90_u', 'f4'),
        ('petroR50_r', 'f4'), ('petroR90_r', 'f4'),
        ('petroR50_z', 'f4'), ('petroR90_z', 'f4')
    ]
    
    data = np.zeros(n_samples, dtype=dtype)
    
    # Generate galaxy classes with realistic proportions
    n_elliptical = int(n_samples * 0.4)
    n_spiral = int(n_samples * 0.5)
    n_merger = n_samples - n_elliptical - n_spiral
    
    classes = ['elliptical'] * n_elliptical + ['spiral'] * n_spiral + ['merger'] * n_merger
    np.random.shuffle(classes)
    data['class'] = classes
    
    # Generate features based on galaxy type
    for i, galaxy_type in enumerate(classes):
        if galaxy_type == 'elliptical':
            # Elliptical galaxies: redder colors, higher ellipticity, more concentrated
            data['u-g'][i] = np.random.normal(1.5, 0.3)
            data['g-r'][i] = np.random.normal(0.8, 0.2)
            data['r-i'][i] = np.random.normal(0.4, 0.1)
            data['i-z'][i] = np.random.normal(0.3, 0.1)
            data['ecc'][i] = np.random.uniform(0.3, 0.8)
            data['m4_r'][i] = np.random.normal(0.1, 0.05)
            data['petroR50_r'][i] = np.random.uniform(2.0, 4.0)
            data['petroR90_r'][i] = np.random.uniform(8.0, 12.0)
            
        elif galaxy_type == 'spiral':
            # Spiral galaxies: bluer colors, moderate ellipticity, less concentrated
            data['u-g'][i] = np.random.normal(0.8, 0.4)
            data['g-r'][i] = np.random.normal(0.5, 0.3)
            data['r-i'][i] = np.random.normal(0.2, 0.2)
            data['i-z'][i] = np.random.normal(0.1, 0.1)
            data['ecc'][i] = np.random.uniform(0.1, 0.6)
            data['m4_r'][i] = np.random.normal(0.05, 0.03)
            data['petroR50_r'][i] = np.random.uniform(3.0, 6.0)
            data['petroR90_r'][i] = np.random.uniform(10.0, 15.0)
            
        else:  # merger
            # Merger galaxies: mixed colors, high ellipticity, irregular
            data['u-g'][i] = np.random.normal(1.2, 0.5)
            data['g-r'][i] = np.random.normal(0.6, 0.4)
            data['r-i'][i] = np.random.normal(0.3, 0.3)
            data['i-z'][i] = np.random.normal(0.2, 0.2)
            data['ecc'][i] = np.random.uniform(0.4, 0.9)
            data['m4_r'][i] = np.random.normal(0.15, 0.1)
            data['petroR50_r'][i] = np.random.uniform(2.5, 5.0)
            data['petroR90_r'][i] = np.random.uniform(9.0, 14.0)
    
    # Generate other filters based on r-band
    data['m4_u'] = data['m4_r'] + np.random.normal(0, 0.02, n_samples)
    data['m4_g'] = data['m4_r'] + np.random.normal(0, 0.01, n_samples)
    data['m4_i'] = data['m4_r'] + np.random.normal(0, 0.01, n_samples)
    data['m4_z'] = data['m4_r'] + np.random.normal(0, 0.02, n_samples)
    
    data['petroR50_u'] = data['petroR50_r'] * np.random.uniform(0.8, 1.2, n_samples)
    data['petroR90_u'] = data['petroR90_r'] * np.random.uniform(0.8, 1.2, n_samples)
    data['petroR50_z'] = data['petroR50_r'] * np.random.uniform(0.8, 1.2, n_samples)
    data['petroR90_z'] = data['petroR90_r'] * np.random.uniform(0.8, 1.2, n_samples)
    
    return data


def main():
    """Main function demonstrating galaxy classification capabilities."""
    
    print("=== Galaxy ML - Galaxy Morphology Classification Example ===\n")
    
    # Create sample data
    print("1. Creating Sample Galaxy Data")
    print("-" * 30)
    
    galaxy_data = create_sample_galaxy_data(500)
    print(f"Created {len(galaxy_data)} sample galaxies")
    print(f"Data shape: {galaxy_data.shape}")
    print(f"Data fields: {galaxy_data.dtype.names}")
    
    # Analyze class distribution
    print("\n2. Class Distribution Analysis")
    print("-" * 30)
    
    from galaxy_ml.classification import analyze_class_distribution
    class_dist = analyze_class_distribution(galaxy_data)
    
    # Initialize classifier
    classifier = GalaxyClassifier(model_type='random_forest', n_estimators=50, random_state=42)
    
    # Extract features and targets
    print("\n3. Feature Extraction")
    print("-" * 30)
    
    features, targets = classifier.extract_features_targets(galaxy_data)
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Feature names: {classifier.GALAXY_FEATURE_NAMES if hasattr(classifier, 'GALAXY_FEATURE_NAMES') else '13 features'}")
    
    # Train the model
    print("\n4. Model Training")
    print("-" * 30)
    
    classifier.train(features, targets)
    print("Model trained successfully!")
    
    # Evaluate model
    print("\n5. Model Evaluation")
    print("-" * 30)
    
    results = classifier.evaluate_model(galaxy_data, cv_folds=5)
    print(f"Cross-validation accuracy: {results['accuracy']:.4f}")
    print(f"Class labels: {results['class_labels']}")
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(
        results['confusion_matrix'], 
        results['class_labels'],
        title="Galaxy Classification Confusion Matrix"
    )
    
    # Compare models
    print("\n6. Model Comparison")
    print("-" * 30)
    
    comparison_results = classifier.compare_models(galaxy_data)
    
    print("Model Comparison Results:")
    print(f"Decision Tree Accuracy: {comparison_results['decision_tree']['accuracy']:.4f}")
    print(f"Random Forest Accuracy: {comparison_results['random_forest']['accuracy']:.4f}")
    
    # Plot comparison confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Decision Tree confusion matrix
    dt_cm = comparison_results['decision_tree']['confusion_matrix']
    dt_labels = comparison_results['decision_tree']['class_labels']
    
    im1 = axes[0].imshow(dt_cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title('Decision Tree')
    axes[0].set_xticks(range(len(dt_labels)))
    axes[0].set_yticks(range(len(dt_labels)))
    axes[0].set_xticklabels(dt_labels, rotation=45)
    axes[0].set_yticklabels(dt_labels)
    axes[0].set_ylabel('True Class')
    axes[0].set_xlabel('Predicted Class')
    
    # Add text annotations
    thresh = dt_cm.max() / 2.
    for i, j in np.ndindex(dt_cm.shape):
        axes[0].text(j, i, f"{dt_cm[i, j]}",
                    ha="center", va="center",
                    color="white" if dt_cm[i, j] > thresh else "black")
    
    # Random Forest confusion matrix
    rf_cm = comparison_results['random_forest']['confusion_matrix']
    rf_labels = comparison_results['random_forest']['class_labels']
    
    im2 = axes[1].imshow(rf_cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1].set_title('Random Forest')
    axes[1].set_xticks(range(len(rf_labels)))
    axes[1].set_yticks(range(len(rf_labels)))
    axes[1].set_xticklabels(rf_labels, rotation=45)
    axes[1].set_yticklabels(rf_labels)
    axes[1].set_ylabel('True Class')
    axes[1].set_xlabel('Predicted Class')
    
    # Add text annotations
    thresh = rf_cm.max() / 2.
    for i, j in np.ndindex(rf_cm.shape):
        axes[1].text(j, i, f"{rf_cm[i, j]}",
                    ha="center", va="center",
                    color="white" if rf_cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.show()
    
    # Feature importance
    print("\n7. Feature Importance Analysis")
    print("-" * 30)
    
    classifier.plot_feature_importance()
    
    # Parameter optimization
    print("\n8. Parameter Optimization")
    print("-" * 30)
    
    opt_results = classifier.optimize_parameters(galaxy_data)
    print(f"Best parameters: {opt_results['best_params']}")
    print(f"Best accuracy: {opt_results['best_accuracy']:.4f}")
    
    # Plot optimization results
    results_df = opt_results['all_results']
    n_est_values = sorted(set([r['n_estimators'] for r in results_df]))
    max_depth_values = sorted(set([r['max_depth'] for r in results_df if r['max_depth'] is not None]))
    
    # Create heatmap of accuracy vs parameters
    accuracy_matrix = np.zeros((len(max_depth_values), len(n_est_values)))
    
    for result in results_df:
        if result['max_depth'] is not None:
            depth_idx = max_depth_values.index(result['max_depth'])
            est_idx = n_est_values.index(result['n_estimators'])
            accuracy_matrix[depth_idx, est_idx] = result['accuracy']
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(accuracy_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Accuracy')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Max Depth')
    plt.title('Parameter Optimization Results')
    plt.xticks(range(len(n_est_values)), n_est_values)
    plt.yticks(range(len(max_depth_values)), max_depth_values)
    plt.show()
    
    # Feature visualization
    print("\n9. Feature Visualization")
    print("-" * 30)
    
    from galaxy_ml.classification import visualize_galaxy_features
    visualize_galaxy_features(galaxy_data)
    
    # Detailed analysis
    print("\n10. Detailed Analysis")
    print("-" * 30)
    
    # Analyze per-class performance
    from galaxy_ml.utils import ValidationUtils
    
    predictions = results['predictions']
    actual = results['targets']
    
    ValidationUtils.print_classification_report(actual, predictions, results['class_labels'])
    
    # Create feature correlation plot
    feature_names = ['u-g', 'g-r', 'r-i', 'i-z', 'ecc', 'm4_r', 'conc_r']
    feature_data = features[:, [0, 1, 2, 3, 4, 7, 11]]  # Select key features
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Color-color diagram
    scatter = axes[0, 0].scatter(feature_data[:, 0], feature_data[:, 1], c=actual, alpha=0.6)
    axes[0, 0].set_xlabel('u-g')
    axes[0, 0].set_ylabel('g-r')
    axes[0, 0].set_title('Color-Color Diagram')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # Ellipticity vs Concentration
    scatter = axes[0, 1].scatter(feature_data[:, 4], feature_data[:, 6], c=actual, alpha=0.6)
    axes[0, 1].set_xlabel('Ellipticity')
    axes[0, 1].set_ylabel('Concentration (r-band)')
    axes[0, 1].set_title('Ellipticity vs Concentration')
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # Fourth moment distribution
    for galaxy_type in results['class_labels']:
        mask = actual == galaxy_type
        axes[1, 0].hist(feature_data[mask, 5], alpha=0.6, label=galaxy_type, bins=20)
    axes[1, 0].set_xlabel('Fourth Moment (r-band)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Fourth Moment Distribution')
    axes[1, 0].legend()
    
    # Prediction accuracy by class
    class_accuracies = []
    for galaxy_type in results['class_labels']:
        mask = actual == galaxy_type
        if np.sum(mask) > 0:
            accuracy = np.mean(predictions[mask] == actual[mask])
            class_accuracies.append(accuracy)
        else:
            class_accuracies.append(0)
    
    bars = axes[1, 1].bar(results['class_labels'], class_accuracies)
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy by Galaxy Class')
    axes[1, 1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracies):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
