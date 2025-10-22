"""
Galaxy Morphology Classification Module

This module provides functionality for classifying galaxy morphologies using
machine learning techniques, specifically random forest classifiers.
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from typing import Tuple, List, Dict, Any


class GalaxyClassifier:
    """
    A class for classifying galaxy morphologies using machine learning.
    
    This class implements both decision tree and random forest classifiers
    for galaxy morphology classification based on SDSS data.
    """
    
    def __init__(self, model_type: str = 'random_forest', n_estimators: int = 50, 
                 max_depth: int = None, random_state: int = 42):
        """
        Initialize the galaxy classifier.
        
        Args:
            model_type (str): Type of model ('decision_tree' or 'random_forest')
            n_estimators (int): Number of trees for random forest
            max_depth (int): Maximum depth of trees
            random_state (int): Random state for reproducibility
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.is_trained = False
        
        # Initialize model
        if model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                random_state=random_state
            )
        else:
            raise ValueError("model_type must be 'decision_tree' or 'random_forest'")
    
    def extract_features_targets(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and targets from galaxy catalog data.
        
        Args:
            data (np.ndarray): Structured array containing galaxy data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (features, targets)
        """
        targets = data['class']
        
        # Initialize features array
        features = np.empty(shape=(len(data), 13))
        
        # Color features
        features[:, 0] = data['u-g']
        features[:, 1] = data['g-r']
        features[:, 2] = data['r-i']
        features[:, 3] = data['i-z']
        
        # Ellipticity
        features[:, 4] = data['ecc']
        
        # Fourth moment features
        features[:, 5] = data['m4_u']
        features[:, 6] = data['m4_g']
        features[:, 7] = data['m4_r']
        features[:, 8] = data['m4_i']
        features[:, 9] = data['m4_z']
        
        # Concentration features (Petrosian radius ratios)
        features[:, 10] = data['petroR50_u'] / data['petroR90_u']
        features[:, 11] = data['petroR50_r'] / data['petroR90_r']
        features[:, 12] = data['petroR50_z'] / data['petroR90_z']
        
        return features, targets
    
    def split_data(self, data: np.ndarray, train_fraction: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            data (np.ndarray): Input data
            train_fraction (float): Fraction of data to use for training
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (training_data, testing_data)
        """
        # Shuffle the data
        np.random.seed(self.random_state)
        shuffled_indices = np.random.permutation(len(data))
        shuffled_data = data[shuffled_indices]
        
        # Split the data
        split_index = int(len(shuffled_data) * train_fraction)
        training_data = shuffled_data[:split_index]
        testing_data = shuffled_data[split_index:]
        
        return training_data, testing_data
    
    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Train the classification model.
        
        Args:
            features (np.ndarray): Training features
            targets (np.ndarray): Training targets
        """
        self.model.fit(features, targets)
        self.is_trained = True
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict galaxy classes for given features.
        
        Args:
            features (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted classes
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for given features.
        
        Args:
            features (np.ndarray): Input features
            
        Returns:
            np.ndarray: Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(features)
    
    def evaluate_model(self, data: np.ndarray, cv_folds: int = 10) -> Dict[str, Any]:
        """
        Evaluate the model using cross-validation.
        
        Args:
            data (np.ndarray): Input data
            cv_folds (int): Number of cross-validation folds
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        features, targets = self.extract_features_targets(data)
        
        # Get cross-validated predictions
        predictions = cross_val_predict(self.model, features, targets, cv=cv_folds)
        
        # Calculate accuracy
        accuracy = accuracy_score(targets, predictions)
        
        # Calculate confusion matrix
        class_labels = list(set(targets))
        cm = confusion_matrix(targets, predictions, labels=class_labels)
        
        # Generate classification report
        report = classification_report(targets, predictions, target_names=class_labels, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'class_labels': class_labels,
            'predictions': predictions,
            'targets': targets,
            'classification_report': report
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_labels: List[str], 
                            normalize: bool = False, title: str = 'Confusion Matrix',
                            save_path: str = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm (np.ndarray): Confusion matrix
            class_labels (List[str]): Class labels
            normalize (bool): Whether to normalize the matrix
            title (str): Plot title
            save_path (str): Optional path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title += ' (Normalized)'
        
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        
        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels, rotation=45)
        plt.yticks(tick_marks, class_labels)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, f"{cm[i, j]:.2f}",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str] = None, 
                               save_path: str = None) -> None:
        """
        Plot feature importance for random forest models.
        
        Args:
            feature_names (List[str]): Names of features
            save_path (str): Optional path to save the plot
        """
        if self.model_type != 'random_forest':
            print("Feature importance is only available for random forest models")
            return
        
        if not self.is_trained:
            print("Model must be trained before plotting feature importance")
            return
        
        if feature_names is None:
            feature_names = [
                'u-g', 'g-r', 'r-i', 'i-z', 'ecc',
                'm4_u', 'm4_g', 'm4_r', 'm4_i', 'm4_z',
                'conc_u', 'conc_r', 'conc_z'
            ]
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self, data: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Compare decision tree and random forest performance.
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            Dict[str, Dict[str, Any]]: Comparison results
        """
        results = {}
        
        # Test decision tree
        dt_classifier = GalaxyClassifier(model_type='decision_tree', random_state=self.random_state)
        dt_results = dt_classifier.evaluate_model(data)
        results['decision_tree'] = dt_results
        
        # Test random forest
        rf_classifier = GalaxyClassifier(model_type='random_forest', random_state=self.random_state)
        rf_results = rf_classifier.evaluate_model(data)
        results['random_forest'] = rf_results
        
        return results
    
    def optimize_parameters(self, data: np.ndarray, 
                          n_estimators_list: List[int] = None,
                          max_depths: List[int] = None) -> Dict[str, Any]:
        """
        Optimize model parameters using grid search.
        
        Args:
            data (np.ndarray): Input data
            n_estimators_list (List[int]): List of n_estimators to test
            max_depths (List[int]): List of max_depths to test
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        if n_estimators_list is None:
            n_estimators_list = [10, 25, 50, 100]
        if max_depths is None:
            max_depths = [5, 10, 15, 20, None]
        
        features, targets = self.extract_features_targets(data)
        
        best_accuracy = 0
        best_params = {}
        results = []
        
        for n_est in n_estimators_list:
            for max_depth in max_depths:
                # Create model with current parameters
                model = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_depth,
                    random_state=self.random_state
                )
                
                # Evaluate using cross-validation
                predictions = cross_val_predict(model, features, targets, cv=5)
                accuracy = accuracy_score(targets, predictions)
                
                results.append({
                    'n_estimators': n_est,
                    'max_depth': max_depth,
                    'accuracy': accuracy
                })
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {'n_estimators': n_est, 'max_depth': max_depth}
        
        return {
            'best_params': best_params,
            'best_accuracy': best_accuracy,
            'all_results': results
        }


def visualize_galaxy_features(data: np.ndarray, save_path: str = None) -> None:
    """
    Create visualizations of galaxy features.
    
    Args:
        data (np.ndarray): Galaxy catalog data
        save_path (str): Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Color-color diagram
    axes[0, 0].scatter(data['u-g'], data['g-r'], c=data['class'], alpha=0.6)
    axes[0, 0].set_xlabel('u-g')
    axes[0, 0].set_ylabel('g-r')
    axes[0, 0].set_title('Color-Color Diagram')
    
    # Ellipticity distribution
    for galaxy_type in set(data['class']):
        mask = data['class'] == galaxy_type
        axes[0, 1].hist(data['ecc'][mask], alpha=0.6, label=galaxy_type, bins=20)
    axes[0, 1].set_xlabel('Ellipticity')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Ellipticity Distribution')
    axes[0, 1].legend()
    
    # Concentration vs Ellipticity
    scatter = axes[1, 0].scatter(data['petroR50_r']/data['petroR90_r'], data['ecc'], 
                               c=data['class'], alpha=0.6)
    axes[1, 0].set_xlabel('Concentration (r-band)')
    axes[1, 0].set_ylabel('Ellipticity')
    axes[1, 0].set_title('Concentration vs Ellipticity')
    
    # Fourth moment distribution
    for galaxy_type in set(data['class']):
        mask = data['class'] == galaxy_type
        axes[1, 1].hist(data['m4_r'][mask], alpha=0.6, label=galaxy_type, bins=20)
    axes[1, 1].set_xlabel('Fourth Moment (r-band)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Fourth Moment Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_class_distribution(data: np.ndarray) -> Dict[str, int]:
    """
    Analyze the distribution of galaxy classes in the dataset.
    
    Args:
        data (np.ndarray): Galaxy catalog data
        
    Returns:
        Dict[str, int]: Class distribution
    """
    unique_classes, counts = np.unique(data['class'], return_counts=True)
    distribution = dict(zip(unique_classes, counts))
    
    print("Galaxy Class Distribution:")
    for class_name, count in distribution.items():
        percentage = (count / len(data)) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    return distribution
