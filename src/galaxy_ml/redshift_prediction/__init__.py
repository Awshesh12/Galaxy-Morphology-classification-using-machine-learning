"""
Redshift Prediction Module

This module provides functionality for predicting galaxy redshifts using
machine learning techniques, specifically decision tree regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import KFold
from typing import Tuple, List, Dict, Any
import pydotplus


class RedshiftPredictor:
    """
    A class for predicting galaxy redshifts using machine learning.
    
    This class implements decision tree regression for redshift prediction
    based on photometric colors from SDSS data.
    """
    
    def __init__(self, max_depth: int = 19, random_state: int = 42):
        """
        Initialize the redshift predictor.
        
        Args:
            max_depth (int): Maximum depth of the decision tree
            random_state (int): Random state for reproducibility
        """
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        self.is_trained = False
    
    def extract_features_targets(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and targets from SDSS galaxy data.
        
        Args:
            data (np.ndarray): Structured array containing SDSS data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (features, targets)
        """
        features = np.zeros((data.shape[0], 4))
        features[:, 0] = data['u'] - data['g']  # u-g color
        features[:, 1] = data['g'] - data['r']  # g-r color
        features[:, 2] = data['r'] - data['i']  # r-i color
        features[:, 3] = data['i'] - data['z']  # i-z color
        
        targets = data['redshift']
        
        return features, targets
    
    def median_absolute_error(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """
        Calculate median absolute error between predicted and actual redshifts.
        
        Args:
            predicted (np.ndarray): Predicted redshift values
            actual (np.ndarray): Actual redshift values
            
        Returns:
            float: Median absolute error
        """
        return np.median(np.abs(predicted - actual))
    
    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Train the redshift prediction model.
        
        Args:
            features (np.ndarray): Training features
            targets (np.ndarray): Training targets
        """
        self.model.fit(features, targets)
        self.is_trained = True
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict redshifts for given features.
        
        Args:
            features (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted redshifts
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(features)
    
    def validate_model(self, features: np.ndarray, targets: np.ndarray, 
                      train_fraction: float = 0.67) -> float:
        """
        Validate the model using a simple train-test split.
        
        Args:
            features (np.ndarray): Input features
            targets (np.ndarray): Target values
            train_fraction (float): Fraction of data to use for training
            
        Returns:
            float: Median absolute error on test set
        """
        split_index = int(train_fraction * len(features))
        
        train_features = features[:split_index]
        test_features = features[split_index:]
        train_targets = targets[:split_index]
        test_targets = targets[split_index:]
        
        # Train the model
        self.model.fit(train_features, train_targets)
        
        # Make predictions
        predictions = self.model.predict(test_features)
        
        # Calculate error
        error = self.median_absolute_error(test_targets, predictions)
        
        return error
    
    def cross_validate(self, features: np.ndarray, targets: np.ndarray, 
                      k_folds: int = 10) -> List[float]:
        """
        Perform k-fold cross validation.
        
        Args:
            features (np.ndarray): Input features
            targets (np.ndarray): Target values
            k_folds (int): Number of folds for cross validation
            
        Returns:
            List[float]: List of median absolute errors for each fold
        """
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)
        errors = []
        
        for train_indices, test_indices in kf.split(features):
            train_features = features[train_indices]
            test_features = features[test_indices]
            train_targets = targets[train_indices]
            test_targets = targets[test_indices]
            
            # Train and predict
            self.model.fit(train_features, train_targets)
            predictions = self.model.predict(test_features)
            
            # Calculate error
            error = self.median_absolute_error(test_targets, predictions)
            errors.append(error)
        
        return errors
    
    def cross_validate_predictions(self, features: np.ndarray, targets: np.ndarray, 
                                  k_folds: int = 10) -> np.ndarray:
        """
        Perform k-fold cross validation and return predictions for all data points.
        
        Args:
            features (np.ndarray): Input features
            targets (np.ndarray): Target values
            k_folds (int): Number of folds for cross validation
            
        Returns:
            np.ndarray: Cross-validated predictions
        """
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)
        all_predictions = np.zeros_like(targets)
        
        for train_indices, test_indices in kf.split(features):
            train_features = features[train_indices]
            test_features = features[test_indices]
            train_targets = targets[train_indices]
            
            # Train and predict
            self.model.fit(train_features, train_targets)
            predictions = self.model.predict(test_features)
            
            # Store predictions
            all_predictions[test_indices] = predictions
        
        return all_predictions
    
    def optimize_depth(self, features: np.ndarray, targets: np.ndarray, 
                      max_depths: List[int] = None) -> Dict[str, Any]:
        """
        Optimize the maximum depth of the decision tree.
        
        Args:
            features (np.ndarray): Input features
            targets (np.ndarray): Target values
            max_depths (List[int]): List of depths to test
            
        Returns:
            Dict[str, Any]: Results dictionary with optimal depth and errors
        """
        if max_depths is None:
            max_depths = list(range(1, 36, 2))
        
        train_errors = []
        test_errors = []
        
        split_index = int(0.5 * len(features))
        train_features = features[:split_index]
        test_features = features[split_index:]
        train_targets = targets[:split_index]
        test_targets = targets[split_index:]
        
        for depth in max_depths:
            # Create model with current depth
            model = DecisionTreeRegressor(max_depth=depth, random_state=self.random_state)
            
            # Train and predict
            model.fit(train_features, train_targets)
            
            train_predictions = model.predict(train_features)
            test_predictions = model.predict(test_features)
            
            train_error = self.median_absolute_error(train_targets, train_predictions)
            test_error = self.median_absolute_error(test_targets, test_predictions)
            
            train_errors.append(train_error)
            test_errors.append(test_error)
        
        # Find optimal depth
        optimal_depth = max_depths[np.argmin(test_errors)]
        
        return {
            'depths': max_depths,
            'train_errors': train_errors,
            'test_errors': test_errors,
            'optimal_depth': optimal_depth,
            'min_error': min(test_errors)
        }
    
    def visualize_tree(self, features: np.ndarray, targets: np.ndarray, 
                      output_path: str = "decision_tree.jpg") -> None:
        """
        Visualize the decision tree structure.
        
        Args:
            features (np.ndarray): Input features
            targets (np.ndarray): Target values
            output_path (str): Path to save the tree visualization
        """
        # Train the model
        self.model.fit(features, targets)
        
        # Export tree to dot format
        dot_data = export_graphviz(
            self.model,
            out_file=None,
            feature_names=['u-g', 'g-r', 'r-i', 'i-z'],
            filled=True,
            rounded=True
        )
        
        # Create graph and save
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_jpg(output_path)
        print(f"Decision tree visualization saved to {output_path}")
    
    def plot_color_redshift(self, data: np.ndarray, save_path: str = None) -> None:
        """
        Create a color-redshift scatter plot.
        
        Args:
            data (np.ndarray): SDSS galaxy data
            save_path (str): Optional path to save the plot
        """
        # Extract colors and redshifts
        u_g = data['u'] - data['g']
        r_i = data['r'] - data['i']
        redshift = data['redshift']
        
        # Create plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(u_g, r_i, s=0.5, c=redshift, cmap='YlOrRd', alpha=0.6)
        
        plt.colorbar(scatter, label='Redshift')
        plt.xlabel('u-g Color Index')
        plt.ylabel('r-i Color Index')
        plt.title('Galaxy Redshift vs Color Indices')
        plt.xlim(-0.5, 2.5)
        plt.ylim(-0.5, 1.0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, targets: np.ndarray, predictions: np.ndarray, 
                        save_path: str = None) -> None:
        """
        Plot predicted vs actual redshifts.
        
        Args:
            targets (np.ndarray): Actual redshifts
            predictions (np.ndarray): Predicted redshifts
            save_path (str): Optional path to save the plot
        """
        plt.figure(figsize=(8, 8))
        plt.scatter(targets, predictions, s=0.4, alpha=0.6)
        
        # Add diagonal line for perfect predictions
        max_val = max(targets.max(), predictions.max())
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        
        plt.xlabel('Measured Redshift')
        plt.ylabel('Predicted Redshift')
        plt.title('Predicted vs Actual Redshifts')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def split_galaxies_qsos(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data into galaxies and QSOs (Quasi-Stellar Objects).
    
    Args:
        data (np.ndarray): SDSS data with 'spec_class' field
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (galaxies, qsos)
    """
    galaxies = data[data['spec_class'] == b'GALAXY']
    qsos = data[data['spec_class'] == b'QSO']
    
    return galaxies, qsos


def compare_galaxy_qso_performance(data: np.ndarray) -> Dict[str, float]:
    """
    Compare redshift prediction performance for galaxies vs QSOs.
    
    Args:
        data (np.ndarray): SDSS data
        
    Returns:
        Dict[str, float]: Performance metrics for galaxies and QSOs
    """
    # Split data
    galaxies, qsos = split_galaxies_qsos(data)
    
    # Create predictor
    predictor = RedshiftPredictor()
    
    # Evaluate galaxies
    galaxy_features, galaxy_targets = predictor.extract_features_targets(galaxies)
    galaxy_errors = predictor.cross_validate(galaxy_features, galaxy_targets)
    galaxy_mean_error = np.mean(galaxy_errors)
    
    # Evaluate QSOs
    qso_features, qso_targets = predictor.extract_features_targets(qsos)
    qso_errors = predictor.cross_validate(qso_features, qso_targets)
    qso_mean_error = np.mean(qso_errors)
    
    return {
        'galaxy_error': galaxy_mean_error,
        'qso_error': qso_mean_error,
        'galaxy_count': len(galaxies),
        'qso_count': len(qsos)
    }
