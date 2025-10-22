"""
Utility Modules for Galaxy Morphology Classification

This module provides common utility functions for data processing,
visualization, and analysis tasks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import warnings


class DataUtils:
    """
    Utility class for data processing and manipulation.
    """
    
    @staticmethod
    def load_numpy_data(file_path: str) -> np.ndarray:
        """
        Load numpy data from file.
        
        Args:
            file_path (str): Path to the numpy file
            
        Returns:
            np.ndarray: Loaded data
        """
        try:
            return np.load(file_path)
        except Exception as e:
            raise ValueError(f"Error loading data from {file_path}: {e}")
    
    @staticmethod
    def save_numpy_data(data: np.ndarray, file_path: str) -> None:
        """
        Save numpy data to file.
        
        Args:
            data (np.ndarray): Data to save
            file_path (str): Path to save the data
        """
        try:
            np.save(file_path, data)
        except Exception as e:
            raise ValueError(f"Error saving data to {file_path}: {e}")
    
    @staticmethod
    def create_directory(path: str) -> None:
        """
        Create directory if it doesn't exist.
        
        Args:
            path (str): Directory path
        """
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def get_file_list(directory: str, extension: str = None) -> List[str]:
        """
        Get list of files in directory.
        
        Args:
            directory (str): Directory path
            extension (str): File extension filter
            
        Returns:
            List[str]: List of file paths
        """
        if not os.path.exists(directory):
            raise ValueError(f"Directory {directory} does not exist")
        
        files = os.listdir(directory)
        if extension:
            files = [f for f in files if f.endswith(extension)]
        
        return [os.path.join(directory, f) for f in files]
    
    @staticmethod
    def validate_data_shape(data: np.ndarray, expected_shape: Tuple[int, ...]) -> bool:
        """
        Validate data shape.
        
        Args:
            data (np.ndarray): Input data
            expected_shape (Tuple[int, ...]): Expected shape
            
        Returns:
            bool: True if shape matches
        """
        return data.shape == expected_shape
    
    @staticmethod
    def remove_outliers(data: np.ndarray, method: str = 'iqr', 
                       factor: float = 1.5) -> np.ndarray:
        """
        Remove outliers from data.
        
        Args:
            data (np.ndarray): Input data
            method (str): Method for outlier detection ('iqr' or 'zscore')
            factor (float): Factor for outlier detection
            
        Returns:
            np.ndarray: Data with outliers removed
        """
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            mask = (data >= lower_bound) & (data <= upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            mask = z_scores < factor
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
        
        return data[mask]
    
    @staticmethod
    def normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize data.
        
        Args:
            data (np.ndarray): Input data
            method (str): Normalization method ('minmax' or 'zscore')
            
        Returns:
            np.ndarray: Normalized data
        """
        if method == 'minmax':
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        elif method == 'zscore':
            return (data - np.mean(data)) / np.std(data)
        else:
            raise ValueError("Method must be 'minmax' or 'zscore'")


class VisualizationUtils:
    """
    Utility class for creating visualizations.
    """
    
    @staticmethod
    def set_plot_style(style: str = 'default') -> None:
        """
        Set matplotlib plot style.
        
        Args:
            style (str): Plot style ('default', 'seaborn', 'ggplot')
        """
        if style == 'seaborn':
            try:
                import seaborn as sns
                sns.set_style("whitegrid")
            except ImportError:
                warnings.warn("Seaborn not available, using default style")
        elif style == 'ggplot':
            plt.style.use('ggplot')
        else:
            plt.style.use('default')
    
    @staticmethod
    def create_subplot_grid(n_plots: int, n_cols: int = 2, 
                           figsize: Tuple[int, int] = (15, 10)) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create a grid of subplots.
        
        Args:
            n_plots (int): Number of plots
            n_cols (int): Number of columns
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            Tuple[plt.Figure, np.ndarray]: Figure and axes array
        """
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        if n_plots == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        return fig, axes
    
    @staticmethod
    def add_colorbar(plot, label: str = None, ax: plt.Axes = None) -> None:
        """
        Add colorbar to plot.
        
        Args:
            plot: Matplotlib plot object
            label (str): Colorbar label
            ax (plt.Axes): Axes object
        """
        if ax is None:
            ax = plt.gca()
        
        cbar = plt.colorbar(plot, ax=ax)
        if label:
            cbar.set_label(label)
    
    @staticmethod
    def save_plot(fig: plt.Figure, file_path: str, dpi: int = 300) -> None:
        """
        Save plot to file.
        
        Args:
            fig (plt.Figure): Figure object
            file_path (str): Output file path
            dpi (int): Resolution
        """
        fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {file_path}")
    
    @staticmethod
    def create_histogram(data: np.ndarray, bins: int = 30, 
                        title: str = "Histogram", xlabel: str = "Value",
                        ylabel: str = "Frequency", save_path: str = None) -> None:
        """
        Create histogram plot.
        
        Args:
            data (np.ndarray): Input data
            bins (int): Number of bins
            title (str): Plot title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            save_path (str): Optional save path
        """
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=bins, alpha=0.7, edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def create_scatter_plot(x: np.ndarray, y: np.ndarray, 
                          c: np.ndarray = None, title: str = "Scatter Plot",
                          xlabel: str = "X", ylabel: str = "Y",
                          save_path: str = None) -> None:
        """
        Create scatter plot.
        
        Args:
            x (np.ndarray): X data
            y (np.ndarray): Y data
            c (np.ndarray): Color data
            title (str): Plot title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            save_path (str): Optional save path
        """
        plt.figure(figsize=(10, 8))
        
        if c is not None:
            scatter = plt.scatter(x, y, c=c, alpha=0.6, cmap='viridis')
            plt.colorbar(scatter)
        else:
            plt.scatter(x, y, alpha=0.6)
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class PerformanceUtils:
    """
    Utility class for performance analysis and timing.
    """
    
    @staticmethod
    def time_function(func, *args, **kwargs) -> Tuple[Any, float]:
        """
        Time a function execution.
        
        Args:
            func: Function to time
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple[Any, float]: (result, execution_time)
        """
        import time
        
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        return result, execution_time
    
    @staticmethod
    def benchmark_functions(functions: List[callable], data: np.ndarray, 
                           n_trials: int = 10) -> Dict[str, float]:
        """
        Benchmark multiple functions.
        
        Args:
            functions (List[callable]): List of functions to benchmark
            data (np.ndarray): Input data
            n_trials (int): Number of trials
            
        Returns:
            Dict[str, float]: Average execution times
        """
        results = {}
        
        for func in functions:
            times = []
            for _ in range(n_trials):
                _, exec_time = PerformanceUtils.time_function(func, data)
                times.append(exec_time)
            
            results[func.__name__] = np.mean(times)
        
        return results
    
    @staticmethod
    def print_performance_summary(results: Dict[str, float]) -> None:
        """
        Print performance summary.
        
        Args:
            results (Dict[str, float]): Performance results
        """
        print("Performance Summary:")
        print("-" * 30)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1])
        
        for func_name, avg_time in sorted_results:
            print(f"{func_name}: {avg_time:.6f}s")
        
        if len(sorted_results) > 1:
            fastest = sorted_results[0][1]
            slowest = sorted_results[-1][1]
            speedup = slowest / fastest
            print(f"\nSpeedup factor: {speedup:.1f}x")


class ValidationUtils:
    """
    Utility class for model validation and evaluation.
    """
    
    @staticmethod
    def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            float: Accuracy score
        """
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def calculate_precision_recall(y_true: np.ndarray, y_pred: np.ndarray, 
                                 labels: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate precision and recall for each class.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            labels (List[str]): Class labels
            
        Returns:
            Dict[str, Dict[str, float]]: Precision and recall for each class
        """
        if labels is None:
            labels = list(set(y_true))
        
        results = {}
        
        for label in labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fp = np.sum((y_true != label) & (y_pred == label))
            fn = np.sum((y_true == label) & (y_pred != label))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            results[label] = {
                'precision': precision,
                'recall': recall,
                'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            }
        
        return results
    
    @staticmethod
    def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                                  labels: List[str] = None) -> None:
        """
        Print detailed classification report.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            labels (List[str]): Class labels
        """
        if labels is None:
            labels = list(set(y_true))
        
        print("Classification Report:")
        print("-" * 50)
        
        # Calculate overall accuracy
        accuracy = ValidationUtils.calculate_accuracy(y_true, y_pred)
        print(f"Overall Accuracy: {accuracy:.4f}")
        print()
        
        # Calculate per-class metrics
        metrics = ValidationUtils.calculate_precision_recall(y_true, y_pred, labels)
        
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 50)
        
        for label in labels:
            precision = metrics[label]['precision']
            recall = metrics[label]['recall']
            f1_score = metrics[label]['f1_score']
            
            print(f"{label:<15} {precision:<10.4f} {recall:<10.4f} {f1_score:<10.4f}")


def create_summary_statistics(data: np.ndarray) -> Dict[str, Any]:
    """
    Create summary statistics for data.
    
    Args:
        data (np.ndarray): Input data
        
    Returns:
        Dict[str, Any]: Summary statistics
    """
    return {
        'count': len(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75)
    }


def print_data_info(data: np.ndarray, name: str = "Data") -> None:
    """
    Print information about data.
    
    Args:
        data (np.ndarray): Input data
        name (str): Data name
    """
    print(f"{name} Information:")
    print("-" * 30)
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Memory usage: {data.nbytes / (1024**2):.2f} MB")
    
    if data.size > 0:
        stats = create_summary_statistics(data.flatten())
        print(f"Mean: {stats['mean']:.4f}")
        print(f"Std: {stats['std']:.4f}")
        print(f"Min: {stats['min']:.4f}")
        print(f"Max: {stats['max']:.4f}")
        print(f"Median: {stats['median']:.4f}")
    print()
