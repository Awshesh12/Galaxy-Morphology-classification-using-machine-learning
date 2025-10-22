"""
Data Processing Module for Galaxy Morphology Classification

This module provides functionality for processing FITS files and performing
image stacking operations for astronomical data analysis.
"""

import os
import time
import statistics
from typing import List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


class FITSProcessor:
    """
    A class for processing FITS files and extracting astronomical data.
    
    This class provides methods for loading FITS files, finding bright sources,
    and performing basic statistical operations on astronomical images.
    """
    
    def __init__(self):
        """Initialize the FITS processor."""
        pass
    
    @staticmethod
    def load_fits(file_path: str) -> np.ndarray:
        """
        Load a FITS file and return the data array.
        
        Args:
            file_path (str): Path to the FITS file
            
        Returns:
            np.ndarray: The data array from the FITS file
        """
        try:
            with fits.open(file_path) as hdu_list:
                data = hdu_list[0].data
            return data
        except Exception as e:
            raise ValueError(f"Error loading FITS file {file_path}: {e}")
    
    @staticmethod
    def find_brightest_pixel(data: np.ndarray) -> Tuple[int, int]:
        """
        Find the coordinates of the brightest pixel in the data.
        
        Args:
            data (np.ndarray): Input data array
            
        Returns:
            Tuple[int, int]: Coordinates (y, x) of the brightest pixel
        """
        return np.unravel_index(np.argmax(data, axis=None), data.shape)
    
    @staticmethod
    def visualize_fits(data: np.ndarray, title: str = "FITS Image", 
                      cmap: str = 'viridis', save_path: str = None) -> None:
        """
        Visualize FITS data as an image.
        
        Args:
            data (np.ndarray): Data to visualize
            title (str): Title for the plot
            cmap (str): Colormap to use
            save_path (str): Optional path to save the image
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(data.T, cmap=cmap, origin='lower')
        plt.colorbar(label='Flux Density')
        plt.title(title)
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ImageStacker:
    """
    A class for stacking astronomical images to improve signal-to-noise ratio.
    
    This class provides methods for mean and median stacking of FITS images,
    with optimized memory usage for large datasets.
    """
    
    def __init__(self):
        """Initialize the image stacker."""
        pass
    
    @staticmethod
    def calculate_mean(data: List[float]) -> float:
        """
        Calculate the mean of a list of numbers.
        
        Args:
            data (List[float]): List of numbers
            
        Returns:
            float: Mean value
        """
        return statistics.mean(data)
    
    @staticmethod
    def calculate_median_mean(data: List[float]) -> Tuple[float, float]:
        """
        Calculate both median and mean of a list of numbers.
        
        Args:
            data (List[float]): List of numbers
            
        Returns:
            Tuple[float, float]: (median, mean) values
        """
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        # Calculate mean
        mean_val = sum(sorted_data) / n
        
        # Calculate median
        if n % 2 == 0:
            median_val = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        else:
            median_val = sorted_data[n//2]
            
        return median_val, mean_val
    
    def mean_stack(self, file_paths: List[str]) -> np.ndarray:
        """
        Calculate mean stack of multiple FITS files.
        
        Args:
            file_paths (List[str]): List of paths to FITS files
            
        Returns:
            np.ndarray: Mean stacked image
        """
        if not file_paths:
            raise ValueError("No files provided for stacking")
        
        # Load all data
        data_arrays = []
        for file_path in file_paths:
            data = FITSProcessor.load_fits(file_path)
            data_arrays.append(data)
        
        # Calculate mean
        sum_data = sum(data_arrays)
        mean_data = sum_data / len(data_arrays)
        
        return mean_data
    
    def median_stack(self, file_paths: List[str]) -> Tuple[np.ndarray, float, float]:
        """
        Calculate median stack of multiple FITS files with timing and memory info.
        
        Args:
            file_paths (List[str]): List of paths to FITS files
            
        Returns:
            Tuple[np.ndarray, float, float]: (median_stack, execution_time, memory_usage_mb)
        """
        if not file_paths:
            raise ValueError("No files provided for stacking")
        
        start_time = time.perf_counter()
        
        # Load all data
        data_arrays = [FITSProcessor.load_fits(file_path) for file_path in file_paths]
        
        # Calculate median using numpy for efficiency
        median_data = np.median(data_arrays, axis=0)
        
        # Calculate memory usage
        memory_usage = sum(array.nbytes for array in data_arrays) / (1024 * 1024)  # MB
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        return median_data, execution_time, memory_usage
    
    def batch_process_directory(self, directory_path: str, 
                              stacking_method: str = 'median') -> dict:
        """
        Process all FITS files in a directory.
        
        Args:
            directory_path (str): Path to directory containing FITS files
            stacking_method (str): 'mean' or 'median'
            
        Returns:
            dict: Results dictionary with stacked image and metadata
        """
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory {directory_path} does not exist")
        
        # Find all FITS files
        fits_files = [f for f in os.listdir(directory_path) if f.endswith('.fits')]
        if not fits_files:
            raise ValueError(f"No FITS files found in {directory_path}")
        
        file_paths = [os.path.join(directory_path, f) for f in fits_files]
        
        if stacking_method == 'mean':
            stacked_image = self.mean_stack(file_paths)
            execution_time = 0  # Not tracked for mean stacking
            memory_usage = 0
        elif stacking_method == 'median':
            stacked_image, execution_time, memory_usage = self.median_stack(file_paths)
        else:
            raise ValueError("stacking_method must be 'mean' or 'median'")
        
        return {
            'stacked_image': stacked_image,
            'num_files': len(fits_files),
            'execution_time': execution_time,
            'memory_usage_mb': memory_usage,
            'method': stacking_method
        }


def performance_comparison():
    """
    Compare performance of different statistical functions.
    
    This function demonstrates the performance difference between
    Python's statistics module and NumPy for large datasets.
    """
    def time_stat(func, size: int, n_trials: int) -> float:
        """Time a statistical function."""
        results = []
        for _ in range(n_trials):
            data = np.random.rand(size)
            start = time.perf_counter()
            func(data)
            end = time.perf_counter() - start
            results.append(end)
        return np.mean(results)
    
    # Compare statistics.mean vs np.mean
    statistics_mean_time = time_stat(statistics.mean, 10**5, 10)
    np_mean_time = time_stat(np.mean, 10**5, 2000)
    
    print(f"statistics.mean: {statistics_mean_time:.6f}s")
    print(f"np.mean: {np_mean_time:.6f}s")
    print(f"NumPy is {statistics_mean_time/np_mean_time:.1f}x faster")
    
    return {
        'statistics_mean': statistics_mean_time,
        'numpy_mean': np_mean_time,
        'speedup_factor': statistics_mean_time/np_mean_time
    }
