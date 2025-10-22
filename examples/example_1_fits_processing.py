#!/usr/bin/env python3
"""
Example 1: FITS Image Processing and Stacking

This example demonstrates how to use the galaxy_ml package for processing
FITS images and performing image stacking operations.
"""

import numpy as np
import matplotlib.pyplot as plt
from galaxy_ml import FITSProcessor, ImageStacker, DataUtils, VisualizationUtils


def main():
    """Main function demonstrating FITS processing capabilities."""
    
    print("=== Galaxy ML - FITS Image Processing Example ===\n")
    
    # Initialize processors
    fits_processor = FITSProcessor()
    image_stacker = ImageStacker()
    
    # Example 1: Basic FITS operations
    print("1. Basic FITS Operations")
    print("-" * 30)
    
    # Create sample data (simulating FITS data)
    print("Creating sample FITS data...")
    sample_data = np.random.rand(100, 100) * 100
    sample_data[50:60, 50:60] += 1000  # Add bright source
    
    # Find brightest pixel
    brightest_coords = fits_processor.find_brightest_pixel(sample_data)
    print(f"Brightest pixel coordinates: {brightest_coords}")
    print(f"Brightest pixel value: {sample_data[brightest_coords]:.2f}")
    
    # Visualize the data
    print("Creating visualization...")
    fits_processor.visualize_fits(sample_data, title="Sample FITS Image")
    
    # Example 2: Image stacking
    print("\n2. Image Stacking Operations")
    print("-" * 30)
    
    # Create multiple sample images
    print("Creating multiple sample images...")
    images = []
    for i in range(5):
        # Create image with different noise patterns
        img = np.random.rand(50, 50) * 50
        img[25:30, 25:30] += 500  # Consistent bright source
        images.append(img)
    
    # Save images as numpy arrays (simulating FITS files)
    file_paths = []
    for i, img in enumerate(images):
        file_path = f"sample_image_{i}.npy"
        DataUtils.save_numpy_data(img, file_path)
        file_paths.append(file_path)
    
    # Perform mean stacking
    print("Performing mean stacking...")
    mean_stack = image_stacker.mean_stack(file_paths)
    print(f"Mean stack shape: {mean_stack.shape}")
    print(f"Mean stack value at center: {mean_stack[25, 25]:.2f}")
    
    # Perform median stacking
    print("Performing median stacking...")
    median_stack, exec_time, memory_usage = image_stacker.median_stack(file_paths)
    print(f"Median stack shape: {median_stack.shape}")
    print(f"Execution time: {exec_time:.4f} seconds")
    print(f"Memory usage: {memory_usage:.2f} MB")
    print(f"Median stack value at center: {median_stack[25, 25]:.2f}")
    
    # Example 3: Performance comparison
    print("\n3. Performance Comparison")
    print("-" * 30)
    
    from galaxy_ml.data_processing import performance_comparison
    
    print("Comparing statistical function performance...")
    perf_results = performance_comparison()
    
    # Example 4: Data analysis
    print("\n4. Data Analysis")
    print("-" * 30)
    
    # Analyze the stacked images
    print("Analyzing stacked images...")
    DataUtils.print_data_info(mean_stack, "Mean Stack")
    DataUtils.print_data_info(median_stack, "Median Stack")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(images[0], cmap='viridis')
    axes[0].set_title('Original Image')
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')
    
    axes[1].imshow(mean_stack, cmap='viridis')
    axes[1].set_title('Mean Stack')
    axes[1].set_xlabel('X (pixels)')
    axes[1].set_ylabel('Y (pixels)')
    
    axes[2].imshow(median_stack, cmap='viridis')
    axes[2].set_title('Median Stack')
    axes[2].set_xlabel('X (pixels)')
    axes[2].set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    plt.show()
    
    # Clean up temporary files
    print("\nCleaning up temporary files...")
    for file_path in file_paths:
        import os
        if os.path.exists(file_path):
            os.remove(file_path)
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
