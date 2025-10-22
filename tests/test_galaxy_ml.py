import pytest
import numpy as np
from galaxy_ml import FITSProcessor, ImageStacker, RedshiftPredictor, GalaxyClassifier


class TestFITSProcessor:
    """Test cases for FITSProcessor class."""
    
    def test_load_fits(self):
        """Test FITS file loading."""
        processor = FITSProcessor()
        # Create sample data
        sample_data = np.random.rand(10, 10)
        # This would test actual FITS loading if we had test files
        assert processor is not None
    
    def test_find_brightest_pixel(self):
        """Test brightest pixel detection."""
        processor = FITSProcessor()
        data = np.random.rand(5, 5)
        data[2, 2] = 100  # Set brightest pixel
        coords = processor.find_brightest_pixel(data)
        assert coords == (2, 2)


class TestImageStacker:
    """Test cases for ImageStacker class."""
    
    def test_calculate_mean(self):
        """Test mean calculation."""
        stacker = ImageStacker()
        data = [1, 2, 3, 4, 5]
        mean_val = stacker.calculate_mean(data)
        assert mean_val == 3.0
    
    def test_calculate_median_mean(self):
        """Test median and mean calculation."""
        stacker = ImageStacker()
        data = [1, 2, 3, 4, 5]
        median, mean = stacker.calculate_median_mean(data)
        assert median == 3.0
        assert mean == 3.0


class TestRedshiftPredictor:
    """Test cases for RedshiftPredictor class."""
    
    def test_extract_features_targets(self):
        """Test feature extraction."""
        predictor = RedshiftPredictor()
        
        # Create sample SDSS data
        dtype = [('u', 'f4'), ('g', 'f4'), ('r', 'f4'), ('i', 'f4'), ('z', 'f4'), ('redshift', 'f4')]
        data = np.array([(20.0, 19.0, 18.5, 18.0, 17.5, 0.1)], dtype=dtype)
        
        features, targets = predictor.extract_features_targets(data)
        assert features.shape == (1, 4)
        assert targets.shape == (1,)
        assert features[0, 0] == 1.0  # u-g
    
    def test_median_absolute_error(self):
        """Test median absolute error calculation."""
        predictor = RedshiftPredictor()
        predicted = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.1, 2.1, 2.9])
        error = predictor.median_absolute_error(predicted, actual)
        assert abs(error - 0.1) < 1e-6


class TestGalaxyClassifier:
    """Test cases for GalaxyClassifier class."""
    
    def test_extract_features_targets(self):
        """Test galaxy feature extraction."""
        classifier = GalaxyClassifier()
        
        # Create sample galaxy data
        dtype = [
            ('class', 'U20'), ('u-g', 'f4'), ('g-r', 'f4'), ('r-i', 'f4'), ('i-z', 'f4'),
            ('ecc', 'f4'), ('m4_r', 'f4'), ('petroR50_r', 'f4'), ('petroR90_r', 'f4')
        ]
        data = np.array([('elliptical', 1.5, 0.8, 0.4, 0.3, 0.5, 0.1, 3.0, 10.0)], dtype=dtype)
        
        features, targets = classifier.extract_features_targets(data)
        assert features.shape == (1, 13)
        assert targets.shape == (1,)
        assert targets[0] == 'elliptical'
    
    def test_split_data(self):
        """Test data splitting."""
        classifier = GalaxyClassifier()
        
        # Create sample data
        dtype = [('class', 'U20'), ('u-g', 'f4')]
        data = np.array([('elliptical', 1.5), ('spiral', 0.8), ('merger', 1.2)], dtype=dtype)
        
        train_data, test_data = classifier.split_data(data, train_fraction=0.67)
        assert len(train_data) == 2
        assert len(test_data) == 1


if __name__ == "__main__":
    pytest.main([__file__])
