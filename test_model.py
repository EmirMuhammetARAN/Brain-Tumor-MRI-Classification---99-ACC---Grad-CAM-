"""
Comprehensive test suite for brain tumor classification model
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from inference import ModelInference, validate_image_input, ValidationError


class TestImageValidation:
    """Test input validation"""
    
    def test_valid_rgb_image(self):
        """Test valid RGB image passes validation"""
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        validate_image_input(image)  # Should not raise
    
    def test_valid_grayscale_image(self):
        """Test valid grayscale image passes validation"""
        image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        validate_image_input(image)  # Should not raise
    
    def test_none_image_raises_error(self):
        """Test None image raises ValidationError"""
        with pytest.raises(ValidationError, match="Image is None"):
            validate_image_input(None)
    
    def test_wrong_type_raises_error(self):
        """Test non-numpy array raises ValidationError"""
        with pytest.raises(ValidationError, match="must be numpy array"):
            validate_image_input([1, 2, 3])
    
    def test_wrong_dimensions_raises_error(self):
        """Test wrong number of dimensions raises ValidationError"""
        with pytest.raises(ValidationError, match="must be 2D or 3D"):
            validate_image_input(np.array([[[1, 2], [3, 4]]]))
    
    def test_empty_image_raises_error(self):
        """Test empty image raises ValidationError"""
        with pytest.raises(ValidationError, match="empty"):
            validate_image_input(np.array([]))
    
    def test_too_small_image_raises_error(self):
        """Test too small image raises ValidationError"""
        with pytest.raises(ValidationError, match="dimensions must be between"):
            validate_image_input(np.random.randint(0, 255, (10, 10, 3)))
    
    def test_too_large_image_raises_error(self):
        """Test too large image raises ValidationError"""
        with pytest.raises(ValidationError, match="dimensions must be between"):
            validate_image_input(np.random.randint(0, 255, (3000, 3000, 3)))


class TestModelInference:
    """Test model inference pipeline"""
    
    @pytest.fixture
    def model(self):
        """Load model for testing"""
        return ModelInference(
            config_path="config.json",
            model_weights_path="best_weights_balanced.h5"
        )
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing"""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_model_loads_successfully(self, model):
        """Test model loads without errors"""
        assert model.model is not None
        assert len(model.class_names) == 4
    
    def test_preprocessing_rgb_image(self, model, sample_image):
        """Test image preprocessing for RGB image"""
        processed = model.preprocess_image(sample_image)
        assert processed.shape == (224, 224, 3)
    
    def test_preprocessing_grayscale_image(self, model):
        """Test image preprocessing converts grayscale to RGB"""
        gray_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        processed = model.preprocess_image(gray_image)
        assert processed.shape == (224, 224, 3)
    
    def test_predict_returns_valid_structure(self, model, sample_image):
        """Test predict returns expected structure"""
        result = model.predict(sample_image, return_gradcam=False)
        
        assert 'prediction' in result
        assert 'metadata' in result
        assert 'clinical_recommendation' in result
        
        assert 'class' in result['prediction']
        assert 'confidence' in result['prediction']
        assert 'probabilities' in result['prediction']
        
        assert result['prediction']['class'] in model.class_names
        assert 0 <= result['prediction']['confidence'] <= 1
    
    def test_predict_probabilities_sum_to_one(self, model, sample_image):
        """Test predicted probabilities sum to 1"""
        result = model.predict(sample_image)
        probs = list(result['prediction']['probabilities'].values())
        assert abs(sum(probs) - 1.0) < 1e-5
    
    def test_predict_with_gradcam(self, model, sample_image):
        """Test prediction with Grad-CAM"""
        result = model.predict(sample_image, return_gradcam=True)
        
        # Grad-CAM might be None if it fails, but shouldn't crash
        if 'gradcam' in result and result['gradcam'] is not None:
            assert isinstance(result['gradcam'], np.ndarray)
    
    def test_batch_predict(self, model):
        """Test batch prediction"""
        images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(3)]
        results = model.batch_predict(images)
        
        assert len(results) == 3
        for result in results:
            assert 'prediction' in result
            assert 'metadata' in result
    
    def test_performance_stats(self, model, sample_image):
        """Test performance statistics tracking"""
        # Make a few predictions
        for _ in range(3):
            model.predict(sample_image)
        
        stats = model.get_performance_stats()
        assert stats['total_inferences'] >= 3
        assert stats['average_inference_time_ms'] > 0
        assert stats['throughput_per_second'] > 0
    
    def test_low_confidence_triggers_review(self, model):
        """Test low confidence predictions trigger review flag"""
        # This is a mock test - in practice you'd need specific images
        # that produce low confidence
        result = model.predict(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        if result['prediction']['confidence'] < model.confidence_threshold:
            assert result['metadata']['requires_human_review'] is True


class TestClinicalValidation:
    """Test clinical validation metrics"""
    
    def test_confusion_matrix_shape(self):
        """Test confusion matrix has correct shape"""
        from clinical_validation import ClinicalValidator
        
        validator = ClinicalValidator(['class1', 'class2', 'class3'])
        
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred_probs = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
            [0.9, 0.05, 0.05],
            [0.2, 0.6, 0.2],
            [0.1, 0.1, 0.8]
        ])
        
        metrics = validator.calculate_clinical_metrics(y_true, y_pred_probs)
        
        assert metrics['overall']['confusion_matrix'].shape == (3, 3)
        assert 'per_class' in metrics
        assert len(metrics['per_class']) == 3
    
    def test_metrics_in_valid_range(self):
        """Test all metrics are in valid ranges"""
        from clinical_validation import ClinicalValidator
        
        validator = ClinicalValidator(['class1', 'class2'])
        
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred_probs = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.85, 0.15],
            [0.1, 0.9],
            [0.95, 0.05],
            [0.3, 0.7]
        ])
        
        metrics = validator.calculate_clinical_metrics(y_true, y_pred_probs)
        
        for class_name, class_metrics in metrics['per_class'].items():
            assert 0 <= class_metrics['sensitivity'] <= 1
            assert 0 <= class_metrics['specificity'] <= 1
            assert 0 <= class_metrics['ppv'] <= 1
            assert 0 <= class_metrics['npv'] <= 1
            assert 0 <= class_metrics['roc_auc'] <= 1


class TestErrorAnalysis:
    """Test error analysis functionality"""
    
    def test_failure_case_identification(self):
        """Test identification of different failure types"""
        from error_analysis import ErrorAnalyzer
        
        analyzer = ErrorAnalyzer(['class1', 'class2'])
        
        # Mock data: some correct, some wrong with varying confidence
        images = np.random.randint(0, 255, (6, 224, 224, 3), dtype=np.uint8)
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred_probs = np.array([
            [0.95, 0.05],  # Correct, high confidence
            [0.05, 0.95],  # Correct, high confidence
            [0.3, 0.7],    # Wrong, high confidence (class1 -> class2)
            [0.8, 0.2],    # Wrong, medium confidence (class2 -> class1)
            [0.6, 0.4],    # Correct but low confidence
            [0.45, 0.55]   # Wrong, low confidence
        ])
        
        failures = analyzer.identify_failure_cases(images, y_true, y_pred_probs)
        
        assert 'high_confidence_errors' in failures
        assert 'low_confidence_errors' in failures
        assert 'borderline_correct' in failures
        assert 'confusion_pairs' in failures


@pytest.mark.integration
class TestAPIEndpoints:
    """Integration tests for API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from api import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'model_loaded' in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert 'version' in data
    
    def test_classes_endpoint(self, client):
        """Test classes endpoint"""
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert 'classes' in data
        assert len(data['classes']) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
