"""
Production-ready inference module with proper error handling and logging
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import tensorflow as tf
from datetime import datetime
import hashlib


class ModelInference:
    """Production-ready inference with monitoring and error handling"""
    
    def __init__(self, config_path: str = "config.json", model_weights_path: str = "best_weights_balanced.h5"):
        """
        Initialize inference pipeline
        
        Args:
            config_path: Path to configuration file
            model_weights_path: Path to trained model weights
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Setup logging
        self._setup_logging()
        
        # Load model
        self.logger.info("Loading model...")
        self.model = self._build_model()
        self.model.load_weights(model_weights_path)
        self.logger.info("Model loaded successfully")
        
        # Configuration parameters
        self.class_names = self.config['model']['class_names']
        self.confidence_threshold = self.config['inference']['confidence_threshold']
        self.high_risk_threshold = self.config['inference']['high_risk_threshold']
        self.input_shape = tuple(self.config['model']['input_shape'])
        
        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
    
    def _setup_logging(self):
        """Configure logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=log_format,
            handlers=[
                logging.FileHandler('inference.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _build_model(self):
        """Reconstruct model architecture"""
        img_size = self.config['data']['image_size']
        n_classes = self.config['model']['num_classes']
        
        inputs = tf.keras.Input(shape=tuple(img_size) + (3,))
        base_model = tf.keras.applications.efficientnet.EfficientNetB3(
            include_top=False,
            weights=None,  # Will load from weights file
            input_tensor=inputs,
            pooling='max'
        )
        
        x = base_model.output
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        return model
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference
        
        Args:
            image: Input image (H, W, C) or (H, W)
        
        Returns:
            Preprocessed image ready for model
        """
        try:
            # Convert grayscale to RGB if needed
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            
            # Resize
            target_size = tuple(self.config['data']['image_size'])
            image = tf.image.resize(image, target_size).numpy()
            
            # Preprocess for EfficientNet
            image = tf.keras.applications.efficientnet.preprocess_input(image)
            
            return image
        
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    def predict(self, image: np.ndarray, return_gradcam: bool = False) -> Dict:
        """
        Run inference on a single image
        
        Args:
            image: Input image
            return_gradcam: Whether to compute Grad-CAM visualization
        
        Returns:
            Dictionary with prediction results and metadata
        """
        start_time = time.time()
        
        try:
            # Preprocess
            processed_image = self.preprocess_image(image)
            batch_image = np.expand_dims(processed_image, axis=0)
            
            # Inference
            predictions = self.model.predict(batch_image, verbose=0)[0]
            
            # Results
            pred_class_idx = int(np.argmax(predictions))
            pred_class_name = self.class_names[pred_class_idx]
            confidence = float(predictions[pred_class_idx])
            
            # Determine if prediction is reliable
            requires_review = confidence < self.confidence_threshold
            high_risk = confidence > self.high_risk_threshold and pred_class_name != "notumor"
            
            # Calculate inference time
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            result = {
                'prediction': {
                    'class': pred_class_name,
                    'class_index': pred_class_idx,
                    'confidence': confidence,
                    'probabilities': {
                        self.class_names[i]: float(predictions[i]) 
                        for i in range(len(self.class_names))
                    }
                },
                'metadata': {
                    'requires_human_review': requires_review,
                    'high_risk_case': high_risk,
                    'inference_time_ms': round(inference_time * 1000, 2),
                    'timestamp': datetime.now().isoformat(),
                    'model_version': self.config['model']['version']
                },
                'clinical_recommendation': self._get_clinical_recommendation(
                    pred_class_name, confidence, requires_review
                )
            }
            
            # Grad-CAM if requested
            if return_gradcam and self.config['deployment']['enable_gradcam']:
                result['gradcam'] = self._compute_gradcam(batch_image, pred_class_idx)
            
            # Log prediction
            if self.config['logging']['log_predictions']:
                self._log_prediction(result)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Inference failed: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_clinical_recommendation(self, pred_class: str, confidence: float, 
                                    requires_review: bool) -> str:
        """Generate clinical recommendation based on prediction"""
        if requires_review:
            return (f"Low confidence prediction ({confidence:.2%}). "
                   "Mandatory radiologist review required before any clinical action.")
        
        if pred_class == "notumor":
            return (f"No tumor detected with {confidence:.2%} confidence. "
                   "Consider clinical correlation and follow-up if symptoms persist.")
        
        else:
            return (f"{pred_class.capitalize()} detected with {confidence:.2%} confidence. "
                   "Recommend urgent radiologist confirmation and appropriate clinical workup.")
    
    def _compute_gradcam(self, image_batch: np.ndarray, pred_class: int) -> np.ndarray:
        """Compute Grad-CAM visualization"""
        try:
            # Find last conv layer
            last_conv_layer = None
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower():
                    last_conv_layer = layer
                    break
            
            if last_conv_layer is None:
                return None
            
            grad_model = tf.keras.models.Model(
                [self.model.inputs],
                [last_conv_layer.output, self.model.output]
            )
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image_batch)
                class_channel = predictions[:, pred_class]
            
            grads = tape.gradient(class_channel, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            return heatmap.numpy()
        
        except Exception as e:
            self.logger.warning(f"Grad-CAM computation failed: {str(e)}")
            return None
    
    def _log_prediction(self, result: Dict):
        """Log prediction for audit trail"""
        log_entry = {
            'timestamp': result['metadata']['timestamp'],
            'prediction': result['prediction']['class'],
            'confidence': result['prediction']['confidence'],
            'requires_review': result['metadata']['requires_human_review'],
            'inference_time_ms': result['metadata']['inference_time_ms']
        }
        self.logger.info(f"Prediction: {json.dumps(log_entry)}")
    
    def batch_predict(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Run inference on multiple images
        
        Args:
            images: List of input images
        
        Returns:
            List of prediction results
        """
        results = []
        for idx, image in enumerate(images):
            self.logger.info(f"Processing image {idx + 1}/{len(images)}")
            result = self.predict(image)
            results.append(result)
        
        return results
    
    def get_performance_stats(self) -> Dict:
        """Get inference performance statistics"""
        avg_time = self.total_inference_time / self.inference_count if self.inference_count > 0 else 0
        
        return {
            'total_inferences': self.inference_count,
            'total_time_seconds': round(self.total_inference_time, 2),
            'average_inference_time_ms': round(avg_time * 1000, 2),
            'throughput_per_second': round(1 / avg_time, 2) if avg_time > 0 else 0
        }


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def validate_image_input(image: np.ndarray) -> None:
    """
    Validate image input meets requirements
    
    Args:
        image: Input image to validate
    
    Raises:
        ValidationError: If image doesn't meet requirements
    """
    if image is None:
        raise ValidationError("Image is None")
    
    if not isinstance(image, np.ndarray):
        raise ValidationError(f"Image must be numpy array, got {type(image)}")
    
    if len(image.shape) not in [2, 3]:
        raise ValidationError(f"Image must be 2D or 3D array, got shape {image.shape}")
    
    if len(image.shape) == 3 and image.shape[2] not in [1, 3]:
        raise ValidationError(f"Image must have 1 or 3 channels, got {image.shape[2]}")
    
    if image.size == 0:
        raise ValidationError("Image is empty")
    
    # Check dimensions are reasonable
    min_size, max_size = 50, 2000
    for dim in image.shape[:2]:
        if dim < min_size or dim > max_size:
            raise ValidationError(f"Image dimensions must be between {min_size} and {max_size}, got {image.shape}")


if __name__ == "__main__":
    # Example usage
    print("Inference Module - Production Ready")
    print("Import and use ModelInference class for predictions")
    
    # Example
    # inferencer = ModelInference()
    # result = inferencer.predict(image)
    # print(result)
