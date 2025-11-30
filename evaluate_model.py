# -*- coding: utf-8 -*-
"""
Evaluate Model with Complete Analysis
Run this after training your model in model.ipynb
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

print("="*80)
print("MODEL EVALUATION SCRIPT")
print("="*80)

# Class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

print("\nSTEPS:")
print("1. Run model.ipynb and train model")
print("2. Make predictions on test set")
print("3. Run this script")

print("\n" + "="*80)
print("OPTION 1: Model and Test Data Available")
print("="*80)

# Load model
try:
    print("\nChecking model file...")
    if not Path('best_weights_balanced.h5').exists():
        raise FileNotFoundError("Model file (best_weights_balanced.h5) not found!")
    
    print("Loading model...")
    # Rebuild model architecture
    from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
    from tensorflow.keras.models import Model
    
    img_size = (224, 224)
    inputs = tf.keras.Input(shape=img_size + (3,))
    base_model = tf.keras.applications.efficientnet.EfficientNetB3(
        include_top=False, 
        weights=None,  # Will load weights later
        input_tensor=inputs, 
        pooling='max'
    )
    base_model.trainable = True
    
    x = base_model.output
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(4, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Load weights
    model.load_weights('best_weights_balanced.h5')
    print("Model loaded successfully!")
    
    # Load test data
    print("\nChecking test data...")
    test_dir = 'Testing'
    
    if Path(test_dir).exists():
        print(f"{test_dir} folder found!")
        
        # Load test data
        img_size = (224, 224)
        batch_size = 32
        
        test_data = tf.keras.preprocessing.image_dataset_from_directory(
            directory=test_dir,
            image_size=img_size,
            label_mode="categorical",
            batch_size=batch_size,
            shuffle=False
        )
        
        print("\nMaking predictions...")
        y_pred_probs = model.predict(test_data, verbose=1)
        
        # Extract true labels
        y_true = []
        for _, labels in test_data:
            y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_true = np.array(y_true)
        
        print(f"Predictions complete!")
        print(f"   Test samples: {len(y_true)}")
        print(f"   Predictions shape: {y_pred_probs.shape}")
        
        # Run analysis
        print("\n" + "="*80)
        print("DETAILED ANALYSIS STARTING...")
        print("="*80)
        
        from run_complete_evaluation import CompleteEvaluationPipeline
        
        # Create pipeline
        pipeline = CompleteEvaluationPipeline(class_names, output_dir='evaluation_results')
        
        # NOTE: We don't have train/val labels from saved model, only running test results
        results = pipeline.run_complete_evaluation(
            y_true=y_true,
            y_pred_probs=y_pred_probs
        )
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nResults saved in 'evaluation_results/' folder:")
        print("   per_class_metrics.png")
        print("   detailed_confusion_matrix.png")
        print("   confusion_pairs.png")
        print("   comprehensive_evaluation_report.txt")
        print("   evaluation_results.json")
        
    else:
        print(f"{test_dir} folder not found!")
        print(f"   Add data folder to project or check path in model.ipynb")
        raise FileNotFoundError(f"{test_dir} folder not found")
        
except FileNotFoundError as e:
    print(f"\nFile/Folder not found: {e}")
    print("\n" + "="*80)
    print("OPTION 2: Manual Evaluation")
    print("="*80)
    print("\nIf you have saved predictions from model.ipynb:")
    print("\n```python")
    print("import numpy as np")
    print("from run_complete_evaluation import CompleteEvaluationPipeline")
    print("")
    print("# Load saved predictions")
    print("y_true = np.load('y_test_true.npy')  # True labels")
    print("y_pred_probs = np.load('y_test_pred.npy')  # Prediction probabilities")
    print("")
    print("# Class names")
    print("class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']")
    print("")
    print("# Run pipeline")
    print("pipeline = CompleteEvaluationPipeline(class_names)")
    print("results = pipeline.run_complete_evaluation(y_true, y_pred_probs)")
    print("```")
    print("\n" + "="*80)
    print("OPTION 3: Run from model.ipynb")
    print("="*80)
    print("\nAdd this code at the end of model.ipynb:")
    print("\n```python")
    print("# Get test predictions")
    print("y_pred_probs = model.predict(test_data)")
    print("")
    print("# Extract true labels")
    print("y_true = []")
    print("for _, labels in test_data:")
    print("    y_true.extend(np.argmax(labels.numpy(), axis=1))")
    print("y_true = np.array(y_true)")
    print("")
    print("# Run analysis script")
    print("from run_complete_evaluation import CompleteEvaluationPipeline")
    print("pipeline = CompleteEvaluationPipeline(class_names)")
    print("results = pipeline.run_complete_evaluation(y_true, y_pred_probs)")
    print("```")

except Exception as e:
    print(f"\nERROR: {str(e)}")
    print("\nSUGGESTION:")
    print("   Run model.ipynb and make test predictions")
    print("   Then run this script again")

print("\n" + "="*80)
