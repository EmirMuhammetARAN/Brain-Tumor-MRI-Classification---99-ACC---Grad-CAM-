"""
Clinical Validation Metrics for Medical AI Model
Includes metrics required for regulatory approval and clinical deployment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from scipy import stats
import tensorflow as tf


class ClinicalValidator:
    """Comprehensive clinical validation for medical AI models"""
    
    def __init__(self, class_names):
        self.class_names = class_names
        self.n_classes = len(class_names)
        
    def calculate_clinical_metrics(self, y_true, y_pred_probs, confidence_level=0.95):
        """
        Calculate clinical-grade metrics with confidence intervals
        
        Args:
            y_true: True labels (one-hot or integer encoded)
            y_pred_probs: Predicted probabilities
            confidence_level: Confidence level for intervals (default 95%)
        
        Returns:
            Dictionary with comprehensive clinical metrics
        """
        # Convert to integer labels if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics with confidence intervals
        metrics_per_class = {}
        
        for i, class_name in enumerate(self.class_names):
            # Create binary classification for this class
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            # True/False Positives/Negatives
            TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            
            # Calculate metrics
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            ppv = TP / (TP + FP) if (TP + FP) > 0 else 0  # Precision
            npv = TN / (TN + FN) if (TN + FN) > 0 else 0
            
            # Calculate confidence intervals using Wilson score method
            n_positive = TP + FN
            n_negative = TN + FP
            
            sens_ci = self._wilson_ci(TP, n_positive, confidence_level)
            spec_ci = self._wilson_ci(TN, n_negative, confidence_level)
            ppv_ci = self._wilson_ci(TP, TP + FP, confidence_level) if (TP + FP) > 0 else (0, 0)
            npv_ci = self._wilson_ci(TN, TN + FN, confidence_level) if (TN + FN) > 0 else (0, 0)
            
            # F1 Score
            f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
            
            # ROC-AUC for this class
            try:
                roc_auc = roc_auc_score(y_true_binary, y_pred_probs[:, i])
            except:
                roc_auc = 0.5
            
            metrics_per_class[class_name] = {
                'sensitivity': sensitivity,
                'sensitivity_ci': sens_ci,
                'specificity': specificity,
                'specificity_ci': spec_ci,
                'ppv': ppv,
                'ppv_ci': ppv_ci,
                'npv': npv,
                'npv_ci': npv_ci,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'support': n_positive,
                'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
            }
        
        # Overall metrics
        accuracy = np.mean(y_pred == y_true)
        acc_ci = self._wilson_ci(np.sum(y_pred == y_true), len(y_true), confidence_level)
        
        # Macro-averaged AUC
        try:
            macro_auc = roc_auc_score(
                tf.keras.utils.to_categorical(y_true, self.n_classes),
                y_pred_probs,
                average='macro',
                multi_class='ovr'
            )
        except:
            macro_auc = np.mean([m['roc_auc'] for m in metrics_per_class.values()])
        
        overall_metrics = {
            'accuracy': accuracy,
            'accuracy_ci': acc_ci,
            'macro_auc': macro_auc,
            'confusion_matrix': cm
        }
        
        return {
            'per_class': metrics_per_class,
            'overall': overall_metrics
        }
    
    def _wilson_ci(self, successes, trials, confidence=0.95):
        """
        Calculate Wilson score confidence interval
        More accurate than normal approximation for small samples
        """
        if trials == 0:
            return (0, 0)
        
        p = successes / trials
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        
        denominator = 1 + z**2 / trials
        centre = (p + z**2 / (2 * trials)) / denominator
        adjustment = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
        
        return (max(0, centre - adjustment), min(1, centre + adjustment))
    
    def plot_roc_curves(self, y_true, y_pred_probs, save_path='roc_curves.png'):
        """Plot ROC curves for all classes"""
        if len(y_true.shape) == 1:
            y_true_onehot = tf.keras.utils.to_categorical(y_true, self.n_classes)
        else:
            y_true_onehot = y_true
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-Class Brain Tumor Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curves saved to {save_path}")
    
    def plot_precision_recall_curves(self, y_true, y_pred_probs, save_path='pr_curves.png'):
        """Plot Precision-Recall curves for all classes"""
        if len(y_true.shape) == 1:
            y_true_onehot = tf.keras.utils.to_categorical(y_true, self.n_classes)
        else:
            y_true_onehot = y_true
        
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_pred_probs[:, i])
            avg_precision = average_precision_score(y_true_onehot[:, i], y_pred_probs[:, i])
            plt.plot(recall, precision, lw=2, label=f'{class_name} (AP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Precision-Recall curves saved to {save_path}")
    
    def generate_clinical_report(self, metrics, save_path='clinical_report.txt'):
        """Generate a comprehensive clinical validation report"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CLINICAL VALIDATION REPORT\n")
            f.write("Brain Tumor Classification Model\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            overall = metrics['overall']
            f.write(f"Accuracy: {overall['accuracy']:.4f} ")
            f.write(f"(95% CI: [{overall['accuracy_ci'][0]:.4f}, {overall['accuracy_ci'][1]:.4f}])\n")
            f.write(f"Macro-averaged AUC: {overall['macro_auc']:.4f}\n\n")
            
            # Per-class metrics
            f.write("PER-CLASS CLINICAL METRICS\n")
            f.write("=" * 80 + "\n\n")
            
            for class_name, class_metrics in metrics['per_class'].items():
                f.write(f"{class_name.upper()}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Support (n): {class_metrics['support']}\n")
                f.write(f"True Positives: {class_metrics['TP']}, False Positives: {class_metrics['FP']}\n")
                f.write(f"False Negatives: {class_metrics['FN']}, True Negatives: {class_metrics['TN']}\n\n")
                
                f.write(f"Sensitivity (Recall): {class_metrics['sensitivity']:.4f} ")
                f.write(f"(95% CI: [{class_metrics['sensitivity_ci'][0]:.4f}, {class_metrics['sensitivity_ci'][1]:.4f}])\n")
                
                f.write(f"Specificity: {class_metrics['specificity']:.4f} ")
                f.write(f"(95% CI: [{class_metrics['specificity_ci'][0]:.4f}, {class_metrics['specificity_ci'][1]:.4f}])\n")
                
                f.write(f"PPV (Precision): {class_metrics['ppv']:.4f} ")
                f.write(f"(95% CI: [{class_metrics['ppv_ci'][0]:.4f}, {class_metrics['ppv_ci'][1]:.4f}])\n")
                
                f.write(f"NPV: {class_metrics['npv']:.4f} ")
                f.write(f"(95% CI: [{class_metrics['npv_ci'][0]:.4f}, {class_metrics['npv_ci'][1]:.4f}])\n")
                
                f.write(f"F1-Score: {class_metrics['f1_score']:.4f}\n")
                f.write(f"ROC-AUC: {class_metrics['roc_auc']:.4f}\n\n")
            
            # Clinical interpretation
            f.write("=" * 80 + "\n")
            f.write("CLINICAL INTERPRETATION\n")
            f.write("-" * 80 + "\n")
            f.write("Sensitivity: Proportion of true positive cases correctly identified\n")
            f.write("Specificity: Proportion of true negative cases correctly identified\n")
            f.write("PPV: Probability that positive prediction is correct\n")
            f.write("NPV: Probability that negative prediction is correct\n")
            f.write("ROC-AUC: Area under ROC curve (0.5=random, 1.0=perfect)\n")
            f.write("=" * 80 + "\n")
        
        print(f"Clinical report saved to {save_path}")
        return save_path
    
    def analyze_prediction_confidence(self, y_pred_probs, save_path='confidence_distribution.png'):
        """Analyze model confidence distribution"""
        max_probs = np.max(y_pred_probs, axis=1)
        predictions = np.argmax(y_pred_probs, axis=1)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Overall confidence distribution
        axes[0, 0].hist(max_probs, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(max_probs), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(max_probs):.3f}')
        axes[0, 0].set_xlabel('Maximum Probability', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Overall Prediction Confidence Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Per-class confidence
        for i, class_name in enumerate(self.class_names):
            class_probs = max_probs[predictions == i]
            axes[0, 1].hist(class_probs, bins=30, alpha=0.5, label=class_name, edgecolor='black')
        
        axes[0, 1].set_xlabel('Maximum Probability', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Confidence Distribution by Predicted Class', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Confidence thresholds analysis
        thresholds = np.arange(0.5, 1.0, 0.05)
        percentages = [np.mean(max_probs >= t) * 100 for t in thresholds]
        
        axes[1, 0].plot(thresholds, percentages, marker='o', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Confidence Threshold', fontsize=11)
        axes[1, 0].set_ylabel('% Predictions Above Threshold', fontsize=11)
        axes[1, 0].set_title('Prediction Coverage vs Confidence Threshold', fontsize=12, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # Statistics box
        stats_text = f"""
        Confidence Statistics:
        
        Mean Confidence: {np.mean(max_probs):.4f}
        Median Confidence: {np.median(max_probs):.4f}
        Std Dev: {np.std(max_probs):.4f}
        
        Predictions > 0.90: {np.mean(max_probs > 0.90)*100:.1f}%
        Predictions > 0.95: {np.mean(max_probs > 0.95)*100:.1f}%
        Predictions > 0.99: {np.mean(max_probs > 0.99)*100:.1f}%
        
        Min Confidence: {np.min(max_probs):.4f}
        Max Confidence: {np.max(max_probs):.4f}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                       verticalalignment='center', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confidence analysis saved to {save_path}")


def run_clinical_validation(model, test_dataset, class_names, output_dir='validation_results'):
    """
    Run complete clinical validation pipeline
    
    Args:
        model: Trained TensorFlow model
        test_dataset: Test dataset
        class_names: List of class names
        output_dir: Directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    print("Generating predictions...")
    y_true_list = []
    y_pred_probs_list = []
    
    for images, labels in test_dataset:
        y_true_list.append(labels.numpy())
        y_pred_probs_list.append(model.predict(images, verbose=0))
    
    y_true = np.concatenate(y_true_list)
    y_pred_probs = np.concatenate(y_pred_probs_list)
    
    # Initialize validator
    validator = ClinicalValidator(class_names)
    
    # Calculate clinical metrics
    print("\nCalculating clinical metrics...")
    metrics = validator.calculate_clinical_metrics(y_true, y_pred_probs)
    
    # Generate visualizations
    print("\nGenerating ROC curves...")
    validator.plot_roc_curves(y_true, y_pred_probs, 
                             save_path=os.path.join(output_dir, 'roc_curves.png'))
    
    print("Generating Precision-Recall curves...")
    validator.plot_precision_recall_curves(y_true, y_pred_probs,
                                          save_path=os.path.join(output_dir, 'pr_curves.png'))
    
    print("Analyzing prediction confidence...")
    validator.analyze_prediction_confidence(y_pred_probs,
                                           save_path=os.path.join(output_dir, 'confidence_analysis.png'))
    
    # Generate report
    print("\nGenerating clinical report...")
    validator.generate_clinical_report(metrics,
                                       save_path=os.path.join(output_dir, 'clinical_report.txt'))
    
    print(f"\n✓ Clinical validation complete. Results saved to '{output_dir}/'")
    return metrics


if __name__ == "__main__":
    print("Clinical Validation Module")
    print("Import this module and use run_clinical_validation() function")
