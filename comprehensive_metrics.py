"""
Comprehensive Model Validation with Cross-Validation and Per-Class Metrics
Addresses overfitting concerns and provides detailed clinical metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
from scipy import stats
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveValidator:
    """Complete validation suite with cross-validation and clinical metrics"""
    
    def __init__(self, class_names):
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.results = {}
        
    def cross_validation_analysis(self, X_data, y_data, model_builder, n_splits=5, epochs=30):
        """
        Perform k-fold cross-validation to assess overfitting
        
        Args:
            X_data: Input images
            y_data: Labels (one-hot or integer)
            model_builder: Function that returns a compiled model
            n_splits: Number of CV folds
            epochs: Training epochs per fold
            
        Returns:
            Dictionary with CV results
        """
        print(f"\n{'='*60}")
        print(f"CROSS-VALIDATION ANALYSIS ({n_splits}-Fold)")
        print(f"{'='*60}")
        
        # Convert to integer labels if needed
        if len(y_data.shape) > 1:
            y_labels = np.argmax(y_data, axis=1)
        else:
            y_labels = y_data
            
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_results = {
            'fold_accuracies': [],
            'fold_precisions': [],
            'fold_recalls': [],
            'fold_f1_scores': [],
            'per_class_metrics': {class_name: {
                'precision': [], 'recall': [], 'f1': []
            } for class_name in self.class_names}
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_labels), 1):
            print(f"\n--- Fold {fold}/{n_splits} ---")
            
            # Split data
            X_train, X_val = X_data[train_idx], X_data[val_idx]
            y_train, y_val = y_data[train_idx], y_data[val_idx]
            
            # Build and train model
            model = model_builder()
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                verbose=0
            )
            
            # Evaluate
            y_pred_probs = model.predict(X_val, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_val_labels = np.argmax(y_val, axis=1) if len(y_val.shape) > 1 else y_val
            
            # Calculate metrics
            accuracy = np.mean(y_pred == y_val_labels)
            precision_macro = precision_score(y_val_labels, y_pred, average='macro')
            recall_macro = recall_score(y_val_labels, y_pred, average='macro')
            f1_macro = f1_score(y_val_labels, y_pred, average='macro')
            
            cv_results['fold_accuracies'].append(accuracy)
            cv_results['fold_precisions'].append(precision_macro)
            cv_results['fold_recalls'].append(recall_macro)
            cv_results['fold_f1_scores'].append(f1_macro)
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision_macro:.4f}")
            print(f"  Recall: {recall_macro:.4f}")
            print(f"  F1-Score: {f1_macro:.4f}")
            
            # Per-class metrics
            for i, class_name in enumerate(self.class_names):
                class_precision = precision_score(y_val_labels, y_pred, 
                                                 labels=[i], average='macro', zero_division=0)
                class_recall = recall_score(y_val_labels, y_pred, 
                                          labels=[i], average='macro', zero_division=0)
                class_f1 = f1_score(y_val_labels, y_pred, 
                                   labels=[i], average='macro', zero_division=0)
                
                cv_results['per_class_metrics'][class_name]['precision'].append(class_precision)
                cv_results['per_class_metrics'][class_name]['recall'].append(class_recall)
                cv_results['per_class_metrics'][class_name]['f1'].append(class_f1)
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
        
        # Summarize results
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"\nOverall Metrics (Mean ± Std):")
        print(f"  Accuracy:  {np.mean(cv_results['fold_accuracies']):.4f} ± {np.std(cv_results['fold_accuracies']):.4f}")
        print(f"  Precision: {np.mean(cv_results['fold_precisions']):.4f} ± {np.std(cv_results['fold_precisions']):.4f}")
        print(f"  Recall:    {np.mean(cv_results['fold_recalls']):.4f} ± {np.std(cv_results['fold_recalls']):.4f}")
        print(f"  F1-Score:  {np.mean(cv_results['fold_f1_scores']):.4f} ± {np.std(cv_results['fold_f1_scores']):.4f}")
        
        print(f"\nPer-Class Metrics (Mean ± Std):")
        for class_name in self.class_names:
            metrics = cv_results['per_class_metrics'][class_name]
            print(f"\n  {class_name}:")
            print(f"    Precision: {np.mean(metrics['precision']):.4f} ± {np.std(metrics['precision']):.4f}")
            print(f"    Recall:    {np.mean(metrics['recall']):.4f} ± {np.std(metrics['recall']):.4f}")
            print(f"    F1-Score:  {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}")
        
        # Overfitting assessment
        std_accuracy = np.std(cv_results['fold_accuracies'])
        print(f"\n{'='*60}")
        print("OVERFITTING ASSESSMENT")
        print(f"{'='*60}")
        print(f"Accuracy Std Dev: {std_accuracy:.4f}")
        if std_accuracy < 0.02:
            print("✓ LOW variance - Model generalizes well")
        elif std_accuracy < 0.05:
            print("⚠ MODERATE variance - Acceptable generalization")
        else:
            print("✗ HIGH variance - Potential overfitting detected")
        
        self.results['cross_validation'] = cv_results
        return cv_results
    
    def calculate_per_class_clinical_metrics(self, y_true, y_pred_probs, confidence_level=0.95):
        """
        Calculate comprehensive per-class clinical metrics with confidence intervals
        
        Args:
            y_true: True labels
            y_pred_probs: Predicted probabilities
            confidence_level: CI confidence level (default 95%)
            
        Returns:
            Dictionary with detailed per-class metrics
        """
        print(f"\n{'='*60}")
        print("PER-CLASS CLINICAL METRICS")
        print(f"{'='*60}")
        
        # Convert to integer labels if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        per_class_metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            print(f"\n{class_name.upper()}")
            print("-" * 40)
            
            # Binary classification for this class
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            # Confusion matrix elements
            TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            
            # Calculate metrics
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # PPV and NPV
            ppv = precision  # Same as precision
            npv = TN / (TN + FN) if (TN + FN) > 0 else 0
            
            # Confidence intervals (Wilson score)
            n_positive = TP + FN
            n_negative = TN + FP
            
            sens_ci = self._wilson_ci(TP, n_positive, confidence_level)
            spec_ci = self._wilson_ci(TN, n_negative, confidence_level)
            ppv_ci = self._wilson_ci(TP, TP + FP, confidence_level) if (TP + FP) > 0 else (0, 0)
            npv_ci = self._wilson_ci(TN, TN + FN, confidence_level) if (TN + FN) > 0 else (0, 0)
            
            # ROC-AUC
            try:
                roc_auc = roc_auc_score(y_true_binary, y_pred_probs[:, i])
            except:
                roc_auc = 0.5
            
            # Store metrics
            per_class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'sensitivity': recall,
                'specificity': specificity,
                'f1_score': f1,
                'ppv': ppv,
                'npv': npv,
                'roc_auc': roc_auc,
                'support': n_positive,
                'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN),
                'sensitivity_ci': sens_ci,
                'specificity_ci': spec_ci,
                'ppv_ci': ppv_ci,
                'npv_ci': npv_ci
            }
            
            # Print metrics
            print(f"  Support: {n_positive} samples")
            print(f"  Confusion Matrix: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
            print(f"\n  Classification Metrics:")
            print(f"    Precision:    {precision:.4f}")
            print(f"    Recall:       {recall:.4f}")
            print(f"    F1-Score:     {f1:.4f}")
            print(f"    ROC-AUC:      {roc_auc:.4f}")
            print(f"\n  Clinical Metrics:")
            print(f"    Sensitivity:  {recall:.4f} (95% CI: {sens_ci[0]:.4f}-{sens_ci[1]:.4f})")
            print(f"    Specificity:  {specificity:.4f} (95% CI: {spec_ci[0]:.4f}-{spec_ci[1]:.4f})")
            print(f"    PPV:          {ppv:.4f} (95% CI: {ppv_ci[0]:.4f}-{ppv_ci[1]:.4f})")
            print(f"    NPV:          {npv:.4f} (95% CI: {npv_ci[0]:.4f}-{npv_ci[1]:.4f})")
        
        self.results['per_class_metrics'] = per_class_metrics
        return per_class_metrics
    
    def _wilson_ci(self, successes, trials, confidence=0.95):
        """Calculate Wilson score confidence interval"""
        if trials == 0:
            return (0, 0)
        
        p = successes / trials
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        
        denominator = 1 + z**2 / trials
        centre = (p + z**2 / (2 * trials)) / denominator
        adjustment = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
        
        return (max(0, centre - adjustment), min(1, centre + adjustment))
    
    def plot_per_class_metrics(self, save_path='per_class_metrics.png'):
        """Create comprehensive visualization of per-class metrics"""
        if 'per_class_metrics' not in self.results:
            print("Error: No per-class metrics calculated yet")
            return
        
        metrics = self.results['per_class_metrics']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Per-Class Clinical Metrics', fontsize=16, fontweight='bold')
        
        classes = list(metrics.keys())
        
        # 1. Sensitivity and Specificity with CI
        ax1 = axes[0, 0]
        sens_values = [metrics[c]['sensitivity'] for c in classes]
        spec_values = [metrics[c]['specificity'] for c in classes]
        sens_cis = [metrics[c]['sensitivity_ci'] for c in classes]
        spec_cis = [metrics[c]['specificity_ci'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        ax1.bar(x - width/2, sens_values, width, label='Sensitivity', alpha=0.8, color='#2E86AB')
        ax1.bar(x + width/2, spec_values, width, label='Specificity', alpha=0.8, color='#A23B72')
        
        # Add error bars for CI
        sens_errors = [[sens_values[i] - sens_cis[i][0] for i in range(len(classes))],
                       [sens_cis[i][1] - sens_values[i] for i in range(len(classes))]]
        spec_errors = [[spec_values[i] - spec_cis[i][0] for i in range(len(classes))],
                       [spec_cis[i][1] - spec_values[i] for i in range(len(classes))]]
        
        ax1.errorbar(x - width/2, sens_values, yerr=sens_errors, fmt='none', 
                     ecolor='black', capsize=5, alpha=0.7)
        ax1.errorbar(x + width/2, spec_values, yerr=spec_errors, fmt='none', 
                     ecolor='black', capsize=5, alpha=0.7)
        
        ax1.set_xlabel('Tumor Type', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('Sensitivity & Specificity (with 95% CI)', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.1])
        
        # 2. Precision, Recall, F1-Score
        ax2 = axes[0, 1]
        precision_values = [metrics[c]['precision'] for c in classes]
        recall_values = [metrics[c]['recall'] for c in classes]
        f1_values = [metrics[c]['f1_score'] for c in classes]
        
        x_pos = np.arange(len(classes))
        width = 0.25
        
        ax2.bar(x_pos - width, precision_values, width, label='Precision', alpha=0.8, color='#F18F01')
        ax2.bar(x_pos, recall_values, width, label='Recall', alpha=0.8, color='#C73E1D')
        ax2.bar(x_pos + width, f1_values, width, label='F1-Score', alpha=0.8, color='#6A994E')
        
        ax2.set_xlabel('Tumor Type', fontweight='bold')
        ax2.set_ylabel('Score', fontweight='bold')
        ax2.set_title('Classification Metrics', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(classes, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim([0, 1.1])
        
        # 3. PPV and NPV with CI
        ax3 = axes[1, 0]
        ppv_values = [metrics[c]['ppv'] for c in classes]
        npv_values = [metrics[c]['npv'] for c in classes]
        ppv_cis = [metrics[c]['ppv_ci'] for c in classes]
        npv_cis = [metrics[c]['npv_ci'] for c in classes]
        
        ax3.bar(x - width/2, ppv_values, width, label='PPV', alpha=0.8, color='#06A77D')
        ax3.bar(x + width/2, npv_values, width, label='NPV', alpha=0.8, color='#D4A574')
        
        ppv_errors = [[ppv_values[i] - ppv_cis[i][0] for i in range(len(classes))],
                      [ppv_cis[i][1] - ppv_values[i] for i in range(len(classes))]]
        npv_errors = [[npv_values[i] - npv_cis[i][0] for i in range(len(classes))],
                      [npv_cis[i][1] - npv_values[i] for i in range(len(classes))]]
        
        ax3.errorbar(x - width/2, ppv_values, yerr=ppv_errors, fmt='none', 
                     ecolor='black', capsize=5, alpha=0.7)
        ax3.errorbar(x + width/2, npv_values, yerr=npv_errors, fmt='none', 
                     ecolor='black', capsize=5, alpha=0.7)
        
        ax3.set_xlabel('Tumor Type', fontweight='bold')
        ax3.set_ylabel('Score', fontweight='bold')
        ax3.set_title('Positive & Negative Predictive Values (with 95% CI)', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(classes, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim([0, 1.1])
        
        # 4. ROC-AUC
        ax4 = axes[1, 1]
        roc_values = [metrics[c]['roc_auc'] for c in classes]
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E']
        
        bars = ax4.bar(classes, roc_values, alpha=0.8, color=colors)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')
        
        ax4.set_xlabel('Tumor Type', fontweight='bold')
        ax4.set_ylabel('AUC Score', fontweight='bold')
        ax4.set_title('ROC-AUC per Class', fontweight='bold')
        ax4.set_xticklabels(classes, rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)
        ax4.set_ylim([0, 1.1])
        ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Classifier')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Per-class metrics plot saved to {save_path}")
        plt.show()
    
    def save_results_to_json(self, save_path='comprehensive_metrics.json'):
        """Save all results to JSON file"""
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        serializable_results = convert_to_serializable(self.results)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        print(f"\n✓ Results saved to {save_path}")


def main():
    """Example usage"""
    print("Comprehensive Model Validation Script")
    print("=" * 60)
    print("\nThis script provides:")
    print("  1. Cross-validation analysis to assess overfitting")
    print("  2. Per-class clinical metrics with confidence intervals")
    print("  3. Detailed sensitivity, specificity, PPV, NPV for each tumor type")
    print("\nUsage:")
    print("  from comprehensive_metrics import ComprehensiveValidator")
    print("  validator = ComprehensiveValidator(class_names)")
    print("  validator.cross_validation_analysis(X, y, model_builder)")
    print("  validator.calculate_per_class_clinical_metrics(y_true, y_pred_probs)")
    print("  validator.plot_per_class_metrics()")


if __name__ == "__main__":
    main()
