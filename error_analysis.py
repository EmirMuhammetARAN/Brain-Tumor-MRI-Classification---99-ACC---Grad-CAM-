"""
Error Analysis and Failure Case Detection
Critical for understanding model limitations and unsafe deployment scenarios
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from pathlib import Path
import tensorflow as tf


class ErrorAnalyzer:
    """Comprehensive error analysis for medical AI models"""
    
    def __init__(self, class_names):
        self.class_names = class_names
        self.n_classes = len(class_names)
    
    def identify_failure_cases(self, images, y_true, y_pred_probs, 
                               filenames=None, confidence_threshold=0.9):
        """
        Identify and categorize different types of failures
        
        Returns:
            Dictionary with different failure categories
        """
        y_pred = np.argmax(y_pred_probs, axis=1)
        max_confidence = np.max(y_pred_probs, axis=1)
        
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        
        # Categories of failures
        failures = {
            'high_confidence_errors': [],  # Wrong but confident - MOST DANGEROUS
            'low_confidence_errors': [],    # Wrong and unsure
            'borderline_correct': [],       # Correct but low confidence - needs review
            'confusion_pairs': {},          # Which classes get confused
        }
        
        for i in range(len(y_true)):
            true_class = y_true[i]
            pred_class = y_pred[i]
            confidence = max_confidence[i]
            
            is_correct = (true_class == pred_class)
            
            # High confidence errors (dangerous!)
            if not is_correct and confidence >= confidence_threshold:
                failures['high_confidence_errors'].append({
                    'index': i,
                    'true_class': self.class_names[true_class],
                    'pred_class': self.class_names[pred_class],
                    'confidence': confidence,
                    'filename': filenames[i] if filenames is not None else f"sample_{i}"
                })
            
            # Low confidence errors
            elif not is_correct and confidence < confidence_threshold:
                failures['low_confidence_errors'].append({
                    'index': i,
                    'true_class': self.class_names[true_class],
                    'pred_class': self.class_names[pred_class],
                    'confidence': confidence,
                    'filename': filenames[i] if filenames is not None else f"sample_{i}"
                })
            
            # Borderline correct (correct but uncertain)
            elif is_correct and confidence < 0.8:
                failures['borderline_correct'].append({
                    'index': i,
                    'true_class': self.class_names[true_class],
                    'confidence': confidence,
                    'filename': filenames[i] if filenames is not None else f"sample_{i}"
                })
            
            # Track confusion pairs
            if not is_correct:
                pair = f"{self.class_names[true_class]} → {self.class_names[pred_class]}"
                if pair not in failures['confusion_pairs']:
                    failures['confusion_pairs'][pair] = []
                failures['confusion_pairs'][pair].append(i)
        
        return failures
    
    def plot_confusion_analysis(self, y_true, y_pred_probs, save_path='confusion_analysis.png'):
        """Detailed confusion matrix analysis"""
        from sklearn.metrics import confusion_matrix
        
        y_pred = np.argmax(y_pred_probs, axis=1)
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_xlabel('Predicted Class', fontsize=12)
        axes[0].set_ylabel('True Class', fontsize=12)
        axes[0].set_title('Confusion Matrix - Absolute Counts', fontsize=13, fontweight='bold')
        
        # Normalized (percentage)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[1], cbar_kws={'label': 'Percentage'}, vmin=0, vmax=1)
        axes[1].set_xlabel('Predicted Class', fontsize=12)
        axes[1].set_ylabel('True Class', fontsize=12)
        axes[1].set_title('Confusion Matrix - Normalized by True Class', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion analysis saved to {save_path}")
    
    def visualize_failure_cases(self, images, failures, output_dir='failure_cases', 
                               max_samples=10):
        """Save visualizations of failure cases"""
        os.makedirs(output_dir, exist_ok=True)
        
        # High confidence errors (CRITICAL)
        if failures['high_confidence_errors']:
            print(f"\n🚨 CRITICAL: {len(failures['high_confidence_errors'])} high-confidence errors found!")
            high_conf_dir = os.path.join(output_dir, 'high_confidence_errors')
            os.makedirs(high_conf_dir, exist_ok=True)
            
            for idx, error in enumerate(failures['high_confidence_errors'][:max_samples]):
                img_idx = error['index']
                img = images[img_idx]
                
                # Denormalize if needed
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                
                plt.figure(figsize=(8, 6))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"TRUE: {error['true_class']} | PREDICTED: {error['pred_class']}\n"
                         f"Confidence: {error['confidence']:.4f} ⚠️ HIGH RISK ERROR",
                         fontsize=12, color='red', fontweight='bold')
                
                save_name = f"critical_{idx}_{error['filename']}.png"
                plt.savefig(os.path.join(high_conf_dir, save_name), dpi=150, bbox_inches='tight')
                plt.close()
        
        # Low confidence errors
        if failures['low_confidence_errors']:
            low_conf_dir = os.path.join(output_dir, 'low_confidence_errors')
            os.makedirs(low_conf_dir, exist_ok=True)
            
            for idx, error in enumerate(failures['low_confidence_errors'][:max_samples]):
                img_idx = error['index']
                img = images[img_idx]
                
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                
                plt.figure(figsize=(8, 6))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"TRUE: {error['true_class']} | PREDICTED: {error['pred_class']}\n"
                         f"Confidence: {error['confidence']:.4f} (Uncertain)",
                         fontsize=12, color='orange', fontweight='bold')
                
                save_name = f"uncertain_{idx}_{error['filename']}.png"
                plt.savefig(os.path.join(low_conf_dir, save_name), dpi=150, bbox_inches='tight')
                plt.close()
        
        print(f"Failure case visualizations saved to '{output_dir}/'")
    
    def generate_error_report(self, failures, metrics, save_path='error_analysis_report.txt'):
        """Generate detailed error analysis report"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ERROR ANALYSIS & FAILURE CASE REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # High confidence errors (MOST IMPORTANT)
            f.write("🚨 HIGH CONFIDENCE ERRORS (CRITICAL - FALSE SENSE OF SECURITY)\n")
            f.write("-" * 80 + "\n")
            f.write("These are cases where the model is WRONG but CONFIDENT.\n")
            f.write("Most dangerous for clinical deployment!\n\n")
            
            if failures['high_confidence_errors']:
                f.write(f"Count: {len(failures['high_confidence_errors'])}\n\n")
                for i, error in enumerate(failures['high_confidence_errors'][:20], 1):
                    f.write(f"{i}. {error['filename']}\n")
                    f.write(f"   True: {error['true_class']} → Predicted: {error['pred_class']}\n")
                    f.write(f"   Confidence: {error['confidence']:.4f}\n\n")
            else:
                f.write("✓ No high-confidence errors found.\n\n")
            
            # Low confidence errors
            f.write("\n" + "=" * 80 + "\n")
            f.write("⚠️  LOW CONFIDENCE ERRORS (Wrong and Uncertain)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Count: {len(failures['low_confidence_errors'])}\n")
            f.write("These cases may benefit from human review in production.\n\n")
            
            # Borderline correct
            f.write("\n" + "=" * 80 + "\n")
            f.write("🔍 BORDERLINE CORRECT PREDICTIONS\n")
            f.write("-" * 80 + "\n")
            f.write("Correct predictions but with low confidence - may need review.\n")
            f.write(f"Count: {len(failures['borderline_correct'])}\n\n")
            
            # Confusion pairs
            f.write("\n" + "=" * 80 + "\n")
            f.write("📊 CLASS CONFUSION ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write("Which classes get confused with each other:\n\n")
            
            sorted_pairs = sorted(failures['confusion_pairs'].items(), 
                                key=lambda x: len(x[1]), reverse=True)
            
            for pair, indices in sorted_pairs:
                f.write(f"{pair}: {len(indices)} cases\n")
            
            # Clinical implications
            f.write("\n" + "=" * 80 + "\n")
            f.write("🏥 CLINICAL IMPLICATIONS & RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n\n")
            
            total_errors = len(failures['high_confidence_errors']) + len(failures['low_confidence_errors'])
            
            if failures['high_confidence_errors']:
                f.write("⚠️  CRITICAL FINDINGS:\n")
                f.write(f"- {len(failures['high_confidence_errors'])} high-confidence misclassifications detected\n")
                f.write("- These represent the highest clinical risk\n")
                f.write("- RECOMMENDATION: Implement mandatory human review for all predictions\n\n")
            
            if len(failures['borderline_correct']) > 0:
                f.write("📋 UNCERTAINTY MANAGEMENT:\n")
                f.write(f"- {len(failures['borderline_correct'])} cases with low confidence despite being correct\n")
                f.write("- RECOMMENDATION: Set confidence threshold at 0.80 for auto-approval\n")
                f.write("- Cases below threshold should trigger radiologist review\n\n")
            
            # Model limitations
            f.write("\n" + "=" * 80 + "\n")
            f.write("⚠️  MODEL LIMITATIONS (Mandatory for FDA/CE Approval)\n")
            f.write("-" * 80 + "\n")
            f.write("1. This model is trained on specific MRI protocols and may not generalize\n")
            f.write("2. Performance may degrade with different imaging equipment or parameters\n")
            f.write("3. Not validated for pediatric cases or rare tumor subtypes\n")
            f.write("4. Should NOT be used as sole diagnostic criterion\n")
            f.write("5. Requires validation on institution-specific data before deployment\n")
            f.write("6. Not suitable for real-time emergency decision making\n")
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"Error analysis report saved to {save_path}")
    
    def analyze_misclassification_patterns(self, y_true, y_pred_probs, save_path='misclassification_patterns.png'):
        """Analyze patterns in misclassifications"""
        y_pred = np.argmax(y_pred_probs, axis=1)
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        
        max_confidence = np.max(y_pred_probs, axis=1)
        is_correct = (y_true == y_pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Error rate by confidence level
        confidence_bins = np.arange(0, 1.1, 0.1)
        bin_errors = []
        bin_counts = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (max_confidence >= confidence_bins[i]) & (max_confidence < confidence_bins[i+1])
            if mask.sum() > 0:
                bin_errors.append((~is_correct[mask]).mean() * 100)
                bin_counts.append(mask.sum())
            else:
                bin_errors.append(0)
                bin_counts.append(0)
        
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        axes[0, 0].bar(bin_centers, bin_errors, width=0.08, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Confidence Level', fontsize=11)
        axes[0, 0].set_ylabel('Error Rate (%)', fontsize=11)
        axes[0, 0].set_title('Error Rate vs Confidence Level', fontsize=12, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Per-class error rates
        class_error_rates = []
        for i in range(self.n_classes):
            class_mask = (y_true == i)
            if class_mask.sum() > 0:
                error_rate = (~is_correct[class_mask]).mean() * 100
                class_error_rates.append(error_rate)
            else:
                class_error_rates.append(0)
        
        axes[0, 1].bar(self.class_names, class_error_rates, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_ylabel('Error Rate (%)', fontsize=11)
        axes[0, 1].set_title('Error Rate by True Class', fontsize=12, fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Confidence distribution: correct vs incorrect
        axes[1, 0].hist([max_confidence[is_correct], max_confidence[~is_correct]], 
                       bins=30, label=['Correct', 'Incorrect'], 
                       color=['green', 'red'], alpha=0.6, edgecolor='black')
        axes[1, 0].set_xlabel('Confidence', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].set_title('Confidence Distribution: Correct vs Incorrect', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Error statistics table
        stats_text = f"""
        MISCLASSIFICATION STATISTICS
        
        Total Samples: {len(y_true)}
        Correct Predictions: {is_correct.sum()}
        Incorrect Predictions: {(~is_correct).sum()}
        
        Overall Error Rate: {(~is_correct).mean()*100:.2f}%
        
        High Confidence Errors (>0.9):
        {((~is_correct) & (max_confidence > 0.9)).sum()} cases
        
        Medium Confidence Errors (0.7-0.9):
        {((~is_correct) & (max_confidence >= 0.7) & (max_confidence <= 0.9)).sum()} cases
        
        Low Confidence Errors (<0.7):
        {((~is_correct) & (max_confidence < 0.7)).sum()} cases
        
        Avg Confidence (Correct): {max_confidence[is_correct].mean():.4f}
        Avg Confidence (Incorrect): {max_confidence[~is_correct].mean():.4f}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
                       verticalalignment='center', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Misclassification pattern analysis saved to {save_path}")


def run_error_analysis(model, test_dataset, class_names, output_dir='error_analysis'):
    """
    Run comprehensive error analysis
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        class_names: List of class names
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Collecting predictions for error analysis...")
    images_list = []
    y_true_list = []
    y_pred_probs_list = []
    
    for images, labels in test_dataset:
        images_list.append(images.numpy())
        y_true_list.append(labels.numpy())
        y_pred_probs_list.append(model.predict(images, verbose=0))
    
    images = np.concatenate(images_list)
    y_true = np.concatenate(y_true_list)
    y_pred_probs = np.concatenate(y_pred_probs_list)
    
    # Initialize analyzer
    analyzer = ErrorAnalyzer(class_names)
    
    # Identify failures
    print("\nIdentifying failure cases...")
    failures = analyzer.identify_failure_cases(images, y_true, y_pred_probs)
    
    print(f"\nFailure Analysis Summary:")
    print(f"  High-confidence errors: {len(failures['high_confidence_errors'])} 🚨")
    print(f"  Low-confidence errors: {len(failures['low_confidence_errors'])}")
    print(f"  Borderline correct: {len(failures['borderline_correct'])}")
    print(f"  Confusion pairs: {len(failures['confusion_pairs'])}")
    
    # Generate visualizations
    print("\nGenerating confusion analysis...")
    analyzer.plot_confusion_analysis(y_true, y_pred_probs,
                                    save_path=os.path.join(output_dir, 'confusion_analysis.png'))
    
    print("Analyzing misclassification patterns...")
    analyzer.analyze_misclassification_patterns(y_true, y_pred_probs,
                                               save_path=os.path.join(output_dir, 'misclassification_patterns.png'))
    
    print("Visualizing failure cases...")
    analyzer.visualize_failure_cases(images, failures,
                                    output_dir=os.path.join(output_dir, 'failure_cases'))
    
    # Generate report
    print("Generating error analysis report...")
    analyzer.generate_error_report(failures, None,
                                  save_path=os.path.join(output_dir, 'error_analysis_report.txt'))
    
    print(f"\n✓ Error analysis complete. Results saved to '{output_dir}/'")
    return failures


if __name__ == "__main__":
    print("Error Analysis Module")
    print("Import this module and use run_error_analysis() function")
