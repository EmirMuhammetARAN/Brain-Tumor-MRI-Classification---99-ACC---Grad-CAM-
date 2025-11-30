"""
Detailed Confusion Matrix Analysis
Identifies which tumor types are confused and provides clinical insights
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class ConfusionAnalyzer:
    """Analyze confusion patterns and misclassifications"""
    
    def __init__(self, class_names):
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.confusion_results = {}
        
    def analyze_confusion_matrix(self, y_true, y_pred_probs, threshold=0.8):
        """
        Comprehensive confusion matrix analysis
        
        Args:
            y_true: True labels
            y_pred_probs: Predicted probabilities
            threshold: Confidence threshold for high-confidence errors
            
        Returns:
            Dictionary with confusion analysis
        """
        print(f"\n{'='*60}")
        print("CONFUSION MATRIX ANALYSIS")
        print(f"{'='*60}")
        
        # Convert to integer labels if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        confidence = np.max(y_pred_probs, axis=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix (row-wise)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Store results
        self.confusion_results['confusion_matrix'] = cm
        self.confusion_results['confusion_matrix_normalized'] = cm_normalized
        
        # Print confusion matrix
        print("\nConfusion Matrix (Counts):")
        print(self._format_confusion_matrix(cm))
        
        print("\nConfusion Matrix (Normalized - Row-wise %):")
        print(self._format_confusion_matrix(cm_normalized, percentage=True))
        
        # Analyze misclassifications
        misclassifications = self._analyze_misclassifications(y_true, y_pred, y_pred_probs)
        
        # Identify most confused pairs
        confused_pairs = self._identify_confused_pairs(cm, cm_normalized)
        
        # High confidence errors (dangerous!)
        high_conf_errors = self._find_high_confidence_errors(y_true, y_pred, confidence, threshold)
        
        # Store all results
        self.confusion_results['misclassifications'] = misclassifications
        self.confusion_results['confused_pairs'] = confused_pairs
        self.confusion_results['high_confidence_errors'] = high_conf_errors
        
        return self.confusion_results
    
    def _format_confusion_matrix(self, cm, percentage=False):
        """Format confusion matrix for printing"""
        df = pd.DataFrame(cm, 
                         index=[f"True: {c}" for c in self.class_names],
                         columns=[f"Pred: {c}" for c in self.class_names])
        
        if percentage:
            return df.applymap(lambda x: f"{x*100:.1f}%")
        else:
            return df.astype(int)
    
    def _analyze_misclassifications(self, y_true, y_pred, y_pred_probs):
        """Analyze all misclassifications"""
        print(f"\n{'='*60}")
        print("MISCLASSIFICATION ANALYSIS")
        print(f"{'='*60}")
        
        misclass_details = defaultdict(list)
        
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                true_class = self.class_names[y_true[i]]
                pred_class = self.class_names[y_pred[i]]
                confidence = y_pred_probs[i][y_pred[i]]
                
                pair = f"{true_class} → {pred_class}"
                misclass_details[pair].append({
                    'index': i,
                    'confidence': confidence,
                    'true_proba': y_pred_probs[i][y_true[i]],
                    'pred_proba': y_pred_probs[i][y_pred[i]]
                })
        
        # Print summary
        total_misclass = sum(len(v) for v in misclass_details.values())
        print(f"\nTotal Misclassifications: {total_misclass}")
        print(f"Total Samples: {len(y_true)}")
        print(f"Error Rate: {total_misclass/len(y_true)*100:.2f}%")
        
        print(f"\nMisclassification Breakdown:")
        for pair, errors in sorted(misclass_details.items(), key=lambda x: len(x[1]), reverse=True):
            avg_conf = np.mean([e['confidence'] for e in errors])
            print(f"  {pair}: {len(errors)} cases (avg confidence: {avg_conf:.3f})")
        
        return misclass_details
    
    def _identify_confused_pairs(self, cm, cm_normalized):
        """Identify which tumor types are most confused"""
        print(f"\n{'='*60}")
        print("MOST CONFUSED TUMOR PAIRS")
        print(f"{'='*60}")
        
        confused_pairs = []
        
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if i != j and cm[i, j] > 0:
                    confusion_rate = cm_normalized[i, j]
                    confused_pairs.append({
                        'true_class': self.class_names[i],
                        'predicted_class': self.class_names[j],
                        'count': int(cm[i, j]),
                        'confusion_rate': confusion_rate,
                        'clinical_impact': self._assess_clinical_impact(i, j)
                    })
        
        # Sort by confusion rate
        confused_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        print("\nTop Confusion Pairs:")
        for idx, pair in enumerate(confused_pairs[:10], 1):
            print(f"\n{idx}. {pair['true_class']} misclassified as {pair['predicted_class']}")
            print(f"   Count: {pair['count']}")
            print(f"   Rate: {pair['confusion_rate']*100:.2f}%")
            print(f"   Clinical Impact: {pair['clinical_impact']}")
        
        return confused_pairs
    
    def _assess_clinical_impact(self, true_idx, pred_idx):
        """Assess clinical impact of misclassification"""
        true_class = self.class_names[true_idx]
        pred_class = self.class_names[pred_idx]
        
        # Define clinical severity
        # Glioma and Meningioma are malignant, Pituitary is benign, No tumor is normal
        severity_map = {
            'glioma': 'high',
            'meningioma': 'high',
            'pituitary': 'medium',
            'notumor': 'none'
        }
        
        true_severity = severity_map.get(true_class.lower(), 'unknown')
        pred_severity = severity_map.get(pred_class.lower(), 'unknown')
        
        # Assess impact
        if true_class.lower() == 'notumor' and pred_class.lower() != 'notumor':
            return "FALSE POSITIVE - May cause unnecessary treatment/anxiety"
        elif true_class.lower() != 'notumor' and pred_class.lower() == 'notumor':
            return "FALSE NEGATIVE - CRITICAL! Missed tumor diagnosis"
        elif true_severity == 'high' and pred_severity == 'medium':
            return "UNDER-ESTIMATION - Malignant tumor classified as less severe"
        elif true_severity == 'medium' and pred_severity == 'high':
            return "OVER-ESTIMATION - May lead to aggressive treatment"
        else:
            return "TUMOR TYPE CONFUSION - Different treatment protocols"
    
    def _find_high_confidence_errors(self, y_true, y_pred, confidence, threshold):
        """Find high-confidence errors (most dangerous)"""
        print(f"\n{'='*60}")
        print(f"HIGH-CONFIDENCE ERRORS (confidence ≥ {threshold})")
        print(f"{'='*60}")
        
        high_conf_errors = []
        
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i] and confidence[i] >= threshold:
                high_conf_errors.append({
                    'index': i,
                    'true_class': self.class_names[y_true[i]],
                    'predicted_class': self.class_names[y_pred[i]],
                    'confidence': confidence[i]
                })
        
        print(f"\nFound {len(high_conf_errors)} high-confidence errors")
        print("⚠️  These are the MOST DANGEROUS errors - model is wrong but confident!\n")
        
        if len(high_conf_errors) > 0:
            for idx, error in enumerate(high_conf_errors[:10], 1):
                print(f"{idx}. Sample {error['index']}: "
                      f"{error['true_class']} → {error['predicted_class']} "
                      f"(confidence: {error['confidence']:.3f})")
        else:
            print("✓ No high-confidence errors found - good model calibration!")
        
        return high_conf_errors
    
    def plot_detailed_confusion_matrix(self, save_path='detailed_confusion_matrix.png'):
        """Create comprehensive confusion matrix visualizations"""
        if 'confusion_matrix' not in self.confusion_results:
            print("Error: Run analyze_confusion_matrix() first")
            return
        
        cm = self.confusion_results['confusion_matrix']
        cm_norm = self.confusion_results['confusion_matrix_normalized']
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        fig.suptitle('Comprehensive Confusion Matrix Analysis', 
                     fontsize=18, fontweight='bold', y=0.995)
        
        # 1. Absolute counts
        ax1 = axes[0, 0]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax1, cbar_kws={'label': 'Count'}, linewidths=0.5,
                   linecolor='gray')
        ax1.set_title('Confusion Matrix (Absolute Counts)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel('True Label', fontweight='bold')
        ax1.set_xlabel('Predicted Label', fontweight='bold')
        
        # 2. Normalized (percentage)
        ax2 = axes[0, 1]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='YlOrRd', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax2, cbar_kws={'label': 'Percentage'}, linewidths=0.5,
                   linecolor='gray')
        ax2.set_title('Confusion Matrix (Normalized - Row %)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylabel('True Label', fontweight='bold')
        ax2.set_xlabel('Predicted Label', fontweight='bold')
        
        # 3. Error matrix (only misclassifications)
        ax3 = axes[1, 0]
        error_matrix = cm.copy()
        np.fill_diagonal(error_matrix, 0)  # Remove correct predictions
        
        sns.heatmap(error_matrix, annot=True, fmt='d', cmap='Reds',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=ax3, cbar_kws={'label': 'Error Count'}, linewidths=0.5,
                   linecolor='gray')
        ax3.set_title('Error Matrix (Misclassifications Only)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax3.set_ylabel('True Label', fontweight='bold')
        ax3.set_xlabel('Predicted Label', fontweight='bold')
        
        # 4. Per-class error rates
        ax4 = axes[1, 1]
        
        # Calculate error rates for each class
        correct_counts = np.diag(cm)
        total_counts = cm.sum(axis=1)
        error_rates = 1 - (correct_counts / total_counts)
        accuracy_rates = correct_counts / total_counts
        
        x_pos = np.arange(len(self.class_names))
        bars = ax4.bar(x_pos, accuracy_rates * 100, alpha=0.7, color='#2E86AB')
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, accuracy_rates)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc*100:.1f}%\n({int(correct_counts[i])}/{int(total_counts[i])})',
                    ha='center', va='bottom', fontweight='bold')
        
        ax4.set_xlabel('Tumor Type', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
        ax4.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold', pad=15)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax4.set_ylim([0, 110])
        ax4.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect')
        ax4.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% threshold')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Detailed confusion matrix saved to {save_path}")
        plt.show()
    
    def plot_confusion_pairs_analysis(self, save_path='confusion_pairs.png'):
        """Visualize most confused pairs"""
        if 'confused_pairs' not in self.confusion_results:
            print("Error: Run analyze_confusion_matrix() first")
            return
        
        pairs = self.confusion_results['confused_pairs']
        
        if len(pairs) == 0:
            print("No confusion pairs to plot!")
            return
        
        # Take top 10 pairs
        top_pairs = sorted(pairs, key=lambda x: x['count'], reverse=True)[:10]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle('Most Confused Tumor Pairs', fontsize=16, fontweight='bold')
        
        # 1. By count
        pair_labels = [f"{p['true_class']} → {p['predicted_class']}" 
                      for p in top_pairs]
        counts = [p['count'] for p in top_pairs]
        
        colors_impact = []
        for p in top_pairs:
            if 'CRITICAL' in p['clinical_impact'] or 'FALSE NEGATIVE' in p['clinical_impact']:
                colors_impact.append('#D62828')  # Red for critical
            elif 'FALSE POSITIVE' in p['clinical_impact']:
                colors_impact.append('#F77F00')  # Orange for FP
            else:
                colors_impact.append('#003049')  # Blue for other
        
        bars1 = ax1.barh(pair_labels, counts, color=colors_impact, alpha=0.7)
        ax1.set_xlabel('Number of Misclassifications', fontweight='bold')
        ax1.set_title('Confusion Pairs by Count', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars1, counts):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2,
                    f' {int(count)}',
                    ha='left', va='center', fontweight='bold')
        
        # 2. By confusion rate
        rates = [p['confusion_rate'] * 100 for p in top_pairs]
        bars2 = ax2.barh(pair_labels, rates, color=colors_impact, alpha=0.7)
        ax2.set_xlabel('Confusion Rate (%)', fontweight='bold')
        ax2.set_title('Confusion Pairs by Rate', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars2, rates):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f' {rate:.1f}%',
                    ha='left', va='center', fontweight='bold')
        
        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#D62828', alpha=0.7, label='Critical (False Negative)'),
            Patch(facecolor='#F77F00', alpha=0.7, label='Warning (False Positive)'),
            Patch(facecolor='#003049', alpha=0.7, label='Other Confusion')
        ]
        fig.legend(handles=legend_elements, loc='lower center', 
                  ncol=3, bbox_to_anchor=(0.5, -0.05))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Confusion pairs analysis saved to {save_path}")
        plt.show()
    
    def generate_clinical_recommendations(self):
        """Generate clinical recommendations based on confusion analysis"""
        if 'confused_pairs' not in self.confusion_results:
            print("Error: Run analyze_confusion_matrix() first")
            return
        
        print(f"\n{'='*60}")
        print("CLINICAL RECOMMENDATIONS")
        print(f"{'='*60}")
        
        pairs = self.confusion_results['confused_pairs']
        
        # Categorize by clinical impact
        critical_pairs = [p for p in pairs if 'CRITICAL' in p['clinical_impact'] 
                         or 'FALSE NEGATIVE' in p['clinical_impact']]
        warning_pairs = [p for p in pairs if 'FALSE POSITIVE' in p['clinical_impact']]
        
        print("\n🚨 CRITICAL ISSUES (False Negatives - Missed Tumors):")
        if critical_pairs:
            for pair in critical_pairs:
                print(f"\n  • {pair['true_class']} misclassified as {pair['predicted_class']}")
                print(f"    Count: {pair['count']} | Rate: {pair['confusion_rate']*100:.2f}%")
                print(f"    Impact: {pair['clinical_impact']}")
                print(f"    ⚠️  Recommendation: ALWAYS require human radiologist review")
        else:
            print("  ✓ No critical false negatives detected")
        
        print("\n⚠️  WARNING ISSUES (False Positives):")
        if warning_pairs:
            for pair in warning_pairs:
                print(f"\n  • {pair['true_class']} misclassified as {pair['predicted_class']}")
                print(f"    Count: {pair['count']} | Rate: {pair['confusion_rate']*100:.2f}%")
                print(f"    Impact: {pair['clinical_impact']}")
                print(f"    Recommendation: Confirm with additional imaging/testing")
        else:
            print("  ✓ No significant false positives detected")
        
        # General recommendations
        print(f"\n{'='*60}")
        print("GENERAL CLINICAL USAGE RECOMMENDATIONS")
        print(f"{'='*60}")
        print("""
1. INTENDED USE:
   • This model should be used as a SCREENING TOOL only
   • NOT for definitive diagnosis
   • Always requires expert radiologist confirmation

2. HIGH-RISK SCENARIOS:
   • Any prediction with confidence < 80% → Mandatory review
   • All tumor-positive predictions → Confirm with MRI sequences
   • Borderline cases → Additional imaging modalities

3. QUALITY CONTROL:
   • Regular audit of misclassified cases
   • Continuous monitoring of confusion patterns
   • Update model with new data quarterly

4. CLINICAL WORKFLOW:
   • Use as first-pass screening
   • Flag suspicious cases for priority review
   • Track false negative rate monthly
   • Document all AI-assisted diagnoses
        """)


def main():
    """Example usage"""
    print("Confusion Matrix Analysis Script")
    print("=" * 60)
    print("\nThis script provides:")
    print("  1. Detailed confusion matrix with counts and percentages")
    print("  2. Identification of most confused tumor pairs")
    print("  3. High-confidence error detection")
    print("  4. Clinical impact assessment")
    print("  5. Clinical recommendations")
    print("\nUsage:")
    print("  from confusion_analysis import ConfusionAnalyzer")
    print("  analyzer = ConfusionAnalyzer(class_names)")
    print("  analyzer.analyze_confusion_matrix(y_true, y_pred_probs)")
    print("  analyzer.plot_detailed_confusion_matrix()")
    print("  analyzer.generate_clinical_recommendations()")


if __name__ == "__main__":
    main()
