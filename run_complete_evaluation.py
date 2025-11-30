"""
Complete Model Evaluation Pipeline
Integrates all analyses: cross-validation, per-class metrics, confusion analysis, and clinical scenarios
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from datetime import datetime

# Import our custom analysis modules
from comprehensive_metrics import ComprehensiveValidator
from confusion_analysis import ConfusionAnalyzer
from dataset_statistics import DatasetAnalyzer


class CompleteEvaluationPipeline:
    """Run complete evaluation pipeline with all analyses"""
    
    def __init__(self, class_names, output_dir='evaluation_results'):
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize analyzers
        self.validator = ComprehensiveValidator(class_names)
        self.confusion_analyzer = ConfusionAnalyzer(class_names)
        self.dataset_analyzer = DatasetAnalyzer(class_names)
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'class_names': class_names
        }
    
    def run_complete_evaluation(self, y_true, y_pred_probs, 
                               train_labels=None, val_labels=None, test_labels=None):
        """
        Run complete evaluation pipeline
        
        Args:
            y_true: True test labels
            y_pred_probs: Predicted probabilities on test set
            train_labels, val_labels, test_labels: Labels for dataset analysis
            
        Returns:
            Dictionary with all results
        """
        print("\n" + "="*80)
        print("COMPLETE MODEL EVALUATION PIPELINE")
        print("="*80)
        
        # 1. Dataset Statistics
        if train_labels is not None or val_labels is not None or test_labels is not None:
            print("\n[1/4] Analyzing dataset statistics...")
            self.dataset_analyzer.analyze_dataset(
                train_labels=train_labels,
                val_labels=val_labels, 
                test_labels=test_labels
            )
            self.dataset_analyzer.plot_dataset_distribution(
                save_path=self.output_dir / 'dataset_distribution.png'
            )
            self.dataset_analyzer.generate_dataset_report(
                save_path=self.output_dir / 'dataset_report.txt'
            )
            self.results['dataset_stats'] = self.dataset_analyzer.stats
        
        # 2. Per-class Clinical Metrics
        print("\n[2/4] Calculating per-class clinical metrics...")
        per_class_metrics = self.validator.calculate_per_class_clinical_metrics(
            y_true, y_pred_probs
        )
        self.validator.plot_per_class_metrics(
            save_path=self.output_dir / 'per_class_metrics.png'
        )
        self.results['per_class_metrics'] = per_class_metrics
        
        # 3. Confusion Matrix Analysis
        print("\n[3/4] Analyzing confusion patterns...")
        confusion_results = self.confusion_analyzer.analyze_confusion_matrix(
            y_true, y_pred_probs
        )
        self.confusion_analyzer.plot_detailed_confusion_matrix(
            save_path=self.output_dir / 'detailed_confusion_matrix.png'
        )
        self.confusion_analyzer.plot_confusion_pairs_analysis(
            save_path=self.output_dir / 'confusion_pairs.png'
        )
        self.results['confusion_analysis'] = confusion_results
        
        # 4. Clinical Recommendations
        print("\n[4/4] Generating clinical recommendations...")
        self.confusion_analyzer.generate_clinical_recommendations()
        
        # Generate comprehensive report
        self._generate_comprehensive_report()
        
        # Save all results
        self._save_results()
        
        print("\n" + "="*80)
        print("✓ EVALUATION COMPLETE")
        print(f"  Results saved to: {self.output_dir}")
        print("="*80)
        
        return self.results
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive evaluation report"""
        report_path = self.output_dir / 'comprehensive_evaluation_report.txt'
        
        lines = []
        lines.append("="*80)
        lines.append("COMPREHENSIVE MODEL EVALUATION REPORT")
        lines.append("="*80)
        lines.append(f"Generated: {self.results['timestamp']}")
        lines.append("")
        
        # Executive Summary
        lines.append("="*80)
        lines.append("EXECUTIVE SUMMARY")
        lines.append("="*80)
        
        if 'per_class_metrics' in self.results:
            lines.append("\nPer-Class Performance:")
            for class_name, metrics in self.results['per_class_metrics'].items():
                lines.append(f"\n{class_name.upper()}:")
                lines.append(f"  Sensitivity:  {metrics['sensitivity']:.4f} "
                           f"(95% CI: {metrics['sensitivity_ci'][0]:.3f}-{metrics['sensitivity_ci'][1]:.3f})")
                lines.append(f"  Specificity:  {metrics['specificity']:.4f} "
                           f"(95% CI: {metrics['specificity_ci'][0]:.3f}-{metrics['specificity_ci'][1]:.3f})")
                lines.append(f"  F1-Score:     {metrics['f1_score']:.4f}")
                lines.append(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
                lines.append(f"  Support:      {metrics['support']} samples")
        
        # Dataset Information
        if 'dataset_stats' in self.results:
            lines.append("\n" + "="*80)
            lines.append("DATASET INFORMATION")
            lines.append("="*80)
            stats = self.results['dataset_stats']
            lines.append(f"\nTotal Samples: {stats.get('total_samples', 'N/A')}")
            lines.append(f"  Training:   {stats.get('train_samples', 0)} "
                       f"({stats.get('split_ratios', {}).get('train', 0)*100:.1f}%)")
            lines.append(f"  Validation: {stats.get('val_samples', 0)} "
                       f"({stats.get('split_ratios', {}).get('val', 0)*100:.1f}%)")
            lines.append(f"  Test:       {stats.get('test_samples', 0)} "
                       f"({stats.get('split_ratios', {}).get('test', 0)*100:.1f}%)")
            
            if 'class_balance' in stats:
                lines.append("\nClass Distribution:")
                total = stats['total_samples']
                for class_name, count in stats['class_balance'].items():
                    pct = (count / total * 100) if total > 0 else 0
                    lines.append(f"  {class_name:15} {count:6} ({pct:5.1f}%)")
        
        # Critical Findings
        lines.append("\n" + "="*80)
        lines.append("CRITICAL FINDINGS")
        lines.append("="*80)
        
        if 'confusion_analysis' in self.results:
            # High confidence errors
            high_conf_errors = self.results['confusion_analysis'].get('high_confidence_errors', [])
            lines.append(f"\nHigh-Confidence Errors: {len(high_conf_errors)}")
            if len(high_conf_errors) > 0:
                lines.append("⚠️  WARNING: Model is overconfident on some errors!")
                for error in high_conf_errors[:5]:
                    lines.append(f"  • {error['true_class']} → {error['predicted_class']} "
                               f"(confidence: {error['confidence']:.3f})")
            else:
                lines.append("✓ No high-confidence errors detected")
            
            # Most confused pairs
            confused_pairs = self.results['confusion_analysis'].get('confused_pairs', [])
            lines.append(f"\nTop 5 Confused Pairs:")
            for idx, pair in enumerate(confused_pairs[:5], 1):
                lines.append(f"\n{idx}. {pair['true_class']} → {pair['predicted_class']}")
                lines.append(f"   Count: {pair['count']} | Rate: {pair['confusion_rate']*100:.2f}%")
                lines.append(f"   Impact: {pair['clinical_impact']}")
        
        # Clinical Recommendations
        lines.append("\n" + "="*80)
        lines.append("CLINICAL DEPLOYMENT RECOMMENDATIONS")
        lines.append("="*80)
        
        lines.append("""
1. REGULATORY CONSIDERATIONS:
   • Model is for RESEARCH USE ONLY - not FDA/CE approved
   • Requires clinical validation study before deployment
   • Must be used under physician supervision
   
2. INTENDED USE:
   • Screening tool for brain tumor detection in MRI images
   • Assists radiologists in prioritizing cases for review
   • NOT for standalone diagnostic decisions
   
3. CONTRAINDICATIONS:
   • Do not use for emergency/critical care decisions
   • Not validated for pediatric cases (if applicable)
   • Not validated for post-treatment monitoring
   
4. REQUIRED SAFEGUARDS:
   • Mandatory radiologist review for all positive predictions
   • Double-reading protocol for low-confidence predictions (< 80%)
   • Regular performance monitoring and recalibration
   
5. QUALITY ASSURANCE:
   • Monthly audit of misclassified cases
   • Quarterly model performance review
   • Annual revalidation with new data
   
6. DOCUMENTATION REQUIREMENTS:
   • Log all AI predictions with confidence scores
   • Track false positive/negative rates
   • Document all cases where AI disagreed with clinician
        """)
        
        lines.append("\n" + "="*80)
        lines.append("END OF REPORT")
        lines.append("="*80)
        
        # Write report
        report_text = "\n".join(lines)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n✓ Comprehensive report saved to {report_path}")
    
    def _save_results(self):
        """Save results to JSON"""
        # Convert numpy types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        serializable_results = convert_to_serializable(self.results)
        
        json_path = self.output_dir / 'evaluation_results.json'
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        print(f"✓ Results JSON saved to {json_path}")


def main():
    """Example usage"""
    print("""
Complete Evaluation Pipeline
=============================

This pipeline integrates all evaluation components:

1. Dataset Statistics Analysis
   - Total samples and distribution
   - Train/val/test split analysis
   - Class balance assessment

2. Per-Class Clinical Metrics
   - Sensitivity, Specificity with 95% CI
   - PPV, NPV with confidence intervals
   - Precision, Recall, F1-Score
   - ROC-AUC per class

3. Confusion Matrix Analysis
   - Detailed confusion patterns
   - Misclassification breakdown
   - High-confidence error detection
   - Clinical impact assessment

4. Clinical Recommendations
   - Deployment guidelines
   - Safety considerations
   - Quality assurance protocols

Usage:
------
from run_complete_evaluation import CompleteEvaluationPipeline

# Initialize
pipeline = CompleteEvaluationPipeline(class_names)

# Run evaluation
results = pipeline.run_complete_evaluation(
    y_true=test_labels,
    y_pred_probs=predictions,
    train_labels=train_labels,
    val_labels=val_labels,
    test_labels=test_labels
)

All results will be saved to 'evaluation_results/' directory:
- Per-class metrics visualization
- Detailed confusion matrices
- Confusion pairs analysis
- Dataset distribution plots
- Comprehensive text report
- Complete JSON results
    """)


if __name__ == "__main__":
    main()
