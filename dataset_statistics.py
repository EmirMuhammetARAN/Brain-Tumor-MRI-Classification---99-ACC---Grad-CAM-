"""
Dataset Statistics and Analysis
Provides comprehensive overview of dataset composition and distribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class DatasetAnalyzer:
    """Analyze dataset statistics and distribution"""
    
    def __init__(self, class_names):
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.stats = {}
        
    def analyze_dataset(self, train_data=None, val_data=None, test_data=None,
                       train_labels=None, val_labels=None, test_labels=None):
        """
        Comprehensive dataset analysis
        
        Args:
            train_data, val_data, test_data: Image datasets (optional)
            train_labels, val_labels, test_labels: Corresponding labels
            
        Returns:
            Dictionary with dataset statistics
        """
        print(f"\n{'='*60}")
        print("DATASET STATISTICS ANALYSIS")
        print(f"{'='*60}")
        
        # Initialize stats
        stats = {
            'total_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'per_class_distribution': {},
            'class_balance': {},
            'split_ratios': {}
        }
        
        # Analyze each split
        splits = {
            'train': (train_labels, train_data),
            'val': (val_labels, val_data),
            'test': (test_labels, test_data)
        }
        
        total_samples = 0
        class_counts_total = Counter()
        
        for split_name, (labels, data) in splits.items():
            if labels is not None:
                # Convert to integer labels if needed
                if len(labels.shape) > 1:
                    labels = np.argmax(labels, axis=1)
                
                n_samples = len(labels)
                total_samples += n_samples
                stats[f'{split_name}_samples'] = n_samples
                
                # Count per class
                class_counts = Counter(labels)
                class_counts_total.update(class_counts)
                
                # Store per-class info
                stats['per_class_distribution'][split_name] = {
                    self.class_names[i]: class_counts.get(i, 0)
                    for i in range(self.n_classes)
                }
                
                print(f"\n{split_name.upper()} SET:")
                print(f"  Total samples: {n_samples}")
                print(f"  Distribution:")
                for i in range(self.n_classes):
                    count = class_counts.get(i, 0)
                    percentage = (count / n_samples * 100) if n_samples > 0 else 0
                    print(f"    {self.class_names[i]}: {count} ({percentage:.1f}%)")
        
        stats['total_samples'] = total_samples
        
        # Calculate split ratios
        if total_samples > 0:
            stats['split_ratios'] = {
                'train': stats['train_samples'] / total_samples,
                'val': stats['val_samples'] / total_samples,
                'test': stats['test_samples'] / total_samples
            }
        
        # Class balance analysis
        print(f"\n{'='*60}")
        print("CLASS BALANCE ANALYSIS")
        print(f"{'='*60}")
        
        total_per_class = {
            self.class_names[i]: class_counts_total.get(i, 0)
            for i in range(self.n_classes)
        }
        
        print(f"\nTotal samples across all splits: {total_samples}")
        print(f"\nPer-class distribution:")
        
        for class_name, count in total_per_class.items():
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Calculate imbalance ratio
        counts = list(total_per_class.values())
        if len(counts) > 0 and min(counts) > 0:
            imbalance_ratio = max(counts) / min(counts)
            print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio < 1.5:
                print("✓ Well-balanced dataset")
            elif imbalance_ratio < 3:
                print("⚠ Moderate imbalance - consider class weighting")
            else:
                print("✗ Significant imbalance - should use balancing techniques")
        
        stats['class_balance'] = total_per_class
        stats['imbalance_ratio'] = imbalance_ratio if len(counts) > 0 and min(counts) > 0 else None
        
        # Split ratio analysis
        print(f"\n{'='*60}")
        print("DATA SPLIT ANALYSIS")
        print(f"{'='*60}")
        
        if total_samples > 0:
            print(f"\nSplit ratios:")
            print(f"  Train: {stats['split_ratios']['train']*100:.1f}%")
            print(f"  Val:   {stats['split_ratios']['val']*100:.1f}%")
            print(f"  Test:  {stats['split_ratios']['test']*100:.1f}%")
            
            # Recommendations
            train_ratio = stats['split_ratios']['train']
            val_ratio = stats['split_ratios']['val']
            test_ratio = stats['split_ratios']['test']
            
            print(f"\nSplit Assessment:")
            if 0.6 <= train_ratio <= 0.8 and 0.1 <= val_ratio <= 0.2 and 0.1 <= test_ratio <= 0.3:
                print("✓ Standard split ratios (good)")
            else:
                print("⚠ Non-standard split ratios")
                if train_ratio < 0.6:
                    print("  → Train set might be too small")
                if val_ratio < 0.1:
                    print("  → Validation set might be too small")
                if test_ratio < 0.1:
                    print("  → Test set might be too small for reliable evaluation")
        
        self.stats = stats
        return stats
    
    def analyze_from_directory(self, data_dir):
        """
        Analyze dataset from directory structure
        Expected structure: data_dir/[split]/[class]/images
        
        Args:
            data_dir: Path to dataset directory
            
        Returns:
            Dictionary with dataset statistics
        """
        print(f"\n{'='*60}")
        print("ANALYZING DATASET FROM DIRECTORY")
        print(f"{'='*60}")
        
        data_path = Path(data_dir)
        
        if not data_path.exists():
            print(f"Error: Directory {data_dir} does not exist")
            return None
        
        stats = {
            'data_directory': str(data_path),
            'splits': {},
            'total_samples': 0,
            'class_distribution': {}
        }
        
        # Look for common split names
        split_names = ['Training', 'Validation', 'Testing', 'train', 'val', 'test']
        
        total_samples = 0
        class_totals = Counter()
        
        for split_name in split_names:
            split_path = data_path / split_name
            
            if split_path.exists():
                print(f"\nFound {split_name} directory")
                split_stats = {
                    'total': 0,
                    'classes': {}
                }
                
                # Count samples per class
                for class_name in self.class_names:
                    class_path = split_path / class_name
                    
                    if class_path.exists():
                        # Count image files
                        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
                        image_files = []
                        for ext in image_extensions:
                            image_files.extend(list(class_path.glob(f'*{ext}')))
                            image_files.extend(list(class_path.glob(f'*{ext.upper()}')))
                        
                        count = len(image_files)
                        split_stats['classes'][class_name] = count
                        split_stats['total'] += count
                        class_totals[class_name] += count
                        
                        print(f"  {class_name}: {count} images")
                
                total_samples += split_stats['total']
                stats['splits'][split_name] = split_stats
                print(f"  Total in {split_name}: {split_stats['total']}")
        
        stats['total_samples'] = total_samples
        stats['class_distribution'] = dict(class_totals)
        
        print(f"\n{'='*60}")
        print(f"TOTAL DATASET SIZE: {total_samples} images")
        print(f"{'='*60}")
        
        self.stats = stats
        return stats
    
    def plot_dataset_distribution(self, save_path='dataset_distribution.png'):
        """Create comprehensive visualization of dataset distribution"""
        if not self.stats:
            print("Error: No statistics calculated yet")
            return
        
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall class distribution
        ax1 = fig.add_subplot(gs[0, :2])
        
        if 'class_balance' in self.stats:
            classes = list(self.stats['class_balance'].keys())
            counts = list(self.stats['class_balance'].values())
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E']
            
            bars = ax1.bar(classes, counts, color=colors, alpha=0.7)
            
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(count)}\n({count/sum(counts)*100:.1f}%)',
                        ha='center', va='bottom', fontweight='bold')
            
            ax1.set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Tumor Type', fontweight='bold')
            ax1.set_ylabel('Number of Samples', fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
        
        # 2. Split distribution
        ax2 = fig.add_subplot(gs[0, 2])
        
        if 'split_ratios' in self.stats:
            splits = ['Train', 'Val', 'Test']
            sizes = [
                self.stats.get('train_samples', 0),
                self.stats.get('val_samples', 0),
                self.stats.get('test_samples', 0)
            ]
            colors_split = ['#2E86AB', '#F18F01', '#6A994E']
            
            wedges, texts, autotexts = ax2.pie(sizes, labels=splits, autopct='%1.1f%%',
                                                colors=colors_split, startangle=90)
            for text in texts:
                text.set_fontweight('bold')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax2.set_title('Data Split Distribution', fontsize=14, fontweight='bold')
        
        # 3. Per-class distribution across splits
        ax3 = fig.add_subplot(gs[1, :])
        
        if 'per_class_distribution' in self.stats:
            splits_data = self.stats['per_class_distribution']
            
            if len(splits_data) > 0:
                split_names = list(splits_data.keys())
                x = np.arange(len(self.class_names))
                width = 0.25
                
                colors_splits = ['#2E86AB', '#F18F01', '#6A994E']
                
                for i, (split_name, color) in enumerate(zip(split_names, colors_splits)):
                    if split_name in splits_data:
                        counts = [splits_data[split_name].get(class_name, 0) 
                                for class_name in self.class_names]
                        offset = (i - 1) * width
                        ax3.bar(x + offset, counts, width, label=split_name.capitalize(),
                               color=color, alpha=0.7)
                
                ax3.set_xlabel('Tumor Type', fontweight='bold', fontsize=12)
                ax3.set_ylabel('Number of Samples', fontweight='bold', fontsize=12)
                ax3.set_title('Class Distribution Across Data Splits', 
                            fontsize=14, fontweight='bold')
                ax3.set_xticks(x)
                ax3.set_xticklabels(self.class_names)
                ax3.legend()
                ax3.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Dataset Distribution Analysis', fontsize=18, fontweight='bold', y=0.995)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Dataset distribution plot saved to {save_path}")
        plt.show()
    
    def generate_dataset_report(self, save_path='dataset_report.txt'):
        """Generate comprehensive text report"""
        if not self.stats:
            print("Error: No statistics calculated yet")
            return
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("COMPREHENSIVE DATASET REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Total samples
        report_lines.append(f"TOTAL SAMPLES: {self.stats.get('total_samples', 0)}")
        report_lines.append("")
        
        # Split information
        report_lines.append("-" * 70)
        report_lines.append("DATA SPLITS")
        report_lines.append("-" * 70)
        report_lines.append(f"Training Set:   {self.stats.get('train_samples', 0)} samples")
        report_lines.append(f"Validation Set: {self.stats.get('val_samples', 0)} samples")
        report_lines.append(f"Test Set:       {self.stats.get('test_samples', 0)} samples")
        report_lines.append("")
        
        if 'split_ratios' in self.stats:
            report_lines.append("Split Ratios:")
            report_lines.append(f"  Train/Val/Test: "
                              f"{self.stats['split_ratios']['train']*100:.1f}% / "
                              f"{self.stats['split_ratios']['val']*100:.1f}% / "
                              f"{self.stats['split_ratios']['test']*100:.1f}%")
            report_lines.append("")
        
        # Class distribution
        report_lines.append("-" * 70)
        report_lines.append("CLASS DISTRIBUTION")
        report_lines.append("-" * 70)
        
        if 'class_balance' in self.stats:
            total = self.stats['total_samples']
            for class_name, count in self.stats['class_balance'].items():
                percentage = (count / total * 100) if total > 0 else 0
                report_lines.append(f"{class_name:15} {count:6} samples ({percentage:5.1f}%)")
            report_lines.append("")
            
            if 'imbalance_ratio' in self.stats and self.stats['imbalance_ratio']:
                report_lines.append(f"Imbalance Ratio: {self.stats['imbalance_ratio']:.2f}:1")
                report_lines.append("")
        
        # Per-split class distribution
        if 'per_class_distribution' in self.stats:
            report_lines.append("-" * 70)
            report_lines.append("PER-SPLIT CLASS DISTRIBUTION")
            report_lines.append("-" * 70)
            
            for split_name, distribution in self.stats['per_class_distribution'].items():
                report_lines.append(f"\n{split_name.upper()} SET:")
                for class_name, count in distribution.items():
                    split_total = sum(distribution.values())
                    percentage = (count / split_total * 100) if split_total > 0 else 0
                    report_lines.append(f"  {class_name:15} {count:6} samples ({percentage:5.1f}%)")
        
        report_lines.append("")
        report_lines.append("=" * 70)
        
        # Write to file
        report_text = "\n".join(report_lines)
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Dataset report saved to {save_path}")
        print("\n" + report_text)
    
    def save_stats_to_json(self, save_path='dataset_stats.json'):
        """Save statistics to JSON file"""
        if not self.stats:
            print("Error: No statistics calculated yet")
            return
        
        with open(save_path, 'w') as f:
            json.dump(self.stats, f, indent=4)
        
        print(f"\n✓ Dataset statistics saved to {save_path}")


def main():
    """Example usage"""
    print("Dataset Statistics Analysis Script")
    print("=" * 60)
    print("\nThis script provides:")
    print("  1. Total sample counts across all splits")
    print("  2. Per-class distribution analysis")
    print("  3. Class balance assessment")
    print("  4. Train/val/test split ratios")
    print("  5. Comprehensive visualizations")
    print("\nUsage:")
    print("  from dataset_statistics import DatasetAnalyzer")
    print("  analyzer = DatasetAnalyzer(class_names)")
    print("  analyzer.analyze_dataset(train_labels, val_labels, test_labels)")
    print("  # OR")
    print("  analyzer.analyze_from_directory('path/to/data')")
    print("  analyzer.plot_dataset_distribution()")
    print("  analyzer.generate_dataset_report()")


if __name__ == "__main__":
    main()
