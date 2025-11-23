"""
Benchmarking and performance analysis for clinical deployment
"""

import time
import numpy as np
import psutil
import json
from typing import Dict, List
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt


class PerformanceBenchmark:
    """Benchmark model performance for deployment"""
    
    def __init__(self, model, config_path: str = "config.json"):
        """
        Initialize benchmark
        
        Args:
            model: TensorFlow model
            config_path: Path to config file
        """
        self.model = model
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def benchmark_inference_time(self, n_samples: int = 100, 
                                 warmup: int = 10) -> Dict:
        """
        Benchmark inference time
        
        Args:
            n_samples: Number of samples to test
            warmup: Number of warmup iterations
        
        Returns:
            Dictionary with timing statistics
        """
        print(f"Benchmarking inference time ({n_samples} samples)...")
        
        # Create random test data
        input_shape = tuple(self.config['model']['input_shape'])
        test_data = np.random.randn(1, *input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(warmup):
            _ = self.model.predict(test_data, verbose=0)
        
        # Benchmark
        times = []
        for _ in range(n_samples):
            start = time.perf_counter()
            _ = self.model.predict(test_data, verbose=0)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        results = {
            'mean_ms': np.mean(times),
            'median_ms': np.median(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'throughput_per_second': 1000 / np.mean(times)
        }
        
        print(f"  Mean: {results['mean_ms']:.2f} ms")
        print(f"  Median: {results['median_ms']:.2f} ms")
        print(f"  P95: {results['p95_ms']:.2f} ms")
        print(f"  P99: {results['p99_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_per_second']:.2f} inferences/sec")
        
        return results
    
    def benchmark_batch_inference(self, batch_sizes: List[int] = [1, 4, 8, 16, 32]) -> Dict:
        """
        Benchmark inference with different batch sizes
        
        Args:
            batch_sizes: List of batch sizes to test
        
        Returns:
            Dictionary with results for each batch size
        """
        print("Benchmarking batch inference...")
        
        results = {}
        input_shape = tuple(self.config['model']['input_shape'])
        
        for batch_size in batch_sizes:
            test_data = np.random.randn(batch_size, *input_shape).astype(np.float32)
            
            # Warmup
            _ = self.model.predict(test_data, verbose=0)
            
            # Benchmark
            times = []
            for _ in range(20):
                start = time.perf_counter()
                _ = self.model.predict(test_data, verbose=0)
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            results[batch_size] = {
                'mean_ms': np.mean(times),
                'mean_ms_per_sample': np.mean(times) / batch_size,
                'throughput': (batch_size * 1000) / np.mean(times)
            }
            
            print(f"  Batch size {batch_size}: {results[batch_size]['mean_ms']:.2f} ms "
                  f"({results[batch_size]['mean_ms_per_sample']:.2f} ms/sample)")
        
        return results
    
    def measure_memory_usage(self) -> Dict:
        """
        Measure model memory usage
        
        Returns:
            Dictionary with memory statistics
        """
        print("Measuring memory usage...")
        
        # Model size
        model_size_mb = sum([tf.size(w).numpy() * w.dtype.size 
                            for w in self.model.weights]) / (1024**2)
        
        # Process memory
        process = psutil.Process()
        memory_info = process.memory_info()
        
        results = {
            'model_size_mb': model_size_mb,
            'rss_mb': memory_info.rss / (1024**2),  # Resident Set Size
            'vms_mb': memory_info.vms / (1024**2),  # Virtual Memory Size
        }
        
        print(f"  Model size: {results['model_size_mb']:.2f} MB")
        print(f"  Process RSS: {results['rss_mb']:.2f} MB")
        print(f"  Process VMS: {results['vms_mb']:.2f} MB")
        
        return results
    
    def quantify_uncertainty(self, test_dataset, n_samples: int = 100) -> Dict:
        """
        Quantify model uncertainty using Monte Carlo Dropout
        
        Args:
            test_dataset: Test dataset
            n_samples: Number of MC samples
        
        Returns:
            Dictionary with uncertainty metrics
        """
        print(f"Quantifying uncertainty (MC Dropout with {n_samples} samples)...")
        
        # Get a batch of test data
        for images, labels in test_dataset.take(1):
            test_images = images.numpy()
            test_labels = labels.numpy()
            break
        
        # Enable dropout during inference
        predictions = []
        for _ in range(n_samples):
            pred = self.model(test_images, training=True).numpy()
            predictions.append(pred)
        
        predictions = np.array(predictions)  # Shape: (n_samples, batch, n_classes)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Entropy as uncertainty measure
        entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=-1)
        
        results = {
            'mean_uncertainty': np.mean(std_pred),
            'max_uncertainty': np.max(std_pred),
            'mean_entropy': np.mean(entropy),
            'predictions_std': std_pred.tolist()
        }
        
        print(f"  Mean uncertainty (std): {results['mean_uncertainty']:.4f}")
        print(f"  Mean entropy: {results['mean_entropy']:.4f}")
        
        return results
    
    def generate_benchmark_report(self, output_path: str = "benchmark_report.json"):
        """
        Generate comprehensive benchmark report
        
        Args:
            output_path: Path to save the report
        """
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE BENCHMARK REPORT")
        print("="*60 + "\n")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_info': {
                'architecture': self.config['model']['architecture'],
                'version': self.config['model']['version'],
                'input_shape': self.config['model']['input_shape']
            }
        }
        
        # Inference time
        report['inference_time'] = self.benchmark_inference_time()
        
        # Batch inference
        report['batch_inference'] = self.benchmark_batch_inference()
        
        # Memory usage
        report['memory_usage'] = self.measure_memory_usage()
        
        # Clinical acceptability thresholds
        report['clinical_acceptability'] = {
            'target_inference_time_ms': 500,  # Target: < 500ms per scan
            'meets_target': report['inference_time']['p95_ms'] < 500,
            'target_throughput_per_sec': 2,  # Target: >= 2 scans/sec
            'meets_throughput': report['inference_time']['throughput_per_second'] >= 2
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Benchmark report saved to {output_path}")
        
        # Clinical assessment
        print("\n" + "="*60)
        print("CLINICAL DEPLOYMENT ASSESSMENT")
        print("="*60)
        
        if report['clinical_acceptability']['meets_target']:
            print("✓ Inference time meets clinical requirements (<500ms)")
        else:
            print("✗ Inference time DOES NOT meet clinical requirements")
            print(f"  Current P95: {report['inference_time']['p95_ms']:.2f} ms (target: <500ms)")
        
        if report['clinical_acceptability']['meets_throughput']:
            print("✓ Throughput meets clinical requirements (≥2 scans/sec)")
        else:
            print("✗ Throughput DOES NOT meet clinical requirements")
        
        return report
    
    def plot_inference_distribution(self, n_samples: int = 1000, 
                                   save_path: str = "inference_distribution.png"):
        """Plot inference time distribution"""
        input_shape = tuple(self.config['model']['input_shape'])
        test_data = np.random.randn(1, *input_shape).astype(np.float32)
        
        # Collect samples
        times = []
        for _ in range(n_samples):
            start = time.perf_counter()
            _ = self.model.predict(test_data, verbose=0)
            times.append((time.perf_counter() - start) * 1000)
        
        # Plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(times, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(times), color='red', linestyle='--', label=f'Mean: {np.mean(times):.2f}ms')
        plt.axvline(np.percentile(times, 95), color='orange', linestyle='--', 
                   label=f'P95: {np.percentile(times, 95):.2f}ms')
        plt.xlabel('Inference Time (ms)', fontsize=11)
        plt.ylabel('Frequency', fontsize=11)
        plt.title('Inference Time Distribution', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(times, vert=True)
        plt.ylabel('Inference Time (ms)', fontsize=11)
        plt.title('Inference Time Box Plot', fontsize=12, fontweight='bold')
        plt.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Inference distribution plot saved to {save_path}")


if __name__ == "__main__":
    print("Performance Benchmark Module")
    print("Example usage:")
    print("""
    from inference import ModelInference
    
    # Load model
    model_inference = ModelInference()
    
    # Create benchmark
    benchmark = PerformanceBenchmark(model_inference.model)
    
    # Run comprehensive benchmark
    report = benchmark.generate_benchmark_report()
    
    # Plot distributions
    benchmark.plot_inference_distribution()
    """)
