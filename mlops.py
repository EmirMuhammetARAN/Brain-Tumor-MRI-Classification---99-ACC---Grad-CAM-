"""
MLOps integration with experiment tracking and model versioning
"""

import mlflow
import mlflow.tensorflow
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from typing import Dict, Any


class MLOpsManager:
    """Manage ML experiments and model lifecycle"""
    
    def __init__(self, experiment_name: str = "brain-tumor-classification", 
                 tracking_uri: str = "mlruns"):
        """
        Initialize MLOps manager
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking server
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
    
    def start_run(self, run_name: str = None) -> str:
        """
        Start a new MLflow run
        
        Args:
            run_name: Optional name for the run
        
        Returns:
            Run ID
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        mlflow.start_run(run_name=run_name)
        return mlflow.active_run().info.run_id
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model: tf.keras.Model, model_name: str = "model"):
        """Log TensorFlow model"""
        mlflow.tensorflow.log_model(model, model_name)
    
    def log_artifacts(self, artifact_path: str):
        """Log artifacts (files, plots, etc.)"""
        mlflow.log_artifact(artifact_path)
    
    def log_clinical_metrics(self, metrics: Dict[str, Any]):
        """
        Log clinical validation metrics
        
        Args:
            metrics: Dictionary from ClinicalValidator
        """
        # Overall metrics
        mlflow.log_metric("accuracy", metrics['overall']['accuracy'])
        mlflow.log_metric("macro_auc", metrics['overall']['macro_auc'])
        
        # Per-class metrics
        for class_name, class_metrics in metrics['per_class'].items():
            prefix = class_name.lower().replace(" ", "_")
            mlflow.log_metric(f"{prefix}_sensitivity", class_metrics['sensitivity'])
            mlflow.log_metric(f"{prefix}_specificity", class_metrics['specificity'])
            mlflow.log_metric(f"{prefix}_ppv", class_metrics['ppv'])
            mlflow.log_metric(f"{prefix}_npv", class_metrics['npv'])
            mlflow.log_metric(f"{prefix}_f1", class_metrics['f1_score'])
            mlflow.log_metric(f"{prefix}_auc", class_metrics['roc_auc'])
    
    def register_model(self, model_uri: str, model_name: str = "brain-tumor-classifier"):
        """
        Register model in MLflow Model Registry
        
        Args:
            model_uri: URI of the model to register
            model_name: Name for the registered model
        
        Returns:
            ModelVersion object
        """
        return mlflow.register_model(model_uri, model_name)
    
    def transition_model_stage(self, model_name: str, version: int, stage: str):
        """
        Transition model to different stage (Staging, Production, Archived)
        
        Args:
            model_name: Name of the registered model
            version: Model version number
            stage: Target stage (Staging, Production, Archived)
        """
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
    
    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()
    
    def log_training_config(self, config_path: str = "config.json"):
        """Log training configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Log model params
        mlflow.log_params({
            f"model_{k}": v for k, v in config['model'].items()
            if not isinstance(v, (list, dict))
        })
        
        # Log training params
        mlflow.log_params({
            f"train_{k}": v for k, v in config['training'].items()
        })
        
        # Log data params
        mlflow.log_params({
            f"data_{k}": v for k, v in config['data'].items()
            if not isinstance(v, (list, dict))
        })


class TrainingCallback(tf.keras.callbacks.Callback):
    """Keras callback for MLflow integration"""
    
    def __init__(self, mlops_manager: MLOpsManager, log_every_n_epochs: int = 1):
        super().__init__()
        self.mlops_manager = mlops_manager
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch"""
        if logs is not None and epoch % self.log_every_n_epochs == 0:
            self.mlops_manager.log_metrics(logs, step=epoch)


def setup_reproducibility(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    import os
    import random
    
    # Python
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # TensorFlow
    tf.random.set_seed(seed)
    
    # Environment variables for TensorFlow
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ Reproducibility setup complete with seed: {seed}")


def create_experiment_summary(output_path: str = "experiment_summary.md"):
    """
    Create a markdown summary of the experiment
    
    Args:
        output_path: Path to save the summary
    """
    client = mlflow.tracking.MlflowClient()
    
    # Get current experiment
    experiment = mlflow.get_experiment_by_name("brain-tumor-classification")
    
    if experiment is None:
        print("No experiment found")
        return
    
    # Get runs
    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=10)
    
    with open(output_path, 'w') as f:
        f.write("# Experiment Summary\n\n")
        f.write(f"Experiment: {experiment.name}\n")
        f.write(f"Experiment ID: {experiment.experiment_id}\n\n")
        
        f.write("## Recent Runs\n\n")
        f.write("| Run ID | Start Time | Accuracy | Macro AUC | Status |\n")
        f.write("|--------|-----------|----------|-----------|--------|\n")
        
        for run in runs:
            run_id = run.info.run_id[:8]
            start_time = datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M')
            accuracy = run.data.metrics.get('accuracy', 'N/A')
            macro_auc = run.data.metrics.get('macro_auc', 'N/A')
            status = run.info.status
            
            f.write(f"| {run_id} | {start_time} | {accuracy:.4f} | {macro_auc:.4f} | {status} |\n")
    
    print(f"Experiment summary saved to {output_path}")


# Weights & Biases integration (alternative to MLflow)
try:
    import wandb
    
    class WandbLogger:
        """Weights & Biases integration"""
        
        def __init__(self, project_name: str = "brain-tumor-classification", 
                     config: Dict = None):
            """
            Initialize W&B logger
            
            Args:
                project_name: W&B project name
                config: Configuration dictionary
            """
            wandb.init(project=project_name, config=config)
            self.run = wandb.run
        
        def log_metrics(self, metrics: Dict[str, float], step: int = None):
            """Log metrics to W&B"""
            wandb.log(metrics, step=step)
        
        def log_image(self, name: str, image: np.ndarray):
            """Log image to W&B"""
            wandb.log({name: wandb.Image(image)})
        
        def log_confusion_matrix(self, y_true, y_pred, class_names):
            """Log confusion matrix"""
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=y_pred,
                    class_names=class_names
                )
            })
        
        def finish(self):
            """Finish W&B run"""
            wandb.finish()
    
except ImportError:
    WandbLogger = None
    print("Weights & Biases not installed. Install with: pip install wandb")


if __name__ == "__main__":
    print("MLOps Module")
    print("Example usage:")
    print("""
    # Setup reproducibility
    setup_reproducibility(seed=42)
    
    # Initialize MLOps manager
    mlops = MLOpsManager()
    
    # Start run
    run_id = mlops.start_run("training_efficientnetb3")
    
    # Log config
    mlops.log_training_config()
    
    # During training
    callback = TrainingCallback(mlops)
    # model.fit(..., callbacks=[callback])
    
    # After validation
    # mlops.log_clinical_metrics(clinical_metrics)
    
    # Log model
    # mlops.log_model(model)
    
    # End run
    mlops.end_run()
    """)
