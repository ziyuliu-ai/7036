"""
Model Training Module for FinBERT Classification

This module provides a complete pipeline for fine-tuning FinBERT models on
stock sentiment classification tasks. Supports 3-class, 5-class, and regression tasks.

Usage Example:
    from model_training import ModelTrainer
    
    trainer = ModelTrainer(
        model_name="ProsusAI/finbert",
        num_labels=5,
        output_dir="./finbert_finetuned"
    )
    
    trainer.train(
        train_file="train_5class.csv",
        val_file="val_5class.csv"
    )
    
    trainer.evaluate(val_file="val_5class.csv")
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Tuple
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from transformers import (
    BertTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from scipy.stats import pearsonr


class ModelTrainer:
    """
    Fine-tune FinBERT models for sentiment classification and regression.
    
    Attributes:
        model_name: Hugging Face model identifier
        num_labels: Number of classification labels (None for regression)
        output_dir: Directory to save fine-tuned models
        tokenizer: BertTokenizer instance
        model: Model instance (classification or regression)
        trainer: Trainer instance for training/evaluation
        is_regression: Whether this is a regression task
    """
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        num_labels: Optional[int] = 5,
        output_dir: str = "./finbert_finetuned",
        max_length: int = 512,
        device: str = "auto"
    ):
        """
        Initialize ModelTrainer.
        
        Args:
            model_name: Hugging Face model identifier
            num_labels: Number of classification labels (None for regression, 3 or 5 for classification)
            output_dir: Directory to save fine-tuned models
            max_length: Maximum sequence length for tokenization
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = output_dir
        self.max_length = max_length
        self.device = device
        self.is_regression = num_labels is None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        if self.is_regression:
            # For regression: use num_labels=1
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1,
                ignore_mismatched_sizes=True
            )
        else:
            # For classification: use specified num_labels
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
        self.trainer = None
    
    def tokenize(self, batch: Dict) -> Dict:
        """
        Tokenize batch of text samples.
        
        Args:
            batch: Dictionary with "text" key
        
        Returns:
            Tokenized batch with input_ids, attention_mask, token_type_ids
        """
        return self.tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
    
    def load_dataset(self, train_file: str, val_file: str):
        """
        Load and tokenize datasets from CSV files.
        
        Args:
            train_file: Path to training CSV file
            val_file: Path to validation CSV file
        
        Returns:
            Tuple of (tokenized_train_dataset, tokenized_val_dataset)
        """
        print(f"[*] Loading dataset from {train_file} and {val_file}")
        
        # Load datasets
        dataset = load_dataset(
            "csv",
            data_files={"train": train_file, "validation": val_file}
        )
        
        # Tokenize
        print("[*] Tokenizing datasets...")
        tokenized_dataset = dataset.map(self.tokenize, batched=True)
        
        # Set PyTorch format
        tokenized_dataset.set_format(
            "torch",
            columns=["input_ids", "attention_mask", "label"]
        )
        
        return tokenized_dataset["train"], tokenized_dataset["validation"]
    
    def train(
        self,
        train_file: str,
        val_file: str,
        learning_rate: float = 2e-5,
        per_device_batch_size: int = 16,
        num_epochs: int = 3,
        weight_decay: float = 0.01,
        save_strategy: str = "epoch",
        eval_strategy: str = "epoch",
        logging_steps: int = 50,
        save_steps: int = 500,
        eval_steps: int = 500
    ):
        """
        Fine-tune model on training data (classification or regression).
        
        Args:
            train_file: Path to training CSV file
            val_file: Path to validation CSV file
            learning_rate: Learning rate for optimization
            per_device_batch_size: Batch size per device
            num_epochs: Number of training epochs
            weight_decay: L2 regularization weight
            save_strategy: Model saving strategy ("epoch", "steps", or "no")
            eval_strategy: Evaluation strategy ("epoch", "steps", or "no")
            logging_steps: Number of steps between logging
            save_steps: Number of steps between model saves
            eval_steps: Number of steps between evaluations
        """
        task_type = "REGRESSION" if self.is_regression else f"{self.num_labels}-CLASS"
        
        print("="*60)
        print(f"TRAINING FinBERT ({task_type})")
        print("="*60)
        
        # Load dataset
        train_dataset, val_dataset = self.load_dataset(train_file, val_file)
        
        print(f"\n[*] Train samples: {len(train_dataset)}")
        print(f"[*] Val samples: {len(val_dataset)}")
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            load_best_model_at_end=True,
        )
        
        # Create Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train
        print("\n[*] Starting training...")
        self.trainer.train()
        
        # Save model and tokenizer
        print(f"\n[✓] Saving model to {self.output_dir}")
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"[✓] Model saved to {self.output_dir}")
        
        return self.trainer
    
    def evaluate(
        self,
        val_file: str,
        batch_size: int = 16,
        show_confusion_matrix: bool = True,
        show_distribution: bool = True
    ):
        """
        Evaluate model on validation dataset (classification or regression).
        
        Args:
            val_file: Path to validation CSV file
            batch_size: Batch size for evaluation
            show_confusion_matrix: Whether to plot confusion matrix (classification only)
            show_distribution: Whether to plot label distribution (classification only)
        
        Returns:
            Dictionary with evaluation metrics and predictions
        """
        task_type = "REGRESSION" if self.is_regression else f"{self.num_labels}-CLASS"
        
        print("="*60)
        print(f"EVALUATING MODEL ({task_type})")
        print("="*60)
        
        # Load validation dataset
        dataset = load_dataset("csv", data_files={"validation": val_file})
        val_dataset = dataset["validation"]
        
        # Tokenize
        tokenized_val = val_dataset.map(self.tokenize, batched=True)
        tokenized_val.set_format(
            "torch",
            columns=["input_ids", "attention_mask", "label"]
        )
        
        # Create Trainer for evaluation
        training_args = TrainingArguments(
            output_dir="./tmp_eval",
            do_train=False,
            do_eval=True,
            per_device_eval_batch_size=batch_size,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=tokenized_val,
        )
        
        # Get predictions
        print("\n[*] Running predictions...")
        predictions = trainer.predict(tokenized_val)
        true_labels = predictions.label_ids
        
        if self.is_regression:
            # Regression: squeeze predictions to 1D
            pred_labels = predictions.predictions.squeeze()
            
            # Calculate metrics
            mse = np.mean((pred_labels - true_labels) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(pred_labels - true_labels))
            pearson_r, pearson_p = pearsonr(pred_labels, true_labels)
            
            print(f"\n[✓] Regression Metrics:")
            print(f"    MSE:  {mse:.6f}")
            print(f"    RMSE: {rmse:.6f}")
            print(f"    MAE:  {mae:.6f}")
            print(f"    Pearson r: {pearson_r:.4f} (p={pearson_p:.4e})")
            
            # Plot predictions vs true values
            plt.figure(figsize=(10, 6))
            plt.scatter(true_labels, pred_labels, alpha=0.5)
            plt.plot([true_labels.min(), true_labels.max()],
                    [true_labels.min(), true_labels.max()],
                    'r--', lw=2, label='Perfect Prediction')
            plt.xlabel("True Values")
            plt.ylabel("Predicted Values")
            plt.title(f"Regression Predictions (R={pearson_r:.4f})")
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            # Residuals plot
            residuals = true_labels - pred_labels
            plt.figure(figsize=(10, 5))
            plt.scatter(pred_labels, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--', lw=2)
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            plt.title("Residual Plot")
            plt.tight_layout()
            plt.show()
            
            return {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "predictions": pred_labels,
                "true_labels": true_labels,
            }
        else:
            # Classification: argmax to get class labels
            pred_labels = predictions.predictions.argmax(-1)
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, pred_labels)
            
            print(f"\n[✓] Accuracy: {accuracy:.4f}")
            
            # Print classification report
            print("\n" + "="*60)
            print("CLASSIFICATION REPORT")
            print("="*60)
            labels_range = list(range(self.num_labels))
            print(classification_report(
                true_labels,
                pred_labels,
                labels=labels_range,
                zero_division=0
            ))
            
            # Confusion matrix
            if show_confusion_matrix:
                print("\n[*] Plotting confusion matrix...")
                cm = confusion_matrix(
                    true_labels,
                    pred_labels,
                    labels=labels_range
                )
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=labels_range,
                    yticklabels=labels_range
                )
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"Confusion Matrix ({self.num_labels}-class)")
                plt.tight_layout()
                plt.show()
            
            # Label distribution
            if show_distribution:
                print("\n[*] Plotting predicted label distribution...")
                unique, counts = np.unique(pred_labels, return_counts=True)
                
                plt.figure(figsize=(8, 5))
                plt.bar(unique, counts, color="steelblue")
                plt.xlabel("Class")
                plt.ylabel("Count")
                plt.title("Predicted Label Distribution")
                plt.xticks(labels_range)
                plt.tight_layout()
                plt.show()
            
            return {
                "accuracy": accuracy,
                "predictions": pred_labels,
                "true_labels": true_labels,
                "confusion_matrix": confusion_matrix(true_labels, pred_labels, labels=labels_range)
            }
    
    def load_pretrained(self, model_path: str):
        """
        Load a previously fine-tuned model.
        
        Args:
            model_path: Path to saved model directory
        """
        print(f"[*] Loading model from {model_path}")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )
        print(f"[✓] Model loaded successfully")


def train_all_models(
    labels_config: Dict = None,
    learning_rate: float = 2e-5,
    per_device_batch_size: int = 16,
    num_epochs: int = 3
):
    """
    Convenience function to train all model types at once.
    
    Args:
        labels_config: Dictionary with label configs (default: 3, 5 classes, and regression)
        learning_rate: Learning rate for all models
        per_device_batch_size: Batch size for all models
        num_epochs: Number of epochs for all models
    
    Returns:
        Dictionary with all trained models
    """
    if labels_config is None:
        labels_config = {
            "5class": {"num_labels": 5, "train_file": "train_5class.csv", "val_file": "val_5class.csv"},
            "3class": {"num_labels": 3, "train_file": "train_3class.csv", "val_file": "val_3class.csv"},
            "timeseries_5class": {"num_labels": 5, "train_file": "train_timeseries_5class.csv", "val_file": "val_timeseries_5class.csv"},
            "timeseries_3class": {"num_labels": 3, "train_file": "train_timeseries_3class.csv", "val_file": "val_timeseries_3class.csv"},
            "regression": {"num_labels": None, "train_file": "train_regression.csv", "val_file": "val_regression.csv"},
            "timeseries_regression": {"num_labels": None, "train_file": "train_timeseries_regression.csv", "val_file": "val_timeseries_regression.csv"},
        }
    
    results = {}
    
    for config_name, config in labels_config.items():
        print(f"\n{'='*60}")
        print(f"TRAINING {config_name.upper()}")
        print(f"{'='*60}\n")
        
        output_dir = f"./finbert_finetuned_{config_name}"
        
        trainer = ModelTrainer(
            model_name="ProsusAI/finbert",
            num_labels=config["num_labels"],
            output_dir=output_dir
        )
        
        trainer.train(
            train_file=config["train_file"],
            val_file=config["val_file"],
            learning_rate=learning_rate,
            per_device_batch_size=per_device_batch_size,
            num_epochs=num_epochs
        )
        
        results[config_name] = {
            "trainer": trainer,
            "output_dir": output_dir
        }
    
    return results


def train_classification_models(
    learning_rate: float = 2e-5,
    per_device_batch_size: int = 16,
    num_epochs: int = 3
):
    """
    Train all classification models (3-class and 5-class, both stratified and timeseries).
    
    Args:
        learning_rate: Learning rate for all models
        per_device_batch_size: Batch size for all models
        num_epochs: Number of epochs for all models
    
    Returns:
        Dictionary with all trained classification models and their results
    """
    classification_config = {
        "5class": {"num_labels": 5, "train_file": "train_5class.csv", "val_file": "val_5class.csv"},
        "3class": {"num_labels": 3, "train_file": "train_3class.csv", "val_file": "val_3class.csv"},
        "timeseries_5class": {"num_labels": 5, "train_file": "train_timeseries_5class.csv", "val_file": "val_timeseries_5class.csv"},
        "timeseries_3class": {"num_labels": 3, "train_file": "train_timeseries_3class.csv", "val_file": "val_timeseries_3class.csv"},
    }
    
    results = {}
    
    for config_name, config in classification_config.items():
        print(f"\n{'='*60}")
        print(f"TRAINING {config_name.upper()} CLASSIFIER")
        print(f"{'='*60}\n")
        
        output_dir = f"./finbert_finetuned_{config_name}"
        
        trainer = ModelTrainer(
            model_name="ProsusAI/finbert",
            num_labels=config["num_labels"],
            output_dir=output_dir
        )
        
        trainer.train(
            train_file=config["train_file"],
            val_file=config["val_file"],
            learning_rate=learning_rate,
            per_device_batch_size=per_device_batch_size,
            num_epochs=num_epochs
        )
        
        # Evaluate
        print(f"\n[*] Evaluating {config_name}...")
        eval_results = trainer.evaluate(val_file=config["val_file"])
        
        results[config_name] = {
            "trainer": trainer,
            "output_dir": output_dir,
            "eval_results": eval_results
        }
    
    return results


def train_regression_models(
    learning_rate: float = 2e-5,
    per_device_batch_size: int = 16,
    num_epochs: int = 3
):
    """
    Train all regression models (stratified and timeseries).
    
    Args:
        learning_rate: Learning rate for all models
        per_device_batch_size: Batch size for all models
        num_epochs: Number of epochs for all models
    
    Returns:
        Dictionary with all trained regression models and their results
    """
    regression_config = {
        "regression": {"num_labels": None, "train_file": "train_regression.csv", "val_file": "val_regression.csv"},
        "timeseries_regression": {"num_labels": None, "train_file": "train_timeseries_regression.csv", "val_file": "val_timeseries_regression.csv"},
    }
    
    results = {}
    
    for config_name, config in regression_config.items():
        print(f"\n{'='*60}")
        print(f"TRAINING {config_name.upper()}")
        print(f"{'='*60}\n")
        
        output_dir = f"./finbert_finetuned_{config_name}"
        
        trainer = ModelTrainer(
            model_name="ProsusAI/finbert",
            num_labels=config["num_labels"],
            output_dir=output_dir
        )
        
        trainer.train(
            train_file=config["train_file"],
            val_file=config["val_file"],
            learning_rate=learning_rate,
            per_device_batch_size=per_device_batch_size,
            num_epochs=num_epochs
        )
        
        # Evaluate
        print(f"\n[*] Evaluating {config_name}...")
        eval_results = trainer.evaluate(val_file=config["val_file"])
        
        results[config_name] = {
            "trainer": trainer,
            "output_dir": output_dir,
            "eval_results": eval_results
        }
    
    return results


if __name__ == "__main__":
    """
    Example usage when running directly:
    
    python model_training.py
    """
    
    print("Starting model training pipeline...\n")
    
    # Train all models
    results = train_all_models()
    
    print("\n" + "="*60)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY")
    print("="*60)
    print("\nTrained models saved to:")
    for config_name, result in results.items():
        print(f"  ✓ {result['output_dir']}")
