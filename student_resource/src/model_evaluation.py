"""
Model Evaluation for Amazon ML Challenge 2025
Comprehensive evaluation using SMAPE and other metrics
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from text_preprocessing import create_text_features
from image_processing import create_image_features
from model_architecture import PricePredictor, create_price_predictor
from advanced_ensemble import create_advanced_ensemble

def smape_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate SMAPE score
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        SMAPE score
    """
    return 100 * np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'SMAPE': smape_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': 100 * np.mean(np.abs((y_true - y_pred) / y_true))
    }
    
    return metrics

class ModelEvaluator:
    """
    Comprehensive model evaluation class
    """
    
    def __init__(self, 
                 data_folder: str = '../dataset/',
                 output_folder: str = '../output/',
                 sample_size: int = 10000):
        """
        Initialize model evaluator
        
        Args:
            data_folder: Path to dataset folder
            output_folder: Path to output folder
            sample_size: Number of samples to use for evaluation
        """
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.sample_size = sample_size
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Initialize variables
        self.X_text = None
        self.X_image = None
        self.y = None
        self.results = {}
        
    def load_and_prepare_data(self):
        """
        Load and prepare data for evaluation
        """
        print("Loading and preparing data for evaluation...")
        
        # Load data
        train_path = os.path.join(self.data_folder, 'train.csv')
        train_data = pd.read_csv(train_path)
        
        # Use sample for evaluation
        if self.sample_size and len(train_data) > self.sample_size:
            train_data = train_data.sample(n=self.sample_size, random_state=42)
            print(f"Using {self.sample_size} samples for evaluation")
        
        # Preprocess data
        train_data = train_data.fillna('')
        train_data['price_log'] = np.log1p(train_data['price'])
        
        # Extract features
        print("Extracting text features...")
        text_features, _ = create_text_features(
            train_data, 'catalog_content', 
            fit_preprocessor=True, 
            max_features=3000,
            batch_size=1000
        )
        
        print("Extracting image features...")
        image_features, _ = create_image_features(
            train_data, 'image_link', '../images'
        )
        
        # Store data
        self.X_text = text_features.values
        self.X_image = image_features.values
        self.y = train_data['price_log'].values
        self.y_original = train_data['price'].values
        
        print(f"Data prepared: Text features {self.X_text.shape}, Image features {self.X_image.shape}")
        
    def evaluate_single_model(self, 
                            model: Any, 
                            model_name: str,
                            X_text: np.ndarray = None,
                            X_image: np.ndarray = None,
                            y: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate a single model
        
        Args:
            model: Model to evaluate
            model_name: Name of the model
            X_text: Text features
            X_image: Image features
            y: Target values
            
        Returns:
            Dictionary of metrics
        """
        if X_text is None:
            X_text = self.X_text
        if X_image is None:
            X_image = self.X_image
        if y is None:
            y = self.y
        
        print(f"Evaluating {model_name}...")
        
        try:
            # Make predictions
            if hasattr(model, 'predict') and len(model.predict.__code__.co_varnames) == 2:
                # Single input model
                X_combined = np.hstack([X_text, X_image])
                y_pred = model.predict(X_combined)
            else:
                # Multi-input model
                y_pred = model.predict(X_text, X_image)
            
            # Convert back from log if needed
            if hasattr(self, 'y_original'):
                y_pred_original = np.expm1(y_pred)
                y_true_original = self.y_original
            else:
                y_pred_original = y_pred
                y_true_original = y
            
            # Calculate metrics
            metrics = calculate_all_metrics(y_true_original, y_pred_original)
            
            print(f"{model_name} - SMAPE: {metrics['SMAPE']:.4f}%, MAE: {metrics['MAE']:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            return {'SMAPE': float('inf'), 'MAE': float('inf'), 'MSE': float('inf'), 
                   'RMSE': float('inf'), 'R2': -float('inf'), 'MAPE': float('inf')}
    
    def evaluate_multiple_models(self, models: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple models
        
        Args:
            models: Dictionary of models to evaluate
            
        Returns:
            Dictionary of results
        """
        print("Evaluating multiple models...")
        
        results = {}
        
        for name, model in models.items():
            results[name] = self.evaluate_single_model(model, name)
        
        return results
    
    def cross_validate_model(self, 
                           model: Any, 
                           model_name: str,
                           cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on a model
        
        Args:
            model: Model to validate
            model_name: Name of the model
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary of CV results
        """
        print(f"Cross-validating {model_name}...")
        
        try:
            # Combine features
            X_combined = np.hstack([self.X_text, self.X_image])
            
            # Perform cross-validation
            from sklearn.model_selection import KFold
            
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in kf.split(X_combined):
                X_train, X_val = X_combined[train_idx], X_combined[val_idx]
                y_train, y_val = self.y[train_idx], self.y[val_idx]
                
                # Train model
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train, y_train)
                
                # Predict
                y_pred = model_copy.predict(X_val)
                
                # Calculate SMAPE
                y_pred_original = np.expm1(y_pred)
                y_val_original = np.expm1(y_val)
                smape = smape_score(y_val_original, y_pred_original)
                cv_scores.append(smape)
            
            cv_results = {
                'mean_smape': np.mean(cv_scores),
                'std_smape': np.std(cv_scores),
                'min_smape': np.min(cv_scores),
                'max_smape': np.max(cv_scores)
            }
            
            print(f"{model_name} CV - Mean SMAPE: {cv_results['mean_smape']:.4f}% ± {cv_results['std_smape']:.4f}%")
            
            return cv_results
            
        except Exception as e:
            print(f"Error in cross-validation for {model_name}: {e}")
            return {'mean_smape': float('inf'), 'std_smape': 0, 'min_smape': float('inf'), 'max_smape': float('inf')}
    
    def plot_results(self, results: Dict[str, Dict[str, float]]):
        """
        Plot evaluation results
        
        Args:
            results: Dictionary of results
        """
        print("Creating evaluation plots...")
        
        # Create plots directory
        plots_dir = os.path.join(self.output_folder, 'evaluation_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Extract metrics
        models = list(results.keys())
        smape_scores = [results[model]['SMAPE'] for model in models]
        mae_scores = [results[model]['MAE'] for model in models]
        r2_scores = [results[model]['R2'] for model in models]
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # SMAPE comparison
        axes[0, 0].bar(models, smape_scores, color='skyblue')
        axes[0, 0].set_title('SMAPE Comparison')
        axes[0, 0].set_ylabel('SMAPE (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[0, 1].bar(models, mae_scores, color='lightgreen')
        axes[0, 1].set_title('MAE Comparison')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R2 comparison
        axes[1, 0].bar(models, r2_scores, color='lightcoral')
        axes[1, 0].set_title('R² Comparison')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Combined metrics
        x = np.arange(len(models))
        width = 0.25
        
        axes[1, 1].bar(x - width, smape_scores, width, label='SMAPE', color='skyblue')
        axes[1, 1].bar(x, mae_scores, width, label='MAE', color='lightgreen')
        axes[1, 1].bar(x + width, r2_scores, width, label='R²', color='lightcoral')
        
        axes[1, 1].set_title('Combined Metrics Comparison')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {plots_dir}")
    
    def generate_report(self, results: Dict[str, Dict[str, float]]):
        """
        Generate comprehensive evaluation report
        
        Args:
            results: Dictionary of results
        """
        print("Generating evaluation report...")
        
        # Create report
        report = {
            'evaluation_summary': {
                'total_models': len(results),
                'best_model': min(results.keys(), key=lambda x: results[x]['SMAPE']),
                'best_smape': min(results[model]['SMAPE'] for model in results),
                'evaluation_date': pd.Timestamp.now().isoformat()
            },
            'model_results': results
        }
        
        # Save report
        report_path = os.path.join(self.output_folder, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total models evaluated: {len(results)}")
        print(f"Best model: {report['evaluation_summary']['best_model']}")
        print(f"Best SMAPE: {report['evaluation_summary']['best_smape']:.4f}%")
        print("\nDetailed Results:")
        print("-" * 50)
        
        for model, metrics in results.items():
            print(f"\n{model}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        print(f"\nFull report saved to {report_path}")

def main():
    """
    Main evaluation pipeline
    """
    print("=== Amazon ML Challenge 2025 - Model Evaluation ===")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        data_folder='../dataset/',
        output_folder='../output/',
        sample_size=5000  # Use smaller sample for evaluation
    )
    
    # Load and prepare data
    evaluator.load_and_prepare_data()
    
    # Create models to evaluate
    models = {
        'Basic_Ensemble': create_price_predictor(use_neural_network=False, use_ensemble=True),
        'Advanced_Ensemble': create_advanced_ensemble(use_stacking=True, use_blending=True, use_neural_network=False),
        'Neural_Network': create_price_predictor(use_neural_network=True, use_ensemble=False)
    }
    
    # Train models
    print("Training models for evaluation...")
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(evaluator.X_text, evaluator.X_image, evaluator.y)
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    # Evaluate models
    results = evaluator.evaluate_multiple_models(models)
    
    # Generate plots
    evaluator.plot_results(results)
    
    # Generate report
    evaluator.generate_report(results)
    
    print("=== Model Evaluation Complete ===")

if __name__ == "__main__":
    main()
