"""
Hyperparameter Tuning for Amazon ML Challenge 2025
Optimizes model parameters for best performance
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer
import optuna
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from text_preprocessing import create_text_features
from image_processing import create_image_features
from model_architecture import EnsembleModel, PricePredictor

def smape_scorer(y_true, y_pred):
    """
    SMAPE scorer for optimization
    """
    return 100 * np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

def create_smape_scorer():
    """Create SMAPE scorer for sklearn"""
    return make_scorer(smape_scorer, greater_is_better=False)

class HyperparameterTuner:
    """
    Comprehensive hyperparameter tuning class
    """
    
    def __init__(self, 
                 data_folder: str = '../dataset/',
                 output_folder: str = '../output/',
                 n_trials: int = 100,
                 cv_folds: int = 5):
        """
        Initialize hyperparameter tuner
        
        Args:
            data_folder: Path to dataset folder
            output_folder: Path to output folder
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
        """
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Initialize variables
        self.X_text = None
        self.X_image = None
        self.y = None
        self.best_params = {}
        self.best_score = float('inf')
        
    def load_and_prepare_data(self, sample_size: int = 10000):
        """
        Load and prepare data for tuning
        
        Args:
            sample_size: Number of samples to use for tuning
        """
        print("Loading and preparing data for hyperparameter tuning...")
        
        # Load data
        train_path = os.path.join(self.data_folder, 'train.csv')
        train_data = pd.read_csv(train_path)
        
        # Use sample for tuning
        if sample_size and len(train_data) > sample_size:
            train_data = train_data.sample(n=sample_size, random_state=42)
            print(f"Using {sample_size} samples for hyperparameter tuning")
        
        # Preprocess data
        train_data = train_data.fillna('')
        train_data['price_log'] = np.log1p(train_data['price'])
        
        # Extract features
        print("Extracting text features...")
        text_features, _ = create_text_features(
            train_data, 'catalog_content', 
            fit_preprocessor=True, 
            max_features=3000,  # Reduced for tuning
            batch_size=1000
        )
        
        print("Extracting image features...")
        image_features, _ = create_image_features(
            train_data, 'image_link', '../images'
        )
        
        # Combine features
        self.X_text = text_features.values
        self.X_image = image_features.values
        self.y = train_data['price_log'].values
        
        print(f"Data prepared: Text features {self.X_text.shape}, Image features {self.X_image.shape}")
        
    def tune_text_preprocessing(self) -> Dict[str, Any]:
        """
        Tune text preprocessing parameters
        """
        print("Tuning text preprocessing parameters...")
        
        def objective(trial):
            # Suggest parameters
            max_features = trial.suggest_int('max_features', 1000, 5000)
            min_df = trial.suggest_int('min_df', 1, 5)
            max_df = trial.suggest_float('max_df', 0.8, 0.99)
            ngram_range = trial.suggest_categorical('ngram_range', [(1, 1), (1, 2), (1, 3)])
            
            # Create preprocessor with suggested parameters
            from text_preprocessing import TextPreprocessor
            preprocessor = TextPreprocessor(
                max_features=max_features,
                remove_stopwords=True,
                use_stemming=True
            )
            
            # Update vectorizer parameters
            preprocessor.tfidf_vectorizer.set_params(
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                ngram_range=ngram_range
            )
            
            # Extract features
            try:
                text_features, _ = create_text_features(
                    pd.DataFrame({'catalog_content': ['sample text'] * 1000}),
                    'catalog_content',
                    fit_preprocessor=True,
                    max_features=max_features
                )
                
                # Simple evaluation (placeholder)
                return 0.0
            except:
                return float('inf')
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        
        return study.best_params
    
    def tune_ensemble_models(self) -> Dict[str, Any]:
        """
        Tune ensemble model parameters
        """
        print("Tuning ensemble model parameters...")
        
        def objective(trial):
            # Suggest parameters for different models
            rf_params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 50, 200),
                'max_depth': trial.suggest_int('rf_max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 5)
            }
            
            xgb_params = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 200),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0)
            }
            
            lgb_params = {
                'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 200),
                'max_depth': trial.suggest_int('lgb_max_depth', 3, 10),
                'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0)
            }
            
            # Create models with suggested parameters
            from sklearn.ensemble import RandomForestRegressor
            import xgboost as xgb
            import lightgbm as lgb
            
            models = {
                'random_forest': RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1),
                'xgboost': xgb.XGBRegressor(**xgb_params, random_state=42, n_jobs=-1),
                'lightgbm': lgb.LGBMRegressor(**lgb_params, random_state=42, n_jobs=-1, verbose=-1)
            }
            
            # Create ensemble
            ensemble = EnsembleModel(models=models)
            
            # Evaluate with cross-validation
            try:
                # Use smaller sample for evaluation
                sample_size = min(5000, len(self.X_text))
                X_sample = np.hstack([self.X_text[:sample_size], self.X_image[:sample_size]])
                y_sample = self.y[:sample_size]
                
                scores = cross_val_score(
                    ensemble, X_sample, y_sample,
                    cv=3, scoring=create_smape_scorer(), n_jobs=1
                )
                
                return scores.mean()
            except Exception as e:
                print(f"Error in trial: {e}")
                return float('inf')
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params
    
    def tune_neural_network(self) -> Dict[str, Any]:
        """
        Tune neural network parameters
        """
        print("Tuning neural network parameters...")
        
        def objective(trial):
            # Suggest architecture parameters
            hidden_dims = []
            n_layers = trial.suggest_int('n_layers', 2, 4)
            
            for i in range(n_layers):
                hidden_dims.append(trial.suggest_int(f'hidden_dim_{i}', 32, 256))
            
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2)
            
            # Create model
            from model_architecture import MultiModalNN
            import torch
            import torch.nn as nn
            import torch.optim as optim
            
            model = MultiModalNN(
                text_input_dim=self.X_text.shape[1],
                image_input_dim=self.X_image.shape[1],
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate
            )
            
            # Training setup
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            # Convert to tensors
            X_text_tensor = torch.FloatTensor(self.X_text[:1000])  # Use sample
            X_image_tensor = torch.FloatTensor(self.X_image[:1000])
            y_tensor = torch.FloatTensor(self.y[:1000]).unsqueeze(1)
            
            # Training loop
            model.train()
            for epoch in range(10):  # Reduced epochs for tuning
                optimizer.zero_grad()
                outputs = model(X_text_tensor, X_image_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                outputs = model(X_text_tensor, X_image_tensor)
                mse = criterion(outputs, y_tensor).item()
            
            return mse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        
        return study.best_params
    
    def tune_all_parameters(self) -> Dict[str, Any]:
        """
        Tune all parameters comprehensively
        """
        print("Starting comprehensive hyperparameter tuning...")
        
        # Load data
        self.load_and_prepare_data(sample_size=5000)
        
        # Tune different components
        text_params = self.tune_text_preprocessing()
        ensemble_params = self.tune_ensemble_models()
        nn_params = self.tune_neural_network()
        
        # Combine all parameters
        all_params = {
            'text_preprocessing': text_params,
            'ensemble_models': ensemble_params,
            'neural_network': nn_params
        }
        
        # Save parameters
        params_path = os.path.join(self.output_folder, 'best_hyperparameters.json')
        with open(params_path, 'w') as f:
            json.dump(all_params, f, indent=2)
        
        print(f"Best parameters saved to {params_path}")
        print("Best parameters:")
        for component, params in all_params.items():
            print(f"\n{component}:")
            for param, value in params.items():
                print(f"  {param}: {value}")
        
        return all_params
    
    def create_optimized_model(self, params: Dict[str, Any]) -> PricePredictor:
        """
        Create model with optimized parameters
        
        Args:
            params: Optimized parameters
            
        Returns:
            Optimized PricePredictor
        """
        print("Creating optimized model...")
        
        # Extract parameters
        text_params = params.get('text_preprocessing', {})
        ensemble_params = params.get('ensemble_models', {})
        nn_params = params.get('neural_network', {})
        
        # Create optimized models
        from sklearn.ensemble import RandomForestRegressor
        import xgboost as xgb
        import lightgbm as lgb
        
        # Extract model parameters
        rf_params = {k.replace('rf_', ''): v for k, v in ensemble_params.items() if k.startswith('rf_')}
        xgb_params = {k.replace('xgb_', ''): v for k, v in ensemble_params.items() if k.startswith('xgb_')}
        lgb_params = {k.replace('lgb_', ''): v for k, v in ensemble_params.items() if k.startswith('lgb_')}
        
        models = {
            'random_forest': RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1),
            'xgboost': xgb.XGBRegressor(**xgb_params, random_state=42, n_jobs=-1),
            'lightgbm': lgb.LGBMRegressor(**lgb_params, random_state=42, n_jobs=-1, verbose=-1)
        }
        
        # Create optimized predictor
        predictor = PricePredictor(
            use_neural_network=True,
            use_ensemble=True,
            text_features_dim=text_params.get('max_features', 3000),
            image_features_dim=1000
        )
        
        return predictor

def main():
    """
    Main hyperparameter tuning pipeline
    """
    print("=== Amazon ML Challenge 2025 - Hyperparameter Tuning ===")
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        data_folder='../dataset/',
        output_folder='../output/',
        n_trials=50,  # Reduced for faster execution
        cv_folds=3
    )
    
    # Run tuning
    best_params = tuner.tune_all_parameters()
    
    # Create optimized model
    optimized_model = tuner.create_optimized_model(best_params)
    
    print("=== Hyperparameter Tuning Complete ===")
    print(f"Best parameters: {best_params}")

if __name__ == "__main__":
    main()
