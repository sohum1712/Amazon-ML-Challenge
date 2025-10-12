"""
Advanced Ensemble Methods for Amazon ML Challenge 2025
Implements stacking, blending, and advanced ensemble techniques
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from text_preprocessing import create_text_features
from image_processing import create_image_features
from model_architecture import MultiModalNN, EnsembleModel

class StackingEnsemble:
    """
    Advanced stacking ensemble with multiple levels
    """
    
    def __init__(self, 
                 base_models: List[Any],
                 meta_model: Any = None,
                 n_folds: int = 5,
                 use_probas: bool = False,
                 random_state: int = 42):
        """
        Initialize stacking ensemble
        
        Args:
            base_models: List of base models
            meta_model: Meta model for stacking
            n_folds: Number of folds for cross-validation
            use_probas: Whether to use probabilities
            random_state: Random state
        """
        self.base_models = base_models
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.n_folds = n_folds
        self.use_probas = use_probas
        self.random_state = random_state
        
        # Fitted models
        self.fitted_base_models = []
        self.fitted_meta_model = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Self
        """
        print("Fitting stacking ensemble...")
        
        # Initialize KFold
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Create out-of-fold predictions
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        # Initialize prediction matrix
        if self.use_probas:
            # For classification with probabilities
            n_classes = len(np.unique(y))
            oof_predictions = np.zeros((n_samples, n_models * n_classes))
        else:
            # For regression
            oof_predictions = np.zeros((n_samples, n_models))
        
        # Train base models with cross-validation
        for i, model in enumerate(self.base_models):
            print(f"Training base model {i+1}/{n_models}: {type(model).__name__}")
            
            # Store fitted model for final prediction
            fitted_model = type(model)(**model.get_params())
            fitted_model.fit(X, y)
            self.fitted_base_models.append(fitted_model)
            
            # Generate out-of-fold predictions
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train on fold
                fold_model = type(model)(**model.get_params())
                fold_model.fit(X_train, y_train)
                
                # Predict on validation set
                if self.use_probas:
                    val_pred = fold_model.predict_proba(X_val)
                    oof_predictions[val_idx, i*n_classes:(i+1)*n_classes] = val_pred
                else:
                    val_pred = fold_model.predict(X_val)
                    oof_predictions[val_idx, i] = val_pred
        
        # Train meta model on out-of-fold predictions
        print("Training meta model...")
        self.fitted_meta_model = type(self.meta_model)(**self.meta_model.get_params())
        self.fitted_meta_model.fit(oof_predictions, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using stacking ensemble
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Generate base model predictions
        n_samples = X.shape[0]
        n_models = len(self.fitted_base_models)
        
        if self.use_probas:
            n_classes = self.fitted_base_models[0].predict_proba(X).shape[1]
            base_predictions = np.zeros((n_samples, n_models * n_classes))
        else:
            base_predictions = np.zeros((n_samples, n_models))
        
        for i, model in enumerate(self.fitted_base_models):
            if self.use_probas:
                pred = model.predict_proba(X)
                base_predictions[:, i*n_classes:(i+1)*n_classes] = pred
            else:
                pred = model.predict(X)
                base_predictions[:, i] = pred
        
        # Meta model prediction
        meta_pred = self.fitted_meta_model.predict(base_predictions)
        
        return meta_pred

class BlendingEnsemble:
    """
    Blending ensemble with weighted combinations
    """
    
    def __init__(self, 
                 models: Dict[str, Any],
                 weights: Optional[Dict[str, float]] = None,
                 optimize_weights: bool = True):
        """
        Initialize blending ensemble
        
        Args:
            models: Dictionary of models
            weights: Model weights
            optimize_weights: Whether to optimize weights
        """
        self.models = models
        self.weights = weights or {name: 1.0 for name in models.keys()}
        self.optimize_weights = optimize_weights
        self.fitted_models = {}
        self.optimized_weights = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BlendingEnsemble':
        """
        Fit the blending ensemble
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Self
        """
        print("Fitting blending ensemble...")
        
        # Fit all models
        for name, model in self.models.items():
            print(f"Training {name}...")
            fitted_model = type(model)(**model.get_params())
            fitted_model.fit(X, y)
            self.fitted_models[name] = fitted_model
        
        # Optimize weights if requested
        if self.optimize_weights:
            self.optimized_weights = self._optimize_weights(X, y)
        else:
            self.optimized_weights = self.weights
        
        self.is_fitted = True
        return self
    
    def _optimize_weights(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Optimize model weights using validation
        """
        from sklearn.model_selection import train_test_split
        
        # Split data for weight optimization
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.fitted_models.items():
            predictions[name] = model.predict(X_val)
        
        # Optimize weights using linear regression
        from sklearn.linear_model import LinearRegression
        
        # Create prediction matrix
        pred_matrix = np.column_stack(list(predictions.values()))
        
        # Fit linear regression to find optimal weights
        lr = LinearRegression()
        lr.fit(pred_matrix, y_val)
        
        # Create weight dictionary
        optimized_weights = {}
        for i, name in enumerate(predictions.keys()):
            optimized_weights[name] = max(0, lr.coef_[i])  # Ensure non-negative
        
        # Normalize weights
        total_weight = sum(optimized_weights.values())
        if total_weight > 0:
            for name in optimized_weights:
                optimized_weights[name] /= total_weight
        
        print(f"Optimized weights: {optimized_weights}")
        return optimized_weights
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using blending ensemble
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.fitted_models.items():
            predictions[name] = model.predict(X)
        
        # Weighted combination
        final_pred = np.zeros(X.shape[0])
        for name, pred in predictions.items():
            weight = self.optimized_weights.get(name, 0)
            final_pred += weight * pred
        
        return final_pred

class AdvancedEnsemble:
    """
    Advanced ensemble combining multiple techniques
    """
    
    def __init__(self, 
                 use_stacking: bool = True,
                 use_blending: bool = True,
                 use_neural_network: bool = True,
                 n_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize advanced ensemble
        
        Args:
            use_stacking: Whether to use stacking
            use_blending: Whether to use blending
            use_neural_network: Whether to use neural network
            n_folds: Number of folds
            random_state: Random state
        """
        self.use_stacking = use_stacking
        self.use_blending = use_blending
        self.use_neural_network = use_neural_network
        self.n_folds = n_folds
        self.random_state = random_state
        
        # Initialize components
        self.stacking_ensemble = None
        self.blending_ensemble = None
        self.neural_network = None
        self.final_ensemble = None
        self.is_fitted = False
        
    def _create_base_models(self) -> List[Any]:
        """Create base models for ensemble"""
        models = [
            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
            Ridge(alpha=1.0),
            ElasticNet(alpha=0.1, l1_ratio=0.5),
            MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        ]
        
        # Add XGBoost if available
        try:
            models.append(xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1))
        except:
            pass
            
        # Add LightGBM if available
        try:
            models.append(lgb.LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, verbose=-1))
        except:
            pass
            
        return models
    
    def _create_meta_models(self) -> List[Any]:
        """Create meta models for stacking"""
        return [
            Ridge(alpha=1.0),
            Lasso(alpha=0.1),
            ElasticNet(alpha=0.1, l1_ratio=0.5),
            MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=300, random_state=42)
        ]
    
    def fit(self, X_text: np.ndarray, X_image: np.ndarray, y: np.ndarray) -> 'AdvancedEnsemble':
        """
        Fit the advanced ensemble
        
        Args:
            X_text: Text features
            X_image: Image features
            y: Target vector
            
        Returns:
            Self
        """
        print("Fitting advanced ensemble...")
        
        # Ensure inputs are numpy arrays and have consistent shapes
        X_text = np.asarray(X_text)
        X_image = np.asarray(X_image)
        y = np.asarray(y)
        
        # Ensure all arrays are 2D
        if X_text.ndim == 1:
            X_text = X_text.reshape(-1, 1)
        if X_image.ndim == 1:
            X_image = X_image.reshape(-1, 1)
        
        # Ensure all values are finite and convert to float
        X_text = np.nan_to_num(X_text, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        X_image = np.nan_to_num(X_image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        # Ensure all arrays have the same number of samples
        min_samples = min(len(X_text), len(X_image), len(y))
        X_text = X_text[:min_samples]
        X_image = X_image[:min_samples]
        y = y[:min_samples]
        
        print(f"Training with {min_samples} samples")
        print(f"Text features shape: {X_text.shape}")
        print(f"Image features shape: {X_image.shape}")
        
        # Combine features
        X_combined = np.hstack([X_text, X_image])
        print(f"Combined features shape: {X_combined.shape}")
        
        # Create base models
        base_models = self._create_base_models()
        
        # Fit stacking ensemble
        if self.use_stacking:
            print("Training stacking ensemble...")
            meta_models = self._create_meta_models()
            
            # Use best meta model (Ridge for simplicity)
            self.stacking_ensemble = StackingEnsemble(
                base_models=base_models,
                meta_model=Ridge(alpha=1.0),
                n_folds=self.n_folds,
                random_state=self.random_state
            )
            self.stacking_ensemble.fit(X_combined, y)
        
        # Fit blending ensemble
        if self.use_blending:
            print("Training blending ensemble...")
            model_dict = {f'model_{i}': model for i, model in enumerate(base_models)}
            
            self.blending_ensemble = BlendingEnsemble(
                models=model_dict,
                optimize_weights=True
            )
            self.blending_ensemble.fit(X_combined, y)
        
        # Fit neural network
        if self.use_neural_network:
            print("Training neural network...")
            self.neural_network = MultiModalNN(
                text_input_dim=X_text.shape[1],
                image_input_dim=X_image.shape[1],
                hidden_dims=[256, 128, 64],
                dropout_rate=0.3
            )
            
            # Convert to tensors
            import torch
            import torch.nn as nn
            import torch.optim as optim
            
            X_text_tensor = torch.FloatTensor(X_text.astype(np.float32))
            X_image_tensor = torch.FloatTensor(X_image.astype(np.float32))
            y_tensor = torch.FloatTensor(y.astype(np.float32)).unsqueeze(1)
            
            # Training
            optimizer = optim.Adam(self.neural_network.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            self.neural_network.train()
            for epoch in range(50):  # Reduced for speed
                optimizer.zero_grad()
                outputs = self.neural_network(X_text_tensor, X_image_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Create final ensemble
        self._create_final_ensemble()
        
        self.is_fitted = True
        return self
    
    def _create_final_ensemble(self):
        """Create final ensemble combining all methods"""
        components = []
        
        if self.stacking_ensemble:
            components.append(('stacking', self.stacking_ensemble))
        if self.blending_ensemble:
            components.append(('blending', self.blending_ensemble))
        if self.neural_network:
            components.append(('neural_network', self.neural_network))
        
        if len(components) > 1:
            # Create final blending ensemble
            model_dict = {name: model for name, model in components}
            self.final_ensemble = BlendingEnsemble(
                models=model_dict,
                optimize_weights=True
            )
        else:
            # Use single component
            self.final_ensemble = components[0][1]
    
    def predict(self, X_text: np.ndarray, X_image: np.ndarray) -> np.ndarray:
        """
        Make predictions using advanced ensemble
        
        Args:
            X_text: Text features
            X_image: Image features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Combine features
        X_combined = np.hstack([X_text, X_image])
        
        if self.final_ensemble:
            if hasattr(self.final_ensemble, 'predict'):
                return self.final_ensemble.predict(X_combined)
            else:
                # Handle neural network
                import torch
                self.neural_network.eval()
                with torch.no_grad():
                    X_text_tensor = torch.FloatTensor(X_text)
                    X_image_tensor = torch.FloatTensor(X_image)
                    outputs = self.neural_network(X_text_tensor, X_image_tensor)
                    return outputs.numpy().flatten()
        else:
            raise ValueError("No ensemble components available")

def create_advanced_ensemble(use_stacking: bool = True,
                           use_blending: bool = True,
                           use_neural_network: bool = True) -> AdvancedEnsemble:
    """
    Create advanced ensemble instance
    
    Args:
        use_stacking: Whether to use stacking
        use_blending: Whether to use blending
        use_neural_network: Whether to use neural network
        
    Returns:
        AdvancedEnsemble instance
    """
    return AdvancedEnsemble(
        use_stacking=use_stacking,
        use_blending=use_blending,
        use_neural_network=use_neural_network
    )

if __name__ == "__main__":
    # Example usage
    import os
    
    # Load sample data
    DATASET_FOLDER = '../dataset/'
    train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    
    # Create advanced ensemble
    ensemble = create_advanced_ensemble()
    
    # Prepare features (small sample for testing)
    print("Preparing features...")
    text_features, _ = create_text_features(train.head(100), 'catalog_content')
    image_features, _ = create_image_features(train.head(100), 'image_link')
    
    # Fit ensemble
    print("Fitting ensemble...")
    ensemble.fit(
        text_features.values,
        image_features.values,
        train.head(100)['price'].values
    )
    
    # Make predictions
    print("Making predictions...")
    predictions = ensemble.predict(
        text_features.values,
        image_features.values
    )
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")
