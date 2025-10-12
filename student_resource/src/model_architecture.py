"""
Multi-Modal Model Architecture for Amazon ML Challenge 2025
Combines text and image features for price prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class MultiModalNN(nn.Module):
    """
    Multi-modal neural network combining text and image features
    """
    
    def __init__(self, 
                 text_input_dim: int,
                 image_input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.3,
                 output_dim: int = 1):
        """
        Initialize multi-modal neural network
        
        Args:
            text_input_dim: Dimension of text features
            image_input_dim: Dimension of image features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate
            output_dim: Output dimension (1 for regression)
        """
        super(MultiModalNN, self).__init__()
        
        self.text_input_dim = text_input_dim
        self.image_input_dim = image_input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Text feature processing
        self.text_layers = nn.ModuleList()
        text_dims = [text_input_dim] + hidden_dims
        for i in range(len(text_dims) - 1):
            self.text_layers.append(nn.Linear(text_dims[i], text_dims[i + 1]))
            self.text_layers.append(nn.ReLU())
            self.text_layers.append(nn.Dropout(dropout_rate))
        
        # Image feature processing
        self.image_layers = nn.ModuleList()
        image_dims = [image_input_dim] + hidden_dims
        for i in range(len(image_dims) - 1):
            self.image_layers.append(nn.Linear(image_dims[i], image_dims[i + 1]))
            self.image_layers.append(nn.ReLU())
            self.image_layers.append(nn.Dropout(dropout_rate))
        
        # Fusion layer
        fusion_dim = hidden_dims[-1] * 2
        self.fusion_layers = nn.ModuleList()
        fusion_dims = [fusion_dim] + [dim // 2 for dim in hidden_dims]
        
        for i in range(len(fusion_dims) - 1):
            self.fusion_layers.append(nn.Linear(fusion_dims[i], fusion_dims[i + 1]))
            self.fusion_layers.append(nn.ReLU())
            self.fusion_layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.output_layer = nn.Linear(fusion_dims[-1], output_dim)
        
    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            text_features: Text feature tensor
            image_features: Image feature tensor
            
        Returns:
            Output tensor
        """
        # Process text features
        text_out = text_features
        for layer in self.text_layers:
            text_out = layer(text_out)
        
        # Process image features
        image_out = image_features
        for layer in self.image_layers:
            image_out = layer(image_out)
        
        # Fusion
        fused = torch.cat([text_out, image_out], dim=1)
        
        # Process fused features
        for layer in self.fusion_layers:
            fused = layer(fused)
        
        # Output
        output = self.output_layer(fused)
        
        return output

class EnsembleModel:
    """
    Ensemble model combining multiple algorithms
    """
    
    def __init__(self, 
                 models: Dict[str, any] = None,
                 weights: Dict[str, float] = None,
                 use_scaling: bool = True):
        """
        Initialize ensemble model
        
        Args:
            models: Dictionary of models to ensemble
            weights: Dictionary of model weights
            use_scaling: Whether to use feature scaling
        """
        self.models = models or self._get_default_models()
        self.weights = weights or {name: 1.0 for name in self.models.keys()}
        self.use_scaling = use_scaling
        self.scalers = {}
        self.is_fitted = False
        
    def _get_default_models(self) -> Dict[str, any]:
        """Get default set of models"""
        return {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleModel':
        """
        Fit all models
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Self
        """
        if self.use_scaling:
            # Fit scalers
            for model_name in self.models.keys():
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)
                self.scalers[model_name] = scaler
        else:
            X_scaled = X
        
        # Fit models
        for model_name, model in self.models.items():
            print(f"Fitting {model_name}...")
            if self.use_scaling:
                model.fit(self.scalers[model_name].transform(X), y)
            else:
                model.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using ensemble
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        for model_name, model in self.models.items():
            if self.use_scaling:
                X_scaled = self.scalers[model_name].transform(X)
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X)
            
            predictions.append(pred * self.weights[model_name])
        
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0) / sum(self.weights.values())
        
        return ensemble_pred
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance from models that support it
        
        Returns:
            Dictionary of feature importance arrays
        """
        importance = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance[model_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance[model_name] = np.abs(model.coef_)
        
        return importance

class PricePredictor:
    """
    Main price prediction class combining all components
    """
    
    def __init__(self, 
                 use_neural_network: bool = True,
                 use_ensemble: bool = True,
                 text_features_dim: int = 1000,
                 image_features_dim: int = 1000):
        """
        Initialize price predictor
        
        Args:
            use_neural_network: Whether to use neural network
            use_ensemble: Whether to use ensemble methods
            text_features_dim: Dimension of text features
            image_features_dim: Dimension of image features
        """
        self.use_neural_network = use_neural_network
        self.use_ensemble = use_ensemble
        self.text_features_dim = text_features_dim
        self.image_features_dim = image_features_dim
        
        # Initialize models
        self.neural_network = None
        self.ensemble_model = None
        self.is_fitted = False
        
        # Feature processors
        self.text_preprocessor = None
        self.image_processor = None
        
    def prepare_features(self, 
                        df: pd.DataFrame,
                        text_column: str = 'catalog_content',
                        image_column: str = 'image_link',
                        fit_preprocessors: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features from DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            image_column: Name of image column
            fit_preprocessors: Whether to fit preprocessors
            
        Returns:
            Tuple of (text_features, image_features)
        """
        from text_preprocessing import create_text_features
        from image_processing import create_image_features
        
        # Process text features
        print("Processing text features...")
        text_features, self.text_preprocessor = create_text_features(
            df, text_column, fit_preprocessors
        )
        
        # Process image features
        print("Processing image features...")
        image_features, self.image_processor = create_image_features(
            df, image_column, '../images'
        )
        
        return text_features, image_features
    
    def fit(self, 
            X_text: np.ndarray, 
            X_image: np.ndarray, 
            y: np.ndarray) -> 'PricePredictor':
        """
        Fit the price predictor
        
        Args:
            X_text: Text features
            X_image: Image features
            y: Target prices
            
        Returns:
            Self
        """
        # Combine features
        X_combined = np.hstack([X_text, X_image])
        
        print(f"Combined features shape: {X_combined.shape}")
        
        # Fit neural network
        if self.use_neural_network:
            print("Training neural network...")
            self.neural_network = MultiModalNN(
                text_input_dim=X_text.shape[1],
                image_input_dim=X_image.shape[1]
            )
            
            # Convert to tensors
            X_text_tensor = torch.FloatTensor(X_text)
            X_image_tensor = torch.FloatTensor(X_image)
            y_tensor = torch.FloatTensor(y).unsqueeze(1)
            
            # Training loop (simplified)
            optimizer = torch.optim.Adam(self.neural_network.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            self.neural_network.train()
            for epoch in range(100):  # Simplified training
                optimizer.zero_grad()
                outputs = self.neural_network(X_text_tensor, X_image_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 20 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Fit ensemble model
        if self.use_ensemble:
            print("Training ensemble model...")
            self.ensemble_model = EnsembleModel()
            self.ensemble_model.fit(X_combined, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X_text: np.ndarray, X_image: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X_text: Text features
            X_image: Image features
            
        Returns:
            Predicted prices
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        # Neural network prediction
        if self.use_neural_network and self.neural_network is not None:
            self.neural_network.eval()
            with torch.no_grad():
                X_text_tensor = torch.FloatTensor(X_text)
                X_image_tensor = torch.FloatTensor(X_image)
                nn_pred = self.neural_network(X_text_tensor, X_image_tensor)
                predictions.append(nn_pred.numpy().flatten())
        
        # Ensemble prediction
        if self.use_ensemble and self.ensemble_model is not None:
            X_combined = np.hstack([X_text, X_image])
            ensemble_pred = self.ensemble_model.predict(X_combined)
            predictions.append(ensemble_pred)
        
        # Combine predictions
        if len(predictions) > 1:
            final_pred = np.mean(predictions, axis=0)
        else:
            final_pred = predictions[0]
        
        return final_pred
    
    def evaluate(self, 
                 X_text: np.ndarray, 
                 X_image: np.ndarray, 
                 y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_text: Text features
            X_image: Image features
            y_true: True prices
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_text, X_image)
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate SMAPE
        smape = np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'SMAPE': smape
        }

def create_price_predictor(use_neural_network: bool = True,
                          use_ensemble: bool = True) -> PricePredictor:
    """
    Create a price predictor instance
    
    Args:
        use_neural_network: Whether to use neural network
        use_ensemble: Whether to use ensemble methods
        
    Returns:
        PricePredictor instance
    """
    return PricePredictor(
        use_neural_network=use_neural_network,
        use_ensemble=use_ensemble
    )

if __name__ == "__main__":
    # Example usage
    import os
    
    # Load sample data
    DATASET_FOLDER = '../dataset/'
    train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    
    # Create predictor
    predictor = create_price_predictor()
    
    # Prepare features (small sample for testing)
    print("Preparing features...")
    text_features, image_features = predictor.prepare_features(train.head(100))
    
    # Fit model
    print("Fitting model...")
    predictor.fit(
        text_features.values,
        image_features.values,
        train.head(100)['price'].values
    )
    
    # Make predictions
    print("Making predictions...")
    predictions = predictor.predict(
        text_features.values,
        image_features.values
    )
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")
