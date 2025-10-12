"""
Training and Evaluation Script for Amazon ML Challenge 2025
Main script for training, validating, and generating predictions
"""

import os
import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from text_preprocessing import create_text_features
from image_processing import create_image_features
from model_architecture import create_price_predictor, PricePredictor

class ModelTrainer:
    """
    Main training class for the price prediction model
    """
    
    def __init__(self, 
                 data_folder: str = '../dataset/',
                 output_folder: str = '../output/',
                 use_log_transform: bool = True,
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize model trainer
        
        Args:
            data_folder: Path to dataset folder
            output_folder: Path to output folder
            use_log_transform: Whether to use log transformation for prices
            test_size: Fraction of data to use for testing
            random_state: Random state for reproducibility
        """
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.use_log_transform = use_log_transform
        self.test_size = test_size
        self.random_state = random_state
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Initialize variables
        self.train_data = None
        self.test_data = None
        self.predictor = None
        self.text_features = None
        self.image_features = None
        self.feature_names = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data
        
        Returns:
            Tuple of (train_data, test_data)
        """
        print("Loading data...")
        
        # Load training data
        train_path = os.path.join(self.data_folder, 'train.csv')
        self.train_data = pd.read_csv(train_path)
        print(f"Training data shape: {self.train_data.shape}")
        
        # Load test data
        test_path = os.path.join(self.data_folder, 'test.csv')
        self.test_data = pd.read_csv(test_path)
        print(f"Test data shape: {self.test_data.shape}")
        
        # Basic data validation
        print(f"Training data columns: {self.train_data.columns.tolist()}")
        print(f"Test data columns: {self.test_data.columns.tolist()}")
        
        # Check for missing values
        print(f"Training data missing values: {self.train_data.isnull().sum().sum()}")
        print(f"Test data missing values: {self.test_data.isnull().sum().sum()}")
        
        return self.train_data, self.test_data
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the data
        
        Returns:
            Tuple of (processed_train_data, processed_test_data)
        """
        print("Preprocessing data...")
        
        # Handle missing values
        self.train_data = self.train_data.fillna('')
        self.test_data = self.test_data.fillna('')
        
        # Apply log transformation to prices if specified
        if self.use_log_transform:
            print("Applying log transformation to prices...")
            self.train_data['price_log'] = np.log1p(self.train_data['price'])
            print(f"Original price range: {self.train_data['price'].min():.2f} - {self.train_data['price'].max():.2f}")
            print(f"Log price range: {self.train_data['price_log'].min():.2f} - {self.train_data['price_log'].max():.2f}")
        
        return self.train_data, self.test_data
    
    def extract_features(self, 
                        sample_size: int = None,
                        use_cached: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract text and image features
        
        Args:
            sample_size: Number of samples to process (None for all)
            use_cached: Whether to use cached features if available
            
        Returns:
            Tuple of (text_features, image_features)
        """
        print("Extracting features...")
        
        # Use sample if specified
        train_sample = self.train_data.head(sample_size) if sample_size else self.train_data
        test_sample = self.test_data.head(sample_size) if sample_size else self.test_data
        
        # Check for cached features
        text_cache_path = os.path.join(self.output_folder, 'text_features.pkl')
        image_cache_path = os.path.join(self.output_folder, 'image_features.pkl')
        
        if use_cached and os.path.exists(text_cache_path) and os.path.exists(image_cache_path):
            print("Loading cached features...")
            with open(text_cache_path, 'rb') as f:
                self.text_features = pickle.load(f)
            with open(image_cache_path, 'rb') as f:
                self.image_features = pickle.load(f)
        else:
            # Extract text features
            print("Extracting text features...")
            text_features_train, text_preprocessor = create_text_features(
                train_sample, 'catalog_content', fit_preprocessor=True
            )
            text_features_test, _ = create_text_features(
                test_sample, 'catalog_content', fit_preprocessor=False
            )
            
            # Extract image features
            print("Extracting image features...")
            image_features_train, image_processor = create_image_features(
                train_sample, 'image_link', '../images'
            )
            image_features_test, _ = create_image_features(
                test_sample, 'image_link', '../images'
            )
            
            # Combine features
            self.text_features = pd.concat([text_features_train, text_features_test], ignore_index=True)
            self.image_features = pd.concat([image_features_train, image_features_test], ignore_index=True)
            
            # Save cached features
            print("Saving cached features...")
            with open(text_cache_path, 'wb') as f:
                pickle.dump(self.text_features, f)
            with open(image_cache_path, 'wb') as f:
                pickle.dump(self.image_features, f)
        
        print(f"Text features shape: {self.text_features.shape}")
        print(f"Image features shape: {self.image_features.shape}")
        
        return self.text_features, self.image_features
    
    def train_model(self, 
                   use_neural_network: bool = True,
                   use_ensemble: bool = True) -> PricePredictor:
        """
        Train the price prediction model
        
        Args:
            use_neural_network: Whether to use neural network
            use_ensemble: Whether to use ensemble methods
            
        Returns:
            Trained predictor
        """
        print("Training model...")
        
        # Split data
        train_size = len(self.train_data)
        text_features_train = self.text_features[:train_size]
        text_features_test = self.text_features[train_size:]
        image_features_train = self.image_features[:train_size]
        image_features_test = self.image_features[train_size:]
        
        # Prepare target variable
        target_column = 'price_log' if self.use_log_transform else 'price'
        y = self.train_data[target_column].values
        
        # Create predictor
        self.predictor = create_price_predictor(
            use_neural_network=use_neural_network,
            use_ensemble=use_ensemble
        )
        
        # Train model
        self.predictor.fit(
            text_features_train.values,
            image_features_train.values,
            y
        )
        
        # Save model
        model_path = os.path.join(self.output_folder, 'trained_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.predictor, f)
        
        print(f"Model saved to {model_path}")
        
        return self.predictor
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Returns:
            Dictionary of evaluation metrics
        """
        print("Evaluating model...")
        
        # Split data for evaluation
        train_size = len(self.train_data)
        text_features_train = self.text_features[:train_size]
        text_features_test = self.text_features[train_size:]
        image_features_train = self.image_features[:train_size]
        image_features_test = self.image_features[train_size:]
        
        # Prepare target variables
        target_column = 'price_log' if self.use_log_transform else 'price'
        y_train = self.train_data[target_column].values
        y_test = self.test_data[target_column].values if target_column in self.test_data.columns else None
        
        # Train/validation split
        X_text_train, X_text_val, X_image_train, X_image_val, y_train_split, y_val_split = train_test_split(
            text_features_train.values,
            image_features_train.values,
            y_train,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Train on split
        self.predictor.fit(X_text_train, X_image_train, y_train_split)
        
        # Evaluate on validation set
        val_predictions = self.predictor.predict(X_text_val, X_image_val)
        
        # Convert back from log if needed
        if self.use_log_transform:
            val_predictions = np.expm1(val_predictions)
            y_val_split = np.expm1(y_val_split)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val_split, val_predictions)
        mse = mean_squared_error(y_val_split, val_predictions)
        rmse = np.sqrt(mse)
        
        # Calculate SMAPE
        smape = np.mean(np.abs(y_val_split - val_predictions) / ((np.abs(y_val_split) + np.abs(val_predictions)) / 2)) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'SMAPE': smape
        }
        
        print("Validation Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save metrics
        metrics_path = os.path.join(self.output_folder, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def generate_predictions(self) -> pd.DataFrame:
        """
        Generate predictions for test data
        
        Returns:
            DataFrame with predictions
        """
        print("Generating predictions...")
        
        # Split data
        train_size = len(self.train_data)
        text_features_test = self.text_features[train_size:]
        image_features_test = self.image_features[train_size:]
        
        # Make predictions
        predictions = self.predictor.predict(
            text_features_test.values,
            image_features_test.values
        )
        
        # Convert back from log if needed
        if self.use_log_transform:
            predictions = np.expm1(predictions)
        
        # Create output DataFrame
        output_df = pd.DataFrame({
            'sample_id': self.test_data['sample_id'],
            'price': predictions
        })
        
        # Save predictions
        output_path = os.path.join(self.output_folder, 'test_out.csv')
        output_df.to_csv(output_path, index=False)
        
        print(f"Predictions saved to {output_path}")
        print(f"Predictions shape: {output_df.shape}")
        print(f"Sample predictions:")
        print(output_df.head())
        
        return output_df
    
    def plot_results(self, predictions: pd.DataFrame = None):
        """
        Plot training results and predictions
        
        Args:
            predictions: DataFrame with predictions
        """
        print("Creating plots...")
        
        # Create plots directory
        plots_dir = os.path.join(self.output_folder, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Price distribution plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.train_data['price'], bins=50, alpha=0.7, label='Training')
        if predictions is not None:
            plt.hist(predictions['price'], bins=50, alpha=0.7, label='Predictions')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.title('Price Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(np.log1p(self.train_data['price']), bins=50, alpha=0.7, label='Training (Log)')
        if predictions is not None:
            plt.hist(np.log1p(predictions['price']), bins=50, alpha=0.7, label='Predictions (Log)')
        plt.xlabel('Log(Price + 1)')
        plt.ylabel('Frequency')
        plt.title('Log Price Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'price_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {plots_dir}")

def main():
    """
    Main training pipeline
    """
    print("=== Amazon ML Challenge 2025 - Training Pipeline ===")
    
    # Initialize trainer
    trainer = ModelTrainer(
        data_folder='../dataset/',
        output_folder='../output/',
        use_log_transform=True,
        test_size=0.2,
        random_state=42
    )
    
    # Load data
    train_data, test_data = trainer.load_data()
    
    # Preprocess data
    train_data, test_data = trainer.preprocess_data()
    
    # Extract features (use small sample for testing)
    print("Using sample of 1000 for feature extraction...")
    text_features, image_features = trainer.extract_features(sample_size=1000)
    
    # Train model
    predictor = trainer.train_model(
        use_neural_network=True,
        use_ensemble=True
    )
    
    # Evaluate model
    metrics = trainer.evaluate_model()
    
    # Generate predictions
    predictions = trainer.generate_predictions()
    
    # Create plots
    trainer.plot_results(predictions)
    
    print("=== Training Complete ===")
    print(f"Final SMAPE: {metrics['SMAPE']:.4f}%")

if __name__ == "__main__":
    main()
