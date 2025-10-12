"""
Text Preprocessing Pipeline for Amazon ML Challenge 2025
Handles catalog_content text processing and feature extraction
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """
    Comprehensive text preprocessing class for product catalog content
    """
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 use_stemming: bool = True,
                 use_lemmatization: bool = False,
                 min_word_length: int = 2,
                 max_features: int = 10000):
        """
        Initialize text preprocessor
        
        Args:
            remove_stopwords: Whether to remove stopwords
            use_stemming: Whether to use stemming
            use_lemmatization: Whether to use lemmatization
            min_word_length: Minimum word length to keep
            max_features: Maximum number of features for vectorization
        """
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.min_word_length = min_word_length
        self.max_features = max_features
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        
        # Initialize vectorizers with memory optimization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,  # Reduced from 2 to handle small samples
            max_df=0.95,
            dtype=np.float32  # Use float32 to save memory
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,  # Reduced from 2 to handle small samples
            max_df=0.95,
            dtype=np.float32  # Use float32 to save memory
        )
        
        # Feature extractors
        self.lda_model = None
        self.is_fitted = False
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespaces
        text = text.strip()
        
        return text
    
    def extract_structured_features(self, text: str) -> Dict[str, any]:
        """
        Extract structured features from catalog content
        
        Args:
            text: Input catalog content
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        if pd.isna(text) or text == '':
            return {
                'text_length': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'has_pack_info': False,
                'has_quantity': False,
                'has_size_info': False,
                'has_brand_info': False,
                'has_weight_info': False,
                'has_color_info': False,
                'has_material_info': False,
                'special_char_ratio': 0,
                'digit_ratio': 0,
                'uppercase_ratio': 0
            }
        
        # Basic text statistics
        features['text_length'] = len(text)
        words = text.split()
        features['word_count'] = len(words)
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # Content pattern detection
        features['has_pack_info'] = bool(re.search(r'pack\s+of|pack\s*:\s*\d+', text, re.IGNORECASE))
        features['has_quantity'] = bool(re.search(r'\d+\s*(oz|lb|kg|g|ml|l|count|piece|item)', text, re.IGNORECASE))
        features['has_size_info'] = bool(re.search(r'size|dimension|inch|cm|mm', text, re.IGNORECASE))
        features['has_brand_info'] = bool(re.search(r'brand|manufacturer|made by', text, re.IGNORECASE))
        features['has_weight_info'] = bool(re.search(r'weight|heavy|light', text, re.IGNORECASE))
        features['has_color_info'] = bool(re.search(r'color|colour|red|blue|green|black|white|yellow', text, re.IGNORECASE))
        features['has_material_info'] = bool(re.search(r'material|fabric|plastic|metal|wood|glass', text, re.IGNORECASE))
        
        # Character composition
        if text:
            features['special_char_ratio'] = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / len(text)
            features['digit_ratio'] = len(re.findall(r'\d', text)) / len(text)
            features['uppercase_ratio'] = len(re.findall(r'[A-Z]', text)) / len(text)
        else:
            features['special_char_ratio'] = 0
            features['digit_ratio'] = 0
            features['uppercase_ratio'] = 0
        
        return features
    
    def extract_quantity_features(self, text: str) -> Dict[str, any]:
        """
        Extract quantity and measurement features
        
        Args:
            text: Input catalog content
            
        Returns:
            Dictionary of quantity features
        """
        features = {}
        
        if pd.isna(text) or text == '':
            return {
                'has_quantity': False,
                'quantity_value': 0,
                'quantity_unit': '',
                'pack_size': 0,
                'has_multiple_quantities': False
            }
        
        # Extract quantity patterns
        quantity_patterns = [
            r'(\d+(?:\.\d+)?)\s*(oz|ounce|ounces)',
            r'(\d+(?:\.\d+)?)\s*(lb|pound|pounds)',
            r'(\d+(?:\.\d+)?)\s*(kg|kilogram|kilograms)',
            r'(\d+(?:\.\d+)?)\s*(g|gram|grams)',
            r'(\d+(?:\.\d+)?)\s*(ml|milliliter|milliliters)',
            r'(\d+(?:\.\d+)?)\s*(l|liter|liters)',
            r'(\d+(?:\.\d+)?)\s*(count|piece|pieces|item|items)',
            r'pack\s+of\s*(\d+)',
            r'(\d+)\s*pack'
        ]
        
        quantities = []
        units = []
        
        for pattern in quantity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    quantities.append(float(match[0]))
                    units.append(match[1].lower())
                else:
                    quantities.append(float(match))
                    units.append('pack')
        
        features['has_quantity'] = len(quantities) > 0
        features['quantity_value'] = max(quantities) if quantities else 0
        features['quantity_unit'] = units[quantities.index(max(quantities))] if quantities else ''
        features['pack_size'] = quantities[0] if quantities else 0
        features['has_multiple_quantities'] = len(quantities) > 1
        
        return features
    
    def tokenize_and_clean(self, text: str) -> List[str]:
        """
        Tokenize and clean text
        
        Args:
            text: Input text string
            
        Returns:
            List of cleaned tokens
        """
        if pd.isna(text) or text == '':
            return []
        
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Remove short tokens
            if len(token) < self.min_word_length:
                continue
            
            # Remove stopwords
            if self.remove_stopwords and token.lower() in self.stop_words:
                continue
            
            # Apply stemming
            if self.use_stemming and self.stemmer:
                token = self.stemmer.stem(token)
            
            # Apply lemmatization
            if self.use_lemmatization and self.lemmatizer:
                token = self.lemmatizer.lemmatize(token)
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for vectorization
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text string
        """
        tokens = self.tokenize_and_clean(text)
        return ' '.join(tokens)
    
    def fit_transform(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit vectorizers and transform texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Tuple of (tfidf_features, count_features)
        """
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Fit and transform with TF-IDF
        tfidf_features = self.tfidf_vectorizer.fit_transform(processed_texts)
        
        # Fit and transform with Count Vectorizer
        count_features = self.count_vectorizer.fit_transform(processed_texts)
        
        self.is_fitted = True
        return tfidf_features, count_features
    
    def transform(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform texts using fitted vectorizers
        
        Args:
            texts: List of text strings
            
        Returns:
            Tuple of (tfidf_features, count_features)
        """
        if not self.is_fitted:
            raise ValueError("Vectorizers must be fitted before transform")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Transform with fitted vectorizers
        tfidf_features = self.tfidf_vectorizer.transform(processed_texts)
        count_features = self.count_vectorizer.transform(processed_texts)
        
        return tfidf_features, count_features
    
    def extract_all_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract all text features
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame with all extracted features
        """
        # Extract structured features
        structured_features = []
        quantity_features = []
        
        for text in texts:
            structured_features.append(self.extract_structured_features(text))
            quantity_features.append(self.extract_quantity_features(text))
        
        # Convert to DataFrames
        structured_df = pd.DataFrame(structured_features)
        quantity_df = pd.DataFrame(quantity_features)
        
        # Combine features
        all_features = pd.concat([structured_df, quantity_df], axis=1)
        
        return all_features
    
    def fit_lda(self, texts: List[str], n_topics: int = 10) -> np.ndarray:
        """
        Fit LDA topic model
        
        Args:
            texts: List of text strings
            n_topics: Number of topics for LDA
            
        Returns:
            Topic distribution features
        """
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create count matrix
        count_matrix = self.count_vectorizer.fit_transform(processed_texts)
        
        # Fit LDA
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        
        topic_features = self.lda_model.fit_transform(count_matrix)
        
        return topic_features
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names from vectorizers
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            return []
        
        tfidf_features = self.tfidf_vectorizer.get_feature_names_out()
        count_features = self.count_vectorizer.get_feature_names_out()
        
        return list(tfidf_features) + list(count_features)


def create_text_features(df: pd.DataFrame, 
                        text_column: str = 'catalog_content',
                        fit_preprocessor: bool = True,
                        max_features: int = 5000,
                        batch_size: int = 5000) -> Tuple[pd.DataFrame, TextPreprocessor]:
    """
    Create comprehensive text features from DataFrame with memory optimization
    
    Args:
        df: Input DataFrame
        text_column: Name of text column
        fit_preprocessor: Whether to fit the preprocessor
        max_features: Maximum number of features to reduce memory usage
        batch_size: Batch size for processing large datasets
        
    Returns:
        Tuple of (features_df, preprocessor)
    """
    preprocessor = TextPreprocessor(max_features=max_features)
    
    # Extract text features
    text_features = preprocessor.extract_all_features(df[text_column].tolist())
    
    if fit_preprocessor:
        # Process in batches to avoid memory issues
        if len(df) > batch_size:
            print(f"Processing {len(df)} samples in batches of {batch_size}")
            
            # Fit on first batch
            first_batch = df[text_column].iloc[:batch_size].tolist()
            tfidf_features, count_features = preprocessor.fit_transform(first_batch)
            
            # Process remaining batches
            all_tfidf = [tfidf_features]
            all_count = [count_features]
            
            for i in range(batch_size, len(df), batch_size):
                batch = df[text_column].iloc[i:i+batch_size].tolist()
                tfidf_batch, count_batch = preprocessor.transform(batch)
                all_tfidf.append(tfidf_batch)
                all_count.append(count_batch)
            
            # Combine all batches
            from scipy.sparse import vstack
            tfidf_features = vstack(all_tfidf)
            count_features = vstack(all_count)
        else:
            # Fit vectorizers and extract TF-IDF and Count features
            tfidf_features, count_features = preprocessor.fit_transform(df[text_column].tolist())
        
        # Convert to DataFrames with memory optimization
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray().astype(np.float32),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        
        count_df = pd.DataFrame(
            count_features.toarray().astype(np.float32),
            columns=[f'count_{i}' for i in range(count_features.shape[1])]
        )
        
        # Ensure all text features are numeric
        for col in text_features.columns:
            if str(text_features[col].dtypes) == 'object':
                # Convert to numeric, replacing non-numeric with 0
                text_features[col] = pd.to_numeric(text_features[col], errors='coerce').fillna(0)
        
        # Combine all features
        all_features = pd.concat([text_features, tfidf_df, count_df], axis=1)
    else:
        all_features = text_features
    
    return all_features, preprocessor


if __name__ == "__main__":
    # Example usage
    import os
    
    # Load sample data
    DATASET_FOLDER = '../dataset/'
    train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    
    # Create text features
    print("Creating text features...")
    text_features, preprocessor = create_text_features(train.head(1000))
    
    print(f"Text features shape: {text_features.shape}")
    print(f"Feature columns: {list(text_features.columns)}")
    print(f"Sample features:")
    print(text_features.head())
