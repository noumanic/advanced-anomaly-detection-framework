"""
Data Preprocessing Module
Handles data loading, normalization, windowing, and missing value handling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Preprocesses time-series data for anomaly detection"""
    
    def __init__(self, 
                 window_size: int = 100,
                 stride: int = 1,
                 normalization: str = 'standard',
                 handle_missing: str = 'forward_fill'):
        """
        Args:
            window_size: Size of sliding window for sequences
            stride: Step size for sliding window
            normalization: 'standard' or 'minmax'
            handle_missing: 'forward_fill', 'backward_fill', 'interpolate', or 'zero'
        """
        self.window_size = window_size
        self.stride = stride
        self.normalization = normalization
        self.handle_missing = handle_missing
        self.scaler = None
        self.feature_names = None
        
    def load_data(self, file_path: str, customer_id: Optional[str] = None) -> pd.DataFrame:
        """
        Load and prepare time-series data
        
        Args:
            file_path: Path to CSV file
            customer_id: Optional specific customer ID to filter
            
        Returns:
            Preprocessed DataFrame
        """
        df = pd.read_csv(file_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Convert DateTime
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        # Convert KWH/hh to numeric (handle spaces)
        kwh_col = 'KWH/hh (per half hour)'
        if kwh_col in df.columns:
            df[kwh_col] = df[kwh_col].astype(str).str.strip().replace('', np.nan)
            df[kwh_col] = pd.to_numeric(df[kwh_col], errors='coerce')
        
        # Filter by customer if specified
        if customer_id:
            df = df[df['LCLid'] == customer_id].copy()
        
        # Sort by DateTime
        df = df.sort_values('DateTime').reset_index(drop=True)
        
        return df
    
    def create_multivariate_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create multivariate features from time-series data
        
        Args:
            df: DataFrame with DateTime and KWH/hh columns
            
        Returns:
            Multivariate feature array
        """
        # Extract temporal features
        df = df.copy()
        df['hour'] = df['DateTime'].dt.hour
        df['day_of_week'] = df['DateTime'].dt.dayofweek
        df['day_of_month'] = df['DateTime'].dt.day
        df['month'] = df['DateTime'].dt.month
        
        # Get energy consumption
        kwh_col = 'KWH/hh (per half hour)'
        if kwh_col not in df.columns:
            raise ValueError(f"Column '{kwh_col}' not found")
        
        # Create rolling statistics as features
        energy = df[kwh_col].values
        
        # Create multivariate features: [energy, hour, day_of_week, rolling_mean, rolling_std]
        features = []
        for i in range(len(df)):
            if i < 24:  # Need enough history for rolling stats
                rolling_mean = np.mean(energy[:i+1]) if i > 0 else energy[i]
                rolling_std = np.std(energy[:i+1]) if i > 0 else 0.0
            else:
                rolling_mean = np.mean(energy[i-24:i+1])
                rolling_std = np.std(energy[i-24:i+1])
            
            features.append([
                energy[i] if not np.isnan(energy[i]) else 0.0,
                df.iloc[i]['hour'] / 23.0,  # Normalize to [0, 1]
                df.iloc[i]['day_of_week'] / 6.0,
                df.iloc[i]['day_of_month'] / 31.0,
                df.iloc[i]['month'] / 12.0,
                rolling_mean,
                rolling_std
            ])
        
        features = np.array(features, dtype=np.float32)
        
        # Handle missing values
        features = self._handle_missing_values(features)
        
        self.feature_names = ['energy', 'hour', 'day_of_week', 'day_of_month', 
                             'month', 'rolling_mean', 'rolling_std']
        
        return features
    
    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Handle missing values in the data"""
        data = data.copy()
        
        if self.handle_missing == 'forward_fill':
            mask = np.isnan(data)
            for i in range(1, len(data)):
                data[i] = np.where(mask[i], data[i-1], data[i])
            data[0] = np.where(mask[0], 0, data[0])
        elif self.handle_missing == 'backward_fill':
            mask = np.isnan(data)
            for i in range(len(data)-2, -1, -1):
                data[i] = np.where(mask[i], data[i+1], data[i])
        elif self.handle_missing == 'interpolate':
            for col in range(data.shape[1]):
                mask = ~np.isnan(data[:, col])
                if mask.any():
                    data[:, col] = np.interp(
                        np.arange(len(data)),
                        np.arange(len(data))[mask],
                        data[mask, col]
                    )
        elif self.handle_missing == 'zero':
            data = np.nan_to_num(data, nan=0.0)
        
        return data
    
    def normalize(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize the data
        
        Args:
            data: Input data array
            fit: Whether to fit the scaler (True for training, False for test)
            
        Returns:
            Normalized data
        """
        if fit:
            if self.normalization == 'standard':
                self.scaler = StandardScaler()
            elif self.normalization == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown normalization: {self.normalization}")
            
            # Reshape for scaler (needs 2D)
            original_shape = data.shape
            data_2d = data.reshape(-1, data.shape[-1])
            data_normalized = self.scaler.fit_transform(data_2d)
            return data_normalized.reshape(original_shape)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            original_shape = data.shape
            data_2d = data.reshape(-1, data.shape[-1])
            data_normalized = self.scaler.transform(data_2d)
            return data_normalized.reshape(original_shape)
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from multivariate time-series
        
        Args:
            data: Multivariate time-series array (T, features)
            
        Returns:
            sequences: (N, window_size, features)
            indices: Original indices for each sequence
        """
        sequences = []
        indices = []
        
        for i in range(0, len(data) - self.window_size + 1, self.stride):
            sequences.append(data[i:i+self.window_size])
            indices.append(i)
        
        return np.array(sequences, dtype=np.float32), np.array(indices)
    
    def prepare_data(self, 
                     file_path: str, 
                     customer_id: Optional[str] = None,
                     train_split: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete data preparation pipeline
        
        Returns:
            X_train, X_test, train_indices, test_indices
        """
        # Load data
        df = self.load_data(file_path, customer_id)
        
        # Create features
        features = self.create_multivariate_features(df)
        
        # Normalize
        features = self.normalize(features, fit=True)
        
        # Create sequences
        sequences, indices = self.create_sequences(features)
        
        # Train/test split
        split_idx = int(len(sequences) * train_split)
        X_train = sequences[:split_idx]
        X_test = sequences[split_idx:]
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        return X_train, X_test, train_indices, test_indices

