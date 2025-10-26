"""
Data preprocessing pipeline - exact from V3 notebook
Fixed for pandas warnings
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


def preprocess_and_split_data(df, test_size=0.3, val_size=0.15, fit_scaler=True, scaler=None):
    """
    Cleans, scales test data as per paper methodology
    Exact implementation from V3 notebook - FIXED for warnings
    """
    logger.info("Starting data preprocessing...")
    
    # Create explicit copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Data cleaning
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Feature selection
    required_cols = ['Source IP', 'Destination IP', 'Timestamp', 'Label', 
                    'Average Packet Size', 'Bwd Packets/s', 'FIN Flag Count',
                    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
                    'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Flow Packets/s']
    
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing columns: {missing}")
    
    df = df[required_cols].copy()  # Explicit copy
    
    # Binary label conversion - use .loc to avoid warning
    df.loc[:, 'Label'] = df['Label'].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)
    
    # Scaling
    num_cols = ['Average Packet Size', 'Bwd Packets/s', 'FIN Flag Count',
                'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
                'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Flow Packets/s']
    
    if fit_scaler:
        scaler = StandardScaler()
        df.loc[:, num_cols] = scaler.fit_transform(df[num_cols])
    else:
        if scaler is None:
            raise ValueError("scaler required when fit_scaler=False")
        df.loc[:, num_cols] = scaler.transform(df[num_cols])
    
    logger.info(f"✅ Preprocessing finished. Processed {len(df)} samples")
    return df, scaler
