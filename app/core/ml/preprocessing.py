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
    FIXED: Label column is now OPTIONAL for inference
    
    Args:
        df: Input DataFrame
        test_size: Test split ratio (unused in inference)
        val_size: Validation split ratio (unused in inference)
        fit_scaler: If True, fit new scaler; if False, use provided scaler
        scaler: Pre-fitted scaler (required if fit_scaler=False)
    
    Returns:
        Processed DataFrame, scaler
    """
    logger.info("Starting data preprocessing...")
    
    # Create explicit copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Data cleaning
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Feature selection - Label is now OPTIONAL
    feature_cols = ['Source IP', 'Destination IP', 'Timestamp', 
                    'Average Packet Size', 'Bwd Packets/s', 'FIN Flag Count',
                    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
                    'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Flow Packets/s']
    
    # Check only for required feature columns (NOT Label)
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # FIXED: Check if Label exists BEFORE selecting columns
    has_label = 'Label' in df.columns
    
    # Select ALL columns we need (features + Label if exists)
    cols_to_keep = feature_cols + (['Label'] if has_label else [])
    df = df[cols_to_keep].copy()
    
    # Add dummy Label ONLY if it doesn't exist
    if not has_label:
        df['Label'] = 1  # Default to attack label
        logger.info("⚠️  No 'Label' column found. Added dummy labels (1 = attack) for processing.")
    
    # Binary label conversion - ONLY apply if we have real labels
    if has_label:
        df.loc[:, 'Label'] = df['Label'].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)
    # else: Label already set to 1 (dummy)
    
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
