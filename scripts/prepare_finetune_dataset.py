import os
import glob
import pandas as pd
import numpy as np

DATASETS_DIR = '../custom_datasets'
OUTPUT_FILE = '../custom_datasets/fine_tuning_dataset.csv'

# Since we have ~1,100 normal flows, let's limit attacks to roughly 2,000 each
# to avoid complete catastrophic forgetting
TARGET_SAMPLES_PER_ATTACK = 2500

def prepare_dataset():
    csv_files = glob.glob(os.path.join(DATASETS_DIR, '*.csv'))
    
    # Exclude the output file if it's already in the directory
    csv_files = [f for f in csv_files if f != OUTPUT_FILE]

    dfs = []

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        dataset_name = os.path.splitext(filename)[0]
        
        print(f"Processing {dataset_name}...")
        df = pd.read_csv(file_path)
        
        # 1. Relabeling
        if dataset_name == 'normal_flows':
            df['Label'] = 0
            # Keep all normal flows
            dfs.append(df)
            print(f"  Target: Normal, Label: 0, Shape: {df.shape}")
        else:
            df['Label'] = 1
            # 2. Undersampling attacks
            if len(df) > TARGET_SAMPLES_PER_ATTACK:
                df = df.sample(n=TARGET_SAMPLES_PER_ATTACK, random_state=42)
            dfs.append(df)
            print(f"  Target: Attack (Undersampled), Label: 1, Shape: {df.shape}")

    # 3. Consolidate
    print("\\nConsolidating datasets...")
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Shuffle the final dataset so it's ready for training
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\\nFinal Dataset Shape: {final_df.shape}")
    print("Label Distribution:")
    print(final_df['Label'].value_counts())
    
    # Save to CSV
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\\nSaved fine-tuning dataset to: {OUTPUT_FILE}")

if __name__ == '__main__':
    prepare_dataset()
