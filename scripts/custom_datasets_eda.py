import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration
DATASETS_DIR = '../custom_datasets'
RESULTS_DIR = '../eda_results'

def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    csv_files = glob.glob(os.path.join(DATASETS_DIR, '*.csv'))
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        dataset_name = os.path.splitext(filename)[0]
        print(f"\\n[{dataset_name}] Starting EDA...")
        
        # Create a directory for this dataset's results
        dataset_results_dir = os.path.join(RESULTS_DIR, dataset_name)
        if not os.path.exists(dataset_results_dir):
            os.makedirs(dataset_results_dir)
            
        # 1. Load Data
        df = pd.read_csv(file_path)
        
        stats_file = os.path.join(dataset_results_dir, f"{dataset_name}_summary.txt")
        with open(stats_file, 'w') as f:
            f.write(f"=== {dataset_name} EDA Summary ===\\n\\n")
            f.write(f"Data Shape: {df.shape}\\n\\n")
            
            # 2. Basic Info & Missing values
            f.write("--- Missing Values ---\\n")
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if not missing.empty:
                f.write(missing.to_string() + "\\n\\n")
            else:
                f.write("No missing values found.\\n\\n")
                
            # 3. Label Distribution
            if 'Label' in df.columns:
                f.write("--- Label Distribution ---\\n")
                f.write(df['Label'].value_counts().to_string() + "\\n\\n")
                
                # Plot Label Distribution
                plt.figure(figsize=(10, 6))
                sns.countplot(data=df, x='Label')
                plt.title(f'{dataset_name} - Label Distribution')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(dataset_results_dir, f"{dataset_name}_label_dist.png"))
                plt.close()
                
            # 4. Summary Statistics for numerical columns
            f.write("--- Numerical Summary ---\\n")
            num_df = df.select_dtypes(include=[np.number])
            f.write(num_df.describe().to_string() + "\\n")
            
        # 5. Plot distributions of some key features (if they exist)
        key_features = ['Flow Duration', 'Total Fwd Packet', 'Total Length of Fwd Packet', 'Flow IAT Mean']
        for feature in key_features:
            if feature in df.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[feature], bins=50, kde=True)
                plt.yscale('log')
                plt.title(f'{dataset_name} - Distribution of {feature} (Log Scale)')
                plt.tight_layout()
                plt.savefig(os.path.join(dataset_results_dir, f"{dataset_name}_{feature.replace(' ', '_')}_dist.png"))
                plt.close()

        # 6. Correlation Heatmap (using correlation of a subset of features to avoid explosion)
        # Select first 10 numerical features + Label (if present)
        cols_to_corr = num_df.columns[:15]
        if len(cols_to_corr) > 1:
            plt.figure(figsize=(12, 10))
            corr = df[cols_to_corr].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title(f'{dataset_name} - Correlation Heatmap (subset)')
            plt.tight_layout()
            plt.savefig(os.path.join(dataset_results_dir, f"{dataset_name}_corr_heatmap.png"))
            plt.close()

        print(f"[{dataset_name}] EDA Complete. Results saved in {dataset_results_dir}")

if __name__ == '__main__':
    # Set seaborn style
    sns.set_theme(style="whitegrid")
    main()
