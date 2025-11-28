"""
Comprehensive Data Visualization Script
Creates visualizations for raw dataset, preprocessed data, and statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.data_preprocessing import DataPreprocessor

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")


class DataVisualizer:
    """Creates comprehensive visualizations for dataset analysis"""
    
    def __init__(self, output_dir: str = 'Visualization'):
        """
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'raw_data').mkdir(exist_ok=True)
        (self.output_dir / 'preprocessed_data').mkdir(exist_ok=True)
        (self.output_dir / 'statistics').mkdir(exist_ok=True)
    
    def visualize_raw_dataset(self, file_path: str, customer_id: str = None, 
                              max_samples: int = 100000):
        """
        Visualize raw dataset statistics and plots
        
        Args:
            file_path: Path to CSV file
            customer_id: Optional customer ID to filter
            max_samples: Maximum samples to load for visualization
        """
        print("="*60)
        print("Visualizing Raw Dataset")
        print("="*60)
        
        # Load data
        print("Loading raw data...")
        df = pd.read_csv(file_path, nrows=max_samples)
        df.columns = df.columns.str.strip()
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        # Filter by customer if specified
        if customer_id:
            df = df[df['LCLid'] == customer_id].copy()
            print(f"Filtered to customer: {customer_id}")
        
        # Convert KWH/hh to numeric
        kwh_col = 'KWH/hh (per half hour)'
        if kwh_col in df.columns:
            df[kwh_col] = df[kwh_col].astype(str).str.strip().replace('', np.nan)
            df[kwh_col] = pd.to_numeric(df[kwh_col], errors='coerce')
        
        df = df.sort_values('DateTime').reset_index(drop=True)
        
        # Extract temporal features
        df['hour'] = df['DateTime'].dt.hour
        df['day_of_week'] = df['DateTime'].dt.dayofweek
        df['day_of_month'] = df['DateTime'].dt.day
        df['month'] = df['DateTime'].dt.month
        df['year'] = df['DateTime'].dt.year
        
        # 1. Dataset Overview Statistics
        print("\n1. Generating dataset overview statistics...")
        self._plot_dataset_overview(df)
        
        # 2. Energy Consumption Time Series
        print("2. Generating energy consumption time series plots...")
        self._plot_energy_time_series(df)
        
        # 3. Energy Consumption Distribution
        print("3. Generating energy consumption distribution plots...")
        self._plot_energy_distribution(df)
        
        # 4. Temporal Patterns
        print("4. Generating temporal pattern analysis...")
        self._plot_temporal_patterns(df)
        
        # 5. Customer Analysis (if multiple customers)
        if df['LCLid'].nunique() > 1:
            print("5. Generating customer analysis...")
            self._plot_customer_analysis(df)
        
        # 6. Missing Values Analysis
        print("6. Generating missing values analysis...")
        self._plot_missing_values(df)
        
        # 7. Statistical Summary
        print("7. Generating statistical summary...")
        self._save_statistical_summary(df)
        
        print(f"\n✓ Raw dataset visualizations saved to: {self.output_dir / 'raw_data'}")
    
    def _plot_dataset_overview(self, df: pd.DataFrame):
        """Plot dataset overview statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Raw Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Data shape and basic info
        ax = axes[0, 0]
        ax.axis('off')
        info_text = f"""
        Dataset Statistics
        
        Total Records: {len(df):,}
        Unique Customers: {df['LCLid'].nunique()}
        Date Range: {df['DateTime'].min()} to {df['DateTime'].max()}
        Time Span: {(df['DateTime'].max() - df['DateTime'].min()).days} days
        
        Columns:
        {chr(10).join([f'  • {col}' for col in df.columns])}
        
        Missing Values:
        {chr(10).join([f'  • {col}: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.2f}%)' 
                      for col in df.columns if df[col].isnull().sum() > 0])}
        """
        ax.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Records per customer
        ax = axes[0, 1]
        customer_counts = df['LCLid'].value_counts().head(10)
        ax.barh(range(len(customer_counts)), customer_counts.values)
        ax.set_yticks(range(len(customer_counts)))
        ax.set_yticklabels(customer_counts.index, fontsize=8)
        ax.set_xlabel('Number of Records')
        ax.set_title('Top 10 Customers by Record Count')
        ax.grid(True, alpha=0.3)
        
        # 3. Records over time
        ax = axes[1, 0]
        df['date'] = df['DateTime'].dt.date
        daily_counts = df.groupby('date').size()
        ax.plot(daily_counts.index, daily_counts.values, linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Records')
        ax.set_title('Records Over Time')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 4. Energy consumption statistics
        ax = axes[1, 1]
        kwh_col = 'KWH/hh (per half hour)'
        if kwh_col in df.columns:
            energy_stats = df[kwh_col].describe()
            stats_text = f"""
            Energy Consumption Statistics
            
            Mean: {energy_stats['mean']:.4f} KWH/hh
            Median: {energy_stats['50%']:.4f} KWH/hh
            Std Dev: {energy_stats['std']:.4f} KWH/hh
            Min: {energy_stats['min']:.4f} KWH/hh
            Max: {energy_stats['max']:.4f} KWH/hh
            25th Percentile: {energy_stats['25%']:.4f} KWH/hh
            75th Percentile: {energy_stats['75%']:.4f} KWH/hh
            """
            ax.axis('off')
            ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'raw_data' / '01_dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_energy_time_series(self, df: pd.DataFrame):
        """Plot energy consumption time series"""
        kwh_col = 'KWH/hh (per half hour)'
        if kwh_col not in df.columns:
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('Energy Consumption Time Series Analysis', fontsize=16, fontweight='bold')
        
        # Sample data for plotting (if too large)
        plot_df = df.copy()
        if len(plot_df) > 50000:
            plot_df = plot_df.sample(n=50000, random_state=42).sort_values('DateTime')
        
        # 1. Full time series
        ax = axes[0]
        ax.plot(plot_df['DateTime'], plot_df[kwh_col], alpha=0.6, linewidth=0.5)
        ax.set_xlabel('DateTime')
        ax.set_ylabel('Energy Consumption (KWH/hh)')
        ax.set_title('Energy Consumption Over Time (Sampled)')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # 2. Daily average
        ax = axes[1]
        plot_df['date'] = plot_df['DateTime'].dt.date
        daily_avg = plot_df.groupby('date')[kwh_col].mean()
        ax.plot(daily_avg.index, daily_avg.values, linewidth=2, color='orange')
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Energy Consumption (KWH/hh)')
        ax.set_title('Daily Average Energy Consumption')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Hourly pattern
        ax = axes[2]
        hourly_avg = plot_df.groupby('hour')[kwh_col].mean()
        ax.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Energy Consumption (KWH/hh)')
        ax.set_title('Average Energy Consumption by Hour of Day')
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'raw_data' / '02_energy_time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_energy_distribution(self, df: pd.DataFrame):
        """Plot energy consumption distribution"""
        kwh_col = 'KWH/hh (per half hour)'
        if kwh_col not in df.columns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Energy Consumption Distribution Analysis', fontsize=16, fontweight='bold')
        
        energy = df[kwh_col].dropna()
        
        # 1. Histogram
        ax = axes[0, 0]
        ax.hist(energy, bins=100, edgecolor='black', alpha=0.7)
        ax.axvline(energy.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {energy.mean():.4f}')
        ax.axvline(energy.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {energy.median():.4f}')
        ax.set_xlabel('Energy Consumption (KWH/hh)')
        ax.set_ylabel('Frequency')
        ax.set_title('Energy Consumption Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Box plot
        ax = axes[0, 1]
        ax.boxplot(energy, vert=True)
        ax.set_ylabel('Energy Consumption (KWH/hh)')
        ax.set_title('Energy Consumption Box Plot')
        ax.grid(True, alpha=0.3)
        
        # 3. Log scale histogram
        ax = axes[1, 0]
        energy_positive = energy[energy > 0]
        if len(energy_positive) > 0:
            ax.hist(np.log10(energy_positive + 1e-6), bins=100, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Log10(Energy Consumption)')
            ax.set_ylabel('Frequency')
            ax.set_title('Energy Consumption Distribution (Log Scale)')
            ax.grid(True, alpha=0.3)
        
        # 4. Q-Q plot
        ax = axes[1, 1]
        from scipy import stats
        stats.probplot(energy, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Normal Distribution)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'raw_data' / '03_energy_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_temporal_patterns(self, df: pd.DataFrame):
        """Plot temporal patterns"""
        kwh_col = 'KWH/hh (per half hour)'
        if kwh_col not in df.columns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Pattern Analysis', fontsize=16, fontweight='bold')
        
        # 1. Hourly pattern
        ax = axes[0, 0]
        hourly_stats = df.groupby('hour')[kwh_col].agg(['mean', 'std'])
        ax.plot(hourly_stats.index, hourly_stats['mean'], marker='o', linewidth=2, label='Mean')
        ax.fill_between(hourly_stats.index, 
                        hourly_stats['mean'] - hourly_stats['std'],
                        hourly_stats['mean'] + hourly_stats['std'],
                        alpha=0.3, label='±1 Std Dev')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Energy Consumption (KWH/hh)')
        ax.set_title('Average Energy Consumption by Hour')
        ax.set_xticks(range(0, 24, 2))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Day of week pattern
        ax = axes[0, 1]
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow_stats = df.groupby('day_of_week')[kwh_col].agg(['mean', 'std'])
        ax.plot(dow_stats.index, dow_stats['mean'], marker='o', linewidth=2, markersize=8)
        ax.fill_between(dow_stats.index,
                       dow_stats['mean'] - dow_stats['std'],
                       dow_stats['mean'] + dow_stats['std'],
                       alpha=0.3)
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Energy Consumption (KWH/hh)')
        ax.set_title('Average Energy Consumption by Day of Week')
        ax.set_xticks(range(7))
        ax.set_xticklabels(day_names)
        ax.grid(True, alpha=0.3)
        
        # 3. Monthly pattern
        ax = axes[1, 0]
        monthly_stats = df.groupby('month')[kwh_col].agg(['mean', 'std'])
        ax.plot(monthly_stats.index, monthly_stats['mean'], marker='o', linewidth=2, markersize=8)
        ax.fill_between(monthly_stats.index,
                       monthly_stats['mean'] - monthly_stats['std'],
                       monthly_stats['mean'] + monthly_stats['std'],
                       alpha=0.3)
        ax.set_xlabel('Month')
        ax.set_ylabel('Energy Consumption (KWH/hh)')
        ax.set_title('Average Energy Consumption by Month')
        ax.set_xticks(range(1, 13))
        ax.grid(True, alpha=0.3)
        
        # 4. Heatmap: Hour vs Day of Week
        ax = axes[1, 1]
        pivot_data = df.pivot_table(values=kwh_col, index='day_of_week', 
                                   columns='hour', aggfunc='mean')
        im = ax.imshow(pivot_data.values, aspect='auto', cmap='viridis')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Day of Week')
        ax.set_title('Energy Consumption Heatmap (Hour vs Day of Week)')
        ax.set_yticks(range(7))
        ax.set_yticklabels(day_names)
        ax.set_xticks(range(0, 24, 2))
        plt.colorbar(im, ax=ax, label='Energy (KWH/hh)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'raw_data' / '04_temporal_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_customer_analysis(self, df: pd.DataFrame):
        """Plot customer analysis"""
        kwh_col = 'KWH/hh (per half hour)'
        if kwh_col not in df.columns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Customer Analysis', fontsize=16, fontweight='bold')
        
        # 1. Average consumption per customer
        ax = axes[0, 0]
        customer_avg = df.groupby('LCLid')[kwh_col].mean().sort_values(ascending=False).head(15)
        ax.barh(range(len(customer_avg)), customer_avg.values)
        ax.set_yticks(range(len(customer_avg)))
        ax.set_yticklabels(customer_avg.index, fontsize=8)
        ax.set_xlabel('Average Energy Consumption (KWH/hh)')
        ax.set_title('Top 15 Customers by Average Consumption')
        ax.grid(True, alpha=0.3)
        
        # 2. Consumption distribution by customer
        ax = axes[0, 1]
        top_customers = df['LCLid'].value_counts().head(5).index
        for customer in top_customers:
            customer_data = df[df['LCLid'] == customer][kwh_col].dropna()
            ax.hist(customer_data, bins=50, alpha=0.5, label=customer, edgecolor='black')
        ax.set_xlabel('Energy Consumption (KWH/hh)')
        ax.set_ylabel('Frequency')
        ax.set_title('Energy Distribution for Top 5 Customers')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 3. Time series for top customers
        ax = axes[1, 0]
        for customer in top_customers[:3]:
            customer_df = df[df['LCLid'] == customer].sort_values('DateTime')
            if len(customer_df) > 10000:
                customer_df = customer_df.sample(n=10000, random_state=42).sort_values('DateTime')
            ax.plot(customer_df['DateTime'], customer_df[kwh_col], 
                   alpha=0.6, linewidth=0.5, label=customer)
        ax.set_xlabel('DateTime')
        ax.set_ylabel('Energy Consumption (KWH/hh)')
        ax.set_title('Time Series for Top 3 Customers')
        ax.legend(fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 4. Customer statistics summary
        ax = axes[1, 1]
        customer_stats = df.groupby('LCLid')[kwh_col].agg(['mean', 'std', 'count']).head(10)
        ax.axis('off')
        stats_text = f"""
        Top 10 Customers Statistics
        
        {'Customer':<15} {'Mean':<10} {'Std':<10} {'Count':<10}
        {'-'*50}
        """
        for idx, row in customer_stats.iterrows():
            stats_text += f"{idx:<15} {row['mean']:<10.4f} {row['std']:<10.4f} {int(row['count']):<10}\n"
        ax.text(0.1, 0.5, stats_text, fontsize=9, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'raw_data' / '05_customer_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_missing_values(self, df: pd.DataFrame):
        """Plot missing values analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Missing Values Analysis', fontsize=16, fontweight='bold')
        
        # 1. Missing values per column
        ax = axes[0]
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        if len(missing_counts) > 0:
            ax.barh(range(len(missing_counts)), missing_counts.values)
            ax.set_yticks(range(len(missing_counts)))
            ax.set_yticklabels(missing_counts.index, fontsize=10)
            ax.set_xlabel('Number of Missing Values')
            ax.set_title('Missing Values per Column')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            ax.axis('off')
        
        # 2. Missing values percentage
        ax = axes[1]
        missing_pct = (df.isnull().sum() / len(df) * 100)
        missing_pct = missing_pct[missing_pct > 0]
        if len(missing_pct) > 0:
            ax.barh(range(len(missing_pct)), missing_pct.values)
            ax.set_yticks(range(len(missing_pct)))
            ax.set_yticklabels(missing_pct.index, fontsize=10)
            ax.set_xlabel('Percentage of Missing Values (%)')
            ax.set_title('Missing Values Percentage per Column')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'raw_data' / '06_missing_values.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_statistical_summary(self, df: pd.DataFrame):
        """Save statistical summary to text file"""
        summary_path = self.output_dir / 'statistics' / 'raw_dataset_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("RAW DATASET STATISTICAL SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Dataset Shape: {df.shape}\n")
            f.write(f"Date Range: {df['DateTime'].min()} to {df['DateTime'].max()}\n")
            f.write(f"Time Span: {(df['DateTime'].max() - df['DateTime'].min()).days} days\n")
            f.write(f"Unique Customers: {df['LCLid'].nunique()}\n\n")
            
            f.write("Column Information:\n")
            f.write("-" * 60 + "\n")
            for col in df.columns:
                f.write(f"\n{col}:\n")
                f.write(f"  Type: {df[col].dtype}\n")
                f.write(f"  Non-null count: {df[col].notna().sum()}\n")
                f.write(f"  Null count: {df[col].isnull().sum()}\n")
                if df[col].dtype in ['int64', 'float64']:
                    f.write(f"  Mean: {df[col].mean():.4f}\n")
                    f.write(f"  Std: {df[col].std():.4f}\n")
                    f.write(f"  Min: {df[col].min():.4f}\n")
                    f.write(f"  25%: {df[col].quantile(0.25):.4f}\n")
                    f.write(f"  50%: {df[col].quantile(0.50):.4f}\n")
                    f.write(f"  75%: {df[col].quantile(0.75):.4f}\n")
                    f.write(f"  Max: {df[col].max():.4f}\n")
            
            kwh_col = 'KWH/hh (per half hour)'
            if kwh_col in df.columns:
                f.write("\n\nEnergy Consumption Detailed Statistics:\n")
                f.write("-" * 60 + "\n")
                f.write(str(df[kwh_col].describe()))
        
        print(f"  ✓ Statistical summary saved to: {summary_path}")
    
    def visualize_preprocessed_data(self, file_path: str, customer_id: str = None,
                                    window_size: int = 100, stride: int = 1):
        """
        Visualize preprocessed dataset
        
        Args:
            file_path: Path to CSV file
            customer_id: Optional customer ID
            window_size: Sequence window size
            stride: Sliding window stride
        """
        print("\n" + "="*60)
        print("Visualizing Preprocessed Dataset")
        print("="*60)
        
        # Preprocess data
        print("Preprocessing data...")
        preprocessor = DataPreprocessor(
            window_size=window_size,
            stride=stride,
            normalization='standard',
            handle_missing='forward_fill'
        )
        
        # Load raw data for comparison
        df_raw = preprocessor.load_data(file_path, customer_id)
        
        # Get preprocessed data
        X_train, X_test, train_indices, test_indices = preprocessor.prepare_data(
            file_path, customer_id, train_split=0.8
        )
        
        # Get features before normalization for comparison
        features_before_norm = preprocessor.create_multivariate_features(df_raw)
        features_after_norm = preprocessor.normalize(features_before_norm.copy(), fit=True)
        
        print(f"Training sequences: {X_train.shape}")
        print(f"Test sequences: {X_test.shape}")
        print(f"Feature names: {preprocessor.feature_names}")
        
        # 1. Feature Comparison (Before/After Normalization)
        print("\n1. Generating feature comparison plots...")
        self._plot_feature_comparison(features_before_norm, features_after_norm, 
                                     preprocessor.feature_names)
        
        # 2. Feature Distributions
        print("2. Generating feature distribution plots...")
        self._plot_feature_distributions(X_train, preprocessor.feature_names)
        
        # 3. Sequence Visualization
        print("3. Generating sequence visualization...")
        self._plot_sequences(X_train, X_test, preprocessor.feature_names)
        
        # 4. Train/Test Split Analysis
        print("4. Generating train/test split analysis...")
        self._plot_train_test_split(X_train, X_test, train_indices, test_indices)
        
        # 5. Feature Correlation
        print("5. Generating feature correlation analysis...")
        self._plot_feature_correlation(X_train, preprocessor.feature_names)
        
        # 6. Preprocessing Statistics
        print("6. Generating preprocessing statistics...")
        self._save_preprocessing_summary(X_train, X_test, preprocessor.feature_names,
                                        features_before_norm, features_after_norm)
        
        print(f"\n✓ Preprocessed data visualizations saved to: {self.output_dir / 'preprocessed_data'}")
    
    def _plot_feature_comparison(self, before_norm: np.ndarray, after_norm: np.ndarray,
                                 feature_names: list):
        """Plot feature comparison before and after normalization"""
        n_features = before_norm.shape[1]
        fig, axes = plt.subplots(n_features, 2, figsize=(16, 4*n_features))
        fig.suptitle('Feature Comparison: Before vs After Normalization', 
                     fontsize=16, fontweight='bold')
        
        # Sample data for visualization
        sample_size = min(10000, len(before_norm))
        indices = np.random.choice(len(before_norm), sample_size, replace=False)
        
        for i, feature_name in enumerate(feature_names):
            # Before normalization
            ax = axes[i, 0]
            ax.hist(before_norm[indices, i], bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel(f'{feature_name} (Original)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{feature_name} - Before Normalization')
            ax.grid(True, alpha=0.3)
            
            # After normalization
            ax = axes[i, 1]
            ax.hist(after_norm[indices, i], bins=50, edgecolor='black', alpha=0.7, color='orange')
            ax.set_xlabel(f'{feature_name} (Normalized)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{feature_name} - After Normalization')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'preprocessed_data' / '01_feature_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_distributions(self, X_train: np.ndarray, feature_names: list):
        """Plot feature distributions"""
        n_features = len(feature_names)
        fig, axes = plt.subplots(2, (n_features + 1) // 2, figsize=(16, 10))
        axes = axes.flatten()
        fig.suptitle('Feature Distributions (Training Data)', fontsize=16, fontweight='bold')
        
        # Flatten sequences for distribution analysis
        X_flat = X_train.reshape(-1, X_train.shape[-1])
        
        for i, feature_name in enumerate(feature_names):
            ax = axes[i]
            data = X_flat[:, i]
            ax.hist(data, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {data.mean():.4f}')
            ax.axvline(np.median(data), color='green', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(data):.4f}')
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{feature_name} Distribution')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'preprocessed_data' / '02_feature_distributions.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sequences(self, X_train: np.ndarray, X_test: np.ndarray, feature_names: list):
        """Plot sample sequences"""
        fig, axes = plt.subplots(len(feature_names), 1, figsize=(16, 3*len(feature_names)))
        if len(feature_names) == 1:
            axes = [axes]
        fig.suptitle('Sample Sequences Visualization', fontsize=16, fontweight='bold')
        
        # Plot a few sample sequences
        n_samples = min(5, len(X_train))
        sample_indices = np.random.choice(len(X_train), n_samples, replace=False)
        
        for i, feature_name in enumerate(feature_names):
            ax = axes[i]
            for idx in sample_indices:
                ax.plot(X_train[idx, :, i], alpha=0.6, linewidth=1)
            ax.set_xlabel('Time Step')
            ax.set_ylabel(feature_name)
            ax.set_title(f'{feature_name} - Sample Training Sequences')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'preprocessed_data' / '03_sample_sequences.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_train_test_split(self, X_train: np.ndarray, X_test: np.ndarray,
                               train_indices: np.ndarray, test_indices: np.ndarray):
        """Plot train/test split analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Train/Test Split Analysis', fontsize=16, fontweight='bold')
        
        # 1. Split statistics
        ax = axes[0, 0]
        ax.axis('off')
        split_text = f"""
        Train/Test Split Statistics
        
        Training Sequences: {len(X_train):,}
        Test Sequences: {len(X_test):,}
        Total Sequences: {len(X_train) + len(X_test):,}
        
        Training Percentage: {len(X_train)/(len(X_train)+len(X_test))*100:.2f}%
        Test Percentage: {len(X_test)/(len(X_train)+len(X_test))*100:.2f}%
        
        Sequence Shape: {X_train.shape[1:]} (time_steps, features)
        """
        ax.text(0.1, 0.5, split_text, fontsize=12, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 2. Pie chart
        ax = axes[0, 1]
        sizes = [len(X_train), len(X_test)]
        labels = ['Training', 'Test']
        colors = ['#66b3ff', '#ff9999']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Train/Test Split')
        
        # 3. Feature mean comparison
        ax = axes[1, 0]
        train_means = X_train.mean(axis=(0, 1))
        test_means = X_test.mean(axis=(0, 1))
        x = np.arange(len(train_means))
        width = 0.35
        ax.bar(x - width/2, train_means, width, label='Train', alpha=0.8)
        ax.bar(x + width/2, test_means, width, label='Test', alpha=0.8)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Mean Value')
        ax.set_title('Feature Means: Train vs Test')
        ax.set_xticks(x)
        ax.set_xticklabels([f'F{i+1}' for i in range(len(train_means))], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Index distribution
        ax = axes[1, 1]
        all_indices = np.concatenate([train_indices, test_indices])
        ax.scatter(train_indices, np.ones(len(train_indices)), 
                  alpha=0.5, s=1, label='Train', color='blue')
        ax.scatter(test_indices, np.ones(len(test_indices)) * 1.1,
                  alpha=0.5, s=1, label='Test', color='red')
        ax.set_xlabel('Original Index')
        ax.set_ylabel('')
        ax.set_title('Train/Test Index Distribution')
        ax.set_yticks([1, 1.1])
        ax.set_yticklabels(['Train', 'Test'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'preprocessed_data' / '04_train_test_split.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_correlation(self, X_train: np.ndarray, feature_names: list):
        """Plot feature correlation matrix"""
        # Flatten sequences and compute correlation
        X_flat = X_train.reshape(-1, X_train.shape[-1])
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Feature Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Correlation matrix
        correlation_matrix = np.corrcoef(X_flat.T)
        
        # 1. Heatmap
        ax = axes[0]
        im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(feature_names)))
        ax.set_yticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_yticklabels(feature_names)
        ax.set_title('Feature Correlation Matrix')
        plt.colorbar(im, ax=ax, label='Correlation')
        
        # Add correlation values
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        # 2. Correlation bar plot
        ax = axes[1]
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        correlations = correlation_matrix[mask]
        pairs = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                pairs.append(f'{feature_names[i][:5]}-{feature_names[j][:5]}')
        
        ax.barh(range(len(correlations)), correlations)
        ax.set_yticks(range(len(correlations)))
        ax.set_yticklabels(pairs, fontsize=8)
        ax.set_xlabel('Correlation Coefficient')
        ax.set_title('Pairwise Feature Correlations')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'preprocessed_data' / '05_feature_correlation.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_preprocessing_summary(self, X_train: np.ndarray, X_test: np.ndarray,
                                   feature_names: list, before_norm: np.ndarray,
                                   after_norm: np.ndarray):
        """Save preprocessing summary"""
        summary_path = self.output_dir / 'statistics' / 'preprocessed_data_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("PREPROCESSED DATA STATISTICAL SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Training Sequences Shape: {X_train.shape}\n")
            f.write(f"Test Sequences Shape: {X_test.shape}\n")
            f.write(f"Number of Features: {len(feature_names)}\n")
            f.write(f"Feature Names: {', '.join(feature_names)}\n\n")
            
            f.write("Feature Statistics (Before Normalization):\n")
            f.write("-" * 60 + "\n")
            for i, name in enumerate(feature_names):
                data = before_norm[:, i]
                f.write(f"\n{name}:\n")
                f.write(f"  Mean: {data.mean():.4f}\n")
                f.write(f"  Std: {data.std():.4f}\n")
                f.write(f"  Min: {data.min():.4f}\n")
                f.write(f"  Max: {data.max():.4f}\n")
            
            f.write("\n\nFeature Statistics (After Normalization):\n")
            f.write("-" * 60 + "\n")
            for i, name in enumerate(feature_names):
                data = after_norm[:, i]
                f.write(f"\n{name}:\n")
                f.write(f"  Mean: {data.mean():.4f}\n")
                f.write(f"  Std: {data.std():.4f}\n")
                f.write(f"  Min: {data.min():.4f}\n")
                f.write(f"  Max: {data.max():.4f}\n")
            
            f.write("\n\nTraining Data Statistics:\n")
            f.write("-" * 60 + "\n")
            X_train_flat = X_train.reshape(-1, X_train.shape[-1])
            for i, name in enumerate(feature_names):
                data = X_train_flat[:, i]
                f.write(f"\n{name}:\n")
                f.write(f"  Mean: {data.mean():.4f}\n")
                f.write(f"  Std: {data.std():.4f}\n")
                f.write(f"  Min: {data.min():.4f}\n")
                f.write(f"  Max: {data.max():.4f}\n")
        
        print(f"  ✓ Preprocessing summary saved to: {summary_path}")


def main():
    """Main function to generate all visualizations"""
    print("="*60)
    print("Advanced Anomaly Detection Framework - Data Visualization")
    print("="*60)
    
    # Initialize visualizer
    visualizer = DataVisualizer(output_dir='Visualization')
    
    # Dataset path
    data_path = 'dataset/LCL-June2015v2_94.csv'
    
    # Visualize raw dataset
    print("\n" + "="*60)
    print("STEP 1: Visualizing Raw Dataset")
    print("="*60)
    visualizer.visualize_raw_dataset(data_path, customer_id=None, max_samples=200000)
    
    # Visualize preprocessed data
    print("\n" + "="*60)
    print("STEP 2: Visualizing Preprocessed Dataset")
    print("="*60)
    visualizer.visualize_preprocessed_data(
        data_path,
        customer_id=None,
        window_size=100,
        stride=1
    )
    
    print("\n" + "="*60)
    print("All visualizations completed!")
    print("="*60)
    print(f"\nVisualizations saved to: {visualizer.output_dir}")
    print("\nDirectory structure:")
    print(f"  {visualizer.output_dir}/")
    print(f"    ├── raw_data/          (Raw dataset visualizations)")
    print(f"    ├── preprocessed_data/ (Preprocessed data visualizations)")
    print(f"    └── statistics/        (Statistical summaries)")


if __name__ == '__main__':
    main()

