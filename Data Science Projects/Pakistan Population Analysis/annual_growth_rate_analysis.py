"""
Annual Growth Rate Analysis for Pakistan Provinces
====================================================
This script analyzes the annual population growth rate across different provinces
in Pakistan, comparing urban and rural growth rates and calculating province-wise
averages.

Author: Data Science Project
Date: 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set up file paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'sub-division_population_of_pakistan.csv')

def load_data():
    """Load and validate the dataset."""
    try:
        df = pd.read_csv(DATA_FILE)
        print("Dataset loaded successfully!")
        print(f"\nDataset shape: {df.shape}")
        print(f"\nColumn names:\n{df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE}")
        return None

def calculate_province_agr(df):
    """
    Calculate average annual growth rate for each province.
    
    Args:
        df: DataFrame containing population data
        
    Returns:
        DataFrame with average AGR by province
    """
    # Create total annual growth rate (urban + rural)
    df['TOTAL_AGR'] = df['ANNUAL GROWTH RATE (URBAN)'] + df['ANNUAL GROWTH RATE (RURAL)']
    
    # Select relevant columns
    df_filtered = df[['PROVINCE', 'TOTAL_AGR', 'ANNUAL GROWTH RATE (URBAN)', 
                       'ANNUAL GROWTH RATE (RURAL)']].copy()
    
    # Get unique provinces (first 4 as per original requirement)
    provinces = df_filtered['PROVINCE'].unique()
    provinces = provinces[0:4]
    
    print(f"\nAnalyzing provinces: {provinces.tolist()}")
    
    # Calculate aggregated statistics
    agr_data = {}
    
    for province in provinces:
        province_data = df_filtered[df_filtered['PROVINCE'] == province]
        
        agr_data[province] = {
            'Average AGR': province_data['TOTAL_AGR'].mean(),
            'Average Urban AGR': province_data['ANNUAL GROWTH RATE (URBAN)'].mean(),
            'Average Rural AGR': province_data['ANNUAL GROWTH RATE (RURAL)'].mean(),
            'Max AGR': province_data['TOTAL_AGR'].max(),
            'Min AGR': province_data['TOTAL_AGR'].min(),
            'Std Dev': province_data['TOTAL_AGR'].std(),
            'Districts': len(province_data)
        }
    
    return pd.DataFrame(agr_data).T, provinces

def visualize_agr(agr_df, provinces):
    """Create visualizations for AGR analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Annual Growth Rate Analysis - Pakistan Provinces', fontsize=16, fontweight='bold')
    
    # 1. Bar chart of average AGR by province
    axes[0, 0].bar(agr_df.index, agr_df['Average AGR'], color='steelblue', alpha=0.7)
    axes[0, 0].set_title('Average Annual Growth Rate by Province')
    axes[0, 0].set_ylabel('Growth Rate (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Urban vs Rural comparison
    x = np.arange(len(agr_df))
    width = 0.35
    axes[0, 1].bar(x - width/2, agr_df['Average Urban AGR'], width, label='Urban', alpha=0.8)
    axes[0, 1].bar(x + width/2, agr_df['Average Rural AGR'], width, label='Rural', alpha=0.8)
    axes[0, 1].set_title('Urban vs Rural Growth Rate')
    axes[0, 1].set_ylabel('Growth Rate (%)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(agr_df.index, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Range and distribution
    axes[1, 0].bar(agr_df.index, agr_df['Max AGR'] - agr_df['Min AGR'], 
                   bottom=agr_df['Min AGR'], alpha=0.7, color='green')
    axes[1, 0].set_title('AGR Range by Province (Min-Max)')
    axes[1, 0].set_ylabel('Growth Rate (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Standard deviation
    axes[1, 1].bar(agr_df.index, agr_df['Std Dev'], color='coral', alpha=0.7)
    axes[1, 1].set_title('Variability (Standard Deviation) by Province')
    axes[1, 1].set_ylabel('Std Dev (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_statistics(agr_df):
    """Print detailed statistics."""
    print("\n" + "="*70)
    print("ANNUAL GROWTH RATE STATISTICS BY PROVINCE")
    print("="*70)
    print(agr_df.to_string())
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    
    # Find provinces with highest/lowest growth
    max_agr_province = agr_df['Average AGR'].idxmax()
    min_agr_province = agr_df['Average AGR'].idxmin()
    
    print(f"\nHighest Average Growth Rate: {max_agr_province} ({agr_df.loc[max_agr_province, 'Average AGR']:.2f}%)")
    print(f"Lowest Average Growth Rate: {min_agr_province} ({agr_df.loc[min_agr_province, 'Average AGR']:.2f}%)")
    
    # Urban vs Rural insight
    print("\nUrban vs Rural Comparison:")
    for province in agr_df.index:
        urban = agr_df.loc[province, 'Average Urban AGR']
        rural = agr_df.loc[province, 'Average Rural AGR']
        diff = urban - rural
        print(f"  {province}: Urban {urban:.2f}% | Rural {rural:.2f}% | Difference: {diff:+.2f}%")

def main():
    """Main execution function."""
    print("Pakistan Population Growth Analysis")
    print("="*70)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Calculate statistics
    agr_df, provinces = calculate_province_agr(df)
    
    # Print detailed statistics
    print_statistics(agr_df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_agr(agr_df, provinces)

if __name__ == "__main__":
    main()
