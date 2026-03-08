"""
Gender Distribution Analysis for Pakistan Provinces
====================================================
This script analyzes the gender distribution (Male/Female) across different
provinces in Pakistan, comparing urban and rural populations.

Author: Data Science Project
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set up file paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'sub-division_population_of_pakistan.csv')

def load_data():
    """Load and validate the dataset."""
    try:
        df = pd.read_csv(DATA_FILE)
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE}")
        return None

def calculate_gender_distribution(df):
    """
    Calculate total male and female population by province.
    
    Args:
        df: DataFrame containing population data
        
    Returns:
        DataFrame with gender statistics by province
    """
    # Calculate total male and female (both urban and rural)
    df['TOTAL_MALE'] = df['MALE (URBAN)'] + df['MALE (RURAL)']
    df['TOTAL_FEMALE'] = df['FEMALE (URBAN)'] + df['FEMALE (RURAL)']
    df['TOTAL_POPULATION'] = df['TOTAL_MALE'] + df['TOTAL_FEMALE']
    
    # Select relevant columns
    df_filtered = df[['PROVINCE', 'TOTAL_MALE', 'TOTAL_FEMALE', 'TOTAL_POPULATION',
                       'MALE (URBAN)', 'MALE (RURAL)', 'FEMALE (URBAN)', 'FEMALE (RURAL)']].copy()
    
    # Get unique provinces (first 4)
    provinces = df_filtered['PROVINCE'].unique()
    provinces = provinces[0:4]
    
    print(f"Analyzing provinces: {provinces.tolist()}")
    
    # Aggregate by province
    gender_data = {}
    
    for province in provinces:
        province_data = df_filtered[df_filtered['PROVINCE'] == province]
        
        total_male = province_data['TOTAL_MALE'].sum()
        total_female = province_data['TOTAL_FEMALE'].sum()
        total_pop = province_data['TOTAL_POPULATION'].sum()
        
        gender_data[province] = {
            'Male': total_male,
            'Female': total_female,
            'Total': total_pop,
            'Male %': (total_male / total_pop * 100) if total_pop > 0 else 0,
            'Female %': (total_female / total_pop * 100) if total_pop > 0 else 0,
            'Male Urban': province_data['MALE (URBAN)'].sum(),
            'Male Rural': province_data['MALE (RURAL)'].sum(),
            'Female Urban': province_data['FEMALE (URBAN)'].sum(),
            'Female Rural': province_data['FEMALE (RURAL)'].sum(),
            'Urban Total': province_data['MALE (URBAN)'].sum() + province_data['FEMALE (URBAN)'].sum(),
            'Rural Total': province_data['MALE (RURAL)'].sum() + province_data['FEMALE (RURAL)'].sum(),
        }
    
    return pd.DataFrame(gender_data).T, provinces

def visualize_gender_distribution(gender_df, provinces):
    """Create visualizations for gender distribution analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gender Distribution Analysis - Pakistan Provinces', 
                 fontsize=16, fontweight='bold')
    
    # 1. Stacked bar chart - Male vs Female
    x = np.arange(len(gender_df))
    axes[0, 0].bar(x, gender_df['Male'], label='Male', alpha=0.8, color='steelblue')
    axes[0, 0].bar(x, gender_df['Female'], bottom=gender_df['Male'], 
                   label='Female', alpha=0.8, color='coral')
    axes[0, 0].set_title('Total Population Distribution by Gender')
    axes[0, 0].set_ylabel('Population')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(gender_df.index, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Percentage distribution pie charts (showing for first province as example)
    province = gender_df.index[0]
    sizes = [gender_df.loc[province, 'Male'], gender_df.loc[province, 'Female']]
    colors = ['steelblue', 'coral']
    axes[0, 1].pie(sizes, labels=['Male', 'Female'], autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[0, 1].set_title(f'Gender Distribution: {province}')
    
    # 3. Urban vs Rural comparison
    width = 0.35
    x_pos = np.arange(len(gender_df))
    axes[1, 0].bar(x_pos - width/2, gender_df['Urban Total'], width, 
                   label='Urban', alpha=0.8, color='green')
    axes[1, 0].bar(x_pos + width/2, gender_df['Rural Total'], width, 
                   label='Rural', alpha=0.8, color='orange')
    axes[1, 0].set_title('Urban vs Rural Population')
    axes[1, 0].set_ylabel('Population')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(gender_df.index, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Gender percentage comparison
    axes[1, 1].bar(x, gender_df['Male %'], label='Male %', alpha=0.8)
    axes[1, 1].bar(x, gender_df['Female %'], bottom=gender_df['Male %'], 
                   label='Female %', alpha=0.8)
    axes[1, 1].set_title('Gender Distribution Percentage')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].set_ylim([0, 100])
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(gender_df.index, rotation=45)
    axes[1, 1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_statistics(gender_df):
    """Print detailed statistics."""
    print("\n" + "="*80)
    print("GENDER DISTRIBUTION STATISTICS BY PROVINCE")
    print("="*80)
    print(gender_df[['Male', 'Female', 'Total', 'Male %', 'Female %']].to_string())
    
    print("\n" + "="*80)
    print("URBAN vs RURAL BREAKDOWN")
    print("="*80)
    print(gender_df[['Male Urban', 'Male Rural', 'Female Urban', 'Female Rural', 
                     'Urban Total', 'Rural Total']].to_string())
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    
    # Overall statistics
    total_male = gender_df['Male'].sum()
    total_female = gender_df['Female'].sum()
    total_pop = gender_df['Total'].sum()
    
    print(f"\nTotal Population Across Regions: {total_pop:,.0f}")
    print(f"Total Males: {total_male:,.0f} ({total_male/total_pop*100:.2f}%)")
    print(f"Total Females: {total_female:,.0f} ({total_female/total_pop*100:.2f}%)")
    print(f"Male-to-Female Ratio: {total_male/total_female:.2f}:1")
    
    # Province-wise insights
    print("\nProvince-wise Insights:")
    for province in gender_df.index:
        male = gender_df.loc[province, 'Male']
        female = gender_df.loc[province, 'Female']
        urban = gender_df.loc[province, 'Urban Total']
        rural = gender_df.loc[province, 'Rural Total']
        total = gender_df.loc[province, 'Total']
        
        print(f"\n  {province}:")
        print(f"    Total Population: {total:,.0f}")
        print(f"    Males: {male:,.0f} ({male/total*100:.1f}%) | Females: {female:,.0f} ({female/total*100:.1f}%)")
        print(f"    Urban: {urban:,.0f} ({urban/total*100:.1f}%) | Rural: {rural:,.0f} ({rural/total*100:.1f}%)")
        print(f"    M/F Ratio: {male/female:.2f}:1")

def main():
    """Main execution function."""
    print("Pakistan Gender Distribution Analysis")
    print("="*80)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Calculate statistics
    gender_df, provinces = calculate_gender_distribution(df)
    
    # Print detailed statistics
    print_statistics(gender_df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_gender_distribution(gender_df, provinces)

if __name__ == "__main__":
    main()
