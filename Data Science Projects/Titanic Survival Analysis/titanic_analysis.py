"""
Titanic Survival Analysis
===========================
This script performs comprehensive analysis of the Titanic dataset, examining
survival rates based on gender, passenger class, and other demographic factors.

The analysis includes:
- Survival rate by gender
- Survival rate by passenger class
- Age distribution analysis
- Cross-tabulation analysis
- Statistical insights

Author: Data Science Project
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

# Set up file paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
GENDER_SUBMISSION_FILE = os.path.join(DATA_DIR, 'gender_submission.csv')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

def load_data():
    """Load all Titanic datasets."""
    try:
        # Prefer train.csv for full analysis (has all columns)
        # Fall back to gender_submission.csv if train.csv not available
        if os.path.exists(TRAIN_FILE):
            df = pd.read_csv(TRAIN_FILE)
            print("✓ Loaded train.csv (Full dataset with demographics)")
            is_full_dataset = True
        elif os.path.exists(GENDER_SUBMISSION_FILE):
            df = pd.read_csv(GENDER_SUBMISSION_FILE)
            print("⚠ Loaded gender_submission.csv (Limited columns)")
            print("  Note: This file only contains PassengerId and Survived columns")
            print("  For full analysis with gender/class breakdown, use train.csv")
            is_full_dataset = False
        else:
            print("Error: No data files found!")
            print(f"Expected locations:")
            print(f"  - {TRAIN_FILE}")
            print(f"  - {GENDER_SUBMISSION_FILE}")
            return None, None
        
        print(f"\nDataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df, is_full_dataset
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def analyze_survival_overview(df):
    """Analyze basic survival statistics."""
    print("\n" + "="*70)
    print("SURVIVAL OVERVIEW")
    print("="*70)
    
    if 'Survived' in df.columns:
        survival_counts = df['Survived'].value_counts()
        survival_rates = df['Survived'].value_counts(normalize=True) * 100
        
        print(f"\nTotal Records: {len(df)}")
        print(f"\nSurvival Status:")
        print(f"  Did Not Survive (0): {survival_counts.get(0, 0):,} ({survival_rates.get(0, 0):.1f}%)")
        print(f"  Survived (1): {survival_counts.get(1, 0):,} ({survival_rates.get(1, 0):.1f}%)")
    else:
        print("'Survived' column not found in dataset")

def analyze_survival_by_gender(df):
    """Analyze survival rates by gender."""
    print("\n" + "="*70)
    print("SURVIVAL ANALYSIS BY GENDER")
    print("="*70)
    
    if 'Sex' in df.columns and 'Survived' in df.columns:
        gender_survival = df.groupby('Sex')['Survived'].agg([
            ('Count', 'count'),
            ('Survived', 'sum'),
            ('Died', lambda x: (x == 0).sum()),
            ('Survival Rate %', lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0)
        ])
        
        print("\n" + gender_survival.to_string())
        
        # Calculate additional statistics
        for gender in df['Sex'].unique():
            gender_data = df[df['Sex'] == gender]
            survived = gender_data['Survived'].sum()
            total = len(gender_data)
            print(f"\n{gender.capitalize()}:")
            print(f"  Total: {total}")
            print(f"  Survived: {survived} ({survived/total*100:.1f}%)")
    else:
        print("Required columns not found")

def analyze_survival_by_class(df):
    """Analyze survival rates by passenger class."""
    print("\n" + "="*70)
    print("SURVIVAL ANALYSIS BY PASSENGER CLASS")
    print("="*70)
    
    if 'Pclass' in df.columns and 'Survived' in df.columns:
        class_survival = df.groupby('Pclass')['Survived'].agg([
            ('Count', 'count'),
            ('Survived', 'sum'),
            ('Died', lambda x: (x == 0).sum()),
            ('Survival Rate %', lambda x: (x.sum() / len(x) * 100) if len(x) > 0 else 0)
        ])
        
        print("\n" + class_survival.to_string())
    else:
        print("Required columns not found")

def analyze_interaction(df):
    """Analyze survival by gender and class interaction."""
    print("\n" + "="*70)
    print("SURVIVAL BY GENDER & CLASS (INTERACTION)")
    print("="*70)
    
    if 'Sex' in df.columns and 'Pclass' in df.columns and 'Survived' in df.columns:
        interaction = df.groupby(['Sex', 'Pclass'])['Survived'].agg([
            ('Count', 'count'),
            ('Survived', 'sum'),
            ('Survival Rate %', lambda x: (x.sum() / len(x) * 100))
        ])
        
        print("\n" + interaction.to_string())
    else:
        print("Required columns not found")

def visualize_survival_analysis(df, is_full_dataset):
    """Create comprehensive visualizations."""
    if 'Survived' not in df.columns:
        print("Cannot create visualizations - 'Survived' column not found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Titanic Survival Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overall Survival Counts
    survival_counts = df['Survived'].value_counts()
    axes[0, 0].bar(['Did Not Survive', 'Survived'], survival_counts.values, 
                   color=['#d62728', '#2ca02c'], alpha=0.7)
    axes[0, 0].set_title('Overall Survival Counts')
    axes[0, 0].set_ylabel('Number of People')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add percentage on bars
    for i, v in enumerate(survival_counts.values):
        pct = v / len(df) * 100
        axes[0, 0].text(i, v + 5, f'{pct:.1f}%', ha='center', fontweight='bold')
    
    # 2. Survival by Gender
    if 'Sex' in df.columns:
        gender_survival = pd.crosstab(df['Sex'], df['Survived'])
        gender_survival.plot(kind='bar', ax=axes[0, 1], color=['#d62728', '#2ca02c'], alpha=0.7)
        axes[0, 1].set_title('Survival by Gender')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xlabel('Gender')
        axes[0, 1].legend(['Did Not Survive', 'Survived'])
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=0)
    else:
        axes[0, 1].text(0.5, 0.5, 'Gender data\nnot available\nin dataset', 
                       ha='center', va='center', fontsize=12, color='gray',
                       transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Survival by Gender (No Data)')
        axes[0, 1].axis('off')
    
    # 3. Survival by Passenger Class
    if 'Pclass' in df.columns:
        class_survival = pd.crosstab(df['Pclass'], df['Survived'])
        class_survival.plot(kind='bar', ax=axes[1, 0], color=['#d62728', '#2ca02c'], alpha=0.7)
        axes[1, 0].set_title('Survival by Passenger Class')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xlabel('Passenger Class')
        axes[1, 0].legend(['Did Not Survive', 'Survived'])
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=0)
    else:
        axes[1, 0].text(0.5, 0.5, 'Passenger Class data\nnot available\nin dataset', 
                       ha='center', va='center', fontsize=12, color='gray',
                       transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Survival by Passenger Class (No Data)')
        axes[1, 0].axis('off')
    
    # 4. Survival Rate % by Gender and Class
    if 'Sex' in df.columns and 'Pclass' in df.columns:
        pivot_data = df.pivot_table(
            values='Survived', 
            index='Pclass', 
            columns='Sex', 
            aggfunc=lambda x: (x.sum()/len(x)*100)
        )
        pivot_data.plot(kind='bar', ax=axes[1, 1], alpha=0.7)
        axes[1, 1].set_title('Survival Rate (%) by Class & Gender')
        axes[1, 1].set_ylabel('Survival Rate (%)')
        axes[1, 1].set_xlabel('Passenger Class')
        axes[1, 1].legend(title='Gender')
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=0)
        axes[1, 1].axhline(y=50, color='red', linestyle='--', alpha=0.5)
    else:
        axes[1, 1].text(0.5, 0.5, 'Gender + Class data\nnot available\nin dataset', 
                       ha='center', va='center', fontsize=12, color='gray',
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Survival by Class & Gender (No Data)')
        axes[1, 1].axis('off')
    
    # Add note if using limited dataset
    if not is_full_dataset:
        fig.text(0.5, 0.02, 'Note: Using limited dataset (gender_submission.csv). For full analysis, use train.csv', 
                ha='center', fontsize=10, style='italic', color='red')
    
    plt.tight_layout()
    if not is_full_dataset:
        plt.subplots_adjust(bottom=0.08)
    plt.show()

def print_key_insights(df):
    """Print key insights from the analysis."""
    print("\n" + "="*70)
    print("KEY INSIGHTS & FINDINGS")
    print("="*70)
    
    if 'Survived' in df.columns:
        total_survived = df['Survived'].sum()
        total_records = len(df)
        survival_rate = total_survived / total_records * 100
        
        print(f"\n1. Overall Statistics:")
        print(f"   - Total Passengers: {total_records}")
        print(f"   - Survivors: {total_survived}")
        print(f"   - Overall Survival Rate: {survival_rate:.1f}%")
        
        if 'Sex' in df.columns:
            print(f"\n2. Gender Differences:")
            for gender in df['Sex'].unique():
                gender_df = df[df['Sex'] == gender]
                gender_survival = gender_df['Survived'].sum() / len(gender_df) * 100
                print(f"   - {gender.capitalize()} Survival Rate: {gender_survival:.1f}%")
        
        if 'Pclass' in df.columns:
            print(f"\n3. Class Differences (\"Women and Children First\" policy effect):")
            for pclass in sorted(df['Pclass'].unique()):
                class_df = df[df['Pclass'] == pclass]
                class_survival = class_df['Survived'].sum() / len(class_df) * 100
                class_name = {1: 'First', 2: 'Second', 3: 'Third'}
                print(f"   - Class {pclass} ({class_name.get(pclass, 'Unknown')}): {class_survival:.1f}%")
            
            print(f"\n   Observation: Higher class passengers had better survival rates,")
            print(f"   reflecting their prioritization during evacuation.")

def main():
    """Main execution function."""
    print("TITANIC SURVIVAL ANALYSIS")
    print("="*70)
    
    # Load data
    df, is_full_dataset = load_data()
    if df is None:
        return
    
    # Remove any rows with missing 'Survived' values
    df = df.dropna(subset=['Survived'])
    
    # Only perform full analysis if we have the necessary columns
    if is_full_dataset:
        # Perform analyses
        analyze_survival_overview(df)
        analyze_survival_by_gender(df)
        analyze_survival_by_class(df)
        analyze_interaction(df)
        print_key_insights(df)
    else:
        print("\n⚠ LIMITED DATASET WARNING ⚠")
        print("="*70)
        print("The gender_submission.csv file contains limited columns.")
        print("Only 'Survived' column is available for analysis.")
        print("\nTo perform full analysis including:")
        print("  • Survival by Gender")
        print("  • Survival by Passenger Class")
        print("  • Gender × Class Interaction Analysis")
        print("\nPlease use train.csv instead by placing it in the data/ folder.")
        print("="*70)
        
        # Still do basic analysis with available data
        analyze_survival_overview(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_survival_analysis(df, is_full_dataset)

if __name__ == "__main__":
    main()
