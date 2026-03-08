"""
Twitter Sentiment Analysis
===========================
This script performs sentiment analysis on Twitter data, analyzing the distribution
of positive, negative, and neutral sentiments across different topics.

The analysis includes:
- Sentiment distribution overview
- Topic-wise sentiment analysis
- Sentiment vs Topic cross-tabulation
- Statistical insights
- Visualizations

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
DATA_FILE = os.path.join(DATA_DIR, 'twitter_training.csv')

def load_data():
    """Load and parse Twitter training dataset."""
    try:
        # Define column names as per dataset specification
        column_names = ['Number', 'Topic', 'Sentiment', 'Comment']
        
        # Try multiple encodings as the file may have non-UTF-8 characters
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']
        df = None
        encoding_used = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(DATA_FILE, names=column_names, encoding=encoding)
                encoding_used = encoding
                print(f"✓ Dataset loaded successfully with '{encoding}' encoding!")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if df is None:
            # If all encodings fail, try with error handling
            print("⚠ Standard encodings failed. Trying with error handling...")
            df = pd.read_csv(DATA_FILE, names=column_names, encoding='utf-8', 
                           errors='replace')
            encoding_used = 'utf-8 (with error handling)'
            print(f"✓ Dataset loaded with error handling!")
        
        print(f"  Encoding used: {encoding_used}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}\n")
        
        # Display sample records
        print("Sample Records:")
        print(df.head())
        
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Clean and validate data."""
    print("\n" + "="*70)
    print("DATA CLEANING & VALIDATION")
    print("="*70)
    
    # Remove rows with missing values
    initial_rows = len(df)
    df = df.dropna()
    removed_rows = initial_rows - len(df)
    
    print(f"\nInitial records: {initial_rows}")
    print(f"Records removed due to missing values: {removed_rows}")
    print(f"Final records: {len(df)}")
    
    # Data type information
    print("\n" + "-"*70)
    print("Data Type Information:")
    print("-"*70)
    print(df.info())
    
    return df

def analyze_sentiment_distribution(df):
    """Analyze overall sentiment distribution."""
    print("\n" + "="*70)
    print("SENTIMENT DISTRIBUTION ANALYSIS")
    print("="*70)
    
    sentiment_counts = df['Sentiment'].value_counts()
    sentiment_pcts = df['Sentiment'].value_counts(normalize=True) * 100
    
    print(f"\nTotal Tweets Analyzed: {len(df)}\n")
    print(f"Sentiment Distribution:")
    for sentiment in sentiment_counts.index:
        count = sentiment_counts[sentiment]
        pct = sentiment_pcts[sentiment]
        print(f"  {sentiment:15} : {count:6,} tweets ({pct:5.1f}%)")
    
    return sentiment_counts, sentiment_pcts

def analyze_topics(df):
    """Analyze unique topics in the dataset."""
    print("\n" + "="*70)
    print("TOPIC ANALYSIS")
    print("="*70)
    
    topics = df['Topic'].unique()
    print(f"\nTotal Unique Topics: {len(topics)}")
    print(f"Topics: {', '.join(topics)}")
    
    # Topic distribution
    topic_counts = df['Topic'].value_counts()
    print(f"\nTweets per Topic:")
    for topic, count in topic_counts.items():
        pct = count / len(df) * 100
        print(f"  {topic:20} : {count:6,} tweets ({pct:5.1f}%)")
    
    return topics, topic_counts

def analyze_topic_sentiment(df):
    """Analyze sentiment distribution by topic."""
    print("\n" + "="*70)
    print("TOPIC-WISE SENTIMENT ANALYSIS")
    print("="*70)
    
    # Create cross-tabulation
    topic_sentiment = pd.crosstab(df['Topic'], df['Sentiment'], margins=True)
    print("\nCross-tabulation (Count):")
    print(topic_sentiment)
    
    # Calculate percentages per topic
    topic_sentiment_pct = pd.crosstab(df['Topic'], df['Sentiment'], normalize='index') * 100
    print("\nDistribution within each Topic (%):")
    print(topic_sentiment_pct.round(2))
    
    return topic_sentiment, topic_sentiment_pct

def analyze_sentiment_intensity(df):
    """Analyze comment length as a proxy for sentiment intensity."""
    print("\n" + "="*70)
    print("COMMENT LENGTH BY SENTIMENT ANALYSIS")
    print("="*70)
    
    # Calculate comment length
    df['Comment_Length'] = df['Comment'].astype(str).str.len()
    df['Word_Count'] = df['Comment'].astype(str).str.split().str.len()
    
    # Statistics by sentiment
    sentiment_stats = df.groupby('Sentiment').agg({
        'Comment_Length': ['mean', 'median', 'std', 'min', 'max'],
        'Word_Count': ['mean', 'median']
    }).round(2)
    
    print("\nComment Statistics by Sentiment:")
    print(sentiment_stats)
    
    return df

def visualize_analysis(df, sentiment_counts, topic_sentiment_pct):
    """Create comprehensive visualizations."""
    # Create larger figure with adjusted height for better label visibility
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # Create axes with custom positioning
    ax1 = fig.add_subplot(gs[0, 0])  # Top-left: Overall Sentiment
    ax2 = fig.add_subplot(gs[0, 1])  # Top-right: Pie Chart
    ax3 = fig.add_subplot(gs[1:, 0])  # Bottom-left: Topic Sentiment (takes 2 rows)
    ax4 = fig.add_subplot(gs[1:, 1])  # Bottom-right: Tweet Count (takes 2 rows)
    
    fig.suptitle('Twitter Sentiment Analysis', fontsize=18, fontweight='bold', y=0.995)
    
    # Color palette
    colors_list = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']  # Green, Red, Orange, Gray
    
    # 1. Overall Sentiment Distribution
    sentiment_counts.plot(kind='bar', ax=ax1, color=colors_list[:len(sentiment_counts)], alpha=0.7)
    ax1.set_title('Overall Sentiment Distribution', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Tweets', fontsize=10)
    ax1.set_xlabel('Sentiment', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    for i, v in enumerate(sentiment_counts.values):
        pct = v / len(df) * 100
        ax1.text(i, v + 50, f'{pct:.1f}%', ha='center', fontweight='bold', fontsize=9)
    
    # 2. Sentiment Distribution Pie Chart
    sentiment_pcts = sentiment_counts / len(df) * 100
    ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=colors_list[:len(sentiment_counts)], startangle=90)
    ax2.set_title('Sentiment Distribution (Pie Chart)', fontsize=12, fontweight='bold')
    
    # 3. Topic-wise Sentiment Stacked Bar (Horizontal for better label readability)
    topic_sentiment_pct_T = topic_sentiment_pct.T  # Transpose for better layout
    topic_sentiment_pct.plot(kind='barh', stacked=True, ax=ax3, 
                              color=colors_list[:len(topic_sentiment_pct.columns)], alpha=0.7)
    ax3.set_title('Sentiment Distribution by Topic (%)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Percentage (%)', fontsize=10)
    ax3.set_ylabel('Topic', fontsize=10)
    ax3.legend(title='Sentiment', fontsize=9, title_fontsize=10, loc='lower right')
    ax3.grid(axis='x', alpha=0.3)
    ax3.set_xlim([0, 100])
    ax3.tick_params(axis='y', labelsize=8)  # Smaller font for topic names
    
    # 4. Topics by Tweet Count (Horizontal bar)
    topic_counts = df['Topic'].value_counts()
    topic_counts.plot(kind='barh', ax=ax4, color='steelblue', alpha=0.7)
    ax4.set_title('Tweet Count by Topic', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Number of Tweets', fontsize=10)
    ax4.set_ylabel('Topic', fontsize=10)
    ax4.grid(axis='x', alpha=0.3)
    ax4.tick_params(axis='y', labelsize=8)  # Smaller font for topic names
    
    # Add count labels on bars (adjusted position for readability)
    for i, v in enumerate(topic_counts.values):
        ax4.text(v + 30, i, f'{v}', va='center', fontweight='bold', fontsize=8)
    
    plt.show()
    plt.show()

def print_key_insights(df, sentiment_counts, topic_sentiment_pct):
    """Print key insights from the analysis."""
    print("\n" + "="*70)
    print("KEY INSIGHTS & FINDINGS")
    print("="*70)
    
    total_tweets = len(df)
    
    print(f"\n1. Overall Sentiment Summary:")
    print(f"   - Total Tweets: {total_tweets:,}")
    
    for sentiment in sentiment_counts.index:
        count = sentiment_counts[sentiment]
        pct = count / total_tweets * 100
        print(f"   - {sentiment}: {count:,} tweets ({pct:.1f}%)")
    
    # Most common sentiment
    most_common_sentiment = sentiment_counts.idxmax()
    print(f"\n2. Dominant Sentiment: {most_common_sentiment}")
    print(f"   The majority of tweets express {most_common_sentiment.lower()} sentiment.")
    
    # Topic analysis
    print(f"\n3. Most Discussed Topics:")
    topic_counts = df['Topic'].value_counts()
    for i, (topic, count) in enumerate(topic_counts.head(3).items(), 1):
        pct = count / total_tweets * 100
        print(f"   {i}. {topic}: {count:,} tweets ({pct:.1f}%)")
    
    # Topic-sentiment insights
    print(f"\n4. Topic-Sentiment Insights:")
    for topic in topic_sentiment_pct.index:
        dominant_sentiment = topic_sentiment_pct.loc[topic].idxmax()
        dominant_pct = topic_sentiment_pct.loc[topic].max()
        print(f"   - {topic}: Predominantly {dominant_sentiment.lower()} ({dominant_pct:.1f}%)")
    
    # Content analysis
    print(f"\n5. Comment Length Analysis:")
    avg_length = df['Comment_Length'].mean()
    avg_words = df['Word_Count'].mean()
    print(f"   - Average comment length: {avg_length:.0f} characters")
    print(f"   - Average words per comment: {avg_words:.1f}")
    
    # Sentiment-specific insights
    if 'Positive' in df['Sentiment'].values:
        positive_words = df[df['Sentiment'] == 'Positive']['Word_Count'].mean()
        print(f"   - Positive tweets average: {positive_words:.1f} words")
    
    if 'Negative' in df['Sentiment'].values:
        negative_words = df[df['Sentiment'] == 'Negative']['Word_Count'].mean()
        print(f"   - Negative tweets average: {negative_words:.1f} words")

def main():
    """Main execution function."""
    print("TWITTER SENTIMENT ANALYSIS")
    print("="*70)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Clean data
    df = clean_data(df)
    
    # Perform analyses
    sentiment_counts, sentiment_pcts = analyze_sentiment_distribution(df)
    topics, topic_counts = analyze_topics(df)
    topic_sentiment, topic_sentiment_pct = analyze_topic_sentiment(df)
    df = analyze_sentiment_intensity(df)
    
    # Print insights
    print_key_insights(df, sentiment_counts, topic_sentiment_pct)
    
    # Create visualizations
    print("\n" + "="*70)
    print("Generating visualizations...")
    print("="*70)
    visualize_analysis(df, sentiment_counts, topic_sentiment_pct)

if __name__ == "__main__":
    main()
