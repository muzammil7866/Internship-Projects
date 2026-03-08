# Twitter Sentiment Analysis

## Overview
This project performs sentiment analysis on Twitter data to understand public opinion and attitudes toward various topics. The analysis categorizes tweets as Positive, Negative, or Neutral and provides insights into sentiment distributions across different discussion topics.

## Use Case
Sentiment analysis is crucial for:
- Brand reputation monitoring
- Customer feedback analysis
- Social media trend analysis
- Public opinion tracking
- Crisis detection and management

## Dataset
**File**: `data/twitter_training.csv`

The dataset contains:
- **Number**: Tweet ID/index
- **Topic**: Main topic or subject of the tweet
- **Sentiment**: Classification (Positive, Negative, Neutral)
- **Comment**: The actual tweet text

## Script

### sentiment_analysis.py
**Purpose**: Comprehensive sentiment analysis with topic-based breakdown

**Key Features**:
- Data loading and validation
- Sentiment distribution analysis
- Topic identification and frequency
- Topic-sentiment cross-tabulation
- Comment length and engagement analysis
- Statistical insights and visualizations

**Analysis Sections**:
1. **Data Cleaning & Validation** - Handles missing values, displays data info
2. **Sentiment Distribution** - Overall positive/negative/neutral breakdown
3. **Topic Analysis** - Unique topics and tweet counts per topic
4. **Topic-Sentiment Analysis** - Sentiment distribution within each topic
5. **Comment Length Analysis** - Text length as proxy for engagement intensity

**Output**:
- Data quality report
- Sentiment distribution statistics (counts and percentages)
- Topic breakdown with percentages
- Cross-tabulation tables (topic × sentiment)
- Key insights about sentiment patterns
- 4-panel visualization showing:
  - Overall sentiment distribution (bar chart)
  - Sentiment pie chart
  - Topic-wise sentiment distribution (stacked percentage bar)
  - Tweet count by topic (horizontal bar)

**How to Run**:
```bash
python sentiment_analysis.py
```

## Key Metrics Analyzed

### 1. Sentiment Distribution
- **Positive**: Tweets expressing favorable opinions
- **Negative**: Tweets expressing critical or unfavorable opinions
- **Neutral**: Tweets that are informational or balanced

### 2. Topic Coverage
- Identifies all unique topics in the dataset
- Calculates percentage of discussion for each topic
- Shows tweet volume per topic

### 3. Topic-Sentiment Relationships
- Determines which topics tend to generate positive vs negative sentiment
- Identifies controversial topics (mixed sentiment)
- Finds consensus topics (dominant sentiment)

### 4. Engagement Patterns
- Average comment length per sentiment
- Word count analysis
- Engagement intensity (longer comments = more engagement)

## Statistical Methods Used
- Frequency distributions
- Cross-tabulation analysis
- Percentage calculations
- Descriptive statistics
- Text length analysis
- Pivot tables

## Requirements
- pandas
- numpy
- matplotlib
- seaborn

## Installation
```bash
pip install pandas numpy matplotlib seaborn
```

## Data Preparation
- Automatic handling of missing values
- Data validation and cleaning steps
- Data type information displayed
- Sample records shown for verification

## Visualizations
Color scheme for sentiment:
- **Green (#2ecc71)**: Positive sentiment
- **Red (#e74c3c)**: Negative sentiment
- **Orange (#f39c12)**: Neutral sentiment

Visualization types:
- Bar charts for distributions
- Pie charts for percentage visualization
- Stacked bars for topic-sentiment relationships
- Horizontal bars for category comparison

## Interpretation Guide

### How to Read the Results
1. **Sentiment Bar Chart**: Shows absolute tweet counts per sentiment class
2. **Sentiment Pie Chart**: Shows percentage distribution at a glance
3. **Topic-Sentiment Stacked Bar**: Shows dominant sentiment for each topic
4. **Topic Count Chart**: Indicates which topics are most discussed

### What Different Sentiments Mean
- **High Positive %**: Topic generates favorable discussion
- **High Negative %**: Topic is controversial or problematic
- **High Neutral %**: Topic is factual/informational

## Business Applications

### Brand Monitoring
- Monitor customer sentiment toward your brand
- Track sentiment changes over time
- Identify emerging issues

### Market Research
- Understand customer opinions on products
- Compare sentiment across competing brands
- Identify customer pain points

### Crisis Management
- Detect negative sentiment spikes
- Track reputation impact
- Guide response strategies

## Future Enhancements
- Time-series sentiment tracking
- Emotion detection (beyond positive/negative)
- Topic modeling (LDA, NMF)
- Named entity recognition (NER)
- Machine learning classification models
- Word frequency and word clouds
- Sentiment polarity scores (not just categories)
- Language detection for multilingual analysis
- Integration with real-time Twitter API
- Comparative analysis across multiple datasets
