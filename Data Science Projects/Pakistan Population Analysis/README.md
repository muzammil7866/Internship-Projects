# Pakistan Population Analysis

## Overview
This project analyzes population data across different provinces in Pakistan, focusing on two key aspects:
1. **Annual Growth Rate (AGR)** - Analyzing year-over-year population growth trends
2. **Gender Distribution** - Examining male and female population distribution across urban and rural areas

## Dataset
**File**: `sub-division_population_of_pakistan.csv`

The dataset contains population statistics at the sub-division level for Pakistan, including:
- Province information
- Urban and rural male population
- Urban and rural female population
- Annual growth rates (urban and rural)

## Scripts

### 1. annual_growth_rate_analysis.py
**Purpose**: Analyze annual population growth rates across provinces

**Key Features**:
- Calculates province-wise average annual growth rates
- Compares urban vs rural growth rates
- Identifies highest and lowest growth regions
- Provides statistical measures (mean, std dev, max, min)
- Creates multi-panel visualizations

**Output**:
- Statistical summary table
- 4-panel visualization showing:
  - Average AGR by province
  - Urban vs Rural comparison
  - Growth rate range (min-max)
  - Variability (standard deviation)

**How to Run**:
```bash
python annual_growth_rate_analysis.py
```

### 2. gender_distribution_analysis.py
**Purpose**: Analyze gender distribution across provinces and urbanization patterns

**Key Features**:
- Calculates total male and female population by province
- Provides gender ratio analysis
- Compares urban vs rural distribution
- Analyzes male-to-female ratios
- Creates detailed visualizations

**Output**:
- Gender statistics table
- Urban vs Rural breakdown
- 4-panel visualization showing:
  - Total population by gender (stacked bar)
  - Pie chart for a selected province
  - Urban vs Rural population
  - Gender percentage distribution

**How to Run**:
```bash
python gender_distribution_analysis.py
```

## Key Insights

### Growth Rate Findings
- Analysis of 4 major provinces (adjustable in code)
- Standard deviation shows variability of growth across districts
- Urban areas often show different growth patterns compared to rural areas

### Gender Distribution Findings
- Overall male-to-female ratios across provinces
- Urbanization trends (urban vs rural population split)
- Gender balance in different regions

## Requirements
- pandas
- matplotlib
- numpy

## Installation
```bash
pip install pandas matplotlib numpy
```

## Data Notes
- Ensure the CSV file is in the same directory as the scripts
- The scripts use the first 4 provinces by default (adjustable)
- Missing values are handled during data processing

## Future Enhancements
- Add time-series analysis if historical data becomes available
- Implement predictive modeling for future population estimates
- Add regression analysis for growth factors
- Create interactive visualizations with plotly
