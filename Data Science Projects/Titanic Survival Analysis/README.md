# Titanic Survival Analysis

## Overview
This project performs a comprehensive analysis of the Titanic dataset to understand factors that influenced passenger survival. The analysis explores how demographics like gender, passenger class, and other factors affected survival rates during the disaster.

## Historical Context
The Titanic sank on April 15, 1912, with over 1,500 deaths. This dataset provides insights into the evacuation process and survival patterns, famously following the "Women and Children First" policy.

## Dataset
**Files**:
- `data/gender_submission.csv` - Submission data with gender information
- `data/train.csv` - Training dataset with demographic information
- `data/test.csv` - Test dataset

The dataset includes:
- Passenger ID
- Passenger Class (1st, 2nd, 3rd)
- Gender
- Age
- Number of siblings/spouse (if applicable)
- Number of parents/children (if applicable)
- Ticket information
- Fare details
- Cabin information
- Embarkation port
- Survival status (0 = Did not survive, 1 = Survived)

## Script

### titanic_analysis.py
**Purpose**: Comprehensive survival analysis with multiple perspectives

**Key Features**:
- Overall survival rate calculation
- Gender-based survival analysis
- Passenger class impact analysis
- Gender × Class interaction analysis
- Statistical insights and visualizations

**Analysis Sections**:
1. **Survival Overview** - Overall survival counts and percentages
2. **Survival by Gender** - How gender affected chances of survival
3. **Survival by Passenger Class** - Class-based survival disparities
4. **Interaction Analysis** - Combined gender and class effects

**Output**:
- Detailed statistical tables
- Key insights about survival factors
- 4-panel visualization showing:
  - Overall survival distribution
  - Survival by gender
  - Survival by passenger class
  - Survival rate percentage grid (class × gender)

**How to Run**:
```bash
python titanic_analysis.py
```

## Key Findings

### 1. Gender Effect
The "Women and Children First" evacuation policy is clearly reflected in the data:
- Female passengers had significantly higher survival rates
- This policy prioritized women and children during evacuation

### 2. Class Effect
Social class played a crucial role:
- 1st Class passengers had highest survival rates (best access to lifeboats)
- 3rd Class passengers had lowest survival rates (poor access, language barriers)
- 2nd Class fell between these extremes

### 3. Combined Effects
Gender and class together determined survival chances:
- 1st Class women: Highest survival rates
- 3rd Class men: Lowest survival rates
- Complex interaction shows the hierarchical nature of evacuation

## Statistical Methods Used
- Frequency distributions
- Cross-tabulation analysis
- Contingency tables
- Percentage calculations
- Aggregation and grouping
- Basic descriptive statistics

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
- The script automatically handles missing values in the 'Survived' column
- Data is loaded from the `data/` subdirectory
- Multiple file format support (gender_submission.csv or train.csv)

## Visualizations
All visualizations are color-coded for clarity:
- Red (#d62728): Did not survive
- Green (#2ca02c): Survived
- Interactive bar and contingency charts

## Historical Significance
This analysis demonstrates:
- The importance of social class in early 20th century society
- Gender-based discrimination (positive in this context)
- The effectiveness of evacuation policies
- Human behavior during crisis situations

## Future Enhancements
- Age-based survival analysis with visualizations
- Fare price correlation with survival
- Embarkation port analysis
- Name pattern analysis (titles, family groups)
- Machine learning predictions with logistic regression
- More sophisticated statistical tests (chi-square, etc.)
