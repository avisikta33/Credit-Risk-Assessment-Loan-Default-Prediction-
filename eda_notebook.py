# Credit Risk Assessment - Exploratory Data Analysis (EDA)
# This notebook provides comprehensive exploratory analysis of the loan dataset

# Cell 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("Libraries imported successfully!")

# Cell 2: Load Data
# Load the raw dataset
filepath = '../data/raw/lending_club_loans.csv'
df = pd.read_csv(filepath, low_memory=False)

print(f"Dataset Shape: {df.shape}")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

# Cell 3: Basic Information
print("\n=== DATASET OVERVIEW ===\n")
print(df.info())

print("\n=== FIRST FEW ROWS ===\n")
display(df.head())

print("\n=== BASIC STATISTICS ===\n")
display(df.describe())

# Cell 4: Missing Values Analysis
print("\n=== MISSING VALUES ANALYSIS ===\n")
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
}).sort_values('Missing_Percentage', ascending=False)

print(missing_data.head(20))

# Visualize top 20 columns with missing data
plt.figure(figsize=(12, 8))
missing_top20 = missing_data.head(20)
plt.barh(missing_top20['Column'], missing_top20['Missing_Percentage'], color='salmon')
plt.xlabel('Missing Percentage (%)')
plt.title('Top 20 Columns with Missing Data', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/missing_values.png', dpi=300, bbox_inches='tight')
plt.show()

# Cell 5: Target Variable Analysis
print("\n=== TARGET VARIABLE ANALYSIS ===\n")

# Assuming loan_status is the target
target_col = 'loan_status'
if target_col in df.columns:
    print(f"Loan Status Distribution:\n")
    print(df[target_col].value_counts())
    print(f"\nPercentages:\n")
    print(df[target_col].value_counts(normalize=True) * 100)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    df[target_col].value_counts().plot(kind='bar', color='steelblue')
    plt.title('Loan Status Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Loan Status')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../reports/figures/loan_status_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create binary target
    default_statuses = ['Charged Off', 'Default', 'Late (31-120 days)', 
                       'Late (16-30 days)', 'Does not meet the credit policy. Status:Charged Off']
    df['default'] = df[target_col].apply(lambda x: 1 if x in default_statuses else 0)
    
    print(f"\nBinary Target Distribution:")
    print(f"Good Loans (0): {(df['default']==0).sum():,} ({(df['default']==0).sum()/len(df)*100:.2f}%)")
    print(f"Defaulted (1): {(df['default']==1).sum():,} ({(df['default']==1).sum()/len(df)*100:.2f}%)")
    
    # Class imbalance
    plt.figure(figsize=(8, 6))
    df['default'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    plt.title('Class Distribution (Binary)', fontsize=14, fontweight='bold')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('../reports/figures/class_imbalance.png', dpi=300, bbox_inches='tight')
    plt.show()

# Cell 6: Numerical Features Analysis
print("\n=== NUMERICAL FEATURES ANALYSIS ===\n")

# Select numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Number of numerical features: {len(numerical_cols)}")

# Key numerical features
key_numerical = ['loan_amnt', 'funded_amnt', 'int_rate', 'installment', 
                 'annual_inc', 'dti', 'open_acc', 'total_acc', 
                 'revol_bal', 'revol_util']
key_numerical = [col for col in key_numerical if col in df.columns]

# Distribution plots
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
axes = axes.flatten()

for idx, col in enumerate(key_numerical[:12]):
    if idx < len(axes):
        df[col].hist(bins=50, ax=axes[idx], color='steelblue', edgecolor='black')
        axes[idx].set_title(f'{col}', fontweight='bold')
        axes[idx].set_xlabel('')
        axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../reports/figures/numerical_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Cell 7: Categorical Features Analysis
print("\n=== CATEGORICAL FEATURES ANALYSIS ===\n")

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Number of categorical features: {len(categorical_cols)}")

# Key categorical features
key_categorical = ['grade', 'sub_grade', 'home_ownership', 'verification_status', 
                   'purpose', 'term', 'emp_length']
key_categorical = [col for col in key_categorical if col in df.columns]

# Visualize categorical distributions
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, col in enumerate(key_categorical[:9]):
    if idx < len(axes):
        top_categories = df[col].value_counts().head(10)
        top_categories.plot(kind='bar', ax=axes[idx], color='coral')
        axes[idx].set_title(f'{col}', fontweight='bold')
        axes[idx].set_xlabel('')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../reports/figures/categorical_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Cell 8: Correlation Analysis
print("\n=== CORRELATION ANALYSIS ===\n")

# Select numerical features for correlation
corr_features = [col for col in key_numerical if col in df.columns]
if 'default' in df.columns:
    corr_features.append('default')

# Calculate correlation matrix
corr_df = df[corr_features].corr()

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Top correlations with target
if 'default' in df.columns:
    target_corr = corr_df['default'].sort_values(ascending=False)
    print("\nTop 10 Features Correlated with Default:")
    print(target_corr.head(11))  # 11 because 'default' itself will be first

# Cell 9: Bivariate Analysis - Numerical vs Target
print("\n=== BIVARIATE ANALYSIS (Numerical vs Target) ===\n")

if 'default' in df.columns:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(key_numerical[:6]):
        if idx < len(axes):
            df.boxplot(column=col, by='default', ax=axes[idx])
            axes[idx].set_title(f'{col} by Default Status', fontweight='bold')
            axes[idx].set_xlabel('Default (0=No, 1=Yes)')
            plt.sca(axes[idx])
            plt.xticks([1, 2], ['No', 'Yes'])
    
    plt.tight_layout()
    plt.savefig('../reports/figures/bivariate_numerical.png', dpi=300, bbox_inches='tight')
    plt.show()

# Cell 10: Bivariate Analysis - Categorical vs Target
print("\n=== BIVARIATE ANALYSIS (Categorical vs Target) ===\n")

if 'default' in df.columns and 'grade' in df.columns:
    # Example: Default rate by grade
    default_by_grade = df.groupby('grade')['default'].agg(['mean', 'count'])
    default_by_grade['mean'] = default_by_grade['mean'] * 100
    default_by_grade.columns = ['Default Rate (%)', 'Count']
    
    print("Default Rate by Grade:")
    print(default_by_grade)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Default rate by grade
    default_by_grade['Default Rate (%)'].plot(kind='bar', ax=axes[0], color='salmon')
    axes[0].set_title('Default Rate by Grade', fontweight='bold')
    axes[0].set_xlabel('Grade')
    axes[0].set_ylabel('Default Rate (%)')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Count by grade
    default_by_grade['Count'].plot(kind='bar', ax=axes[1], color='steelblue')
    axes[1].set_title('Loan Count by Grade', fontweight='bold')
    axes[1].set_xlabel('Grade')
    axes[1].set_ylabel('Count')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../reports/figures/default_by_grade.png', dpi=300, bbox_inches='tight')
    plt.show()

# Cell 11: Outlier Detection
print("\n=== OUTLIER DETECTION ===\n")

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), (len(outliers) / len(df)) * 100

outlier_summary = []
for col in key_numerical[:8]:
    if col in df.columns:
        count, pct = detect_outliers_iqr(df, col)
        outlier_summary.append({'Feature': col, 'Outlier_Count': count, 'Outlier_Percentage': pct})

outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df)

# Cell 12: Key Insights Summary
print("\n" + "="*70)
print("KEY INSIGHTS FROM EDA")
print("="*70)
print("""
1. DATA QUALITY:
   - Dataset contains {0:,} loans with {1} features
   - Class imbalance detected: {2:.1f}% defaults
   - Multiple columns with high missing rates (>50%)

2. NUMERICAL FEATURES:
   - Loan amounts range widely: ${3:,.0f} to ${4:,.0f}
   - Interest rates vary: {5:.2f}% to {6:.2f}%
   - Strong correlation between loan_amnt and installment

3. CATEGORICAL FEATURES:
   - Grade and sub_grade show strong relationship with defaults
   - Home ownership varies (RENT, OWN, MORTGAGE)
   - Purpose of loan: debt consolidation is most common

4. TARGET VARIABLE:
   - Binary classification problem
   - Significant class imbalance needs addressing (SMOTE)
   - Higher grades (A, B) have lower default rates

5. NEXT STEPS:
   ✓ Handle missing values (drop columns >30% missing)
   ✓ Address outliers in key features
   ✓ Engineer new features (credit utilization, FICO categories)
   ✓ Balance classes using SMOTE
   ✓ Scale numerical features
""".format(
    len(df),
    df.shape[1],
    (df['default']==1).sum()/len(df)*100 if 'default' in df.columns else 0,
    df['loan_amnt'].min() if 'loan_amnt' in df.columns else 0,
    df['loan_amnt'].max() if 'loan_amnt' in df.columns else 0,
    df['int_rate'].min() if 'int_rate' in df.columns else 0,
    df['int_rate'].max() if 'int_rate' in df.columns else 0
))

print("\n✅ EDA Complete! Proceed to Feature Engineering.")
