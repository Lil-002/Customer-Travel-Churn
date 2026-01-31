import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from scipy import stats

#
# Setting up the Output Directory
#

# Create output directories with timestamp for unique runs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_output_dir = f"data/eda_output_{timestamp}"
tables_dir = os.path.join(base_output_dir, "tables")
visuals_dir = os.path.join(base_output_dir, "visuals")

# Create directories if they don't exist
os.makedirs(tables_dir, exist_ok=True)
os.makedirs(visuals_dir, exist_ok=True)

print(f"Output will be saved to: {base_output_dir}")
print(f"Tables directory: {tables_dir}")
print(f"Visuals directory: {visuals_dir}")

#
# Load data .. no need to check column names
# No missing values .. column names Ok ..
# ============================================================================

df = pd.read_csv('Customertravel.csv')

if 'Ages' in df.columns and 'Age' not in df.columns:
    df.rename(columns={'Ages': 'Age'}, inplace=True)
# quick sanity print to confirm
print("Columns after rename:", list(df.columns))


print("\nFirst 5 rows:")
print(df.head())
print("\nColumn types:")
print(df.dtypes)

#
# Helper functions - to save csv files to /tables dir
# and charts to /visuals
#

def save_table_to_csv(dataframe, filename, description=""):
    """Save DataFrame to CSV with description"""
    filepath = os.path.join(tables_dir, filename)
    dataframe.to_csv(filepath, index=True)
    print(f"  ✓ Saved table: {filename}")
    if description:
        print(f"    Description: {description}")
    return filepath


def save_plot_to_png(fig, filename, description=""):
    """Save matplotlib figure to PNG"""
    filepath = os.path.join(visuals_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved visual: {filename}")
    if description:
        print(f"    Description: {description}")
    return filepath


#
# 1. Target variable analysis
#

print("\n" + "=" * 80)
print("1. TARGET VARIABLE ANALYSIS")
print("=" * 80)

# Create a copy of Target with labels for better visualization
df['Target_Label'] = df['Target'].map({0: 'No Churn', 1: 'Churned'})

# Target distribution summary
target_summary = df['Target'].value_counts().reset_index()
target_summary.columns = ['Churn', 'Count']
target_summary['Percentage'] = (target_summary['Count'] / len(df) * 100).round(2)
target_summary['Churn_Label'] = target_summary['Churn'].map({0: 'No Churn', 1: 'Churned'})

# Save target summary table
save_table_to_csv(target_summary, "1_target_summary.csv",
                  "Distribution of churn vs no churn in dataset")

# Visualize target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
axes[0].pie(target_summary['Count'], labels=target_summary['Churn_Label'],
            autopct='%1.1f%%', colors=['skyblue', 'salmon'], startangle=90)
axes[0].set_title('Churn Distribution (Pie Chart)', fontsize=14, fontweight='bold')

# Bar chart
sns.barplot(data=target_summary, x='Churn_Label', y='Count', hue='Churn_Label',
            palette=['skyblue', 'salmon'], legend=False, ax=axes[1])
axes[1].set_title('Churn Distribution (Bar Chart)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Churn Status')
axes[1].set_ylabel('Count')

# Add count labels on bars
for i, v in enumerate(target_summary['Count']):
    axes[1].text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
save_plot_to_png(fig, "1_target_distribution.png",
                 "Bar Chart - Churn Distribution")
plt.show()

#
# 2. Age Analysis
#

print("\n" + "=" * 80)
print("2. AGE ANALYSIS")
print("=" * 80)

# Create age groups
df['AgeGroup'] = pd.cut(df['Age'],
                        bins=[20, 30, 40, 50, 60],
                        labels=['20-29', '30-39', '40-49', '50-59'])

# Age statistics by churn status with MODE
age_stats_by_churn = df.groupby('Target')['Age'].agg([
    'count', 'mean', 'median',
    lambda x: stats.mode(x, keepdims=False).mode,
    'std', 'min', 'max',
    lambda x: x.quantile(0.25),
    lambda x: x.quantile(0.75)
]).round(2)
age_stats_by_churn.columns = ['Count', 'Mean', 'Median', 'Mode', 'Std', 'Min', 'Max', 'Q1', 'Q3']
age_stats_by_churn.index = age_stats_by_churn.index.map({0: 'No Churn', 1: 'Churned'})

# Overall age statistics with MODE
age_mode = stats.mode(df['Age'], keepdims=False).mode
overall_age_stats = pd.DataFrame({
    'Statistic': ['Count', 'Mean', 'Median', 'Mode', 'Std', 'Min', 'Max', 'Q1', 'Q3'],
    'Value': [
        len(df['Age']),
        round(df['Age'].mean(), 2),
        round(df['Age'].median(), 2),
        round(age_mode, 2),
        round(df['Age'].std(), 2),
        df['Age'].min(),
        df['Age'].max(),
        round(df['Age'].quantile(0.25), 2),
        round(df['Age'].quantile(0.75), 2)
    ]
})

# Churn rate by age group
age_group_churn = df.groupby('AgeGroup')['Target'].agg(['count', 'mean', 'sum']).round(3)
age_group_churn.columns = ['Count', 'Churn_Rate', 'Churned_Count']
age_group_churn['No_Churn_Count'] = age_group_churn['Count'] - age_group_churn['Churned_Count']

# Save age analysis tables
save_table_to_csv(age_stats_by_churn, "2_age_stats_by_churn.csv",
                  "Age statistics grouped by churn status")
save_table_to_csv(overall_age_stats, "2_overall_age_stats.csv",
                  "Overall age statistics for entire dataset")
save_table_to_csv(age_group_churn, "2_age_group_churn_rates.csv",
                  "Churn rates by age group (binned)")

# Visualize age analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Histogram with KDE - using Target_Label for proper coloring
sns.histplot(data=df, x='Age', hue='Target_Label', kde=True, element='step',
             palette=['skyblue', 'salmon'], ax=axes[0, 0])
axes[0, 0].set_title('Age Distribution by Churn Status', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

# Boxplot - using Target_Label for proper coloring
sns.boxplot(data=df, x='Target_Label', y='Age', hue='Target_Label',
            palette=['skyblue', 'salmon'], legend=False, ax=axes[0, 1])
axes[0, 1].set_title('Age Distribution by Churn Status (Boxplot)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Churn Status')
axes[0, 1].set_ylabel('Age')

# Churn rate by age group
sns.barplot(data=age_group_churn.reset_index(), x='AgeGroup', y='Churn_Rate',
            color='steelblue', ax=axes[1, 0])
axes[1, 0].set_title('Churn Rate by Age Group', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Age Group')
axes[1, 0].set_ylabel('Churn Rate')

# Add churn rate labels on bars
for i, v in enumerate(age_group_churn['Churn_Rate']):
    axes[1, 0].text(i, v + 0.01, f'{v:.2%}', ha='center', va='bottom', fontweight='bold')

#
# 3 - Categorical analysis
#

print("\n" + "=" * 80)
print("3. CATEGORICAL VARIABLES ANALYSIS")
print("=" * 80)

categorical_cols = ['FrequentFlyer', 'AnnualIncomeClass',
                    'AccountSyncedToSocialMedia', 'BookedHotelOrNot']

# Initialize dictionary to store all categorical analysis results
categorical_analysis = {}

for col in categorical_cols:
    print(f"\nAnalyzing: {col}")

    # Calculate churn rates
    churn_stats = df.groupby(col)['Target'].agg([
        ('Count', 'count'),
        ('Churned', 'sum'),
        ('Not_Churned', lambda x: (x == 0).sum()),
        ('Churn_Rate', 'mean'),
        ('Std_Error', lambda x: np.std(x) / np.sqrt(len(x)))
    ]).round(3)

    churn_stats['Churn_Rate_Pct'] = (churn_stats['Churn_Rate'] * 100).round(1)

    # Save individual table
    save_table_to_csv(churn_stats, f"3_{col.lower()}_churn_stats.csv",
                      f"Churn statistics for {col}")

    # Store for combined analysis
    categorical_analysis[col] = churn_stats

# Create combined categorical summary
combined_categorical = []
for col in categorical_cols:
    temp_df = categorical_analysis[col].copy()
    temp_df['Variable'] = col
    temp_df['Category'] = temp_df.index
    combined_categorical.append(temp_df.reset_index(drop=True))

combined_categorical_df = pd.concat(combined_categorical, ignore_index=True)
combined_categorical_df = combined_categorical_df[['Variable', 'Category', 'Count',
                                                   'Churned', 'Not_Churned',
                                                   'Churn_Rate', 'Churn_Rate_Pct',
                                                   'Std_Error']]

save_table_to_csv(combined_categorical_df, "3_combined_categorical_analysis.csv",
                  "Combined churn analysis for all categorical variables")

# Visualize categorical variables
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for idx, col in enumerate(categorical_cols):
    # Create stacked bar chart
    crosstab = pd.crosstab(df[col], df['Target'], normalize='index')
    crosstab = crosstab[[0, 1]]  # Ensure correct order

    crosstab.plot(kind='bar', stacked=True, ax=axes[idx],
                  color=['skyblue', 'salmon'], width=0.7)

    axes[idx].set_title(f'{col} - Churn Distribution', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Proportion')
    axes[idx].legend(['No Churn', 'Churned'], loc='upper right')
    axes[idx].grid(axis='y', alpha=0.3)

    # Add percentage labels
    for i, (index, row) in enumerate(crosstab.iterrows()):
        height_accum = 0
        for j, val in enumerate(row):
            if val > 0.05:  # Only label if segment is large enough
                axes[idx].text(i, height_accum + val / 2,
                               f'{val:.1%}',
                               ha='center', va='center',
                               fontsize=9, fontweight='bold',
                               color='white' if j == 1 else 'black')
            height_accum += val

plt.tight_layout()
save_plot_to_png(fig, "3_categorical_variables_stacked.png",
                 "Stacked bar charts showing churn distribution for all categorical variables")
plt.show()

#
# 4. Services Opted analysis
#

print("\n" + "=" * 80)
print("4. SERVICES OPTED ANALYSIS")
print("=" * 80)

# ServicesOpted statistics by churn status with MODE
services_stats_by_churn = df.groupby('Target')['ServicesOpted'].agg([
    'count', 'mean', 'median',
    lambda x: stats.mode(x, keepdims=False).mode,
    'std', 'min', 'max',
    lambda x: x.quantile(0.25),
    lambda x: x.quantile(0.75)
]).round(2)

services_stats_by_churn.columns = ['Count', 'Mean', 'Median', 'Mode', 'Std', 'Min', 'Max', 'Q1', 'Q3']
services_stats_by_churn.index = services_stats_by_churn.index.map({0: 'No Churn', 1: 'Churned'})

# ServicesOpted frequency distribution
services_freq = df['ServicesOpted'].value_counts().sort_index().reset_index()
services_freq.columns = ['Services_Opted', 'Count']
services_freq['Percentage'] = (services_freq['Count'] / len(df) * 100).round(2)

# Churn rate by ServicesOpted value
services_churn_rate = df.groupby('ServicesOpted')['Target'].agg([
    ('Count', 'count'),
    ('Churned', 'sum'),
    ('Churn_Rate', 'mean')
]).round(3)
services_churn_rate['Churn_Rate_Pct'] = (services_churn_rate['Churn_Rate'] * 100).round(1)

# Save ServicesOpted tables
save_table_to_csv(services_stats_by_churn, "4_services_stats_by_churn.csv",
                  "ServicesOpted statistics grouped by churn status")
save_table_to_csv(services_freq, "4_services_frequency_distribution.csv",
                  "Frequency distribution of ServicesOpted values")
save_table_to_csv(services_churn_rate, "4_services_churn_rates.csv",
                  "Churn rates by ServicesOpted value")

# Visualize ServicesOpted analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Histogram with KDE
sns.histplot(data=df, x='ServicesOpted', hue='Target_Label', kde=True, element='step',
             palette=['skyblue', 'salmon'], bins=range(1, 8), ax=axes[0, 0])
axes[0, 0].set_title('ServicesOpted Distribution by Churn Status', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Number of Services Opted')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_xticks(range(1, 7))

# Boxplot
sns.boxplot(data=df, x='Target_Label', y='ServicesOpted', hue='Target_Label',
            palette=['skyblue', 'salmon'], legend=False, ax=axes[0, 1])
axes[0, 1].set_title('ServicesOpted Distribution (Boxplot)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Churn Status')
axes[0, 1].set_ylabel('Services Opted')

# Churn rate by ServicesOpted value
sns.barplot(data=services_churn_rate.reset_index(), x='ServicesOpted', y='Churn_Rate',
            color='steelblue', ax=axes[1, 0])
axes[1, 0].set_title('Churn Rate by ServicesOpted Value', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Number of Services Opted')
axes[1, 0].set_ylabel('Churn Rate')
axes[1, 0].set_xticks(range(0, 6))

# Add churn rate labels on bars
for i, v in enumerate(services_churn_rate['Churn_Rate']):
    axes[1, 0].text(i, v + 0.01, f'{v:.2%}', ha='center', va='bottom', fontweight='bold')

#
# 5. Outlier Detection ANalyss
#

print("\n" + "=" * 80)
print("5. OUTLIER DETECTION ANALYSIS (Tukey's Method)")
print("=" * 80)

numeric_cols = ['Age', 'ServicesOpted']
outlier_results = []

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df[col] < lower_fence) | (df[col] > upper_fence)]
    outlier_count = len(outliers)

    # Create outlier details
    outlier_details = outliers[[col, 'Target']].copy()
    outlier_details['Outlier_Type'] = np.where(outlier_details[col] < lower_fence, 'Low', 'High')

    # Save individual outlier details
    save_table_to_csv(outlier_details, f"5_{col.lower()}_outliers_detail.csv",
                      f"Detailed outlier information for {col}")

    # Calculate mode for summary
    mode_val = stats.mode(df[col], keepdims=False).mode

    # Summary statistics with MODE
    outlier_results.append({
        'Variable': col,
        'Q1': round(Q1, 2),
        'Q3': round(Q3, 2),
        'IQR': round(IQR, 2),
        'Lower_Fence': round(lower_fence, 2),
        'Upper_Fence': round(upper_fence, 2),
        'Outlier_Count': outlier_count,
        'Outlier_Percentage': round((outlier_count / len(df)) * 100, 2),
        'Min_Value': round(df[col].min(), 2),
        'Max_Value': round(df[col].max(), 2),
        'Mean': round(df[col].mean(), 2),
        'Median': round(df[col].median(), 2),
        'Mode': round(mode_val, 2)  # Added mode
    })

# Create outlier summary table
outlier_summary_df = pd.DataFrame(outlier_results)
save_table_to_csv(outlier_summary_df, "5_outlier_summary.csv",
                  "Summary of outlier detection using Tukey's method")

# Visualize outliers
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, col in enumerate(numeric_cols):
    # Boxplot showing outliers
    boxplot_data = df[[col, 'Target_Label']].copy()

    sns.boxplot(data=boxplot_data, x='Target_Label', y=col, hue='Target_Label',
                palette=['skyblue', 'salmon'], legend=False, ax=axes[idx])

    # Calculate outlier boundaries
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    # Add fence lines
    axes[idx].axhline(y=lower_fence, color='red', linestyle='--', alpha=0.5, label='Lower Fence')
    axes[idx].axhline(y=upper_fence, color='red', linestyle='--', alpha=0.5, label='Upper Fence')

    axes[idx].set_title(f'{col} - Boxplot with Outlier Boundaries', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Churn Status')
    axes[idx].set_ylabel(col)

    if idx == 0:
        axes[idx].legend()

plt.tight_layout()
save_plot_to_png(fig, "5_outlier_detection_boxplots.png",
                 "Boxplots showing outliers with Tukey's fences")
plt.show()

#
# 6. Paired interaction analysis
#

print("\n" + "=" * 80)
print("6. INTERACTION ANALYSIS")
print("=" * 80)

# Interaction 1: AnnualIncomeClass × FrequentFlyer
interaction_1 = pd.crosstab([df['AnnualIncomeClass'], df['FrequentFlyer']],
                            df['Target'],
                            normalize='index')[1].reset_index()
interaction_1.columns = ['IncomeClass', 'FrequentFlyer', 'Churn_Rate']

# Add count information
interaction_counts = df.groupby(['AnnualIncomeClass', 'FrequentFlyer']).size().reset_index()
interaction_counts.columns = ['IncomeClass', 'FrequentFlyer', 'Count']
interaction_1 = pd.merge(interaction_1, interaction_counts, on=['IncomeClass', 'FrequentFlyer'])

save_table_to_csv(interaction_1, "6_interaction_income_frequentflyer.csv",
                  "Interaction analysis: Income Class × FrequentFlyer Status")

# Interaction 2: ServicesOpted × BookedHotelOrNot
df['ServicesGroup'] = pd.cut(df['ServicesOpted'],
                             bins=[0, 2, 4, 6],
                             labels=['Low (1-2)', 'Medium (3-4)', 'High (5-6)'])

interaction_2 = pd.crosstab([df['ServicesGroup'], df['BookedHotelOrNot']],
                            df['Target'],
                            normalize='index')[1].reset_index()
interaction_2.columns = ['ServicesGroup', 'BookedHotel', 'Churn_Rate']

# Add count information
interaction_counts_2 = df.groupby(['ServicesGroup', 'BookedHotelOrNot']).size().reset_index()
interaction_counts_2.columns = ['ServicesGroup', 'BookedHotel', 'Count']
interaction_2 = pd.merge(interaction_2, interaction_counts_2, on=['ServicesGroup', 'BookedHotel'])

save_table_to_csv(interaction_2, "6_interaction_services_hotel.csv",
                  "Interaction analysis: Services Group × BookedHotel Status")

# Visualize interactions
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Interaction 1: Income × FrequentFlyer
sns.barplot(data=interaction_1, x='IncomeClass', y='Churn_Rate', hue='FrequentFlyer',
            palette='viridis', ax=axes[0])
axes[0].set_title('Churn Rate: Income Class × FrequentFlyer Status',
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Income Class')
axes[0].set_ylabel('Churn Rate')
axes[0].legend(title='Frequent Flyer')

# Add count labels
for container in axes[0].containers:
    for bar in container:
        height = bar.get_height()
        if not np.isnan(height) and height > 0:
            axes[0].text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                         f'{height:.2%}', ha='center', va='bottom', fontsize=9)

# Interaction 2: Services × Hotel
sns.barplot(data=interaction_2, x='ServicesGroup', y='Churn_Rate', hue='BookedHotel',
            palette='coolwarm', ax=axes[1])
axes[1].set_title('Churn Rate: Services Group × BookedHotel Status',
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Services Group')
axes[1].set_ylabel('Churn Rate')
axes[1].legend(title='Booked Hotel')

# Add count labels
for container in axes[1].containers:
    for bar in container:
        height = bar.get_height()
        if not np.isnan(height) and height > 0:
            axes[1].text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                         f'{height:.2%}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
save_plot_to_png(fig, "6_interaction_analysis.png",
                 "Interaction analysis between key variables")
plt.show()

#
# 7. Correlation analysis
#

print("\n" + "=" * 80)
print("7. CORRELATION ANALYSIS")
print("=" * 80)

# Prepare data for correlation analysis
df_corr = df.copy()

# Encode categorical variables for correlation
encoding_map = {
    'FrequentFlyer': {'Yes': 1, 'No': 0, 'No Record': 0},
    'AccountSyncedToSocialMedia': {'Yes': 1, 'No': 0},
    'BookedHotelOrNot': {'Yes': 1, 'No': 0},
    'AnnualIncomeClass': {'Low Income': 0, 'Middle Income': 1, 'High Income': 2}
}

for col, mapping in encoding_map.items():
    if col in df_corr.columns:
        df_corr[col + '_Encoded'] = df_corr[col].map(mapping)

# Select columns for correlation
corr_columns = ['Age', 'ServicesOpted', 'Target',
                'FrequentFlyer_Encoded', 'AccountSyncedToSocialMedia_Encoded',
                'BookedHotelOrNot_Encoded', 'AnnualIncomeClass_Encoded']

corr_matrix = df_corr[corr_columns].corr()

# Rename columns for better readability
column_names = {
    'Age': 'Age',
    'ServicesOpted': 'ServicesOpted',
    'Target': 'Churn',
    'FrequentFlyer_Encoded': 'FrequentFlyer',
    'AccountSyncedToSocialMedia_Encoded': 'SocialMediaSync',
    'BookedHotelOrNot_Encoded': 'BookedHotel',
    'AnnualIncomeClass_Encoded': 'IncomeClass'
}

corr_matrix = corr_matrix.rename(columns=column_names, index=column_names)

# Save correlation matrix
save_table_to_csv(corr_matrix, "7_correlation_matrix.csv",
                  "Pearson correlation matrix for all variables (encoded)")

# Extract correlations with target
target_correlations = corr_matrix['Churn'].sort_values(ascending=False)
target_corr_df = pd.DataFrame({
    'Variable': target_correlations.index,
    'Correlation_with_Churn': target_correlations.values
}).round(3)

save_table_to_csv(target_corr_df, "7_correlations_with_target.csv",
                  "Correlation coefficients between each variable and churn")

# Visualize correlation matrix
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True, mask=mask, ax=axes[0],
            cbar_kws={"shrink": 0.8})
axes[0].set_title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')

# Bar chart of correlations with target
sorted_corr = target_corr_df[target_corr_df['Variable'] != 'Churn'].sort_values('Correlation_with_Churn')
colors = ['salmon' if x > 0 else 'skyblue' for x in sorted_corr['Correlation_with_Churn']]
bars = axes[1].barh(sorted_corr['Variable'], sorted_corr['Correlation_with_Churn'], color=colors)
axes[1].set_title('Correlation with Churn', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Correlation Coefficient')
axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)

# Add value labels
for bar in bars:
    width = bar.get_width()
    axes[1].text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                 f'{width:.3f}', va='center',
                 ha='left' if width >= 0 else 'right',
                 fontweight='bold')

plt.tight_layout()
save_plot_to_png(fig, "7_correlation_analysis.png",
                 "Correlation heatmap and bar chart showing relationships with churn")
plt.show()

print('Analysis Complete!')