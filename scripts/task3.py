import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
import os
import sys

# Set up the file path to the data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "../notebooks/insurance_data.csv")


# Adjust the delimiter based on the file format (e.g., ',' for CSV, '\t' for tab-separated)
data = pd.read_csv(data_path, delimiter='|') # Change ',' to '\t' if tab-separated

# Data Cleaning (Handle missing values)
data.dropna(subset=['Province', 'PostalCode', 'Gender', 'StatutoryRiskType', 'Premium', 'Total_Claim'], inplace=True)

# Check for required columns
required_columns = ['Province', 'PostalCode', 'Gender', 'StatutoryRiskType', 'Premium', 'Total_Claim']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    print(f"Warning: Missing columns in the dataset: {missing_columns}")
    
    # Remove missing columns from the dropna subset
    required_columns = [col for col in required_columns if col in data.columns]


# Data Cleaning (Handle missing values only for available columns)
data.dropna(subset=required_columns, inplace=True)

# Calculate Profit Margin
data['ProfitMargin'] = data['Premium'] - data['Total_Claim']

# Segmentation: Control vs Test
control_group = data[data['CoverCategory'] == 'Basic']  # Example feature
test_group = data[data['CoverCategory'] == 'Enhanced']

# Null Hypothesis 1: No risk differences across provinces
def chi_squared_test(data, column):
    contingency_table = pd.crosstab(data['CoverCategory'], data[column])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    return p_value

p_value_province = chi_squared_test(data, 'Province')
print(f"Risk Differences Across Provinces: p-value = {p_value_province}")

# Null Hypothesis 2: No risk differences between postal codes
p_value_postal = chi_squared_test(data, 'PostalCode')
print(f"Risk Differences Between Postal Codes: p-value = {p_value_postal}")

# Null Hypothesis 3: No significant profit margin difference by postal codes
def t_test(group_a, group_b, column):
    t_stat, p_value = ttest_ind(group_a[column], group_b[column], equal_var=False)
    return p_value

p_value_profit_postal = t_test(control_group, test_group, 'ProfitMargin')
print(f"Profit Margin Differences by Postal Code: p-value = {p_value_profit_postal}")

# Null Hypothesis 4: No significant risk difference between genders
p_value_gender = chi_squared_test(data, 'Gender')
print(f"Risk Differences Between Genders: p-value = {p_value_gender}")

# Output Results
results = {
    "Risk Differences Across Provinces": p_value_province,
    "Risk Differences Between Postal Codes": p_value_postal,
    "Profit Margin Differences by Postal Code": p_value_profit_postal,
    "Risk Differences Between Genders": p_value_gender,
}


# Interpretation
for key, p_value in results.items():
    if p_value < 0.05:
        print(f"{key}: Reject the null hypothesis (p-value={p_value:.3f})")
    else:
        print(f"{key}: Fail to reject the null hypothesis (p-value={p_value:.3f})")