import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
import os
import sys

data_path = "MachineLearningRating_v3.txt"

data = pd.read_csv(data_path, delimiter='|') 

# Check for required columns
required_columns = ['Province', 'PostalCode', 'Gender', 'StatutoryRiskType', 'Premium', 'Total_Claim']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    print(f"Warning: Missing columns in the dataset: {missing_columns}")


# Function to calculate Claims Ratio
def calculate_claims_ratio(df):
    """Calculate the claims ratio: TotalClaims / TotalPremium"""
    return df["TotalClaims"].sum() / df["TotalPremium"].sum()

# Function to calculate Profit Margin
def calculate_profit_margin(df):
    """Calculate the profit margin: (TotalPremium - TotalClaims) / TotalPremium"""
    return (df["TotalPremium"].sum() - df["TotalClaims"].sum()) / df["TotalPremium"].sum()

# Function to test risk differences across provinces
def test_risk_across_provinces(data):
    print("\nTesting Risk Differences Across Provinces")
    group_a = data[data["Province"] == "Gauteng"]
    group_b = data[data["Province"] == "Western Cape"]
    
    if group_a.empty or group_b.empty:
        print("Insufficient data for one or both groups.")
        return

    kpi_a = calculate_claims_ratio(group_a)
    kpi_b = calculate_claims_ratio(group_b)
    
    # Perform t-test
    t_stat, p_value = ttest_ind(group_a["TotalClaims"] / group_a["TotalPremium"],
                                group_b["TotalClaims"] / group_b["TotalPremium"],
                                equal_var=False)
    
    print(f"KPI Group A (Gauteng): {kpi_a:.2f}")
    print(f"KPI Group B (Western Cape): {kpi_b:.2f}")
    print(f"P-value: {p_value:.5f}")
    
    if p_value < 0.05:
        print("Reject Null Hypothesis: Significant risk differences across provinces.")
    else:
        print("Fail to Reject Null Hypothesis: No significant risk differences across provinces.")


# Function to test risk differences between zip codes
def test_risk_across_zip_codes(data):
    print("\nTesting Risk Differences Across Zip Codes")
    group_a = data[data["PostalCode"].astype(str).str.startswith("1")]
    group_b = data[data["PostalCode"].astype(str).str.startswith("2")]
    
    if group_a.empty or group_b.empty:
        print("Insufficient data for one or both groups.")
        return

    kpi_a = calculate_claims_ratio(group_a)
    kpi_b = calculate_claims_ratio(group_b)
    
    # Perform t-test
    t_stat, p_value = ttest_ind(group_a["TotalClaims"] / group_a["TotalPremium"],
                                group_b["TotalClaims"] / group_b["TotalPremium"],
                                equal_var=False)
    
    print(f"KPI Group A (PostalCode starts with 1): {kpi_a:.2f}")
    print(f"KPI Group B (PostalCode starts with 2): {kpi_b:.2f}")
    print(f"P-value: {p_value:.5f}")
    
    if p_value < 0.05:
        print("Reject Null Hypothesis: Significant risk differences between zip codes.")
    else:
        print("Fail to Reject Null Hypothesis: No significant risk differences between zip codes.")

# Function to test profit margin differences between zip codes
def test_profit_margin_across_zip_codes(data):
    print("\nTesting Profit Margin Differences Across Zip Codes")
    group_a = data[data["PostalCode"].astype(str).str.startswith("1")]
    group_b = data[data["PostalCode"].astype(str).str.startswith("2")]
    
    if group_a.empty or group_b.empty:
        print("Insufficient data for one or both groups.")
        return

    kpi_a = calculate_profit_margin(group_a)
    kpi_b = calculate_profit_margin(group_b)

    # Perform t-test
    t_stat, p_value = ttest_ind(group_a["TotalPremium"] - group_a["TotalClaims"],
                                group_b["TotalPremium"] - group_b["TotalClaims"],
                                equal_var=False)

    print(f"KPI Group A (PostalCode starts with 1): {kpi_a:.2f}")
    print(f"KPI Group B (PostalCode starts with 2): {kpi_b:.2f}")
    print(f"P-value: {p_value:.5f}")
    
    if p_value < 0.05:
        print("Reject Null Hypothesis: Significant profit margin differences between zip codes.")
    else:
        print("Fail to Reject Null Hypothesis: No significant profit margin differences between zip codes.")

# Function to test risk differences across genders
def test_risk_across_genders(data):
    print("\nTesting Risk Differences Across Genders")
    group_a = data[data["Gender"] == "Female"]
    group_b = data[data["Gender"] == "Male"]
    
    if group_a.empty or group_b.empty:
        print("Insufficient data for one or both groups.")
        return

    kpi_a = calculate_claims_ratio(group_a)
    kpi_b = calculate_claims_ratio(group_b)
    
    # Perform t-test
    t_stat, p_value = ttest_ind(group_a["TotalClaims"] / group_a["TotalPremium"],
                                group_b["TotalClaims"] / group_b["TotalPremium"],
                                equal_var=False)
    
    print(f"KPI Group A (Female): {kpi_a:.2f}")
    print(f"KPI Group B (Male): {kpi_b:.2f}")
    print(f"P-value: {p_value:.5f}")
    
    if p_value < 0.05:
        print("Reject Null Hypothesis: Significant risk differences between genders.")
    else:
        print("Fail to Reject Null Hypothesis: No significant risk differences between genders.")

# Run all tests
if __name__ == "__main__":
    test_risk_across_provinces(data)
    test_risk_across_zip_codes(data)
    test_profit_margin_across_zip_codes(data)
    test_risk_across_genders(data)
