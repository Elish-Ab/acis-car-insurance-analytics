import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap

# Step 1: Load Data
data_path = "MachineLearningRating_v3.txt"
data = pd.read_csv(data_path, sep='|', engine='python')

# Step 2: Data Preparation
# Handling Missing Data
data.fillna(data.median(numeric_only=True), inplace=True)  
data.dropna(subset=['TotalPremium', 'TotalClaims'], inplace=True)  

# Encoding Categorical Data
categorical_cols = data.select_dtypes(include=['object']).columns
encoded_data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Feature Engineering
encoded_data['PremiumToClaimsRatio'] = encoded_data['TotalPremium'] / (encoded_data['TotalClaims'] + 1)

# Train-Test Split
target = 'TotalPremium'
features = encoded_data.drop(columns=['TotalPremium'])
X_train, X_test, y_train, y_test = train_test_split(features, encoded_data[target], test_size=0.3, random_state=42)

# Step 3: Model Building
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate Models
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {'MAE': mae, 'MSE': mse, 'R2': r2}

# Display Results
results_df = pd.DataFrame(results).T
print("\nModel Evaluation Results:")
print(results_df)

# Step 4: Model Interpretability
best_model_name = results_df['R2'].idxmax()
best_model = models[best_model_name]

print(f"\nAnalyzing feature importance for {best_model_name}...")
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test)

# Visualize Feature Importance
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)

# Save Results
results_df.to_csv("model_evaluation_results.csv", index=True)
