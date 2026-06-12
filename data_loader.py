import os
import pandas as pd
import numpy as np



current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "data", "Churn_Modelling.csv")
df = pd.read_csv(csv_path)




# Validate the dataset
def get_or_download_data():
    validated_df = validate_dataset(df)
    return validated_df

def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Performs ingestion validation and logs structural findings."""
    # 1. Column Presence Check
    required_cols = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure", 
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", 
        "EstimatedSalary", "Exited"
    ]
    for col in required_cols:
        assert col in df.columns, f"Validation Error: Missing required column '{col}'"
        
    # 2. Null Value Check
    null_counts = df[required_cols].isnull().sum()
    assert null_counts.sum() == 0, f"Validation Error: Null values detected: \n{null_counts[null_counts > 0]}"
    
    # 3. Binary Value Consistency Check
    binary_cols = ["HasCrCard", "IsActiveMember", "Exited"]
    for col in binary_cols:
        unique_vals = df[col].unique()
        invalid_vals = [v for v in unique_vals if v not in [0, 1]]
        assert len(invalid_vals) == 0, f"Validation Error: Column '{col}' contains non-binary values: {invalid_vals}"
        
    # 4. Product Fields Integrity Check
    unique_products = df["NumOfProducts"].unique()
    assert df["NumOfProducts"].min() >= 1, "Validation Error: Customer cannot have fewer than 1 product."
    assert df["NumOfProducts"].max() <= 4, f"Validation Error: Product count exceed bounds (Max: 4). Found: {unique_products}"
    
    # 5. Financial Boundary Integrity Check
    assert df["Balance"].min() >= 0, "Validation Error: Negative bank balance detected."
    assert df["EstimatedSalary"].min() >= 0, "Validation Error: Negative salary detected."
    
    # 6. Demographics Integrity Check
    assert df["Age"].min() >= 18, "Validation Error: Age under 18 detected."
    assert df["Age"].max() <= 120, "Validation Error: Age over 120 detected."
    assert df["Tenure"].min() >= 0, "Validation Error: Negative tenure detected."
    
    # Print validation success summary
    print("--- DATA VALIDATION REPORT ---")
    print(f"Total Records Ingested: {len(df)}")
    print(f"Exited Classes: {df['Exited'].value_counts().to_dict()} (Retained: {len(df[df['Exited']==0])}, Churned: {len(df[df['Exited']==1])})")
    print(f"Engagement Classes: {df['IsActiveMember'].value_counts().to_dict()} (Active: {len(df[df['IsActiveMember']==1])}, Inactive: {len(df[df['IsActiveMember']==0])})")
    print("Data validation completed successfully without errors.")
    print("------------------------------")
    
    return df

if __name__ == "__main__":
    # Test script directly
    get_or_download_data()
