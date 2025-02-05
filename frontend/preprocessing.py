import pandas as pd

# Function to perform preprocessing
def preprocess_data(df):
    report = []
    
    # Handling Missing Values
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0].index.tolist()
    for col in missing_cols:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
            report.append(f"Filled missing values in {col} with mode")
        else:
            df[col].fillna(df[col].median(), inplace=True)
            report.append(f"Filled missing values in {col} with median")
    
    # Encoding Categorical Variables
    cat_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    report.append(f"Applied One-Hot Encoding to categorical variables: {list(cat_cols)}")
    
    # Scaling Numerical Features
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = (df[num_cols] - df[num_cols].min()) / (df[num_cols].max() - df[num_cols].min())
    report.append(f"Applied Min-Max Scaling to numerical features: {list(num_cols)}")
    
    return df, report