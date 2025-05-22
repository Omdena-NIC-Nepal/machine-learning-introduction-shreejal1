import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def cap_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    return df

def preprocess_data(input_path='../data/BostonHousing.csv', output_path='../data/preprocessed_data.pkl'):
    data = pd.read_csv(input_path)
    
    # Cap outliers
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    data = cap_outliers(data, numeric_cols)

    # Convert CHAS to integer if needed
    data['CHAS'] = data['CHAS'].astype(int)

    # Feature-target split
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Save to disk
    joblib.dump((X_train, X_test, y_train, y_test, scaler), output_path)

if __name__ == '__main__':
    preprocess_data()
