import joblib
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path='../data/trained_model.pkl', data_path='../data/preprocessed_data.pkl'):
    # Load model and data
    model = joblib.load(model_path)
    X_train, X_test, y_train, y_test, scaler = joblib.load(data_path)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel("Predicted MEDV")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()

    # Actual vs Predicted plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='black', linestyle='--')
    plt.xlabel("Actual MEDV")
    plt.ylabel("Predicted MEDV")
    plt.title("Actual vs Predicted MEDV")
    plt.show()

if __name__ == '__main__':
    evaluate_model()
