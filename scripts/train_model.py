import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def train_and_save_model(preprocessed_data_path='../data/preprocessed_data.pkl', model_output_path='../data/trained_model.pkl'):
    # Load preprocessed data
    X_train, X_test, y_train, y_test, scaler = joblib.load(preprocessed_data_path)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model trained. Test RMSE: {rmse:.2f}")

    # Save model
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == '__main__':
    train_and_save_model()
