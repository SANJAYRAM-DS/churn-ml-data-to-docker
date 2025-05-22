import pandas as pd
import joblib
import yaml
import os

#Loading Model
def load_model(model_path='4_models/CatBoost.pkl'):
    return joblib.load(model_path)

#loading Selected Features
def load_features(config_path='7_config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['selected_features']

#Predict function
def predict_churn(input_data, model=None, feature_list=None):
    """
    input_data: pd.DataFrame with same structure as training data (before selection)
    Returns: prediction (0 or 1), probability
    """
    if model is None:
        model = load_model()
    if feature_list is None:
        feature_list - load_features()
        
    input_data = input_data[feature_list]
    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)[:,1]
    return pred,prob

if __name__ == "__main__":
    test_input = pd.read_csv("1_data/processed/final_features_df.csv").iloc[[0]]  # 1 sample
    model = load_model()
    features = load_features()

    pred, prob = predict_churn(test_input, model, features)
    print(f"Prediction: {pred[0]} | Probability of churn: {prob[0]:.2f}")
