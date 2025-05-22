import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import load_model, load_features, predict_churn

def test_prediction_pipeline():
    try:
        #Load Model +features
        model = load_model("4_models/CatBoost.pkl")
        features = load_features("7_config/config.yaml")
        
        #Load a test input
        df = pd.read_csv("1_data/processed/final_features_df.csv")
        sample = df[features].iloc[[0]] #to get the first row
        
        #run prediction
        pred, prob = predict_churn(sample, model, features)
        assert pred[0] in [0,1], "Prediction should be 0 or 1"
        assert 0.0 <= prob[0] <= 1.0, "Probability should between 0 and 1"
        
        print("Pipeline Test Passed!")
    except Exception as e:
        print("Pipeline Test Failed!")
        print(str(e))
        
if __name__ == "__main__":
    test_prediction_pipeline()