                                                #Customer Churn Prediction App
An end-to-end, production-ready machine learning project to predict customer churn based on historical behavior. Includes feature engineering, model comparison, testing, and a fully deployable Streamlit dashboard — containerized via Docker.

#Project Overview
This project aims to predict whether a customer will churn ("Yes") or stay ("No") using machine learning. It leverages the Telco Customer Churn dataset and follows FAANG-level MLOps standards.

#Project Structure
churn_prediction/
├── data/                  # Raw and processed datasets
├── notebooks/             # EDA, Feature Engineering, Model Training
├── src/                   # Core logic for preprocessing, training, predicting
├── models/                # Saved models (CatBoost, RF, etc.)
├── outputs/               # Prediction results and reports
├── deployment/            # Streamlit app + Docker
├── config/                # config.yaml with feature list
├── tests/                 # Unit test for pipeline
└── README.md              # You are here

#Key Features
•	Cleaned & engineered data: Manual + automated feature selection
•	 5 model comparison: CatBoost, LightGBM, Random Forest, Logistic Regression, XGBoost
•	 Testable pipeline: Includes unit tests for model and prediction logic
•	 Streamlit dashboard: Manual and batch predictions with CSV download
•	 Docker-ready: Easily deployable with a single Docker container

#Models Compared
-  CatBoostClassifier (Best)
-  RandomForestClassifier
-  LogisticRegression
-  LightGBM
-  XGBoost
The best model (CatBoost) is saved and used for all predictions.

#Streamlit App (Local Run)
From root of project:
cd deployment
streamlit run app.py

#Docker Run
cd deployment
docker build -t churn-app .
docker run -p 8501:8501 churn-app
Then go to: http://localhost:8501

#Test the Pipeline
python tests/test_pipeline.py

#Requirements
pip install -r deployment/requirements.txt

#Dataset
Source: Kaggle - Telco Customer Churn (https://www.kaggle.com/blastchar/telco-customer-churn)
Place raw file inside: data/raw/

#Credits
Built with ❤️ by Sanjay as part of his ML learning journey.
Guided by best practices in production machine learning, modularity, and deployment.

 #Status
•	Completed
•	Ready to deploy
•	Maintained actively

