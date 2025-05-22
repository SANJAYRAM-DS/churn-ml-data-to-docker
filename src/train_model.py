import pandas as pd
import numpy as np
import yaml
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import joblib

# Load data
df = pd.read_csv('1_data/processed/final_features_df.csv')

with open('7_config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

features = config['selected_features']
X = df[features]
y = pd.read_csv('1_data/processed/cleaned_telco.csv')['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

# Define models
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
}

metrics = {}

# Train & evaluate
for name, model in models.items():
    print(f"Training: {name}")
    if name == 'CatBoost':
        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    else:
        model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics[name] = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'f1_score': f1_score(y_test, preds),
        'roc_auc': roc_auc_score(y_test, probs)
    }

# Convert to DataFrame
results_df = pd.DataFrame(metrics).T.sort_values('roc_auc', ascending=False)
print("\nModel Comparison:\n", results_df)

# Save metrics
os.makedirs('5_output/reports', exist_ok=True)
with open('5_output/reports/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# Plot
os.makedirs('5_output/figures', exist_ok=True)
results_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].plot(kind='bar', figsize=(12,6))
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('5_output/figures/model_comparison.png')
plt.show()

# Save best model
best_model_name = results_df.index[0]
best_model = models[best_model_name]
os.makedirs('4_models', exist_ok=True)
joblib.dump(best_model, f'4_models/{best_model_name}.pkl')
print(f"\nBest model '{best_model_name}' saved to 4_models/{best_model_name}.pkl")
