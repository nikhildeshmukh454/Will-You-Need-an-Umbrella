import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import numpy as np

class RainPrediction:
    def __init__(self, classifiers=None, test_size=0.2, random_state=42):
        if classifiers is None:
            classifiers = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50),
                'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=random_state)
            }
        self.classifiers = classifiers
        self.test_size = test_size
        self.random_state = random_state
        self.trained_models = {}

    def train_models(self):
        train_data = pd.read_csv("modified_data.csv")
        self.feature_names = train_data.drop(['RainTomorrow'], axis=1).columns
        x_train = train_data.drop(['RainTomorrow'], axis=1)
        y_train = train_data['RainTomorrow']
        
        x_train_split, x_val, y_train_split, y_val = train_test_split(x_train, y_train, test_size=self.test_size, random_state=self.random_state)

        for name, clf in self.classifiers.items():
            clf.fit(x_train_split, y_train_split)
            accuracy = clf.score(x_val, y_val)
            self.trained_models[name] = clf
            print(f"{name} Accuracy: {accuracy:.2f}")

    def predict(self, test_data):
        self.train_models()
        test_data = np.array(test_data, dtype=float).reshape(1, -1)
        
        # Convert the test data to a DataFrame to match the feature names
        test_data_df = pd.DataFrame(test_data, columns=self.feature_names)
        
        predictions = {}
        for clf_name, clf in self.trained_models.items():
            prediction = clf.predict(test_data_df)[0]
            predictions[clf_name] = prediction
        
        # Aggregate predictions (e.g., majority vote or averaging)
        avg_prediction = round(sum(predictions.values()) / len(predictions))
        
        return avg_prediction


Rain_pred = RainPrediction()
import pickle
with open('rain_prediction_model.pkl', 'wb') as f:
    pickle.dump(Rain_pred, f)