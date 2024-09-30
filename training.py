# Import necessary libraries for training
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load the dataset
df = pd.read_csv('conv.csv')

# Features and target variable
X = df.drop('Time_Between_Flares',axis=1)
y = df['Time_Between_Flares']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Get predictions and probabilities
y_pred = rf_classifier.predict(X_test)
y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]  # Probability of flare

# Evaluate the model
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Save the trained model
joblib.dump(rf_classifier, 'rf_flare_predictor.pkl')
