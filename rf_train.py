#=================
# Import Libraries
#=================
import numpy as np
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


#=================
# Load Dataset
#=================
df = pd.read_csv('water_potability.csv')


#=================
# Train-test split
#=================
X = df.drop('Potability', axis=1)
y = df['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# Impute the missing values with respective median
impute = SimpleImputer(strategy='median')
# Use RobustScaler
scaler = RobustScaler()


#=================
# Model Selection: Random Forest
#=================
# We got best Parameters from Notebook through gridsearch: 
# {'model__max_depth': 20, 'model__min_samples_split': 2, 'model__n_estimators': 200}
rf_model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=20, 
    min_samples_split=2, 
    random_state=42, 
    class_weight='balanced')


#=================
# Pipeline Creation
#=================
rf_pipe = Pipeline([
    ('imputer', impute),
    ('scaler', scaler),
    ('model', rf_model)
])


#=================
# Fit the model
#=================
rf_pipe.fit(X_train, y_train)


#=================
# Evaluate the best model
#=================
y_pred = rf_pipe.predict(X_test)
y_prob = rf_pipe.predict_proba(X_test)[:, 1]

print("\n--- Final Test Set Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#=================
# Save Model
#=================
with open("rf_potability.pkl", "wb") as f:
    pickle.dump(rf_pipe, f)
print("✅ Random Forest pipe saved as rf_potability.pkl")