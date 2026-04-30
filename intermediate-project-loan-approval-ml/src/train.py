import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# STEP 1: Load cleaned dataset
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, 'data', 'cleaned_data.csv')

df = pd.read_csv(file_path)

# STEP 2: Split features & target

# Drop ID column if present
if 'Loan_ID' in df.columns:
    df = df.drop('Loan_ID', axis=1)

from sklearn.preprocessing import LabelEncoder

# Encode all categorical columns
for col in df.select_dtypes(include='object'):
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

print(df.dtypes)

# STEP 3: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 4: Train Multiple Models

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Random Forest (BEST)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# STEP 5: Predictions

lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# STEP 6: Evaluation
print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))


# STEP 7: Save Best Model

import os

model_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(model_dir, exist_ok=True)  # creates folder if not exists

model_path = os.path.join(model_dir, 'model.pkl')

pickle.dump(rf_model, open(model_path, 'wb'))
print("\nModel saved successfully!")