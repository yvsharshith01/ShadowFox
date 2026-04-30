import pickle
import numpy as np
import os

# Load model safely

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')

model = pickle.load(open(model_path, 'rb'))


# Sample Input (IMPORTANT)
# Order must match training data


import pandas as pd

sample = pd.DataFrame([{
    "Gender": 1,
    "Married": 1,
    "Dependents": 0,
    "Education": 1,
    "Self_Employed": 0,
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 0,
    "LoanAmount": 120,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": 2
}])

# Prediction
prediction = model.predict(sample)

# Output

if prediction[0] == 1:
    print(" Loan Approved")
else:
    print(" Loan Rejected")