# train_model.py

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import shap
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("corr360_20250731_m.csv", low_memory=False)

df.columns = df.columns.str.strip()
df['ADCNO'] = df['ADCNO'].astype(str).str.zfill(6)
# df["DISABILiTY"] = df["DISABILiTY"].map({"Y": 1, "NULL": 0})
df["Disciplinary"] = df["Disciplinary"].map({"Y": 1, "NULL": 0})
# print(f"Largest value in NUM_DEPENDENTS: {df['NUM_DEPENDENTS'].max()}")

train_df = df[df["RECIDIVISM"].isin(["Y", "N"])].copy()
train_df["RECIDIVISM"] = train_df["RECIDIVISM"].map({"Y": 1, "N": 0})
train_df["RECIDIVISM"] = train_df["RECIDIVISM"].astype("int")

drop_cols = ["ADCNO", "DOC_ID", "RECIDIVISM", "MONTHS", "ADMISSION_DATE", "DATE_RELEASED", "RECOMMIT_DATE", "AGE_AT_OFFENSE", "DOB", "SMI", "SUICIDE_HIST", "NUM_ESCAPES", "GANG_SCORE", "SEX_OFFENDER"]
X = pd.get_dummies(train_df.drop(columns=drop_cols), drop_first=True)
X = X.drop(columns=[col for col in ['MENTALHEALTH_3C','MENTALHEALTH_5','MENTALHEALTH_U','Race_Other','Race_Unknown'] if col in X], errors='ignore')
columns_to_drop = ['Marital_Status_Single', 'CUSTODY_Maximum', 'CUSTODY_Medium', 'CUSTODY_Minimum']
X = X.drop(columns=[col for col in columns_to_drop if col in X.columns], errors='ignore')

X = X.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
y = train_df["RECIDIVISM"]
model_input_columns = X.columns.tolist()

# print("Overall RECIDIVISM value counts:")
# print(y.value_counts())

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# print("RECIDIVISM value counts in test set:")
# print(pd.Series(y_test).value_counts())

rf_model = RandomForestClassifier(n_estimators=130, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
cal_rf = CalibratedClassifierCV(rf_model, method='sigmoid', cv=5)
cal_rf.fit(X_train, y_train)

explainer = shap.TreeExplainer(rf_model)

# The code below is all for printing predictions and model accuracy
# y_pred_probs = cal_rf.predict_proba(X_test)[:, 1]
# y_pred = (y_pred_probs >= 0.513).astype(int)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy:.4f}")
# print("Recidivism distribution in training data:")
# print(train_df['RECIDIVISM'].value_counts(normalize=True))
# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# print(f"Correct Predictions: {tp + tn}")
# print(f"Incorrect Predictions: {fp + fn}")
# print(f"Predicted to come back and did (True Negatives): {tp}")
# print(f"Predicted to come back but didn’t (False Negatives): {fp}")
# print(f"Predicted to not come back but did (False Positives): {fn}")
# print(f"Predicted to not come back and didn’t (True Positives): {tn}")

# Evaluate model
y_pred_probs = cal_rf.predict_proba(X_test)[:, 1]
y_pred = (y_pred_probs >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_probs)

print(f"Model Accuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("Recidivism distribution in training data:")
print(train_df['RECIDIVISM'].value_counts(normalize=True))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"Correct Predictions: {tp + tn}")
print(f"Incorrect Predictions: {fp + fn}")
print(f"Predicted to come back and did (True Negatives): {tp}")
print(f"Predicted to come back but didn’t (False Negatives): {fp}")
print(f"Predicted to not come back but did (False Positives): {fn}")
print(f"Predicted to not come back and didn’t (True Positives): {tn}")

joblib.dump(cal_rf, 'models/calibrated_rf_model.pkl')
joblib.dump(imputer, 'models/imputer.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(model_input_columns, 'models/input_columns.pkl')
joblib.dump(explainer, 'models/explainer.pkl')