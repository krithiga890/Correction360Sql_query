# predict_inmate.py

import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Load model and preprocessing artifacts
cal_rf = joblib.load("models/calibrated_rf_model.pkl")
imputer = joblib.load("models/imputer.pkl")
scaler = joblib.load("models/scaler.pkl")
model_input_columns = joblib.load("models/input_columns.pkl")
explainer = joblib.load("models/explainer.pkl")

# Load full inmate dataset
df = pd.read_csv("corr360_20250731_m.csv", low_memory=False)
df['ADCNO'] = df['ADCNO'].astype(str).str.zfill(6)

def preprocess_new_inmate(inmate_dict):
    df_new = pd.DataFrame([inmate_dict])
    for col in model_input_columns:
        if col not in df_new.columns:
            df_new[col] = np.nan
    df_new = df_new[model_input_columns]
    X_imputed = imputer.transform(df_new)
    X_scaled = scaler.transform(X_imputed)
    return pd.DataFrame(X_scaled, columns=model_input_columns)

def predict_reoffend(input_data):
    df_new = preprocess_new_inmate(input_data)
    prob = cal_rf.predict_proba(df_new)[0][1]
    prediction = "No" if prob >= 0.512 else "Yes"
    confidence = prob if prediction == "No" else 1 - prob
    return prediction, confidence

def explain_factors(input_data):
    df_new = preprocess_new_inmate(input_data)
    shap_values = explainer(df_new)
    contrib = shap_values.values[0, :, 1] if len(shap_values.values.shape) == 3 else shap_values.values[0]
    df = pd.DataFrame({'feature': model_input_columns, 'contribution': contrib})
    df['abs'] = df.contribution.abs()
    df = df.sort_values('abs', ascending=False)
    return df[['feature', 'contribution']].to_dict(orient='records')

def convert_nan_to_none(d):
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing ADC number"}))
        return
    adc = sys.argv[1].zfill(6)
    inmate_row = df[df['ADCNO'] == adc]
    if inmate_row.empty:
        print(json.dumps({"error": f"No record found for ADC {adc}"}))
        return

    latest = inmate_row.sort_values("DOC_ID", ascending=False).iloc[0]
    features = ['EPISODE_NUM', 'AGE_AT_RELEASE', 'Marital_Status', 'CUSTODY', 'MEDSCORE', 'MENTALHEALTH', 'SMI', 'SUICIDE_HIST', 'NUM_COM_JUV_FEL', 'NUM_PRIOR_AZ_FEL', 'NUM_ESCAPES', 'COMMIT_CRIME_SUPPORT_HABIT', 'MOST_SERIOUS_OFF', 'MOST_SER_PRIOR_OFF', 'ESC_HIST_SCORE', 'GANG_SCORE', 'SEX_OFFENDER', 'RISK_FLAG', 'VIOLENT_OFF', 'FELONY_CLASS', 'PROGRAMS', 'GED', 'ED_SCORE', 'Disciplinary', 'VisitationScore', 'Abscond', 'SUBS_Abuse', 'SO_Severity', 'Ed_Achievement']
    input_data = {k: latest[k] for k in features if k in latest}
    input_data = {k: (v.item() if isinstance(v, (np.integer, np.floating)) else v) for k, v in input_data.items()}

    try:
        prediction, confidence = predict_reoffend(input_data)
        factors = explain_factors(input_data)
        input_data = convert_nan_to_none(input_data)
        print(json.dumps({
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "factors": factors,
            "allFactors": factors,
            "fullData": input_data
        }))
    except Exception as e:
        print(json.dumps({"error": f"Error in prediction: {str(e)}"}))

if __name__ == "__main__":
    main()