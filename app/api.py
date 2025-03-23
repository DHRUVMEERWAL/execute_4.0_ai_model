# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import numpy as np
# import joblib
# import pandas as pd
# from app.rules import apply_business_rules

# app = FastAPI()

# # Load pre-trained models and preprocessors
# imputer = joblib.load('models/imputer.pkl')
# scaler = joblib.load('models/scaler.pkl')
# isolation_forest = joblib.load('models/isolation_forest.pkl')

# # Load all models
# models = {
#     "random_forest": joblib.load('models/random_forest_model.pkl'),
#     "logistic_regression": joblib.load('models/logistic_regression_model.pkl'),
#     "neural_network": joblib.load('models/neural_network_model.pkl'),
#     "xgboost": joblib.load('models/xgboost_model.pkl'),
#     "ensemble": joblib.load('models/ensemble_model.pkl')
# }

# # Define expected input
# class Transaction(BaseModel):
#     transaction_amount: float
#     transaction_date: str
#     transaction_channel: int
#     transaction_payment_mode: int
#     payment_gateway_bank: int
#     payer_browser: int
#     payer_email: int
#     payee_ip_anonymous: int
#     payer_mobile: int
#     transaction_id: int
#     payee_id: int

# @app.post("/predict")
# async def predict(transaction: Transaction):
#     try:
#         input_dict = transaction.dict()
#         df = pd.DataFrame([input_dict])

#         df['transaction_date'] = pd.to_datetime(df['transaction_date'])
#         df['transaction_hour'] = df['transaction_date'].dt.hour
#         payee_avg_amount = df['transaction_amount']
#         df['amount_ratio_to_payee_avg'] = df['transaction_amount'] / (payee_avg_amount + 1e-9)
#         df['payee_txn_velocity'] = 1
#         df.drop(columns=['transaction_date', 'transaction_id'], inplace=True)

#         X = imputer.transform(df)
#         X_scaled = scaler.transform(X)

#         fraud_votes = []
#         fraud_reason = []
#         max_score = 0.0

#         # Apply external rule engine
#         rule_result = apply_business_rules(input_dict)
#         if rule_result['triggered']:
#             fraud_reason.extend(rule_result['reasons'])

#         # ML models
#         for model_name, bundle in models.items():
#             model = bundle['model']
#             threshold = bundle['threshold']
#             proba = model.predict_proba(X_scaled)[:, 1][0]
#             pred = int(proba > threshold)
#             fraud_votes.append(pred)
#             if proba > max_score:
#                 max_score = proba
#             if pred:
#                 fraud_reason.append(model_name)

#         anomaly_score = isolation_forest.decision_function(X_scaled)[0]
#         anomaly_flag = int(anomaly_score < 0)
#         if anomaly_flag:
#             fraud_reason.append("isolation_forest")

#         is_fraud = bool(sum(fraud_votes) >= 2 or anomaly_flag == 1 or rule_result['triggered'])

#         return {
#             "transaction_id": str(input_dict['transaction_id']),
#             "is_fraud": is_fraud,
#             "fraud_source": "rule" if rule_result['triggered'] else "model",
#             "fraud_reason": ", ".join(fraud_reason) if fraud_reason else "clean",
#             "fraud_score": round(max_score, 4)
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# def root():
#     return {"message": "Fraud detection API is live! 🚀"}


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import numpy as np
# import joblib
# import pandas as pd
# from app.rules import apply_business_rules

# app = FastAPI()

# # Load pre-trained models and preprocessors
# imputer = joblib.load('app/models/imputer.pkl')
# scaler = joblib.load('app/models/scaler.pkl')
# isolation_forest = joblib.load('app/models/isolation_forest.pkl')

# # Load all models
# models = {
#     "random_forest": joblib.load('app/models/random_forest_model.pkl'),
#     "logistic_regression": joblib.load('app/models/logistic_regression_model.pkl'),
#     "neural_network": joblib.load('app/models/neural_network_model.pkl'),
#     "xgboost": joblib.load('app/models/xgboost_model.pkl'),
#     "ensemble": joblib.load('app/models/ensemble_model.pkl')
# }

# class Transaction(BaseModel):
#     transaction_amount: float
#     transaction_date: str
#     transaction_channel: int
#     transaction_payment_mode: int
#     payment_gateway_bank: int
#     payer_browser: int
#     payer_email: int
#     payee_ip_anonymous: int
#     payer_mobile: int
#     transaction_id: int
#     payee_id: int

# @app.post("/predict")
# async def predict(transaction: Transaction):
#     try:
#         input_dict = transaction.dict()
#         df = pd.DataFrame([input_dict])

#         df['transaction_date'] = pd.to_datetime(df['transaction_date'])
#         df['transaction_hour'] = df['transaction_date'].dt.hour
#         payee_avg_amount = df['transaction_amount']
#         df['amount_ratio_to_payee_avg'] = df['transaction_amount'] / (payee_avg_amount + 1e-9)
#         df['payee_txn_velocity'] = 1
#         df.drop(columns=['transaction_date', 'transaction_id'], inplace=True)

#         X = imputer.transform(df)
#         X_scaled = scaler.transform(X)

#         fraud_votes = []
#         fraud_reason = []
#         max_score = 0.0

#         rule_result = apply_business_rules(input_dict)
#         if rule_result['triggered']:
#             fraud_reason.extend(rule_result['reasons'])

#         for model_name, bundle in models.items():
#             model = bundle['model']
#             threshold = bundle['threshold']
#             proba = model.predict_proba(X_scaled)[:, 1][0]
#             pred = int(proba > threshold)
#             fraud_votes.append(pred)
#             if proba > max_score:
#                 max_score = proba
#             if pred:
#                 fraud_reason.append(model_name)

#         anomaly_score = isolation_forest.decision_function(X_scaled)[0]
#         anomaly_flag = int(anomaly_score < 0)
#         if anomaly_flag:
#             fraud_reason.append("isolation_forest")

#         is_fraud = bool(sum(fraud_votes) >= 2 or anomaly_flag == 1 or rule_result['triggered'])

#         return {
#             "transaction_id": str(input_dict['transaction_id']),
#             "is_fraud": is_fraud,
#             "fraud_source": "rule" if rule_result['triggered'] else "model",
#             "fraud_reason": ", ".join(fraud_reason) if fraud_reason else "clean",
#             "fraud_score": round(max_score, 4)
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# def root():
#     return {"message": "Fraud detection API is live! 🚀"}

# # Required by Vercel
# handler = app



from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd

app = FastAPI()

# Load pre-trained models and preprocessors from the app/models_2 folder
imputer = joblib.load('app/models_2/imputer.pkl')
scaler = joblib.load('app/models_2/scaler.pkl')
isolation_forest = joblib.load('app/models_2/isolation_forest.pkl')

# Load all models (each saved as a dict with keys "model" and "threshold")
models = {
    "random_forest": joblib.load('app/models_2/random_forest_model.pkl'),
    "logistic_regression": joblib.load('app/models_2/logistic_regression_model.pkl'),
    "neural_network": joblib.load('app/models_2/neural_network_model.pkl'),
    "xgboost": joblib.load('app/models_2/xgboost_model.pkl'),
    "ensemble": joblib.load('app/models_2/ensemble_model.pkl')
}

# Define expected input using client-friendly keys.
class Transaction(BaseModel):
    transaction_id: str
    transaction_date: str
    transaction_amount: float
    transaction_channel: str
    transaction_payment_mode: int
    payment_gateway_bank: int
    payer_email: str
    payer_mobile: str
    payer_browser: int
    payee_id: str
    payee_ip: str

@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid input payload. Please check required fields and data types.",
            "errors": exc.errors()
        },
    )

@app.post("/predict")
async def predict(transaction: Transaction):
    try:
        input_dict = transaction.dict()
        df = pd.DataFrame([input_dict])

        # Rename to match training columns exactly
        df.rename(columns={
            "payee_ip": "payee_ip_anonymous"
        }, inplace=True)

        # Encoding steps as per training pipeline
        df['transaction_channel'] = df['transaction_channel'].astype('category').cat.codes
        df['payer_email'] = df['payer_email'].astype('category').cat.codes
        df['payer_mobile'] = df['payer_mobile'].astype('category').cat.codes
        df['payee_ip_anonymous'] = df['payee_ip_anonymous'].astype('category').cat.codes
        df['payee_id'] = df['payee_id'].astype('category').cat.codes

        # Feature engineering
        df['transaction_date'] = pd.to_datetime(input_dict['transaction_date'])
        df['transaction_hour'] = df['transaction_date'].dt.hour
        df['amount_ratio_to_payee_avg'] = df['transaction_amount'] / (df['transaction_amount'] + 1e-9)
        df['payee_txn_velocity'] = 1
        df.drop(columns=['transaction_date', 'transaction_id'], inplace=True)

        # Final expected feature order exactly as in training
        expected_features = [
            "transaction_amount",
            "transaction_channel",
            "transaction_payment_mode",
            "payment_gateway_bank",
            "payer_email",
            "payer_mobile",
            "payer_browser",
            "payee_ip_anonymous",
            "payee_id",
            "amount_ratio_to_payee_avg",
            "payee_txn_velocity",
            "transaction_hour"
        ]
        df = df[expected_features]

        # Preprocessing pipeline
        X = imputer.transform(df)
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=expected_features)

        # Model predictions
        fraud_votes = []
        fraud_reason = []
        max_score = 0.0

        for model_name, bundle in models.items():
            model = bundle['model']
            threshold = bundle['threshold']
            proba = model.predict_proba(X_scaled_df)[:, 1][0]
            pred = int(proba > threshold)
            fraud_votes.append(pred)
            if proba > max_score:
                max_score = proba
            if pred:
                fraud_reason.append(model_name)

        anomaly_score = isolation_forest.decision_function(X_scaled_df)[0]
        anomaly_flag = int(anomaly_score < 0)
        if anomaly_flag:
            fraud_reason.append("isolation_forest")

        is_fraud = bool(sum(fraud_votes) >= 2 or anomaly_flag == 1)

        return {
            "transaction_id": input_dict['transaction_id'],
            "is_fraud": is_fraud,
            "fraud_source": "model",
            "fraud_reason": ", ".join(fraud_reason) if fraud_reason else "clean",
            "fraud_score": round(max_score, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Fraud detection API is live! 🚀"}
