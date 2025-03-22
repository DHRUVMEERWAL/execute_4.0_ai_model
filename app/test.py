import pandas as pd
import numpy as np
import joblib
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder

# 1. Load the test data
df_test = pd.read_csv('transactions_test_wo_target.csv')
submission = pd.read_csv('test_submission_template.csv')

# 2. Rename columns to match training
column_mapping = {
    'transaction_payment_mode_anonymous': 'transaction_payment_mode',
    'payment_gateway_bank_anonymous': 'payment_gateway_bank',
    'payer_email_anonymous': 'payer_email',
    'payer_mobile_anonymous': 'payer_mobile',
    'payer_browser_anonymous': 'payer_browser',
    'payee_id_anonymous': 'payee_id',
    'transaction_id_anonymous': 'transaction_id'
}
df_test.rename(columns=column_mapping, inplace=True)

# 3. Feature Engineering
df_test['transaction_date'] = pd.to_datetime(df_test['transaction_date'])
df_test['transaction_hour'] = df_test['transaction_date'].dt.hour

payee_avg_amount = df_test.groupby('payee_id')['transaction_amount'].transform('mean')
df_test['amount_ratio_to_payee_avg'] = df_test['transaction_amount'] / (payee_avg_amount + 1e-9)
payee_transaction_count = df_test.groupby('payee_id')['transaction_id'].transform('count')
df_test['payee_txn_velocity'] = payee_transaction_count

# 4. Encode categoricals
categorical_cols = [col for col in df_test.columns if df_test[col].dtype == 'object' and col != 'transaction_date']
le = LabelEncoder()
for col in categorical_cols:
    df_test[col] = le.fit_transform(df_test[col].astype(str))

# 5. Preprocessing: impute + scale using saved preprocessors
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')

feature_cols = [col for col in df_test.columns if col not in ['transaction_date', 'transaction_id']]
X_test_imputed = imputer.transform(df_test[feature_cols])
X_test_scaled = scaler.transform(X_test_imputed)

# 6. Load calibrated models & thresholds (make sure all models exist in your 'models' folder)
model_names = ['random_forest', 'logistic_regression', 'neural_network', 'xgboost', 'ensemble']
model_preds = []

for name in model_names:
    model_file = f'{name}_model.pkl'
    data = joblib.load(model_file)
    model = data['model']
    threshold = data['threshold']
    
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob > threshold).astype(int)
    model_preds.append(y_pred)

# 7. Majority Voting
all_preds = np.vstack(model_preds)
final_preds, _ = mode(all_preds, axis=0)
final_preds = final_preds.flatten()

# 8. Save submission
submission['prediction'] = final_preds
submission.to_csv('final_submission.csv', index=False)

print("✅ Final predictions saved to final_submission.csv")
