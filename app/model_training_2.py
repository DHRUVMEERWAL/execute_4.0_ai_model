# import pandas as pd
# import joblib
# import time
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, f1_score
# from imblearn.combine import SMOTEENN
# from xgboost import XGBClassifier

# # Load dataset
# df = pd.read_csv('transactions_train.csv')

# # Rename columns to remove '_anonymous' suffix from key columns
# column_mapping = {
#     'transaction_payment_mode_anonymous': 'transaction_payment_mode',
#     'payment_gateway_bank_anonymous': 'payment_gateway_bank',
#     'payer_email_anonymous': 'payer_email',
#     'payer_mobile_anonymous': 'payer_mobile',
#     'payer_browser_anonymous': 'payer_browser',
#     'payee_id_anonymous': 'payee_id',
#     'transaction_id_anonymous': 'transaction_id'
# }
# df.rename(columns=column_mapping, inplace=True)

# # Feature Engineering
# df['transaction_date'] = pd.to_datetime(df['transaction_date'])
# df['transaction_hour'] = df['transaction_date'].dt.hour
# payee_avg_amount = df.groupby('payee_id')['transaction_amount'].transform('mean')
# df['amount_ratio_to_payee_avg'] = df['transaction_amount'] / (payee_avg_amount + 1e-9)
# payee_transaction_count = df.groupby('payee_id')['transaction_id'].transform('count')
# df['payee_txn_velocity'] = payee_transaction_count

# print("✅ Engineered features")

# # Encode categorical variables (all object type columns except transaction_date)
# categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'transaction_date']
# le = LabelEncoder()
# for col in categorical_cols:
#     df[col] = le.fit_transform(df[col].astype(str))
#     print(f"✅ Encoded: {col}")

# # Define features and target
# feature_cols = [col for col in df.columns if col not in ['transaction_date', 'transaction_id', 'is_fraud']]
# X = df[feature_cols]
# y = df['is_fraud']

# # Preprocessing: Imputation and Scaling
# imputer = SimpleImputer(strategy='mean')
# X_imputed = imputer.fit_transform(X)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_imputed)
# joblib.dump(imputer, 'models_2/imputer.pkl')
# joblib.dump(scaler, 'models_2/scaler.pkl')

# # Split data into train, validation, and test sets.
# X_train_full, X_test, y_train_full, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, stratify=y, random_state=42
# )
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
# )

# print("⚠️ Train Distribution Before Resampling:", pd.Series(y_train).value_counts().to_dict())

# # Apply SMOTE-ENN for handling imbalanced data.
# resampler = SMOTEENN(random_state=42)
# X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
# print("✅ Train Distribution After SMOTEENN:", pd.Series(y_train_resampled).value_counts().to_dict())

# # Define supervised models.
# imbalance_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
# rf = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
# lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
# xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
#                     scale_pos_weight=imbalance_ratio, eval_metric='logloss', random_state=42)
# nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
# ensemble = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('lr', lr)], voting='soft')

# # Train unsupervised model: Isolation Forest.
# iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
# iso_forest.fit(X_train_full)
# joblib.dump(iso_forest, 'models_2/isolation_forest.pkl')
# print("✅ Isolation Forest trained & saved")

# # Train and auto-tune each supervised model.
# models = {"random_forest": rf, "logistic_regression": lr, "neural_network": nn, "xgboost": xgb, "ensemble": ensemble}

# for name, model in models.items():
#     print(f"\n🚀 Training: {name.upper()}")
#     start_time = time.time()
#     model.fit(X_train_resampled, y_train_resampled)
#     duration = time.time() - start_time

#     # Calibrate the model using Platt scaling.
#     calibrated = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=3)
#     calibrated.fit(X_train_resampled, y_train_resampled)
#     y_prob = calibrated.predict_proba(X_val)[:, 1]

#     # Auto-tune threshold based on maximizing the F1-score.
#     thresholds = np.linspace(0.1, 0.9, 50)
#     f1_scores = [f1_score(y_val, (y_prob > t).astype(int), zero_division=0) for t in thresholds]
#     best_threshold = thresholds[np.argmax(f1_scores)]

#     y_pred = (y_prob > best_threshold).astype(int)

#     print(f"⏱️ Training time: {round(duration, 2)} seconds")
#     print(f"⚙️ Auto-tuned threshold on validation: {round(best_threshold, 3)}")
#     print(classification_report(y_val, y_pred, zero_division=0))
#     print("ROC AUC Score:", roc_auc_score(y_val, y_prob))
#     precision, recall, _ = precision_recall_curve(y_val, y_prob)
#     pr_auc = auc(recall, precision)
#     print("Precision-Recall AUC:", pr_auc)

#     # Save the calibrated model and its threshold.
#     joblib.dump({'model': calibrated, 'threshold': best_threshold}, f'models_2/{name}_model.pkl')
#     print(f"✅ Saved {name}_model.pkl")

# print("\n🎯 Hybrid supervised + anomaly detection pipeline complete!")


import pandas as pd
import joblib
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, f1_score
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv('transactions_train.csv')

# Rename columns
df.rename(columns={
    'transaction_payment_mode_anonymous': 'transaction_payment_mode',
    'payment_gateway_bank_anonymous': 'payment_gateway_bank',
    'payer_email_anonymous': 'payer_email',
    'payer_mobile_anonymous': 'payer_mobile',
    'payer_browser_anonymous': 'payer_browser',
    'payee_id_anonymous': 'payee_id',
    'transaction_id_anonymous': 'transaction_id'
}, inplace=True)

# Feature Engineering
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df['transaction_hour'] = df['transaction_date'].dt.hour
payee_avg_amount = df.groupby('payee_id')['transaction_amount'].transform('mean')
df['amount_ratio_to_payee_avg'] = df['transaction_amount'] / (payee_avg_amount + 1e-9)
payee_transaction_count = df.groupby('payee_id')['transaction_id'].transform('count')
df['payee_txn_velocity'] = payee_transaction_count
print("✅ Engineered features")

# Encode categoricals
categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'transaction_date']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))
    print(f"✅ Encoded: {col}")

# Features & Target
feature_cols = [
    "transaction_amount", "transaction_channel", "transaction_payment_mode", "payment_gateway_bank",
    "payer_email", "payer_mobile", "payer_browser", "payee_ip_anonymous", "payee_id",
    "amount_ratio_to_payee_avg", "payee_txn_velocity", "transaction_hour"
]
X = df[feature_cols]
y = df['is_fraud']

# Preprocessing
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
joblib.dump(imputer, 'models_2/imputer.pkl')
joblib.dump(scaler, 'models_2/scaler.pkl')

# Splitting
total_train, X_test, total_y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(total_train, total_y_train, test_size=0.2, stratify=total_y_train, random_state=42)
print("⚠️ Train Distribution Before Resampling:", pd.Series(y_train).value_counts().to_dict())

# SMOTE-ENN
resampler = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
print("✅ Train Distribution After SMOTEENN:", pd.Series(y_train_resampled).value_counts().to_dict())

# Models
imbalance_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
rf = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=imbalance_ratio, eval_metric='logloss', random_state=42)
nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
ensemble = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('lr', lr)], voting='soft')

# Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso_forest.fit(total_train)
joblib.dump(iso_forest, 'models_2/isolation_forest.pkl')
print("✅ Isolation Forest trained & saved")

# Train + Calibrate models
models = {"random_forest": rf, "logistic_regression": lr, "neural_network": nn, "xgboost": xgb, "ensemble": ensemble}
for name, model in models.items():
    print(f"\n🚀 Training: {name.upper()}")
    start_time = time.time()
    model.fit(X_train_resampled, y_train_resampled)
    duration = time.time() - start_time

    calibrated = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=3)
    calibrated.fit(X_train_resampled, y_train_resampled)
    y_prob = calibrated.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores = [f1_score(y_val, (y_prob > t).astype(int), zero_division=0) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]

    y_pred = (y_prob > best_threshold).astype(int)

    print(f"⏱️ Training time: {round(duration, 2)} seconds")
    print(f"⚙️ Auto-tuned threshold on validation: {round(best_threshold, 3)}")
    print(classification_report(y_val, y_pred, zero_division=0))
    print("ROC AUC Score:", roc_auc_score(y_val, y_prob))
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    pr_auc = auc(recall, precision)
    print("Precision-Recall AUC:", pr_auc)

    joblib.dump({'model': calibrated, 'threshold': best_threshold}, f'models_2/{name}_model.pkl')
    print(f"✅ Saved {name}_model.pkl")

print("\n🎯 Hybrid supervised + anomaly detection pipeline complete!")
