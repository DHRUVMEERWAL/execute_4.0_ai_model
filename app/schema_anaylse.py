import joblib
model = joblib.load('gradient_boosting_model.pkl')
print(type(model))  # Should show: <class 'sklearn.ensemble._gb.GradientBoostingClassifier'>
print(model.feature_importances_)
