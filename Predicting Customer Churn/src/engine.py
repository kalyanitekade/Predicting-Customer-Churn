import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import pickle
from sklearn.model_selection import train_test_split
from ML_Pipeline.utils import read_data, null_values, inspection
from ML_Pipeline.visualise import plot_scatter, full_diagnostic
from ML_Pipeline.feature_engg import encode_cat
from ML_Pipeline.ml_model import run_model_and_plot_roc, prepare_model_smote
from ML_Pipeline.evaluate_metrics import evaluate_confusion_matrix

df = pd.read_csv("./Data/data_regression.csv")
x = inspection(df)
df = null_values(df)
print(df.isnull().sum())

encode_cat (df,['gender','multi_screen','mail_subscribed'])

#plot_scatter(df, 'churn', ['customer_id','phone_no', 'year'])
#full_diagnostic(df, 'churn')

#Train model smote
X_train, X_test, y_train, y_test = prepare_model_smote(df,target_col='churn', col_to_exclude=['customer_id','phone_no', 'year'])

#Running Logistic model
log_model, y_pred=run_model_and_plot_roc('Logistic', X_train, X_test, y_train, y_test, w=None)
log_conf = evaluate_confusion_matrix(y_test, y_pred)

#Running Logistic balnced model
log_bal_model, y_pred=run_model_and_plot_roc('Logistic_balanced', X_train, X_test, y_train, y_test, w="balanced")
log_bal_conf = evaluate_confusion_matrix(y_test, y_pred)

#Running Logistic weighted model
log_wt_model, y_pred=run_model_and_plot_roc('Logistic', X_train, X_test, y_train, y_test, w={0:90,1:10})
log_wt_conf = evaluate_confusion_matrix(y_test, y_pred)

#Running Random Forest
rf_model, y_pred=run_model_and_plot_roc('Random', X_train, X_test, y_train, y_test, w=None)
rf_conf = evaluate_confusion_matrix(y_test, y_pred)

#Running Adaboost
ab_model, y_pred=run_model_and_plot_roc('Adaboost', X_train, X_test, y_train, y_test, w=None)
feature_importances = ab_model.feature_importances_
top_features_indices = np.argsort(feature_importances)[-3:][::-1]
feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else np.arange(X_train.shape[1])
top_features = [(feature_names[i], feature_importances[i]) for i in top_features_indices]
print("Top 3 features and their importance scores:")
for feature, importance in top_features:
    print(f"Feature: {feature}, Importance: {importance}")
ab_conf = evaluate_confusion_matrix(y_test, y_pred)

#Running Gradient
grad_model, y_pred=run_model_and_plot_roc('Gradient', X_train, X_test, y_train, y_test, w=None)
grad_conf = evaluate_confusion_matrix(y_test, y_pred)