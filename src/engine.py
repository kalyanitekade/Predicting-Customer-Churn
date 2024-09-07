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
from ML_Pipeline.Lime import plot_feature_importances
from ML_Pipeline.Lime import lime_explanation
from ML_Pipeline.Lime import lime_explanation_list


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
log_model, y_pred=run_model_and_plot_roc('Logistic', X_train, X_test, y_train, y_test)
log_conf = evaluate_confusion_matrix(y_test, y_pred)
log_lime = lime_explanation(log_model,X_train,X_test,['Not Churn','Churn'],1)


#Running Decision Tree Model 
dt_model, y_pred = run_model_and_plot_roc('Decision', X_train, X_test, y_train, y_test)
dt_conf = evaluate_confusion_matrix(y_test, y_pred)
dt = plot_feature_importances(dt_model)
dt_lime = lime_explanation(dt_model, X_train, X_test, ['Not Churn', 'Churn'], 1)


#Running Random Forest
rf_model, y_pred=run_model_and_plot_roc('Random', X_train, X_test, y_train, y_test)
rf_conf = evaluate_confusion_matrix(y_test, y_pred)
rf = plot_feature_importances(rf_model)
rf_lime = lime_explanation(rf_model, X_train, X_test,['Not Churn', 'Churn'], 1)

#Running Adaboost
ab_model, y_pred=run_model_and_plot_roc('Adaboost', X_train, X_test, y_train, y_test)
ab_conf = evaluate_confusion_matrix(y_test, y_pred)
ab = plot_feature_importances(ab_model)
ab_lime = lime_explanation(ab_model, X_train, X_test, ['Not Churn', 'Churn'], 1)

#Running Gradient
grad_model, y_pred=run_model_and_plot_roc('Gradient', X_train, X_test, y_train, y_test)
grad_conf = evaluate_confusion_matrix(y_test, y_pred)
gd = plot_feature_importances(grad_model)
gd_lime = lime_explanation(grad_model, X_train, X_test, ['Not Churn', 'Churn'], 1)
