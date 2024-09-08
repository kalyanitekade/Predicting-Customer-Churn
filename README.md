# Customer Churn Prediction with Machine Learning and Streamlit

This repository contains the code for a comprehensive Customer Churn Prediction project that leverages various machine learning models to predict customer churn. The project is designed to aid businesses in identifying at-risk customers and developing effective retention strategies. The models are trained using balanced datasets (SMOTE) and provide transparent, interpretable results using LIME.

Key Features:
Multiple Machine Learning Models: Implemented and compared several classification models, including Logistic Regression, Random Forest, AdaBoost, and Gradient Boosting, to identify the best performing model.

Data Balancing with SMOTE: Applied the Synthetic Minority Over-sampling Technique (SMOTE) to handle class imbalance, leading to a significant improvement in model performance.

Feature Engineering: Performed feature engineering to improve model accuracy, with techniques including one-hot encoding for categorical features.

Model Interpretability with LIME: Integrated LIME to provide local, interpretable model explanations, enabling transparent predictions and aiding decision-making.

Interactive Web Application: Developed a Streamlit web app to allow users to input key customer data and receive real-time churn predictions with feature explanations, improving transparency and usability.

Visualizations: Generated comprehensive visualizations for ROC curves, feature importance, and model performance metrics to evaluate and compare model effectiveness.

Technologies Used:
Python: Core language for model development and data manipulation.
Scikit-learn: For implementing machine learning models and SMOTE.
LIME: For generating model interpretability and explanations.
Streamlit: To build a user-friendly web interface for the churn prediction system.
Pandas & NumPy: For data manipulation and feature engineering.
Matplotlib & Seaborn: For data visualization and performance analysis.
