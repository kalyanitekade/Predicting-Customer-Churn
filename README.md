# Customer Churn Prediction with Machine Learning and Streamlit

This repository contains the code for a comprehensive Customer Churn Prediction project that leverages various machine learning models to predict customer churn. The project is designed to aid businesses in identifying at-risk customers and developing effective retention strategies. The models are trained using balanced datasets (SMOTE) and provide transparent, interpretable results using LIME.

# Key Features:
1. Multiple Machine Learning Models: Implemented and compared several classification models, including Logistic Regression, Random Forest, AdaBoost, and Gradient Boosting, to identify the best performing model.
2. Data Balancing with SMOTE: Applied the Synthetic Minority Over-sampling Technique (SMOTE) to handle class imbalance, leading to a significant improvement in model performance.
3. Feature Engineering: Performed feature engineering to improve model accuracy, with techniques including one-hot encoding for categorical features.
4. Model Interpretability with LIME: Integrated LIME to provide local, interpretable model explanations, enabling transparent predictions and aiding decision-making.
5. Interactive Web Application: Developed a Streamlit web app to allow users to input key customer data and receive real-time churn predictions with feature explanations, improving transparency and usability.
6. Visualizations: Generated comprehensive visualizations for ROC curves, feature importance, and model performance metrics to evaluate and compare model effectiveness.

# Technologies Used:
1. Python: Core language for model development and data manipulation.
2. Scikit-learn: For implementing machine learning models and SMOTE.
3. LIME: For generating model interpretability and explanations.
4. Streamlit: To build a user-friendly web interface for the churn prediction system.
5. Pandas & NumPy: For data manipulation and feature engineering.
6. Matplotlib & Seaborn: For data visualization and performance analysis.

# Results 
Below are the screenshot from the UI. User are given choice to input 5 features and Random Forest algorithm is used in the background as it has the highest predictive capability compared to other models. 

![ss1](https://github.com/user-attachments/assets/9eb142ed-3f5c-42b6-8b4f-a22b67e69e36)

![ss](https://github.com/user-attachments/assets/f8419494-2b72-4b8e-8d5e-4af3165bbf98)
