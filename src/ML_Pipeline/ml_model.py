import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os 
import pickle 

#Function to prepare smote model
def prepare_model_smote(df, target_col, col_to_exclude):
    cols = df.select_dtypes(include=np.number).columns.tolist()
    X=df[cols]
    X=X[X.columns.difference([target_col])]
    X=X[X.columns.difference(col_to_exclude)]
    
    y=df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
    sm=SMOTE(random_state=0, sampling_strategy=1.0)
    X_train, y_train=sm.fit_resample(X_train, y_train)
    return (X_train, X_test, y_train, y_test)


output = './output'

def run_model_and_plot_roc(model, X_train, X_test, y_train, y_test):
    roc_curve_folder = os.path.join(output, 'ROC_curves')
    models_folder = os.path.join(output, 'models')
    os.makedirs(roc_curve_folder, exist_ok=True)
    os.makedirs(models_folder, exist_ok=True)
    
    if model == "Logistic":
        log_reg = LogisticRegression(random_state=0)
        log_reg.fit(X_train, y_train)
        y_pred = log_reg.predict(X_test)
        model_instance = log_reg
        
    elif model == "Decision":
        decision_tree = DecisionTreeClassifier(max_depth=5, random_state=0)
        decision_tree.fit(X_train, y_train)
        y_pred = decision_tree.predict(X_test)
        model_instance = decision_tree


    elif model == "Random":
        random_forest = RandomForestClassifier(max_depth=5, random_state=0)
        random_forest.fit(X_train, y_train)
        y_pred = random_forest.predict(X_test)
        model_instance = random_forest

    elif model == "Adaboost":
        adaboost = AdaBoostClassifier(n_estimators=100, random_state=0)
        adaboost.fit(X_train, y_train)
        y_pred = adaboost.predict(X_test)
        model_instance = adaboost

    elif model == "Gradient":
        grad = GradientBoostingClassifier(random_state=0)
        grad.fit(X_train, y_train)
        y_pred = grad.predict(X_test)
        model_instance = grad

    else:
        print(f"Please choose one from (Logistic, Decision, Random, Adaboost, Gradient)")
        return

    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)

    # Print classification report
    print(classification_report(y_test, y_pred))
    print(f"Area under ROC curve is {roc_auc}")

    # Plotting the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model} (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model}')
    plt.legend(loc="lower right")
    
    # Save the ROC curve plot
    roc_curve_path = os.path.join(roc_curve_folder, f'ROC_Curve_{model}.png')
    plt.savefig(roc_curve_path)
    plt.show()

    # Save the model using pickle
    model_path = os.path.join(models_folder, f'model_{model}.pkl')
    with open(model_path, 'wb') as model_file:
        pickle.dump(model_instance, model_file)

    return model_instance, y_pred