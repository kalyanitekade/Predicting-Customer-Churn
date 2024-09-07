import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import lime
import lime.lime_tabular
import os

output = './output'

# define a function for plotting the feature importances
def plot_feature_importances(model):
  feature_imp_folder = os.path.join(output, "feat_imp")
  os.makedirs(feature_imp_folder, exist_ok=True)
  feature_importances = pd.Series(model.feature_importances_, index=model.feature_names_in_)
  
  feature_importances = feature_importances.sort_values(axis=0, ascending=False)
  fig, ax = plt.subplots()
  feature_importances.plot.bar()
  ax.set_title("Feature importances")
  fig.tight_layout()
  
  # Save the Feature importance plot
  feat_imp_path = os.path.join(feature_imp_folder, f'feat_imp_{model}.png')
  plt.savefig(feat_imp_path)
  plt.show()
  

def lime_explanation(model, X_train, X_test, class_names, chosen_index):
    lime_folder = os.path.join(output, "lime")
    os.makedirs(lime_folder, exist_ok=True)
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values, feature_names=X_train.columns, class_names=class_names, kernel_width=5)
    
    choosen_instance = X_test.loc[[chosen_index]].values[0]
    exp = explainer.explain_instance(choosen_instance, lambda x: model.predict_proba(x).astype(float), num_features=10)
    
    # Save the full explanation as an HTML file
    lime_path_html = os.path.join(lime_folder, f'Lime_{model}.html')
    exp.save_to_file(lime_path_html)
    
    print(f"LIME explanation saved to {lime_path_html}. You can open this file in a web browser.")

  
  
# define a function for creating lime list
def lime_explanation_list(model,X_train,X_test,class_names,chosen_index):
  explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,feature_names = X_train.columns,class_names=class_names,kernel_width=5)
  choosen_instance = X_test.loc[[chosen_index]].values[0]
  exp = explainer.explain_instance(choosen_instance, lambda x: model.predict_proba(x).astype(float),num_features=10)
  return exp.as_list()
  