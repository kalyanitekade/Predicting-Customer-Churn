import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

def evaluate_confusion_matrix(y_test, y_pred):
    cm = sk_confusion_matrix(y_test, y_pred)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"TP is {tp}")
    print(f"TN is {tn}")
    print(f"FP is {fp}")
    print(f"FN is {fn}")
    return cm
    

