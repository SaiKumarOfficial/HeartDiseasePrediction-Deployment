import os
import pandas as pd  
import warnings        
import argparse
from get_data import read_params
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics 
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,plot_roc_curve,accuracy_score
from sklearn.metrics import average_precision_score
from numpyencoder import NumpyEncoder
import joblib 
import json 
import math



def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config['split_data']['train_path']
    test_data_path = config['split_data']['test_path']
    target = config['base']['target_col']
    model_dir = config['model_dir']

    train_df = pd.read_csv(train_data_path, sep=",")
    test_df = pd.read_csv(test_data_path, sep=",")

    X_train = train_df.drop(target,axis =1)
    y_train = train_df[target]     

    X_test = test_df.drop(target,axis =1)
    y_test = test_df[target] 

    model = LogisticRegression()
    model.fit(X_train,y_train)
    # Report training set score
    train_score = model.score(X_train, y_train) * 100
    print(train_score)
    # Report test set score
    test_score = model.score(X_test, y_test) * 100
    print(test_score)

    predicted_val = model.predict(X_test)
    precision, recall, prc_thresholds = metrics.precision_recall_curve(y_test, predicted_val)
    # print('precision value:', precision)
    # print('recall value:', recall)
    fpr, tpr, roc_thresholds = metrics.roc_curve(y_test, predicted_val)
    avg_prec = metrics.average_precision_score(y_test, predicted_val)
    roc_auc = metrics.roc_auc_score(y_test, predicted_val)

    scores_file = config["reports"]["scores"]
    prc_file = config["reports"]["prc"]
    roc_file = config["reports"]["roc"]
    auc_file = config["reports"]["auc"]
    # cm_file = config["reports"]["cm"]

    # with open(scores_file, "w") as fd:
    #     json.dump({"avg_prec": avg_prec, "roc_auc": roc_auc}, fd, indent=4)

    
    nth_point = math.ceil(len(prc_thresholds)/1000)
    prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]    
    
    
    with open(prc_file, "w") as fd:
        prcs = {
                "prc": [
                    {"precision": p, "recall": r, "threshold": t}
                    for p, r, t in prc_points
                ]
            }
        json.dump(prcs, fd, indent=4, cls=NumpyEncoder)
        

    with open(roc_file, "w") as fd:
        rocs = {
                "roc": [
                    {"fpr": fp, "tpr": tp, "threshold": t}
                    for fp, tp, t in zip(fpr, tpr, roc_thresholds)
                ]
            }
        json.dump(rocs, fd, indent=4, cls=NumpyEncoder)
    print(classification_report(y_test, predicted_val))

    # Confusion Matrix and plot
    cm = confusion_matrix(y_test, predicted_val)
    print(cm)

        
    df1 = pd.DataFrame(predicted_val, columns = ['Predicted'])
    df_cm = pd.concat([y_test, df1], axis=1)
    print(df_cm)
    
    # with open(cm_file, "w") as fd:
    #     df1 = pd.DataFrame(predicted_val, columns = ['Predicted'])
    #     df_cm = pd.concat([test_y, df1], axis=1)
    #     # print(df_cm)
        
    df_cm.to_csv('cm.csv', index = False)

    # with open(auc_file, "w") as fd:
    #     json.dump(df_cm.to_json(), fd, indent=4, cls=NumpyEncoder)

    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print('ROC_AUC:{0:0.2f}'.format(roc_auc))

    Logistic_Accuracy = accuracy_score(y_test, predicted_val)
    print('Logistic Regression Model Accuracy:{0:0.2f}'.format(Logistic_Accuracy))

    # Average precision score
    average_precision = average_precision_score(y_test, predicted_val)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    
    with open(scores_file, "w") as f:
        scores = {
            "train_score": train_score,
            "test_score": test_score,
            "roc_auc": roc_auc,
            #"Precision": precision,
            #"Recall": recall,            "Average precision": average_precision,
            "Logistic Accuracy": Logistic_Accuracy
            # "Random Forest Accuracy": RF_Accuracy                                 
            
        }
        json.dump(scores, f, indent=4)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(model, model_path)

    
if __name__=="__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--config",default="params.yaml")
    parse_args = arg.parse_args()
    train_and_evaluate(config_path=parse_args.config)