import pandas as pd
from EvaluationMetrics.evaluation_metrics import evaluate_model
from Models.LogisticRegression.logistic_regression import train_logistic_regression
from Models.DecisionTrees.decision_trees import train_decision_trees
from Models.SupportVectorMachines.support_vector_machines import train_support_vector_machines
from Models.KNearestNeighbors.k_nearest_neighbors import train_k_nearest_neighbors

def evaluate_all_models():
    df = pd.read_csv('Datasets/loan_train.csv')
    X = df.drop('Target', axis=1)
    y = df['Target']

    log_reg_model = train_logistic_regression(X, y)
    dec_tree_model = train_decision_trees(X, y)
    svm_model = train_support_vector_machines(X, y)
    knn_model = train_k_nearest_neighbors(X, y)

    y_pred_log_reg = log_reg_model.predict(X)
    y_pred_dec_tree = dec_tree_model.predict(X)
    y_pred_svm = svm_model.predict(X)
    y_pred_knn = knn_model.predict(X)

    evaluate_model(y, y_pred_log_reg)
    evaluate_model(y, y_pred_dec_tree)
    evaluate_model(y, y_pred_svm)
    evaluate_model(y, y_pred_knn)

if __name__ == "__main__":
    evaluate_all_models()