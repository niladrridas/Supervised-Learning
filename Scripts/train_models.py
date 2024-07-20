import pandas as pd
from Models.LogisticRegression.logistic_regression import train_logistic_regression
from Models.DecisionTrees.decision_trees import train_decision_trees
from Models.SupportVectorMachines.support_vector_machines import train_support_vector_machines
from Models.KNearestNeighbors.k_nearest_neighbors import train_k_nearest_neighbors

def train_all_models():
    df = pd.read_csv('Datasets/loan_train.csv')
    X = df.drop('Target', axis=1)
    y = df['Target']

    log_reg_model = train_logistic_regression(X, y)
    dec_tree_model = train_decision_trees(X, y)
    svm_model = train_support_vector_machines(X, y)
    knn_model = train_k_nearest_neighbors(X, y)

    # Save models to files
    import pickle
    with open('Models/LogisticRegression/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(log_reg_model, f)
    with open('Models/DecisionTrees/decision_trees_model.pkl', 'wb') as f:
        pickle.dump(dec_tree_model, f)
    with open('Models/SupportVectorMachines/support_vector_machines_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    with open('Models/KNearestNeighbors/k_nearest_neighbors_model.pkl', 'wb') as f:
        pickle.dump(knn_model, f)

if __name__ == "__main__":
    train_all_models()