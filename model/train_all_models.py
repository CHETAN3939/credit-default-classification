import os
import joblib
import pandas as pd

from data_loader import load_and_split_data
from logistic_model import train_logistic
from decision_tree_model import train_tree
from knn_model import train_knn
from naive_bayes_model import train_nb
from random_forest_model import train_rf
from xgboost_model import train_xgb


dataset_file = "default of credit card clients.xls"

X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_split_data(dataset_file)

os.makedirs("saved_models", exist_ok=True)

results = {}

logistic_model, logistic_metrics = train_logistic(X_train_scaled, X_test_scaled, y_train, y_test)
results["Logistic Regression"] = logistic_metrics

tree_model, tree_metrics = train_tree(X_train, X_test, y_train, y_test)
results["Decision Tree"] = tree_metrics

knn_model, knn_metrics = train_knn(X_train_scaled, X_test_scaled, y_train, y_test)
results["KNN"] = knn_metrics

bayes_model, bayes_metrics = train_nb(X_train, X_test, y_train, y_test)
results["Naive Bayes"] = bayes_metrics

forest_model, forest_metrics = train_rf(X_train, X_test, y_train, y_test)
results["Random Forest"] = forest_metrics

xgb_model, xgb_metrics = train_xgb(X_train, X_test, y_train, y_test)
results["XGBoost"] = xgb_metrics


joblib.dump(logistic_model, "saved_models/logistic.pkl")
joblib.dump(tree_model, "saved_models/tree.pkl")
joblib.dump(knn_model, "saved_models/knn.pkl")
joblib.dump(bayes_model, "saved_models/bayes.pkl")
joblib.dump(forest_model, "saved_models/forest.pkl")
joblib.dump(xgb_model, "saved_models/xgboost.pkl")
joblib.dump(scaler, "saved_models/scaler.pkl")


print("\n===== MODEL COMPARISON TABLE =====\n")
comparison_df = pd.DataFrame(results).T
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
print(comparison_df)

print("\nAll models trained and saved successfully!")

