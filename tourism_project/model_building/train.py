
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from xgboost import XGBClassifier


# ---------------------------------------------------------
# 1. Load train and test data
# ---------------------------------------------------------

X_train = pd.read_csv("Xtrain.csv")
X_test = pd.read_csv("Xtest.csv")

y_train = pd.read_csv("ytrain.csv").iloc[:, 0].astype(int)
y_test = pd.read_csv("ytest.csv").iloc[:, 0].astype(int)

print("Training data shape:", X_train.shape)
print("Testing data shape :", X_test.shape)


# ---------------------------------------------------------
# 2. Identify numerical and categorical columns
# ---------------------------------------------------------

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["int64", "float64"]
).columns.tolist()

print("Numerical features:", numeric_features)
print("Categorical features:", categorical_features)


# ---------------------------------------------------------
# 3. Create preprocessing pipeline
# ---------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ---------------------------------------------------------
# 4. Define XGBoost model
# ---------------------------------------------------------

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=2,
    tree_method="hist"
)


# ---------------------------------------------------------
# 5. Create complete ML pipeline
# ---------------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ---------------------------------------------------------
# 6. Define hyperparameter grid
# ---------------------------------------------------------

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8],
    "model__colsample_bytree": [0.8]
}


# ---------------------------------------------------------
# 7. Configure MLflow
# ---------------------------------------------------------

mlflow.set_experiment("Visit_with_Us_Wellness_Tourism")


# ---------------------------------------------------------
# 8. Hyperparameter tuning and experiment tracking
# ---------------------------------------------------------

with mlflow.start_run(run_name="XGBoost_GridSearch"):

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=2,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\nBest Parameters:")
    print(grid_search.best_params__)


    # -----------------------------------------------------
    # 9. Evaluate best model
    # -----------------------------------------------------

    y_pred = best_model.predict(X_test)

    y_probability = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )


    # -----------------------------------------------------
    # 10. Log parameters and metrics to MLflow
    # -----------------------------------------------------

    mlflow.log_params(grid_search.best_params_)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(
        best_model,
        "model"
    )


    # -----------------------------------------------------
    # 11. Display evaluation results
    # -----------------------------------------------------

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


# ---------------------------------------------------------
# 12. Save best model
# ---------------------------------------------------------

deployment_dir = "tourism_project/deployment"

os.makedirs(
    deployment_dir,
    exist_ok=True
)

model_path = os.path.join(
    deployment_dir,
    "best_model.joblib"
)

joblib.dump(
    best_model,
    model_path
)

print("\nBest model saved successfully:")
print(model_path)
