"""
train.py
--------
Run this file BY HAND (python train.py) to train and save the models.
Flask never runs this file.

It follows the same workflow as the Colab analysis (untitled2.py):

    1. Load dataset
    2. Inspect data
    3. Select features (X) and targets (y)
    4. Train/Test split
    5. Build model  (Random Forest)
    6. Train
    7. Evaluate     (placement: Accuracy, Classification Report
                     salary:    MAE, R²)
    8. Save models  (joblib -> models/ folder)
"""

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score,
)
from sklearn.model_selection import train_test_split

import ml_config
from predict import build_metadata


def main():
    # ==================================================
    # STEP 1: LOAD DATASET
    # ==================================================
    if not ml_config.DATASET_PATH.exists():
        print(f"Dataset not found: {ml_config.DATASET_PATH}")
        print("Copy global_placement.csv into the data/ folder first.")
        return

    df = pd.read_csv(ml_config.DATASET_PATH)

    # ==================================================
    # STEP 2: INSPECT DATA
    # ==================================================
    print("Rows, columns:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    print(df[ml_config.EMPLOYMENT_TARGET].value_counts().to_string(), "\n")

    feature_cols = ml_config.CATEGORICAL_FEATURES + ml_config.NUMERICAL_FEATURES
    missing = [c for c in feature_cols + [ml_config.EMPLOYMENT_TARGET, ml_config.SALARY_TARGET]
               if c not in df.columns]
    if missing:
        print("These columns are missing from the dataset:", missing)
        return

    # ==================================================
    # STEP 3: FEATURES (X) AND TARGETS (y)
    # ==================================================
    # Placed = 1, Not Placed = 0
    df["placement_label"] = (df[ml_config.EMPLOYMENT_TARGET] == ml_config.POSITIVE_CLASS).astype(int)

    # Salary model: only students who were actually placed
    df_placed = df[df["placement_label"] == 1].copy()

    cat = ml_config.CATEGORICAL_FEATURES
    df_encoded = pd.get_dummies(df, columns=cat, drop_first=True)
    df_placed_encoded = pd.get_dummies(df_placed, columns=cat, drop_first=True)

    # Salary is dropped from the placement features (it would leak the answer)
    drop_cols = [ml_config.EMPLOYMENT_TARGET, "placement_label", ml_config.SALARY_TARGET]
    X_placement = df_encoded.drop(columns=drop_cols)
    y_placement = df_encoded["placement_label"]

    X_salary = df_placed_encoded.drop(columns=drop_cols)
    y_salary = df_placed_encoded[ml_config.SALARY_TARGET]

    # ==================================================
    # STEP 4: TRAIN/TEST SPLIT
    # ==================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_placement, y_placement, test_size=ml_config.TEST_SIZE,
        random_state=ml_config.RANDOM_STATE, stratify=y_placement,
    )
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_salary, y_salary, test_size=ml_config.TEST_SIZE,
        random_state=ml_config.RANDOM_STATE,
    )
    print(f"Placement model: {len(X_train)} train / {len(X_test)} test")
    print(f"Salary model:    {len(X_train_s)} train / {len(X_test_s)} test\n")

    # ==================================================
    # STEP 5 + 6: BUILD AND TRAIN (Random Forest)
    # ==================================================
    placement_model = RandomForestClassifier(
        n_estimators=ml_config.N_ESTIMATORS, random_state=ml_config.RANDOM_STATE, n_jobs=-1,
    )
    placement_model.fit(X_train, y_train)

    salary_model = RandomForestRegressor(
        n_estimators=ml_config.N_ESTIMATORS, random_state=ml_config.RANDOM_STATE, n_jobs=-1,
    )
    salary_model.fit(X_train_s, y_train_s)

    # ==================================================
    # STEP 7: EVALUATE
    # ==================================================
    placement_pred = placement_model.predict(X_test)
    accuracy = accuracy_score(y_test, placement_pred)
    baseline = max(y_test.mean(), 1 - y_test.mean())

    print("=== Placement model (Random Forest Classifier) ===")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Baseline (always guess majority class): {baseline:.2%}")
    print(classification_report(y_test, placement_pred, target_names=["Not Placed", "Placed"]))
    print("Confusion matrix [rows = actual, cols = predicted]:")
    print(confusion_matrix(y_test, placement_pred), "\n")

    salary_pred = salary_model.predict(X_test_s)
    mae = mean_absolute_error(y_test_s, salary_pred)
    r2 = r2_score(y_test_s, salary_pred)
    avg_salary = y_test_s.mean()

    print("=== Salary model (Random Forest Regressor) ===")
    print(f"MAE: {mae:.2f}")
    print(f"R² score: {r2:.4f}")
    print(f"Average salary in test set: {avg_salary:.2f}")
    print(f"MAE as % of average salary: {mae / avg_salary * 100:.2f}%\n")

    # ==================================================
    # STEP 8: SAVE MODELS
    # ==================================================
    ml_config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(placement_model, ml_config.PLACEMENT_MODEL_PATH)
    joblib.dump(salary_model, ml_config.SALARY_MODEL_PATH)
    joblib.dump(X_placement.columns.tolist(), ml_config.PLACEMENT_COLUMNS_PATH)
    joblib.dump(X_salary.columns.tolist(), ml_config.SALARY_COLUMNS_PATH)
    joblib.dump(cat, ml_config.CATEGORICAL_COLS_PATH)

    metadata = build_metadata(
        df, placement_model, X_placement.columns.tolist(),
        salary_model, X_salary.columns.tolist(),
    )
    metadata["metrics"] = {
        "placement_accuracy": round(float(accuracy), 4),
        "placement_baseline": round(float(baseline), 4),
        "salary_mae": round(float(mae), 2),
        "salary_r2": round(float(r2), 4),
        "salary_avg": round(float(avg_salary), 2),
    }
    with open(ml_config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Models and metadata saved to {ml_config.MODELS_DIR}/")


if __name__ == "__main__":
    main()
