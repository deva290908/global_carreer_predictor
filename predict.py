"""
predict.py
----------
Used by Flask (app.py) to make predictions from the web form.

It:
    1. Loads the saved models from the models/ folder (ONCE, when Flask starts)
    2. Checks the form inputs and converts them into a one-row table
       encoded exactly like the training data (pd.get_dummies + column alignment)
    3. Predicts Placed / Not Placed + probability (Random Forest Classifier)
    4. If placed, predicts salary (Random Forest Regressor)
    5. Builds explainable "AI Profile Insights" from the dataset and the
       models' feature importances

It never trains anything (that is train.py's job).
"""

import json
import math

import joblib
import pandas as pd

import ml_config


# ==================================================
# METADATA (category values, number ranges, feature importances)
# ==================================================

def _group_importances(model, columns):
    """
    One-hot encoding splits a text column into many columns
    (country_India, country_USA, ...). Add their importances back
    together so we get one importance per ORIGINAL feature.
    """
    grouped = {}
    for col, imp in zip(columns, model.feature_importances_):
        feature = col
        for cat in ml_config.CATEGORICAL_FEATURES:
            if col.startswith(cat + "_"):
                feature = cat
                break
        grouped[feature] = grouped.get(feature, 0.0) + float(imp)
    return dict(sorted(grouped.items(), key=lambda kv: kv[1], reverse=True))


def build_metadata(df, placement_model, placement_columns, salary_model, salary_columns):
    """Everything the website needs to know about the dataset and the models."""
    categories = {
        c: sorted(df[c].dropna().astype(str).unique().tolist())
        for c in ml_config.CATEGORICAL_FEATURES
    }
    numeric = {}
    for c in ml_config.NUMERICAL_FEATURES:
        s = df[c].dropna()
        numeric[c] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "p25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "p75": float(s.quantile(0.75)),
        }
    return {
        "n_rows": int(len(df)),
        "categories": categories,
        "numeric": numeric,
        "placement_importance": _group_importances(placement_model, placement_columns),
        "salary_importance": _group_importances(salary_model, salary_columns),
    }


# ==================================================
# LOAD MODELS ONCE
# ==================================================

_state = {
    "ready": False,
    "error": None,
    "placement_model": None,
    "salary_model": None,
    "placement_columns": None,
    "salary_columns": None,
    "metadata": None,
}


def load_models():
    """Load models + metadata into memory. Called once when Flask starts."""
    required = [
        ml_config.PLACEMENT_MODEL_PATH,
        ml_config.SALARY_MODEL_PATH,
        ml_config.PLACEMENT_COLUMNS_PATH,
        ml_config.SALARY_COLUMNS_PATH,
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        _state["error"] = (
            "Trained model files are missing (" + ", ".join(missing) + "). "
            "Place global_placement.csv in data/ and run: python train.py"
        )
        return False

    try:
        _state["placement_model"] = joblib.load(ml_config.PLACEMENT_MODEL_PATH)
        _state["salary_model"] = joblib.load(ml_config.SALARY_MODEL_PATH)
        _state["placement_columns"] = list(joblib.load(ml_config.PLACEMENT_COLUMNS_PATH))
        _state["salary_columns"] = list(joblib.load(ml_config.SALARY_COLUMNS_PATH))

        if ml_config.METADATA_PATH.exists():
            with open(ml_config.METADATA_PATH) as f:
                _state["metadata"] = json.load(f)
        elif ml_config.DATASET_PATH.exists():
            # e.g. the Colab .pkl files were copied in without running train.py
            _state["metadata"] = build_metadata(
                pd.read_csv(ml_config.DATASET_PATH),
                _state["placement_model"], _state["placement_columns"],
                _state["salary_model"], _state["salary_columns"],
            )
        else:
            _state["error"] = (
                "Model metadata is missing. Place global_placement.csv in data/ "
                "and run: python train.py"
            )
            return False
    except Exception as exc:  # corrupted file, version mismatch, ...
        _state["error"] = f"Could not load the trained models ({type(exc).__name__})."
        return False

    _state["ready"] = True
    _state["error"] = None
    return True


def is_ready():
    return _state["ready"]


def get_error():
    return _state["error"]


def get_metadata():
    return _state["metadata"]


# ==================================================
# INPUT VALIDATION
# ==================================================

def _fmt(x):
    """8.0 -> '8', 7.5 -> '7.5'"""
    return f"{x:g}"


def validate_input(form):
    """
    Check the raw form values.
    Returns (student_dict, errors). student_dict is None if there are errors.
    """
    meta = _state["metadata"]
    labels = ml_config.FEATURE_LABELS
    student, errors = {}, []

    for c in ml_config.NUMERICAL_FEATURES:
        raw = (form.get(c) or "").strip()
        if raw == "":
            errors.append(f"{labels[c]} is required.")
            continue
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"{labels[c]} must be a number.")
            continue
        if not math.isfinite(value):
            errors.append(f"{labels[c]} must be a number.")
            continue
        if c in ml_config.INTEGER_FEATURES and not value.is_integer():
            errors.append(f"{labels[c]} must be a whole number.")
            continue
        lo, hi = meta["numeric"][c]["min"], meta["numeric"][c]["max"]
        if not lo <= value <= hi:
            errors.append(
                f"{labels[c]} must be between {_fmt(lo)} and {_fmt(hi)} "
                f"(the range seen in the training data)."
            )
            continue
        student[c] = int(value) if c in ml_config.INTEGER_FEATURES else value

    for c in ml_config.CATEGORICAL_FEATURES:
        raw = (form.get(c) or "").strip()
        if raw == "":
            errors.append(f"{labels[c]} is required.")
        elif raw not in meta["categories"][c]:
            errors.append(f"'{raw}' is not a valid {labels[c]}.")
        else:
            student[c] = raw

    return (None, errors) if errors else (student, [])


# ==================================================
# PREDICTION
# ==================================================

def _encode(students, columns):
    """One-hot encode like the training data and align to the model's columns."""
    df = pd.DataFrame(students)
    encoded = pd.get_dummies(df, columns=ml_config.CATEGORICAL_FEATURES)
    return encoded.reindex(columns=columns, fill_value=0)


def _placement_proba(students):
    model = _state["placement_model"]
    X = _encode(students, _state["placement_columns"])
    placed_idx = list(model.classes_).index(1)
    return model.predict_proba(X)[:, placed_idx]


def _salary(students):
    X = _encode(students, _state["salary_columns"])
    return _state["salary_model"].predict(X)


def predict_student_outcome(student):
    """
    Same logic as predict_student_outcome() in the Colab analysis:
    placement first, salary only if predicted Placed.
    """
    model = _state["placement_model"]
    X = _encode([student], _state["placement_columns"])
    placed = int(model.predict(X)[0]) == 1
    proba = float(_placement_proba([student])[0])

    result = {
        "placed": placed,
        "placement_status": "Placed" if placed else "Not Placed",
        "placement_probability": round(proba * 100, 1),
        "predicted_salary": round(float(_salary([student])[0])) if placed else None,
    }
    result["insights"] = profile_insights(student)
    result["key_factors"] = key_factors()
    result["what_if"] = what_if_analysis(student, proba, placed)
    return result


# ==================================================
# AI PROFILE INSIGHTS (explainability layer)
# ==================================================
# These compare the profile with the training data and describe how the
# model behaves. They are NOT cause-and-effect claims.

HIGHER_IS_STRONGER = {
    "cgpa": "CGPA",
    "internship_count": "internship exposure",
    "aptitude_score": "aptitude score",
    "communication_score": "communication score",
    "internship_quality_score": "internship quality score",
}


def profile_insights(student):
    """Where each number sits compared with the students in the dataset."""
    stats = _state["metadata"]["numeric"]
    insights = []

    backlogs = student["backlogs"]
    if backlogs == 0:
        insights.append({"kind": "strength", "text": "No backlogs"})
    elif backlogs >= stats["backlogs"]["p75"]:
        insights.append({"kind": "attention",
                         "text": f"{backlogs} backlogs: in the upper 25% of the dataset"})
    else:
        insights.append({"kind": "neutral",
                         "text": f"{backlogs} backlog(s): within the typical range of the dataset"})

    for c, name in HIGHER_IS_STRONGER.items():
        v, s = student[c], stats[c]
        if v >= s["p75"]:
            insights.append({"kind": "strength",
                             "text": f"Strong {name} ({_fmt(v)}): top 25% of the dataset"})
        elif v <= s["p25"]:
            insights.append({"kind": "attention",
                             "text": f"{name[0].upper() + name[1:]} ({_fmt(v)}): lower 25% of the dataset"})
        else:
            insights.append({"kind": "neutral",
                             "text": f"{name[0].upper() + name[1:]} ({_fmt(v)}): typical range "
                                     f"(dataset median {_fmt(s['median'])})"})
    return insights


def key_factors(top_n=5):
    """The features each Random Forest relies on most (relative importance)."""
    meta = _state["metadata"]
    labels = ml_config.FEATURE_LABELS

    def top(importances):
        return [{"feature": labels.get(f, f), "importance": round(v * 100, 1)}
                for f, v in list(importances.items())[:top_n]]

    return {
        "placement": top(meta["placement_importance"]),
        "salary": top(meta["salary_importance"]),
    }


def _what_if_step(c):
    """How much to change a number in a what-if scenario (about 10% of its range)."""
    if c in ml_config.INTEGER_FEATURES:
        return 1
    s = _state["metadata"]["numeric"][c]
    step = (s["max"] - s["min"]) * 0.10
    return round(step, 1) if step < 5 else round(step)


def what_if_analysis(student, base_proba, base_placed):
    """
    Model sensitivity: change ONE input at a time and re-run the model.
    This shows how the MODEL responds, not what would happen in real life.
    """
    stats = _state["metadata"]["numeric"]
    labels = ml_config.FEATURE_LABELS
    scenarios = []

    if student["backlogs"] > 0:
        scenarios.append(("backlogs", 0, "Backlogs: 0"))

    for c in HIGHER_IS_STRONGER:
        step = _what_if_step(c)
        new = min(student[c] + step, stats[c]["max"])
        if new > student[c]:
            new = int(new) if c in ml_config.INTEGER_FEATURES else round(new, 2)
            scenarios.append((c, new, f"{labels[c]}: {_fmt(student[c])} → {_fmt(new)}"))

    if not scenarios:
        return []

    variants = [{**student, c: v} for c, v, _ in scenarios]
    probas = _placement_proba(variants)
    base_salary = float(_salary([student])[0]) if base_placed else None
    salaries = _salary(variants) if base_placed else [None] * len(variants)

    results = []
    for (c, v, label), p, sal in zip(scenarios, probas, salaries):
        item = {
            "change": label,
            "probability": round(float(p) * 100, 1),
            "prob_delta": round((float(p) - base_proba) * 100, 1),
        }
        if base_placed:
            item["salary_delta"] = round(float(sal) - base_salary)
        results.append(item)

    results.sort(key=lambda r: abs(r["prob_delta"]), reverse=True)
    return results
