# models/

`python train.py` saves the trained models here (same names as the Colab analysis):

- `placement_model.pkl`: Random Forest Classifier (Placed / Not Placed)
- `salary_model.pkl`: Random Forest Regressor (salary, trained on placed students only)
- `placement_columns.pkl`, `salary_columns.pkl`: one-hot column order used by each model
- `categorical_cols.pkl`: the categorical columns that were one-hot encoded
- `model_metadata.json`: category values, number ranges, metrics and feature importances (used by the web form and AI insights)

The Colab `.pkl` files can also be copied here directly; the app then rebuilds
the metadata from `data/global_placement.csv`.

Flask loads these files once at startup; it never trains.
