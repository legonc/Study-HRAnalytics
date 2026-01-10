from hr_analytics.data.load import load_job_satisfaction_data
from hr_analytics.features.build_features import build_preprocessor
from hr_analytics.models.train import train_decision_tree

DATA_PATH = "data/raw"

train, X_test, y_test = load_job_satisfaction_data(DATA_PATH)

X = train.drop(columns=["job_satisfaction_rate"])
y = train["job_satisfaction_rate"]

num_cols = ["employment_years", "supervisor_evaluation", "salary"]
ord_cols = ["workload", "level"]
ohe_cols = ["dept", "last_year_promo", "last_year_violations"]

preprocessor = build_preprocessor(num_cols, ord_cols, ohe_cols)

model = train_decision_tree(X, y, preprocessor)

print("Best SMAPE:", -model.best_score_)