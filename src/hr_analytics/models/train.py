from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from hr_analytics.utils.metrics import smape
from hr_analytics.config import RANDOM_STATE


def train_decision_tree(X, y, preprocessor):

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeRegressor(random_state=RANDOM_STATE))
    ])

    params = {
        "model__max_depth": range(8, 15),
        "model__min_samples_split": range(8, 15),
        "model__min_samples_leaf": range(2, 10),
    }

    scorer = make_scorer(smape, greater_is_better=False)

    grid = GridSearchCV(
        pipe,
        params,
        scoring=scorer,
        cv=8,
        n_jobs=-1
    )

    grid.fit(X, y)
    return grid