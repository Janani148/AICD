"""Classical baselines: Logistic Regression, SVM, CatBoost x {TF-IDF, AST, both}."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def make_classifier(name: str, seed: int = 0):
    if name == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000, random_state=seed))
    if name == "svm":
        return make_pipeline(StandardScaler(), LinearSVC(max_iter=2000, dual="auto", random_state=seed))
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, random_seed=seed, verbose=0,
                                  loss_function="MultiClass")
    raise ValueError(name)
