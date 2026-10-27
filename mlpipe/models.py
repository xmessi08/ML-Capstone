from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from .config import SEED

CANDIDATES = {
    "baseline": DummyClassifier(strategy="most_frequent"),
    "logreg": LogisticRegression(max_iter=1000),
    "forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
    "boosting": GradientBoostingClassifier(random_state=SEED),
}
