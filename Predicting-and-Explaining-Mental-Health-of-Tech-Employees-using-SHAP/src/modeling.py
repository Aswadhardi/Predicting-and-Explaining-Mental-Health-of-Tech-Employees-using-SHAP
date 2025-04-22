from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_models():
    return {
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'n_estimators': [100, 200],
                'max_depth': [None, 10],
                'max_features': ['sqrt']
            }
        },
        "XGBoost": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            "params": {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1]
            }
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, solver='lbfgs'),
            "params": {
                'C': [0.1, 1.0, 10.0]
            }
        }
    }


