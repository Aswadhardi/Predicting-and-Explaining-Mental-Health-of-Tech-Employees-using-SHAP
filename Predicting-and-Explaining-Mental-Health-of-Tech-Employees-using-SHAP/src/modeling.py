from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_models():
    tuned_models = {

        "Logistic Regression (Tuned)": {
            "model": LogisticRegression(max_iter=1000, solver='saga', random_state=42),
            "params": {
                'penalty': ['l1', 'l2'],
            }
        },
        "Random Forest (Tuned)": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'max_features': ['sqrt', 'log2'],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }
        },
        "Decision Tree Classifier (Tuned)": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini']
            }
        },
        "AdaBoost Classifier (Tuned)": {
            "model": AdaBoostClassifier(random_state=42),
            "params": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0]
            }
        }, "XGBoost (Tuned)": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            "params": {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                
            }
        },

    }

    non_tuned_models = {
        "Logistic Regression (Baseline)": {
            "model": LogisticRegression(),
            "params": {}
        },
        "Random Forest Classifier (Baseline)": {
            "model": RandomForestClassifier(n_estimators=60, random_state=42),
            "params": {}
        },
        "Decision Tree Classifier": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {}
        },
        "AdaBoost Classifier": {
            "model": AdaBoostClassifier(),
            "params": {}
        },
        "XGBoost Classifier": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            "params": {}
        }
    }

    return tuned_models, non_tuned_models
