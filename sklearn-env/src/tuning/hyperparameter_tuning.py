from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
    }
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
    grid.fit(X_train, y_train)
    print("Best Params:", grid.best_params_)
    return grid.best_estimator_
