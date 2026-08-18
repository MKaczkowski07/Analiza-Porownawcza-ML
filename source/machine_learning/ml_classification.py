import pandas as pd
import numpy as np
import sys
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neural_networks.data_preprocessing_cls import TitanicPreprocessing


def run_parameter_analysis(model_class, param_name, values, X_train, y_train, X_test, y_test, **defaults):
    results = []
    print(f"Analiza parametru: {param_name} dla {model_class.__name__}")

    for val in values:
        params = defaults.copy()
        params[param_name] = val
        model = model_class(**params)

        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train) * 100
        test_acc = model.score(X_test, y_test) * 100
        diff = train_acc - test_acc

        results.append({
            'Wartość': val,
            'Dokładność Uczący (%)': round(train_acc, 2),
            'Dokładność Testowy (%)': round(test_acc, 2),
            'Różnica (p.p.)': round(diff, 2)
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("-" * 50)
    return df


def main():
    train_path = '../../data/titanic/train.csv'
    preprocessor = TitanicPreprocessing(data_path=train_path)
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = preprocessor.get_processed_data(test_size=0.2)

    X_train, X_test = X_train_nn.T, X_test_nn.T
    y_train, y_test = y_train_nn.T.ravel(), y_test_nn.T.ravel()

    # --- 1. K-Najbliższych Sąsiadów ---
    knn_defaults = {'n_neighbors': 5, 'p': 2, 'leaf_size': 30}
    run_parameter_analysis(KNeighborsClassifier, 'n_neighbors', [3, 5, 10, 20], X_train, y_train, X_test, y_test,
                           **knn_defaults)
    run_parameter_analysis(KNeighborsClassifier, 'p', [1, 2, 3, 5], X_train, y_train, X_test, y_test, **knn_defaults)
    run_parameter_analysis(KNeighborsClassifier, 'leaf_size', [10, 20, 30, 50], X_train, y_train, X_test, y_test,
                           **knn_defaults)

    # --- 2. Drzewo Decyzyjne ---
    tree_defaults = {'random_state': 42}
    run_parameter_analysis(DecisionTreeClassifier, 'max_depth', [3, 5, 10, None], X_train, y_train, X_test, y_test,
                           **tree_defaults)

    tree_defaults_with_limit = {'random_state': 42, 'max_depth': 10}
    run_parameter_analysis(DecisionTreeClassifier, 'min_samples_split', [2, 5, 10, 20], X_train, y_train, X_test,
                           y_test, **tree_defaults_with_limit)
    run_parameter_analysis(DecisionTreeClassifier, 'min_samples_leaf', [1, 2, 4, 8], X_train, y_train, X_test, y_test,
                           **tree_defaults_with_limit)

    # --- 3. Las Losowy ---
    rf_defaults = {'n_estimators': 100, 'random_state': 42}
    run_parameter_analysis(RandomForestClassifier, 'n_estimators', [10, 50, 100, 200], X_train, y_train, X_test, y_test,
                           **rf_defaults)
    run_parameter_analysis(RandomForestClassifier, 'max_depth', [5, 10, 20, None], X_train, y_train, X_test, y_test,
                           **rf_defaults)

    rf_defaults_with_limit = {'n_estimators': 100, 'max_depth': 20, 'random_state': 42}
    run_parameter_analysis(RandomForestClassifier, 'min_samples_split', [2, 5, 10, 20], X_train, y_train, X_test,
                           y_test, **rf_defaults_with_limit)

    # --- 4. SVC ---
    svc_defaults = {'kernel': 'rbf', 'C': 1.0,
                    'random_state': 42}
    run_parameter_analysis(SVC, 'kernel', ['linear', 'poly', 'rbf', 'sigmoid'], X_train, y_train, X_test, y_test,
                           **svc_defaults)
    run_parameter_analysis(SVC, 'C', [0.1, 1.0, 10.0, 100.0], X_train, y_train, X_test, y_test, **svc_defaults)
    run_parameter_analysis(SVC, 'gamma', ['scale', 'auto', 0.1, 1.0], X_train, y_train, X_test, y_test, **svc_defaults)

if __name__ == "__main__":
    main()