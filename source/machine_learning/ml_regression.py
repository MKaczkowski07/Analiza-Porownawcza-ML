import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neural_networks.data_preprocessing_reg import DataPreprocessing

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import validation_curve


def generate_validation_curve_data(model_name, estimator, param_name, param_range, X, y, preprocessor):
    print(f"\n{'=' * 80}")
    print(f"Krzywa Walidacji (CV=5) dla: {model_name} | Parametr: {param_name}")
    print(f"{'=' * 80}")

    print(f"Obliczanie wyników dla {param_name} = {param_range}...")

    train_scores, val_scores = validation_curve(
        estimator=estimator,
        X=X,
        y=y,
        param_name=param_name,
        param_range=param_range,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    train_scores_mean = np.mean(np.abs(train_scores), axis=1)
    val_scores_mean = np.mean(np.abs(val_scores), axis=1)

    results = []

    for i, val in enumerate(param_range):
        train_rmse_usd = preprocessor.inverse_transform_target(train_scores_mean[i])
        val_rmse_usd = preprocessor.inverse_transform_target(val_scores_mean[i])

        overfitting_usd = val_rmse_usd - train_rmse_usd

        results.append({
            param_name: val,
            "Błąd Uczący (USD)": round(train_rmse_usd, 2),
            "Błąd Walidacyjny (USD)": round(val_rmse_usd, 2),
            "Przeuczenie (USD)": round(overfitting_usd, 2)
        })

    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False) + "\n")


def main():
    train_path = '../../data/housing/train.csv'

    print("Pobieranie i przetwarzanie danych...")
    preprocessor = DataPreprocessing(train_path=train_path)

    X_train_T, X_val_T, y_train_T, y_val_T = preprocessor.get_processed_data(test_size=0.2)

    X_train = X_train_T.T
    y_train = y_train_T.T.ravel()

    print("\nGenerowanie danych do Krzywych Walidacji (Regression) - 3 parametry per metoda...\n")

    # ==========================================
    # 1. Metoda: k-Najbliższych Sąsiadów (k-NN)
    # ==========================================
    generate_validation_curve_data(
        'k-NN', KNeighborsRegressor(),
        'n_neighbors', [3, 5, 10, 15],
        X_train, y_train, preprocessor
    )
    generate_validation_curve_data(
        'k-NN', KNeighborsRegressor(n_neighbors=5),
        'p', [1, 2, 3, 4],
        X_train, y_train, preprocessor
    )
    generate_validation_curve_data(  # Nowy parametr 3
        'k-NN', KNeighborsRegressor(n_neighbors=5, p=1),
        'leaf_size', [10, 20, 30, 50],
        X_train, y_train, preprocessor
    )

    # ==========================================
    # 2. Metoda: Drzewo Decyzyjne
    # ==========================================
    generate_validation_curve_data(
        'Decision Tree', DecisionTreeRegressor(random_state=42),
        'max_depth', [5, 10, 20, 50],
        X_train, y_train, preprocessor
    )
    generate_validation_curve_data(
        'Decision Tree', DecisionTreeRegressor(random_state=42, max_depth=10),
        'min_samples_split', [2, 5, 10, 20],
        X_train, y_train, preprocessor
    )
    generate_validation_curve_data(  # Nowy parametr 3
        'Decision Tree', DecisionTreeRegressor(random_state=42, max_depth=10),
        'min_samples_leaf', [1, 2, 5, 10],
        X_train, y_train, preprocessor
    )

    # ==========================================
    # 3. Metoda: Random Forest
    # ==========================================
    generate_validation_curve_data(
        'Random Forest', RandomForestRegressor(random_state=42),
        'n_estimators', [10, 50, 100, 200],
        X_train, y_train, preprocessor
    )
    generate_validation_curve_data(
        'Random Forest', RandomForestRegressor(random_state=42, n_estimators=100),
        'max_depth', [5, 10, 20, 50],
        X_train, y_train, preprocessor
    )
    generate_validation_curve_data(  # Nowy parametr 3
        'Random Forest', RandomForestRegressor(random_state=42, n_estimators=100, max_depth=20),
        'min_samples_split', [2, 5, 10, 20],
        X_train, y_train, preprocessor
    )

    # ==========================================
    # 4. Metoda: Support Vector Regressor (SVR)
    # ==========================================
    generate_validation_curve_data(
        'SVR', SVR(),
        'kernel', ['linear', 'poly', 'rbf', 'sigmoid'],
        X_train, y_train, preprocessor
    )
    generate_validation_curve_data(
        'SVR', SVR(kernel='linear'),
        'C', [0.1, 1.0, 10.0, 100.0],
        X_train, y_train, preprocessor
    )
    generate_validation_curve_data(  # Nowy parametr 3
        'SVR', SVR(kernel='linear', C=1.0),
        'epsilon', [0.01, 0.1, 0.5, 1.0],
        X_train, y_train, preprocessor
    )


if __name__ == "__main__":
    main()