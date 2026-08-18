# Analiza porównawcza autorskiej sieci neuronowej i klasycznych metod ML


## Cel projektu
Implementacja od podstaw sztucznej sieci neuronowej oraz porównanie jej skuteczności z klasycznymi modelami uczenia maszynowego (scikit-learn) w zadaniach regresji oraz klasyfikacji.

## Dane
* **Regresja (Ames Housing):** Predykcja cen nieruchomości (244 cechy po preprocessingu).
* **Klasyfikacja (Titanic):** Predykcja przeżywalności pasażerów.

## Modele i narzędzia
* **Technologie:** Python, NumPy, Pandas, Scikit-learn, Matplotlib
* **Autorska sieć neuronowa (NumPy):** Wielowarstwowy perceptron (MLP), aktywacja ReLU / Sigmoid, spadek gradientu (SGD) z propagacją wsteczną.
* **Klasyczne modele ML (scikit-learn):**
  * k-Najbliższych Sąsiadów (k-NN)
  * Drzewo Decyzyjne (Decision Tree)
  * Las Losowy (Random Forest)
  * Maszyna Wektorów Nośnych (SVM / SVR)

## Metodyka
1. **Baseline:** Wyznaczenie punktu odniesienia dla modeli na domyślnych parametrach.
2. **Analiza wrażliwości:** Badanie wpływu pojedynczych hiperparametrów na zjawisko przeuczenia (overfitting).
3. **Optymalizacja:** Wielowymiarowe przeszukiwanie siatki parametrów z 5-krotną walidacją krzyżową.
