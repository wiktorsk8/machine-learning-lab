# LAB 1.2: Feature Engineering i Preprocessing Danych

## PREZENTACJA

**Sztuczna Inteligencja - Informatyka, Semestr V**  
**Prowadzący:** Łukasz Grala

---

### Slajd 1: Agenda

**Laboratorium 1.2: Feature Engineering & Data Preprocessing**

**Czego się nauczysz:**
- ✅ Czym jest Feature Engineering i dlaczego jest kluczowy
- ✅ Techniki preprocessingu danych
- ✅ Kodowanie zmiennych kategorycznych
- ✅ Skalowanie i normalizacja
- ✅ Tworzenie nowych features
- ✅ Feature selection
- ✅ Pierwsze modele ML (scikit-learn)

**Narzędzia:** Google Colab, Pandas, NumPy, Scikit-learn

---

### Slajd 2: Co to jest Feature Engineering?

**Feature Engineering** = Proces przekształcania surowych danych w features (cechy), które lepiej reprezentują problem dla modeli ML

**Dlaczego to ważne?**
> "Better data beats more data and better algorithms"
> 
> "Feature engineering is the most important factor in ML competitions" - Kaggle Winners

**Przykład:**
- Surowe dane: data urodzenia "1995-03-15"
- Po feature engineering:
  - Wiek: 28 lat
  - Miesiąc urodzenia: Marzec (sezonowość)
  - Dzień tygodnia: Środa
  - Czy pełnoletni: True
  - Generacja: Millennial

**Impact:** Dobry feature engineering może zwiększyć accuracy o 10-30%!

---

### Slajd 3: Proces Machine Learning Pipeline

```
Dane surowe → Preprocessing → Feature Engineering → Model → Predykcja
     ↓             ↓                  ↓              ↓
Brakujące    Czyszczenie      Tworzenie nowych   Uczenie
Outliers     Transformacje    cech               Walidacja
Duplikaty    Skalowanie       Selekcja           Tuning
```

**Dzisiejsze zajęcia:** Skupiamy się na pierwszych 3 krokach!

**W praktyce:** 80% czasu w projektach ML to preprocessing i feature engineering, tylko 20% to modelowanie

---

### Slajd 4: Typy danych

**1. Numeryczne (Quantitative)**
- **Continuous:** wiek, cena, temperatura (mogą przyjąć dowolną wartość)
- **Discrete:** liczba dzieci, liczba pokoi (tylko liczby całkowite)

**2. Kategoryczne (Qualitative)**
- **Nominal:** kolor, miasto, marka (bez kolejności)
- **Ordinal:** ocena (niska/średnia/wysoka), rozmiar (S/M/L/XL)

**3. Temporalne**
- Data i czas: timestamp, datetime

**4. Tekstowe**
- Opisy, recenzje, dokumenty

**5. Binarne**
- True/False, 0/1, Tak/Nie

**Różne typy → różne techniki preprocessingu!**

---

### Slajd 5: Brakujące dane (Missing Values)

**Przyczyny:**
- Błędy w zbieraniu danych
- Dane niedostępne
- Dane nie mają sensu (np. dochód dla dziecka)

**Strategie:**

**1. Usuwanie (Deletion)**
```python
df.dropna()              # Usuń wiersze z NaN
df.dropna(axis=1)        # Usuń kolumny z NaN
df.dropna(thresh=5)      # Usuń jeśli < 5 wartości
```

**2. Imputacja (Imputation)**
```python
# Podstawowa
df.fillna(0)
df.fillna(df.mean())
df.fillna(df.median())

# Zaawansowana
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
```

**3. Predykcja**
- Użyj ML do przewidzenia brakujących wartości

**Kiedy co?**
- < 5% brakujących → usuń wiersze
- 5-40% → imputacja
- \> 40% → usuń kolumnę lub użyj zaawansowanych metod

---

### Slajd 6: Outliers (Wartości odstające)

**Co to jest outlier?**
Wartość znacząco odbiegająca od pozostałych

**Wykrywanie:**

**1. Metoda IQR (Interquartile Range)**
```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Outlier: < Q1 - 1.5*IQR lub > Q3 + 1.5*IQR
```

**2. Z-score**
```
outlier jeśli |z-score| > 3
```

**3. Wizualne (Box plot)**

**Obsługa:**
- **Usuń** - jeśli to błąd pomiarowy
- **Zachowaj** - jeśli to prawdziwe ekstremalne wartości
- **Transformuj** - log, sqrt
- **Cap** - ograniczenie do percentyla (np. 95th)

---

### Slajd 7: Kodowanie zmiennych kategorycznych

**Problem:** ML modele działają tylko na liczbach!

**Metody:**

**1. Label Encoding (Ordinal)**
```python
# Dla danych z kolejnością
size = ['S', 'M', 'L', 'XL']
→ [0, 1, 2, 3]
```

**2. One-Hot Encoding (Nominal)**
```python
# Dla danych bez kolejności
color = ['red', 'blue', 'green']

         red  blue  green
red   →   1    0     0
blue  →   0    1     0
green →   0    0     1
```

**3. Binary Encoding**
Dla zmiennych z wieloma kategoriami (>10)

**4. Target/Mean Encoding**
Zastąp kategorię średnią target variable

**⚠️ Uwaga:** One-hot może tworzyć dużo kolumn (curse of dimensionality)!

---

### Slajd 8: Skalowanie i Normalizacja

**Dlaczego?**
- Różne features mają różne zakresy (wiek: 0-100, dochód: 0-1000000)
- Modele bazujące na odległości (KNN, SVM, Neural Networks) są wrażliwe na skalę
- Gradient descent szybciej zbiega dla przeskalowanych danych

**Metody:**

**1. Min-Max Scaling (Normalization)**
```
X_scaled = (X - X_min) / (X_max - X_min)
```
→ Zakres [0, 1]

**2. Standardization (Z-score normalization)**
```
X_scaled = (X - μ) / σ
```
→ Średnia=0, std=1

**3. Robust Scaling**
Używa mediany i IQR (odporny na outliers)

**Kiedy co?**
- **Min-Max:** gdy znamy zakres, nie ma outliers
- **Standardization:** gdy dane mają rozkład normalny
- **Robust:** gdy są outliers

---

### Slajd 9: Feature Creation (Tworzenie nowych cech)

**1. Z daty/czasu:**
```python
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5,6])
```

**2. Interakcje (Interactions):**
```python
df['BMI'] = df['weight'] / (df['height'] ** 2)
df['price_per_sqm'] = df['price'] / df['area']
```

**3. Binning (Discretization):**
```python
df['age_group'] = pd.cut(df['age'], 
                         bins=[0, 18, 35, 60, 100],
                         labels=['Child', 'Young', 'Adult', 'Senior'])
```

**4. Polynomial Features:**
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
# X, X^2, X*Y, Y^2
```

**5. Agregacje:**
```python
df['total_spent'] = df.groupby('user_id')['amount'].transform('sum')
```

---

### Slajd 10: Feature Selection

**Dlaczego usuwać features?**
- Zmniejsza overfitting
- Przyspiesza trening
- Upraszcza model (interpretability)
- Curse of dimensionality

**Metody:**

**1. Filter Methods**
- Korelacja z target
- Variance threshold
- Chi-square test
- Mutual information

**2. Wrapper Methods**
- Forward selection
- Backward elimination
- Recursive Feature Elimination (RFE)

**3. Embedded Methods**
- Lasso (L1 regularization)
- Random Forest feature importance
- XGBoost feature importance

**4. Dimensional Reduction**
- PCA (Principal Component Analysis)
- t-SNE, UMAP

---

### Slajd 11: Scikit-learn API

**Scikit-learn** = najpopularniejsza biblioteka ML w Pythonie

**Podstawowy wzorzec:**
```python
from sklearn.xxx import SomeModel

# 1. Stwórz model
model = SomeModel(param1=value1, param2=value2)

# 2. Trenuj (fit)
model.fit(X_train, y_train)

# 3. Predykuj
predictions = model.predict(X_test)

# 4. Ewaluuj
score = model.score(X_test, y_test)
```

**Transformers (preprocessing):**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Uwaga: tylko transform!
```

**⚠️ WAŻNE:** NIGDY nie fit na test set!

---

### Slajd 12: Train-Test Split

**Dlaczego?**
Aby ocenić jak model działa na nowych, niewidzianych danych!

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% na test
    random_state=42,    # Reprodukowalność
    stratify=y          # Zachowaj proporcje klas
)
```

**Typowe podziały:**
- 80/20 (train/test)
- 70/30
- 60/20/20 (train/validation/test)

**Best practice:**
1. Split danych NA POCZĄTKU
2. Wszystkie transformacje fit na train
3. Test zostaje nietknięty do końca

---

### Slajd 13: Pierwszy model - Regresja Liniowa

**Regresja Liniowa** = przewidywanie wartości ciągłej

**Równanie:** y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

**Przykłady:**
- Przewidywanie ceny mieszkania (na podstawie metrażu, lokalizacji)
- Przewidywanie temperatury
- Prognoza sprzedaży

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Trenuj
model = LinearRegression()
model.fit(X_train, y_train)

# Predykuj
y_pred = model.predict(X_test)

# Ewaluuj
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")  # 0-1, bliżej 1 = lepiej
```

---

### Slajd 14: Pierwszy model - Regresja Logistyczna

**Regresja Logistyczna** = klasyfikacja (przewidywanie kategorii)

**Używana gdy:** Target jest binarny (0/1, True/False, Yes/No)

**Przykłady:**
- Czy email to spam?
- Czy klient odejdzie? (churn)
- Diagnoza medyczna (chory/zdrowy)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Trenuj
model = LogisticRegression()
model.fit(X_train, y_train)

# Predykuj
y_pred = model.predict(X_test)

# Prawdopodobieństwa
probabilities = model.predict_proba(X_test)

# Ewaluuj
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
```

---

### Slajd 15: Metryki ewaluacji

**Dla Regresji:**

**MSE (Mean Squared Error):**
- Średnia kwadratów błędów
- Im mniejsze, tym lepiej
- Karze duże błędy

**RMSE (Root Mean Squared Error):**
- Pierwiastek z MSE
- W tych samych jednostkach co target

**MAE (Mean Absolute Error):**
- Średnia wartości bezwzględnych błędów
- Mniej wrażliwy na outliers

**R² Score (Coefficient of Determination):**
- 0 do 1 (może być ujemny dla złych modeli)
- 1 = perfekcyjne dopasowanie
- 0 = model nie lepszy niż średnia

---

### Slajd 16: Metryki ewaluacji - Klasyfikacja

**Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Procent poprawnych predykcji

**Precision (Precyzja):**
```
Precision = TP / (TP + FP)
```
Z przewidzianych pozytywnych, ile jest naprawdę pozytywnych?

**Recall (Czułość, Sensitivity):**
```
Recall = TP / (TP + FN)
```
Z rzeczywistych pozytywnych, ile udało się wykryć?

**F1-Score:**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
Średnia harmoniczna precision i recall

**Confusion Matrix:**
```
                Predicted
                0       1
Actual    0     TN      FP
          1     FN      TP
```

---

### Slajd 17: Kiedy która metoda?

**Preprocessing:**
| Problem | Rozwiązanie |
|---------|-------------|
| Brakujące dane | Imputacja / usunięcie |
| Outliers | IQR, Z-score, transformacje |
| Różne skale | Standaryzacja / Normalizacja |
| Zmienne kategoryczne | One-hot / Label encoding |
| Zbyt wiele cech | Feature selection / PCA |

**Modele:**
| Zadanie | Model |
|---------|-------|
| Przewidywanie liczby | Linear Regression |
| Klasyfikacja binarna | Logistic Regression |
| Klasyfikacja wieloklasowa | Logistic Regression / Trees |
| Nieliniowe zależności | Polynomial Features + Linear |

---

### Slajd 18: Pipeline w Scikit-learn

**Problem:** Dużo kroków preprocessing → łatwo się pomylić

**Rozwiązanie:** Pipeline!

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Definicja pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Wszystko w jednym kroku!
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

**Zalety:**
- Mniej kodu
- Brak wycieków danych (data leakage)
- Łatwe do wdrożenia w produkcji
- Można użyć w GridSearch

---

### Slajd 19: Best Practices

**✅ DO:**
- Zawsze rób train-test split NA POCZĄTKU
- Fit preprocessing TYLKO na train
- Zapisuj random_state dla reprodukowalności
- Wizualizuj dane przed i po preprocessingu
- Dokumentuj wszystkie transformacje
- Zaczynaj od prostych modeli (baseline)

**❌ DON'T:**
- Nie fituj na test set (data leakage!)
- Nie usuwaj outliers bez analizy
- Nie normalizuj target variable dla regression
- Nie używaj test accuracy do optymalizacji
- Nie zapomnij o feature scaling dla neural networks

**Złota zasada:**
> "Garbage in, garbage out" - jakość danych > algorytm

---

### Slajd 20: Zadania praktyczne na dziś

**Zadanie 1:** Czyszczenie i eksploracja danych (20 min)
- Obsługa missing values
- Wykrywanie outliers
- Podstawowe statystyki

**Zadanie 2:** Feature Engineering (25 min)
- Kodowanie kategorii
- Tworzenie nowych features
- Skalowanie

**Zadanie 3:** Pierwszy model (25 min)
- Train-test split
- Regresja liniowa
- Regresja logistyczna
- Ewaluacja

**Zadanie 4:** Pipeline (20 min)
- Kompleksowy pipeline
- Cross-validation
- Optymalizacja

**Mini-projekt:** Analiza i predykcja (30 min)
- End-to-end ML project

---

### Slajd 21: Źródła danych do ćwiczeń

**Klasyczne datasety:**
- **Boston Housing** - regresja (ceny mieszkań)
- **Titanic** - klasyfikacja (survival prediction)
- **Iris** - klasyfikacja (gatunki kwiatów)
- **Wine Quality** - klasyfikacja/regresja

**Kaggle:**
- House Prices: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
- Titanic: https://www.kaggle.com/c/titanic

**UCI Repository:**
- https://archive.ics.uci.edu/ml/

**Scikit-learn built-in:**
```python
from sklearn.datasets import load_boston, load_iris
data = load_boston()
```

---

### Slajd 22: Przydatne zasoby

**Dokumentacja:**
- Scikit-learn: https://scikit-learn.org/
- Pandas: https://pandas.pydata.org/

**Kursy:**
- Kaggle Learn: https://www.kaggle.com/learn
- Google ML Crash Course

**Książki:**
- "Hands-On Machine Learning" - Aurélien Géron
- "Feature Engineering for Machine Learning" - Alice Zheng

**Kaggle Competitions:**
- Najlepszy sposób na naukę feature engineering!

---

### Slajd 23: Na następne zajęcia

**Lab 2.1: Wprowadzenie do Sieci Neuronowych**

**Przygotuj się:**
- Powtórz algebrę liniową (mnożenie macierzy)
- Podstawy pochodnych (gradient)
- Czym jest funkcja aktywacji

**Będziemy implementować:**
- Perceptron od zera
- Backpropagation
- Pierwszą sieć neuronową

**Do zrobienia:**
- Dokończyć dzisiejsze zadania
- Przesłać mini-projekt

**Pytania?** 
📧 maksymilian.marcinowski@cdv.pl

---

### Slajd 24: Podsumowanie

**Dzisiaj nauczyłeś się:**
✅ Czym jest feature engineering  
✅ Jak radzić sobie z brakującymi danymi  
✅ Jak wykrywać i obsługiwać outliers  
✅ Kodowanie zmiennych kategorycznych  
✅ Skalowanie i normalizacja  
✅ Tworzenie nowych features  
✅ Podstawy scikit-learn  
✅ Pierwszy model ML!  

**Kluczowe wnioski:**
- Feature engineering > algorytm
- Preprocessing to 80% pracy
- Zawsze waliduj na test set
- Pipeline = best practice

**Następny krok:** Sieci neuronowe! 🧠

---

## NOTATKI DLA PROWADZĄCEGO

**Timing (120 min):**
- Prezentacja: 45 min
- Zadania praktyczne: 65 min
- Podsumowanie i Q&A: 10 min

**Kluczowe punkty:**
- Podkreśl znaczenie feature engineering (często ważniejsze niż model)
- Pokazuj przykłady na żywych danych
- Demonstruj data leakage i dlaczego jest zły
- Zachęcaj do eksperymentowania

**Live coding:**
- Pokaż jak missing values wpływają na model
- Pokaż różnicę między skalowaniem a brakiem skalowania
- Zademonstruj overfitting gdy nie ma train-test split

**Typowe błędy studentów:**
- Fit preprocessing na całym datasecie (data leakage)
- Zapominanie o random_state
- One-hot encoding bez drop_first (dummy variable trap)
- Skalowanie target variable w regresji

**Interakcja:**
- Pytaj studentów o ich pomysły na features
- Niech eksperymentują z różnymi transformacjami
- Grupowa dyskusja o outliers - usunąć czy nie?
- Porównanie wyników między studentami

**Materiały dodatkowe:**
- Cheat sheet scikit-learn
- Lista najczęściej używanych features
- Przykłady feature engineering z Kaggle

**Na zakończenie:**
- Podkreśl że to praktyczna umiejętność
- W prawdziwych projektach spędzą tu najwięcej czasu
- Feature engineering to sztuka + nauka
- Wymaga eksperymentowania i kreatywności
