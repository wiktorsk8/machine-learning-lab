# LAB 1.1: Materiały informacyjne
## Środowisko Google Colab i biblioteki podstawowe

---

## 1. GOOGLE COLABORATORY

### 1.1 Czym jest Google Colab?

Google Colaboratory (w skrócie Colab) to darmowe środowisko do uruchamiania notebooków Jupyter w chmurze Google. Jest to idealne narzędzie dla studentów i naukowców pracujących z Machine Learning i Deep Learning, ponieważ oferuje darmowy dostęp do mocy obliczeniowej GPU/TPU.

**Główne zalety:**
- Brak konieczności instalacji - działa w przeglądarce
- Darmowy dostęp do GPU (NVIDIA Tesla T4 z 15GB VRAM)
- Automatyczne zapisywanie do Google Drive
- Współpraca w czasie rzeczywistym (jak Google Docs)
- Preinstalowane najpopularniejsze biblioteki ML/DL
- Integracja z GitHub

**Ograniczenia wersji darmowej:**
- Maksymalnie 12h sesji ciągłej
- Może zostać przerwana przy dużym obciążeniu serwerów
- Ograniczone zasoby pamięci RAM
- Brak gwarancji dostępności GPU

### 1.2 Rozpoczęcie pracy

1. Wejdź na https://colab.research.google.com
2. Zaloguj się kontem Google
3. Utwórz nowy notebook: File → New notebook
4. Zmień nazwę: kliknij "Untitled.ipynb" u góry

### 1.3 Typy komórek

**Komórka kodu (Code Cell):**
- Zawiera wykonywalny kod Python
- Uruchamianie: Shift + Enter lub przycisk Play
- Wyniki wyświetlane poniżej komórki

**Komórka tekstowa (Text Cell):**
- Zawiera tekst formatowany Markdown
- Służy do dokumentacji i opisów
- Wspiera LaTeX dla wzorów matematycznych

### 1.4 Wybór runtime

**Zmiana na GPU:**
1. Runtime → Change runtime type
2. Hardware accelerator → GPU (T4)
3. Save

**Sprawdzenie dostępności GPU:**
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

### 1.5 Google Drive

**Montowanie dysku:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

Po wykonaniu otrzymasz link do autoryzacji. Po zalogowaniu Twój Google Drive będzie dostępny w `/content/drive/MyDrive/`

**Przykład użycia:**
```python
# Zapisywanie pliku
df.to_csv('/content/drive/MyDrive/dane.csv')

# Wczytywanie pliku
df = pd.read_csv('/content/drive/MyDrive/dane.csv')
```

### 1.6 Magiczne komendy

**Komendy shell (z prefiksem !):**
```python
!ls                    # Lista plików
!pwd                   # Obecny katalog
!mkdir folder_name     # Utworzenie folderu
!pip install pakiet    # Instalacja pakietu
!wget url              # Pobranie pliku z URL
```

**Magiczne komendy IPython (z prefiksem %):**
```python
%time kod              # Czas wykonania linii
%%time                 # Czas wykonania całej komórki
%matplotlib inline     # Wyświetlanie wykresów w notebooku
%load_ext            # Ładowanie rozszerzeń
%who                   # Lista zmiennych
```

---

## 2. NUMPY - NUMERICAL PYTHON

### 2.1 Wprowadzenie

NumPy jest fundamentalną biblioteką do obliczeń numerycznych w Pythonie. Dostarcza wydajne struktury danych (tablice wielowymiarowe) oraz funkcje do operacji na nich.

**Dlaczego NumPy jest szybkie?**
- Implementacja w C/Fortran
- Vectorization - operacje na całych tablicach
- Brak pętli Pythona
- Ciągła alokacja pamięci

**Import:**
```python
import numpy as np
```

### 2.2 Tablice NumPy (ndarray)

**Tworzenie z list:**
```python
# 1D array
arr1d = np.array([1, 2, 3, 4, 5])

# 2D array (macierz)
arr2d = np.array([[1, 2, 3], 
                  [4, 5, 6]])

# 3D array
arr3d = np.array([[[1, 2], [3, 4]], 
                  [[5, 6], [7, 8]]])
```

**Funkcje generujące:**
```python
# Zera
np.zeros((3, 4))          # Macierz 3x4 zer
np.zeros_like(arr)        # Takich samych wymiarów jak arr

# Jedynki
np.ones((2, 3))
np.ones_like(arr)

# Niezinicjalizowane (szybsze)
np.empty((3, 3))

# Zakres wartości
np.arange(0, 10, 2)       # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)      # 5 równomiernie rozmieszczonych wartości

# Macierz jednostkowa
np.eye(4)                 # Macierz jednostkowa 4x4
np.identity(4)            # To samo

# Losowe wartości
np.random.rand(3, 3)      # Rozkład jednostajny [0, 1)
np.random.randn(3, 3)     # Rozkład normalny N(0,1)
np.random.randint(0, 10, (3, 3))  # Losowe int z zakresu
```

### 2.3 Atrybuty tablic

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape      # (2, 3) - wymiary
arr.ndim       # 2 - liczba wymiarów
arr.size       # 6 - całkowita liczba elementów
arr.dtype      # dtype('int64') - typ danych
arr.itemsize   # 8 - rozmiar elementu w bajtach
arr.nbytes     # 48 - całkowity rozmiar w bajtach
```

### 2.4 Typy danych

```python
# Określenie typu przy tworzeniu
arr_int = np.array([1, 2, 3], dtype=np.int32)
arr_float = np.array([1, 2, 3], dtype=np.float64)

# Zmiana typu
arr_converted = arr_int.astype(np.float32)

# Popularne typy:
# int8, int16, int32, int64
# uint8, uint16, uint32, uint64
# float16, float32, float64
# bool
# complex64, complex128
```

### 2.5 Operacje matematyczne

**Element-wise (element po elemencie):**
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Podstawowe operacje
a + b          # [5, 7, 9]
a - b          # [-3, -3, -3]
a * b          # [4, 10, 18]
a / b          # [0.25, 0.4, 0.5]
a ** 2         # [1, 4, 9]

# Funkcje uniwersalne (ufunc)
np.sqrt(a)     # Pierwiastek
np.exp(a)      # Eksponenta
np.log(a)      # Logarytm naturalny
np.sin(a)      # Sinus
np.abs(a)      # Wartość bezwzględna
```

**Operacje agregujące:**
```python
arr = np.array([[1, 2, 3], 
                [4, 5, 6]])

np.sum(arr)              # 21 - suma wszystkich
np.sum(arr, axis=0)      # [5, 7, 9] - suma kolumn
np.sum(arr, axis=1)      # [6, 15] - suma wierszy

np.mean(arr)             # Średnia
np.std(arr)              # Odchylenie standardowe
np.var(arr)              # Wariancja
np.min(arr)              # Minimum
np.max(arr)              # Maximum
np.argmin(arr)           # Indeks minimum
np.argmax(arr)           # Indeks maximum
```

### 2.6 Broadcasting

Broadcasting pozwala NumPy na wykonywanie operacji na tablicach różnych rozmiarów.

**Zasady broadcasting:**
1. Jeśli tablice mają różną liczbę wymiarów, kształt mniejszej tablicy jest uzupełniany jedynkami po lewej stronie
2. Rozmiary wzdłuż każdego wymiaru muszą być równe lub jeden z nich musi wynosić 1
3. Tablice są rozszerzane wzdłuż wymiarów o rozmiarze 1

**Przykłady:**
```python
# Skalar + tablica
arr = np.array([1, 2, 3])
result = arr + 10        # [11, 12, 13]

# Tablica 1D + tablica 2D
a = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])
result = a + b           # [[11, 22, 33],
                         #  [14, 25, 36]]

# Kolumna + wiersz
col = np.array([[1], [2], [3]])  # (3, 1)
row = np.array([10, 20, 30])     # (3,)
result = col + row               # (3, 3)
# [[11, 21, 31],
#  [12, 22, 32],
#  [13, 23, 33]]
```

### 2.7 Indeksowanie i slicing

**Indeksowanie 1D:**
```python
arr = np.array([10, 20, 30, 40, 50])

arr[0]         # 10 - pierwszy element
arr[-1]        # 50 - ostatni element
arr[1:4]       # [20, 30, 40] - slice
arr[::2]       # [10, 30, 50] - co drugi
arr[::-1]      # [50, 40, 30, 20, 10] - odwrócenie
```

**Indeksowanie 2D:**
```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

arr[0, 1]      # 2 - element (0,1)
arr[0]         # [1, 2, 3] - pierwszy wiersz
arr[:, 0]      # [1, 4, 7] - pierwsza kolumna
arr[0:2, 1:]   # [[2, 3], [5, 6]] - submatrix
```

**Boolean indexing:**
```python
arr = np.array([1, 2, 3, 4, 5])
mask = arr > 3
print(arr[mask])    # [4, 5]

# Jednolinijkowo
arr[arr % 2 == 0]   # [2, 4] - parzyste
arr[(arr > 2) & (arr < 5)]  # [3, 4]
```

**Fancy indexing:**
```python
arr = np.array([10, 20, 30, 40, 50])
indices = [0, 2, 4]
arr[indices]        # [10, 30, 50]

# 2D
arr2d = np.array([[1, 2], [3, 4], [5, 6]])
rows = [0, 2]
cols = [1, 0]
arr2d[rows, cols]   # [2, 5]
```

### 2.8 Zmiana kształtu

```python
arr = np.arange(12)  # [0, 1, 2, ..., 11]

# Reshape
arr_2d = arr.reshape(3, 4)
# [[0, 1, 2, 3],
#  [4, 5, 6, 7],
#  [8, 9, 10, 11]]

# Flatten
arr_flat = arr_2d.flatten()  # [0, 1, 2, ..., 11]
arr_ravel = arr_2d.ravel()   # To samo (widok, nie kopia)

# Transpozycja
arr_T = arr_2d.T
# [[0, 4, 8],
#  [1, 5, 9],
#  [2, 6, 10],
#  [3, 7, 11]]

# Dodanie wymiaru
arr_expanded = arr[:, np.newaxis]  # (12,) -> (12, 1)
```

### 2.9 Algebra liniowa

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Mnożenie macierzy
C = np.dot(A, B)   # lub A @ B
# [[19, 22],
#  [43, 50]]

# Transpozycja
A_T = A.T

# Wyznacznik
det_A = np.linalg.det(A)  # -2.0

# Odwrotność macierzy
A_inv = np.linalg.inv(A)

# Wartości i wektory własne
eigenvalues, eigenvectors = np.linalg.eig(A)

# Rozkład SVD
U, S, VT = np.linalg.svd(A)

# Rozwiązywanie układu równań Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)
```

### 2.10 Zastosowania w ML/AI

**Normalizacja danych:**
```python
# Min-Max scaling do zakresu [0, 1]
X = np.random.rand(100, 5)
X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Standaryzacja (z-score)
X_std = (X - X.mean(axis=0)) / X.std(axis=0)
```

**Podział na batche:**
```python
def create_batches(X, batch_size):
    n_batches = len(X) // batch_size
    for i in range(n_batches):
        yield X[i*batch_size:(i+1)*batch_size]
```

**Inicjalizacja wag sieci neuronowej:**
```python
# Xavier/Glorot initialization
n_in, n_out = 784, 128
weights = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
```

---

## 3. PANDAS

### 3.1 Wprowadzenie

Pandas to biblioteka do manipulacji i analizy danych strukturalnych. Nazwa pochodzi od "Panel Data" - ekonometrycznego terminu oznaczającego wielowymiarowe dane strukturalne.

**Import:**
```python
import pandas as pd
```

### 3.2 Struktury danych

**Series - jednowymiarowa:**
```python
# Z listy
s = pd.Series([10, 20, 30, 40, 50])

# Z indeksami
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

# Ze słownika
s = pd.Series({'a': 10, 'b': 20, 'c': 30})

# Dostęp
s['a']        # 10
s[0]          # 10 (dostęp pozycyjny)
s['a':'c']    # Slice po indeksach
```

**DataFrame - dwuwymiarowa:**
```python
# Ze słownika
data = {
    'Name': ['Anna', 'Bartek', 'Celina'],
    'Age': [23, 25, 22],
    'City': ['Poznań', 'Warszawa', 'Kraków']
}
df = pd.DataFrame(data)

# Z NumPy array
arr = np.random.rand(4, 3)
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])

# Z pliku CSV
df = pd.read_csv('data.csv')

# Z Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
```

### 3.3 Podstawowe operacje

**Podgląd danych:**
```python
df.head(10)        # Pierwsze 10 wierszy
df.tail(5)         # Ostatnie 5 wierszy
df.sample(3)       # 3 losowe wiersze
df.info()          # Informacje o typach i null values
df.describe()      # Statystyki opisowe dla kolumn numerycznych
df.shape           # (wiersze, kolumny)
df.columns         # Nazwy kolumn
df.index           # Indeksy wierszy
df.dtypes          # Typy danych kolumn
```

**Selekcja:**
```python
# Pojedyncza kolumna (zwraca Series)
df['Name']
df.Name  # Alternatywna składnia

# Wiele kolumn (zwraca DataFrame)
df[['Name', 'Age']]

# Wiersze po indeksie
df.loc[0]          # Po etykiecie indeksu
df.iloc[0]         # Po pozycji (integer location)

# Zakres wierszy
df.loc[0:5]        # Wiersze od 0 do 5 (włącznie)
df.iloc[0:5]       # Wiersze od 0 do 4 (wyłącznie 5)

# Selekcja kombinowana
df.loc[0:5, ['Name', 'Age']]
df.iloc[0:5, [0, 1]]

# Boolean indexing
df[df['Age'] > 23]
df[(df['Age'] > 22) & (df['City'] == 'Poznań')]
```

### 3.4 Modyfikacja danych

**Dodawanie kolumny:**
```python
df['Grade'] = ['A', 'B', 'A']
df['Year'] = 2024
df['Score'] = np.random.randint(60, 100, len(df))

# Na podstawie istniejącej kolumny
df['Age_Group'] = df['Age'].apply(lambda x: 'Young' if x < 25 else 'Adult')
```

**Usuwanie kolumny:**
```python
df.drop('Grade', axis=1, inplace=True)
df.drop(columns=['Grade', 'Year'], inplace=True)
```

**Zmiana nazw:**
```python
df.rename(columns={'Name': 'Student_Name'}, inplace=True)
df.columns = ['col1', 'col2', 'col3']
```

### 3.5 Brakujące dane

**Wykrywanie:**
```python
df.isnull()           # Maska bool (True dla NaN)
df.isna()             # Alias dla isnull()
df.isnull().sum()     # Liczba NaN w każdej kolumnie
df.isnull().any()     # Które kolumny mają NaN
```

**Usuwanie:**
```python
df.dropna()           # Usuń wiersze z jakimkolwiek NaN
df.dropna(axis=1)     # Usuń kolumny z jakimkolwiek NaN
df.dropna(subset=['Age'])  # Usuń tylko jeśli NaN w 'Age'
df.dropna(thresh=2)   # Usuń jeśli < 2 wartości non-null
```

**Wypełnianie:**
```python
df.fillna(0)                      # Wypełnij zerami
df.fillna(df.mean())              # Wypełnij średnią
df.fillna(method='ffill')         # Forward fill
df.fillna(method='bfill')         # Backward fill
df['Age'].fillna(df['Age'].median(), inplace=True)
```

### 3.6 Grupowanie i agregacja

```python
# Grupowanie po jednej kolumnie
grouped = df.groupby('City')

# Agregacja
grouped['Age'].mean()
grouped['Age'].agg(['mean', 'std', 'min', 'max'])

# Wiele kolumn
grouped[['Age', 'Score']].mean()

# Własna funkcja agregująca
grouped['Age'].agg(lambda x: x.max() - x.min())

# Grupowanie po wielu kolumnach
df.groupby(['City', 'Grade'])['Age'].mean()
```

### 3.7 Sortowanie

```python
# Po wartościach
df.sort_values('Age')
df.sort_values('Age', ascending=False)
df.sort_values(['City', 'Age'])

# Po indeksie
df.sort_index()
```

### 3.8 Łączenie DataFrames

**Concatenation:**
```python
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

# Pionowo (wiersze)
pd.concat([df1, df2], axis=0)

# Poziomo (kolumny)
pd.concat([df1, df2], axis=1)
```

**Merge (jak SQL JOIN):**
```python
left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
right = pd.DataFrame({'key': ['A', 'B', 'D'], 'value': [4, 5, 6]})

# Inner join (domyślny)
pd.merge(left, right, on='key')

# Left join
pd.merge(left, right, on='key', how='left')

# Right join
pd.merge(left, right, on='key', how='right')

# Outer join
pd.merge(left, right, on='key', how='outer')
```

### 3.9 Funkcje apply, map, applymap

```python
# apply() - na kolumnie lub wierszu
df['Age_Squared'] = df['Age'].apply(lambda x: x**2)

# apply() na całym DataFrame
df.apply(np.sum, axis=0)  # Suma kolumn

# map() - tylko dla Series
df['Age_Category'] = df['Age'].map({23: 'Young', 25: 'Adult', 22: 'Young'})

# applymap() - element-wise dla DataFrame (deprecated w nowszych wersjach)
df_numeric.map(lambda x: x * 2)  # Nowa składnia
```

### 3.10 Zastosowania w ML

**Train-test split:**
```python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**One-hot encoding:**
```python
df_encoded = pd.get_dummies(df, columns=['City', 'Grade'])
```

**Normalizacja:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age', 'Score']] = scaler.fit_transform(df[['Age', 'Score']])
```

---

## 4. MATPLOTLIB & SEABORN

### 4.1 Matplotlib - Podstawy

```python
import matplotlib.pyplot as plt

# Prosty wykres liniowy
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.figure(figsize=(10, 6))
plt.plot(x, y, color='blue', linewidth=2, linestyle='--', marker='o')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Simple Line Plot')
plt.grid(True)
plt.legend(['Line 1'])
plt.show()
```

### 4.2 Typy wykresów

**Scatter plot:**
```python
plt.scatter(x, y, c='red', s=100, alpha=0.5, marker='x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.show()
```

**Bar plot:**
```python
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]

plt.bar(categories, values, color=['red', 'blue', 'green', 'orange'])
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Bar Plot')
plt.show()
```

**Histogram:**
```python
data = np.random.randn(1000)

plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()
```

**Box plot:**
```python
data = [np.random.randn(100) for _ in range(4)]

plt.boxplot(data, labels=['Group A', 'Group B', 'Group C', 'Group D'])
plt.ylabel('Value')
plt.title('Box Plot')
plt.show()
```

### 4.3 Subplots

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1
axes[0, 0].plot(x, y)
axes[0, 0].set_title('Line Plot')

# Plot 2
axes[0, 1].scatter(x, y)
axes[0, 1].set_title('Scatter Plot')

# Plot 3
axes[1, 0].bar(x, y)
axes[1, 0].set_title('Bar Plot')

# Plot 4
axes[1, 1].hist(y, bins=5)
axes[1, 1].set_title('Histogram')

plt.tight_layout()
plt.show()
```

### 4.4 Seaborn - Wyższy poziom

```python
import seaborn as sns

# Ustawienie stylu
sns.set_style('whitegrid')
sns.set_palette('pastel')

# Heatmap korelacji
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Pairplot
sns.pairplot(df, hue='Grade')
plt.show()

# Distribution plot
sns.histplot(df['Age'], kde=True, bins=10)
plt.title('Age Distribution')
plt.show()

# Box plot
sns.boxplot(x='City', y='Score', data=df)
plt.title('Score by City')
plt.show()

# Violin plot
sns.violinplot(x='City', y='Score', data=df)
plt.title('Score Distribution by City')
plt.show()

# Count plot
sns.countplot(x='Grade', data=df)
plt.title('Grade Counts')
plt.show()
```

---

## 5. BEST PRACTICES

### 5.1 Organizacja kodu

```python
# 1. Importy na początku
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Ustawienia globalne
np.random.seed(42)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# 3. Ładowanie danych
df = pd.read_csv('data.csv')

# 4. Eksploracja
print(df.head())
print(df.info())

# 5. Preprocessing
# ...

# 6. Analiza/Modelowanie
# ...

# 7. Wyniki
# ...
```

### 5.2 Reprodukowalność

```python
# Zawsze ustawiaj seed dla losowości
import random
random.seed(42)
np.random.seed(42)

# W TensorFlow/Keras
import tensorflow as tf
tf.random.set_seed(42)

# W PyTorch
import torch
torch.manual_seed(42)
```

### 5.3 Dokumentacja

```python
def process_data(df, threshold=0.5):
    """
    Przetwarza dane zgodnie z określonym progiem.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame wejściowy do przetworzenia
    threshold : float, default=0.5
        Próg dla filtrowania wartości
    
    Returns:
    --------
    pandas.DataFrame
        Przetworzony DataFrame
    
    Examples:
    ---------
    >>> df_processed = process_data(df, threshold=0.7)
    """
    # Implementacja
    pass
```

### 5.4 Typowe pułapki i błędy

**SettingWithCopyWarning:**
```python
# ZŁE
df[df['Age'] > 25]['Score'] = 100

# DOBRE
df.loc[df['Age'] > 25, 'Score'] = 100
```

**Mutowanie oryginału:**
```python
# ZŁE
df_copy = df  # To tylko referencja!
df_copy['New'] = 1  # Modyfikuje też df!

# DOBRE
df_copy = df.copy()  # Prawdziwa kopia
```

**Inplace operations:**
```python
# Zwraca None, modyfikuje oryginalny DataFrame
df.dropna(inplace=True)

# Zwraca nowy DataFrame, oryginalny niezmieniony
df_clean = df.dropna()
```

---

## 6. PRZYDATNE ZASOBY

**Dokumentacja:**
- NumPy: https://numpy.org/doc/stable/
- Pandas: https://pandas.pydata.org/docs/
- Matplotlib: https://matplotlib.org/stable/contents.html
- Seaborn: https://seaborn.pydata.org/

**Cheat Sheets:**
- https://www.datacamp.com/cheat-sheet/numpy-cheat-sheet-data-analysis-in-python
- https://www.datacamp.com/cheat-sheet/pandas-cheat-sheet-for-data-science-in-python

**Kursy online:**
- Kaggle Learn: https://www.kaggle.com/learn
- DataCamp
- Fast.ai

**Zbiory danych:**
- Kaggle: https://www.kaggle.com/datasets
- UCI ML Repository: https://archive.ics.uci.edu/ml/index.php
- Google Dataset Search: https://datasetsearch.research.google.com/

---

**Powodzenia na zajęciach!** 🚀
