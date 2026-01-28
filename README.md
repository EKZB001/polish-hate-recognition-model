# Wykrywanie Mowy Nienawiści (Cyberbullying Detection)

Projekt zaliczeniowy z przedmiotu Uczenie Maszynowe.
Celem projektu jest stworzenie modelu opartego o architekturę Transformer (HerBERT), służącego do automatycznej klasyfikacji komentarzy na "Neutralne" oraz "Hejt".

## 📂 Struktura Projektu
* `data/` - folder na dane (tu znajduje się skrypt i próbki, pełny plik CSV nie jest dołączony do repozytorium)
* `models/` - tu zostanie zapisany wytrenowany model (folder ignorowany przez git ze względu na rozmiar)
* `src/` - kody źródłowe:
  * `download_model.py` - pobieranie i rozpakowywanie modelu
  * `download_data.py` - pobieranie i rozpakowywanie danych
  * `prepare_data.py` - czyszczenie i podział danych (Train/Val/Test)
  * `train.py` - fine-tuning modelu HerBERT
  * `infer.py` - skrypt do interaktywnego testowania modelu
  * `evaluate.py` - generowanie raportu wyników (Macierz Pomyłek)
* `requirements.txt` - lista zależności niezbędnych do uruchomienia
* `reports/` - miejsce zapisu wykresów i raportów

## 🚀 Instalacja i Uruchomienie

### 1. Przygotowanie środowiska
Zalecane jest użycie Python 3.10 lub 3.11.

pip install -r requirements.txt

**Uwaga dot. GPU:** Aby znacznie przyspieszyć trening, zalecane jest posiadanie wersji PyTorch z obsługą CUDA. Domyślna instalacja z `requirements.txt` może zainstalować wersję CPU. Aby wymusić wersję GPU:
`pip install torch --index-url https://download.pytorch.org/whl/cu124`

### 2. Przygotowanie danych
Pobierz i rozpakuj dane:

python src/download_data.py

Upewnij się, że plik `BAN-PL.csv` znajduje się w folderze `data/raw/`. Następnie uruchom:

python src/prepare_data.py

### 3. Trening modelu (Fine-tuning)
Skrypt pobierze model `allegro/herbert-base-cased` i douczy go na przygotowanych danych.

python src/train.py

LUB

Tym skryptem można już pobrać przetrenowany model z Google Drive

python src/download_model.py

### 4. Ewaluacja i Testy
Aby sprawdzić jakość modelu na zbiorze testowym i wygenerować macierz pomyłek:

python src/evaluate.py

Aby uruchomić tryb interaktywny (wpisywanie własnych zdań):

python src/infer.py

## 📊 Wyniki
Model osiągnął następujące wyniki na zbiorze testowym:
* **Accuracy:** 92.71%
* **F1-Score:** 0.93
Szczegółowa analiza znajduje się w pliku `sprawozdanie.pdf`.