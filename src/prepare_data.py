import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- KONFIGURACJA ---
# Upewnij się, że nazwa pliku tutaj zgadza się z tym, co masz w folderze
RAW_DATA_PATH = 'data/raw/BAN-PL.csv' 
OUTPUT_DIR = 'data/processed'
SAMPLE_DIR = 'data/samples'

def clean_text(text):
    """Usuwa zbędne białe znaki (entery, tabulatory, spacje na końcach)."""
    if isinstance(text, str):
        return text.strip()
    return ""

def prepare_data():
    print(">>> Wczytywanie danych...")
    
    # 1. Wczytanie danych (Twój plik to CSV rozdzielany przecinkami)
    try:
        df = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku {RAW_DATA_PATH}. Sprawdź nazwę i lokalizację!")
        return

    print(f"Wczytano {len(df)} wierszy.")
    print("Kolumny w pliku:", df.columns.tolist())

    # 2. Czyszczenie i mapowanie nazw
    # Twoje kolumny to 'Text' i 'Class'. Zmienimy je na standardowe 'text' i 'label'
    df = df.rename(columns={'Text': 'text', 'Class': 'label'})
    
    # Wybieramy tylko te dwie kolumny, 'id' jest nam niepotrzebne do treningu
    df = df[['text', 'label']]

    # Usuwamy puste wiersze
    df = df.dropna(subset=['text', 'label'])
    
    # Czyścimy tekst z dziwnych znaków (entery, tabulatory)
    df['text'] = df['text'].apply(clean_text)
    
    # Usuwamy puste stringi, które mogły zostać po czyszczeniu
    df = df[df['text'] != ""]

    print("Przykładowe dane po czyszczeniu:")
    print(df.head())

    # 3. Podział danych (80% Train, 10% Val, 10% Test)
    # Parametr stratify=df['label'] zadba o to, żeby w każdym zbiorze był taki sam % hejtu
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    )

    print(f"Podział gotowy: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 4. Zapisywanie
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_csv(f'{OUTPUT_DIR}/train.csv', index=False)
    val_df.to_csv(f'{OUTPUT_DIR}/val.csv', index=False)
    test_df.to_csv(f'{OUTPUT_DIR}/test.csv', index=False)
    
    # Próbka dla prowadzącego
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    sample = df.head(5)
    sample.to_csv(f'{SAMPLE_DIR}/sample_io.csv', index=False)
    print(">>> Gotowe! Pliki zapisane w data/processed")

if __name__ == "__main__":
    prepare_data()