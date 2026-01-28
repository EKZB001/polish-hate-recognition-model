import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# --- KONFIGURACJA ---
HF_DATASET_ID = "ptaszynski/PolishCyberbullyingDataset"
OUTPUT_DIR = 'data/processed_ptaszynski'

def prepare_hf_data():
    print(f">>> Pobieranie zbioru {HF_DATASET_ID} z Hugging Face Hub...")
    
    # 1. Pobranie danych
    try:
        dataset = load_dataset(HF_DATASET_ID)
    except Exception as e:
        print(f"BŁĄD pobierania: {e}")
        return

    print(f">>> Pobrano. Dostępne podziały: {dataset.keys()}")

    # 2. Konwersja do Pandas i wybór kolumn
    
    # Funkcja pomocnicza do czyszczenia
    def process_split(ds_split):
        df = ds_split.to_pandas()

        print(f"Kolumny w zbiorze: {df.columns.tolist()}")
        
        # Mapowanie nazw kolumn
        text_col = next((c for c in df.columns if c.lower() in ['text', 'content', 'tweet']), None)
        label_col = next((c for c in df.columns if 'general' in c.lower() or 'label' in c.lower()), None)
        
        if not text_col or not label_col:
            print(f"BŁĄD: Nie rozpoznano kolumn tekst/etykieta. Dostępne: {df.columns}")
            return None

        # Zmiana nazw na nasze standardowe
        df = df.rename(columns={text_col: 'text', label_col: 'label'})
        
        # Wybieramy TYLKO te dwie kolumny
        df = df[['text', 'label']]
        
        # Usuwanie błędów (puste wiersze)
        df = df.dropna()
        
        # Upewnienie się, że etykiety to liczby całkowite
        df['label'] = df['label'].astype(int)
        
        return df

    # Przetwarzanie Train i Test
    print(">>> Przetwarzanie splitu TRAIN...")
    df_train_full = process_split(dataset['train'])
    
    print(">>> Przetwarzanie splitu TEST...")
    if 'test' in dataset:
        df_test = process_split(dataset['test'])
    else:
        print("UWAGA: Brak splitu 'test'. Dzielę 'train' na części.")
        df_train_full, df_test = train_test_split(df_train_full, test_size=0.15, random_state=42)

    # 3. Podział Train na Train/Validation (90%/10%)
    train_df, val_df = train_test_split(
        df_train_full, test_size=0.1, random_state=42, stratify=df_train_full['label']
    )

    print(f"--- Statystyki ---")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(df_test)}")

    # 4. Zapis do CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_csv(f'{OUTPUT_DIR}/train.csv', index=False)
    val_df.to_csv(f'{OUTPUT_DIR}/val.csv', index=False)
    df_test.to_csv(f'{OUTPUT_DIR}/test.csv', index=False)
    
    print(f">>> Gotowe! Czyste dane zapisane w: {OUTPUT_DIR}")
    print(">>> Teraz możesz uruchomić: python src/train_v2.py")

if __name__ == "__main__":
    prepare_hf_data()