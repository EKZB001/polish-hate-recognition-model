import os
import gdown
import zipfile

# --- KONFIGURACJA ---
FILE_ID = '1-WjJE7Ld9QpXFuy2u6QAd8omJMW45nVN' 
OUTPUT_FOLDER = 'models'
ARCHIVE_NAME = 'downloaded_model_v2.zip'

def download_model():
    print(f">>> Rozpoczynam pobieranie modelu z Google Drive...")
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    zip_path = os.path.join(OUTPUT_FOLDER, ARCHIVE_NAME)
    
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    
    try:
        # 1. POBIERANIE NA DYSK
        # Parametr fuzzy=True pomaga znaleźć plik nawet jak Google marudzi
        print(">>> Pobieranie pliku ZIP na dysk...")
        output = gdown.download(url, zip_path, quiet=False, fuzzy=True)
        
        if not output:
            print("BŁĄD: Nie udało się pobrać pliku. Sprawdź ID i uprawnienia.")
            return

        print(">>> Pobieranie zakończone. Rozpakowywanie (to może chwilę potrwać)...")
        
        # 2. ROZPAKOWYWANIE Z DYSKU
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(OUTPUT_FOLDER)
            
        print(f">>> Sukces! Model rozpakowany w folderze: {OUTPUT_FOLDER}")
        
        # 3. SPRZĄTANIE
        print(">>> Usuwanie pliku tymczasowego ZIP...")
        os.remove(zip_path)
        print(">>> Gotowe.")
        
    except Exception as e:
        print(f"\nBŁĄD KRYTYCZNY: {e}")
        print("Upewnij się, że:")
        print("1. Plik na Google Drive jest publiczny (Każdy mający link).")
        print("2. Masz miejsce na dysku.")

if __name__ == "__main__":
    download_model()