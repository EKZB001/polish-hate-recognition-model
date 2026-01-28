import os
import requests
import zipfile
import io
import shutil

URL = "https://github.com/ZILiAT-NASK/BAN-PL/blob/main/data/BAN-PL_1.zip?raw=true"
ZIP_PASSWORD = b"BAN-PL_1"
RAW_DATA_DIR = 'data/raw'
TARGET_FILENAME = 'BAN-PL.csv'

def download_and_extract():
    print(f">>> Rozpoczynam pobieranie z: {URL}")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    try:
        response = requests.get(URL)
        response.raise_for_status()
        print(">>> Pobrano plik ZIP. Rozpakowywanie...")

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            temp_extract_path = os.path.join(RAW_DATA_DIR, "temp_extracted")
            z.extractall(path=temp_extract_path, pwd=ZIP_PASSWORD)
            
            found_csv = None
            for root, dirs, files in os.walk(temp_extract_path):
                for file in files:
                    if file.endswith(".csv"):
                        found_csv = os.path.join(root, file)
                        break
            
            if found_csv:
                target_path = os.path.join(RAW_DATA_DIR, TARGET_FILENAME)
                shutil.move(found_csv, target_path)
                print(f">>> Sukces! Plik gotowy w: {target_path}")
            else:
                print("BŁĄD: Nie znaleziono pliku CSV.")
            
            shutil.rmtree(temp_extract_path)

    except Exception as e:
        print(f"BŁĄD: {e}")

if __name__ == "__main__":
    download_and_extract()