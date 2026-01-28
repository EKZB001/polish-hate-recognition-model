import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- KONFIGURACJA V2 ---
MODEL_PATH = "models/my_hate_model_v2"
TEST_DATA = "data/processed_ptaszynski/testmix.csv"
REPORT_IMG = "reports/confusion_matrix_v2.png"

def evaluate():
    print(f">>> Ewaluacja modelu V2 ({MODEL_PATH}) na danych Ptaszczyńskiego...")
    
    if not os.path.exists(MODEL_PATH):
        print("Błąd: Nie znaleziono modelu v2. Uruchom najpierw src/train_v2.py")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, use_safetensors=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    df = pd.read_csv(TEST_DATA)
    texts = df['text'].astype(str).tolist()
    true_labels = df['label'].tolist()
    
    predictions = []
    
    print(">>> Generowanie predykcji...")
    model.eval()
    batch_size = 16
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())

    # Raport
    print("\n" + "="*40)
    print(" WYNIKI MODELU V2 (DOUCZONEGO)")
    print("="*40)
    print(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")
    print(classification_report(true_labels, predictions, target_names=["Neutralny", "Hejt"]))
    
    # Macierz
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=["Neutralny", "Hejt"], yticklabels=["Neutralny", "Hejt"])
    plt.title('Macierz Pomyłek - Model V2 (Ptaszczyński)')
    os.makedirs('reports', exist_ok=True)
    plt.savefig(REPORT_IMG)
    print(f">>> Wykres zapisano w: {REPORT_IMG}")

if __name__ == "__main__":
    evaluate()