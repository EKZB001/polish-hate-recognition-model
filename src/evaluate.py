import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_PATH = "models/my_hate_model"
TEST_DATA = "data/processed/test.csv"

def evaluate():
    print(">>> Ładowanie modelu i danych testowych...")
    
    # 1. Ładowanie
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, use_safetensors=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    df = pd.read_csv(TEST_DATA)
    # Upewniamy się, że dane są tekstowe
    texts = df['text'].astype(str).tolist()
    true_labels = df['label'].tolist()
    
    print(f">>> Rozpoczynam ocenę na {len(texts)} przykładach...")
    
    predictions = []
    
    # 2. Predykcja w pętli (batchami, żeby nie zapchać pamięci przy dużej ilości)
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
            
            if i % 100 == 0:
                print(f"Przetworzono {i}/{len(texts)}...", end='\r')

    # 3. Raportowanie
    print("\n\n" + "="*30)
    print(" RAPORT KOŃCOWY (ZBIÓR TESTOWY)")
    print("="*30)
    
    acc = accuracy_score(true_labels, predictions)
    print(f"Dokładność (Accuracy): {acc:.4f}")
    
    print("\nSzczegółowy raport:")
    print(classification_report(true_labels, predictions, target_names=["Neutralny", "Hejt"]))
    
    # 4. Macierz Pomyłek (Tekstowa)
    cm = confusion_matrix(true_labels, predictions)
    print("\nMacierz Pomyłek (Confusion Matrix):")
    print(f"Prawdziwy Neutralny: {cm[0][0]} | Fałszywy Hejt (Błąd): {cm[0][1]}")
    print(f"Fałszywy Neutralny (Błąd): {cm[1][0]} | Prawdziwy Hejt: {cm[1][1]}")

    # Opcjonalnie: Zapis macierzy do pliku (do wklejenia w raport)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Neutralny", "Hejt"], yticklabels=["Neutralny", "Hejt"])
    plt.ylabel('Prawdziwa etykieta')
    plt.xlabel('Przewidziana etykieta')
    plt.title('Macierz Pomyłek')
    plt.savefig('reports/confusion_matrix.png')
    print("\n>>> Wykres macierzy zapisano w reports/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()