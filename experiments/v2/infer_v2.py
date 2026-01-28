import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Ścieżka do Twojego wytrenowanego modelu
MODEL_PATH = "models/my_hate_model_v2"

def predict(text, model, tokenizer, device):
    # Tokenizacja
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    
    # Przeniesienie na GPU (jeśli dostępne)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predykcja (wyłączenie liczenia gradientów dla szybkości)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Zamiana logitów na prawdopodobieństwa (Softmax)
    probs = F.softmax(outputs.logits, dim=-1)
    
    # Pobranie klasy z najwyższym prawdopodobieństwem
    pred_label = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_label].item()
    
    return pred_label, confidence

def main():
    print(">>> Ładowanie modelu... (może chwilę potrwać)")
    
    # Ładowanie modelu i tokenizera
    # use_safetensors=True jest ważne, bo tak jest zapisany model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, use_safetensors=True)
    except OSError:
        print(f"Błąd: Nie znaleziono modelu w {MODEL_PATH}. Uruchom najpierw train.py!")
        return

    # Wybór urządzenia (GPU/CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f">>> Model gotowy na: {device}")
    
    print("-" * 50)
    print("Wpisz zdanie do analizy (lub 'q' aby wyjść).")
    print("-" * 50)

    while True:
        text = input("\nPodaj tekst: ")
        if text.lower() in ['q', 'quit', 'exit']:
            break
            
        if not text.strip():
            continue

        label_id, score = predict(text, model, tokenizer, device)
        
        # Interpretacja wyniku (0 = Neutralny, 1 = Hejt - wg danych)
        label_name = "HEJT" if label_id == 1 else "NEUTRALNY"
        color = "\033[91m" if label_id == 1 else "\033[92m" # Czerwony dla hejtu, zielony dla ok
        reset = "\033[0m"
        
        print(f"Wynik: {color}{label_name}{reset} (Pewność: {score:.2%})")

if __name__ == "__main__":
    main()