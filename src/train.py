import os
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- KONFIGURACJA ---
MODEL_NAME = 'allegro/herbert-base-cased'  # Używamy HerBERTa - najlepszy do polskiego
DATA_DIR = 'data/processed'
OUTPUT_DIR = 'models/my_hate_model'        # Zapisywanie wytrenowanego modelu
LOG_DIR = 'logs'

# Parametry treningu
EPOCHS = 3               # Ile razy model zobaczy cały zbiór danych
BATCH_SIZE = 8           # Ile zdań naraz przetwarza (zmiana do 4, jeśli braknie pamięci RAM/GPU)
LEARNING_RATE = 2e-5     # Jak szybko model się uczy (standard dla BERTa)

def compute_metrics(pred):
    """
    Funkcja do obliczania jakości modelu podczas treningu.
    Liczy: Accuracy (dokładność), Precision, Recall i F1.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    print(f">>> Rozpoczynam trening modelu: {MODEL_NAME}")
    
    # 1. Wczytanie danych z plików CSV przygotowanych wcześniej
    # Używamy biblioteki 'datasets' od HuggingFace
    data_files = {
        "train": os.path.join(DATA_DIR, "train.csv"),
        "validation": os.path.join(DATA_DIR, "val.csv")
    }
    
    dataset = load_dataset("csv", data_files=data_files)
    print(f">>> Dane wczytane: {dataset}")

    # 2. Tokenizacja
    # Pobieramy tokenizer dedykowany do HerBERTa
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        # max_length=128 - ucinamy zbyt długie komentarze, padding="max_length" - wyrównujemy krótkie
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print(">>> Rozpoczynam tokenizację danych...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 3. Przygotowanie modelu
    # num_labels=2, bo mamy klasyfikację binarną (0 - ok, 1 - hejt)
    model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2, 
    use_safetensors=True
)

    # Jeśli masz kartę graficzną NVIDIA, skrypt użyje GPU. Jeśli nie - CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f">>> Model załadowany na urządzenie: {device}")

    # 4. Konfiguracja treningu (TrainingArguments)
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}_checkpoints", # Tymczasowe zapisy w trakcie
        eval_strategy="epoch",                  # Sprawdzaj jakość co epokę
        save_strategy="epoch",                  # Zapisuj model co epokę
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,            # Na koniec wczytaj najlepszą wersję modelu
        metric_for_best_model="f1",             # Najlepszy = ten z najwyższym F1 score
        logging_dir=LOG_DIR,
        save_total_limit=2,                     # Trzymaj tylko 2 ostatnie checkpointy (oszczędność miejsca)
    )

    # 5. Inicjalizacja Trenera
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    # 6. Uruchomienie treningu
    print(">>> START TRENINGU...")
    trainer.train()

    # 7. Zapisanie finalnego modelu i tokenizera
    print(f">>> Zapisywanie modelu do {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(">>> GOTOWE! Model zapisany.")

if __name__ == "__main__":
    main()