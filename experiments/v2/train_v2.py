import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- KONFIGURACJA TRANSFER LEARNINGU ---
BASE_MODEL_PATH = 'models/my_hate_model'
NEW_DATA_DIR = 'data/processed_ptaszynski'
OUTPUT_DIR = 'models/my_hate_model_v2'
LOG_DIR = 'logs_v2'

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def main():
    print(f">>> Ładowanie modelu bazowego z: {BASE_MODEL_PATH}")
    
    # 1. Wczytanie tokenizera i modelu z folderu
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        # use_safetensors=True jest ważne, bo tak jest zapisany model v1
        model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_PATH, num_labels=2, use_safetensors=True)
    except OSError:
        print("BŁĄD: Nie znaleziono modelu v1! Uruchom najpierw src/train.py")
        return

    # 2. Wczytanie nowych danych
    print(f">>> Wczytywanie danych Ptaszczyńskiego z: {NEW_DATA_DIR}")
    dataset = load_dataset('csv', data_files={
        'train': f'{NEW_DATA_DIR}/train.csv',
        'validation': f'{NEW_DATA_DIR}/val.csv'
    })

    # 3. Tokenizacja
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 4. Parametry treningu
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,        # Mniej epok wystarczy
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=LOG_DIR,
        save_total_limit=1,
        report_to="none"
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
    )

    # 6. Start treningu
    print(">>> Rozpoczynam douczanie (Stage 2)...")
    trainer.train()

    # 7. Zapis modelu v2
    print(f">>> Zapisywanie modelu v2 do: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()