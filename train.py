import json

import numpy as np
import torch
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer,
)
from sklearn.preprocessing import LabelEncoder


# Constants
BASE_MODEL = "FacebookAI/xlm-roberta-base"
DATASET = load_dataset('clean_datasets/', data_files={
    'train': 'train.csv',
    'validation': 'validate.csv',
    'test': 'test.csv'
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading Tokenizer and Model
TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL)
MODEL = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)
MODEL.to(device)

# Label Encoding
label_encoder = LabelEncoder()
all_labels = DATASET['train']['sentiment'] + DATASET['validation']['sentiment'] + DATASET['test']['sentiment']
label_encoder.fit(all_labels)
num_labels = len(label_encoder.classes_)


def compute_metrics(eval_preds):
    """
    Turns logits to human-readable
    and compares to the labels for said item to compute
    the accuracy
    """
    print("Eval Preds: ", eval_preds)
    metric = evaluate.load('accuracy')
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)


def tokenize_dataset_function(x):
    tokenized = TOKENIZER(x['title'], x['text'], truncation=True)
    tokenized['labels'] = label_encoder.transform(x['sentiment'])
    return tokenized


tokenized_dataset = DATASET.map(tokenize_dataset_function, batched=True)
tokenized_dataset.set_format('torch')
data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)
training_args = TrainingArguments(
    output_dir="D:/CryptoSentimentAnalysisTrainingOutput/results/",
    eval_strategy='epoch',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='D:/CryptoSentimentAnalysisTrainingOutput/logs/',
    logging_steps=100,
)

trainer = Trainer(
    MODEL,
    training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=data_collator,
    tokenizer=TOKENIZER,
    compute_metrics=compute_metrics,
)


if __name__ == "__main__":
    trainer.train()
    # compute_metrics()
