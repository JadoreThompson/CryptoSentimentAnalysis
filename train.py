import json
import torch
from datasets import load_dataset, DatasetDict
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

TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL)
MODEL = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)
label_encoder = LabelEncoder()
all_labels = DATASET['train']['sentiment'] + DATASET['validation']['sentiment'] + DATASET['test']['sentiment']
label_encoder.fit(all_labels)
num_labels = len(label_encoder.classes_)


def tokenize_dataset_function(x):
    tokenized = TOKENIZER(x['title'], x['text'], truncation=True)
    tokenized['labels'] = label_encoder.transform(x['sentiment'])
    return tokenized


tokenized_dataset = DATASET.map(tokenize_dataset_function, batched=True)
tokenized_dataset.set_format('torch')
data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)
training_args = TrainingArguments(
    output_dir="results/",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='logs/',
    logging_steps=1,
)
trainer = Trainer(
    MODEL,
    training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=data_collator,
    tokenizer=TOKENIZER
)


if __name__ == "__main__":
   trainer.train()
