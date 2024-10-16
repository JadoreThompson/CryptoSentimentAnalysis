import torch
from transformers import AutoModelForSequenceClassification

# Local
from train import TOKENIZER


def tokenize_func(x):
    try:
        inputs = TOKENIZER(x, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        return inputs
    except Exception as e:
        print(f"[TOKENIZE FUNCTION][ERROR] >> {str(e)}")
        return None


def test(x):
    """
    Turns logits to human-readable
    and compares to the labels for said item to compute
    the accuracy
    """
    try:
        inputs = tokenize_func(x)
        if inputs:
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).tolist()
                return [['negative', 'neutral', 'positive'][pred] for pred in predicted_class]
    except Exception as e:
        print(f"[TEST][ERROR] >> {str(e)}")


if __name__ == "__main__":
    # Loading Model
    model_path = 'D:/CryptoSentimentAnalysisTrainingOutput/results/checkpoint-4656'
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to('cuda')

    # Testing
    from tqdm import tqdm
    import pandas as pd

    tqdm.pandas()
    testing_data = pd.read_csv('clean_datasets/test.csv')
    batch_size = 1000
    testing_size = len(testing_data)
    correct = 0

    for i in tqdm(range(0, len(testing_data), batch_size)):
        sentiment_batch = testing_data['sentiment'][i: i + batch_size].tolist()
        batch = testing_data['text'][i: i + batch_size].tolist()
        result = test(batch)

        correct += sum(
            1 for i in range(0, len(result))
            if sentiment_batch[i] == result[i]
        )

        testing_data.loc[i: (i + batch_size) - 1, 'tested_outcome'] = result
    testing_data.to_csv('tested_outcome.csv', index=False)
