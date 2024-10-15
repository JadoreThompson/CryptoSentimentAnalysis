import evaluate
import numpy as np
from train import trainer, tokenized_dataset


def test():
    """
    Turns logits to human-readable
    and compares to the labels for said item to compute
    the accuracy
    """
    predictions = trainer.predict(tokenized_dataset['validation'])
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load('accuracy')
    return metric.compute(predictions=preds, references=predictions.label_ids)


if __name__ == "__main__":
    test()
