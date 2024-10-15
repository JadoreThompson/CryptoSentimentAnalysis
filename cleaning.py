import json
from tqdm import tqdm
import pandas as pd


def serialise(x):
    """
    Converts JSON string to JSON
    :param x:
    :return:
    """
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        x = x.replace("'", '"')
        return json.loads(x)


def get_sentiment_class(x):
    """Returns sentiment value"""
    return x['class']


tqdm.pandas()
df = pd.read_csv('raw_datasets/cryptonews.csv')
pd.set_option('display.max_columns', None)

df['sentiment'] = df['sentiment'].progress_apply(serialise)
df['sentiment_class'] = df['sentiment'].progress_apply(get_sentiment_class)

df = df.drop('sentiment', axis=1)
df = df.rename(columns={'sentiment_class': 'sentiment'})

# Saving
test_size = round(len(df) * 0.2)
train = df[test_size:]
test = df[:test_size]
validate = df[:test_size]

train.to_csv('clean_datasets/train.csv', index=False)
test.to_csv('clean_datasets/test.csv', index=False)
validate.to_csv('clean_datasets/validate.csv', index=False)
