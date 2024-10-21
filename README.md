## **Description**

This project uses the transformers module from [Hugging Face](https://huggingface.co/) and the Facebook RoBERTa model to predict sentiment in cryptocurrency news.

### **Why BERT?**

Unlike GPT and other well-known LLMs, BERT (Bidirectional Encoder Representations from Transformers) is bidirectional, meaning it can read text both forward and backward. This allows it to capture relationships between words across a passage, even if they appear far apart, making it especially effective for understanding context.

BERT was trained on Masked Language Modeling (MLM), which forces the model to predict missing words in text, enhancing its ability to generate strong contextual embeddings. Unlike static word embeddings, BERT dynamically generates embeddings based on context, so the word "free" might have different representations depending on usage.

This specific BERT model has only 300 million parameters, making it lightweight enough for my system while maintaining efficiency after removing the Next Sentence Prediction (NSP) task.

## **Prerequisites**

- **Training:** I recommend having a GPU, as training can be slow without one.
- **News API Key:** The model is served through a FastAPI endpoint connected to my [Trading Platform](https://github.com/JadoreThompson/Trading-Portfolio-Platform) and uses the News API to fetch recent news data.
    - Get an API key from [NewsAPI](https://newsapi.org/docs)

## **Installation**

```bash
# Clone the repo
git clone <https://github.com/JadoreThompson/CryptoSentimentAnalysis.git>

# Install dependencies
pip install -r requirements.txt

```

## **Contact**

If you're interested in collaborating or have any opportunities, feel free to contact me at [jadorethompson6@gmail.com](mailto:jadorethompson6@gmail.com) or connect with me on [LinkedIn](https://www.linkedin.com/in/jadore-t-49379a295/).

## Evaluation

In the future I’m looking to build a list and phrases that indicate sentiment in an attempt to aid the model then combined with the CIMAWA model mentioned in the FBert paper. All in all this was a good experience for getting hands on with ML following my [Property Valuation Project](https://github.com/JadoreThompson/property-valuation-v2) where I used Gemini to feed a chain of prompts gathering data from a Postgres DB and leveraging SERP to harbour more data.
