# Movie-reviews-sentiment-analysis
This project fine-tunes an LLM over a movie reviews dataset for sentiment analysis.

## FineTuningFinalNotebook.ipynb: 
Finetunes DistilBERT LLM from Huggingface library over an IMDB movie reviews dataset from Huggingface. Then, it saves the finetuned model locally in a zip file.

## StreamApp.py:
A streamlit application that takes the zipped model as path and uses it to analyze the sentiment of movie reviews.

## Important:
Ensure that the zip folder containing the finetuned model is in the same directory as StreamApp.py, else modify the path in the code.
