import streamlit as st
import os
from transformers.models.distilbert.modeling_distilbert import DistilBertForSequenceClassification
from transformers.models.distilbert.tokenization_distilbert import DistilBertTokenizer
import torch
from safetensors.torch import load_file

def generate_sentiment(model_path, text):

    try:
        text=text

        FineTunedModel = DistilBertForSequenceClassification.from_pretrained(model_path)

        FineTunedTokenizer = DistilBertTokenizer.from_pretrained(model_path)

        encoded_text = FineTunedTokenizer(text, return_tensors="pt") 
        with torch.no_grad():
            outputs = FineTunedModel(**encoded_text)
            predictions = torch.argmax(outputs.logits, dim=-1)

        predicted_sentiment = predictions.item()

        sentiment_map = {0: "Negative", 1: "Positive"}
        predicted_sentiment = sentiment_map[predicted_sentiment]
        return predicted_sentiment
        return sentiment
    except Exception as e:
        st.error(f"Error: An error occurred during sentiment analysis. {e}")
        return None

# Set the app title
st.title('CineSense')
st.write('Sentiment Analysis for Cinema!')

user_input = st.text_input('Would you like to share your thoughts?', 'Amazing movie!')

zip_path = "./finetuned_final_bert_model"

if st.button("Analyze Sentiment"):
        sentiment = generate_sentiment(zip_path, user_input)
        if sentiment=='Negative':
            st.markdown("Sentiment: :red[Negative!]")
        else:
            st.markdown("Sentiment: :green[Positive!]")


    
    
