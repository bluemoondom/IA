# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 21:05:49 2025

@author: dominika
"""

import pandas as pd
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import torch
from transformers import MarianMTModel, MarianTokenizer, pipeline
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = pd.read_csv("chat_data.csv")
nlp = spacy.load("en_core_web_sm")

def extract_product_mentions(text):
    doc = nlp(text)
    #doc = nltk.word_tokenize(text, language='czech')
    products = [ent.text for ent in doc.ents if ent.label_ == "PRODUCT"]
    return products

data['product_mentions'] = data['message'].apply(extract_product_mentions)
print("\nData with Product Mentions:")
print(data[['message', 'product_mentions']])

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

tokenizer_cs_en = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-cs-en')
model_cs_en = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-cs-en').to(device)
def translate_cs_to_en(mytext):
    translated_to_en = model_cs_en.generate(**tokenizer_cs_en(mytext
                                                              , return_tensors='pt'
                                                              , padding=True
                                                              , truncation=True
                                                              , max_length=512).to(device))
    return[tokenizer_cs_en.decode(t, skip_special_tokens=True) for t in translated_to_en]

def analyze_sentiment(text):
    print(translate_cs_to_en(text)[0])
    score = sia.polarity_scores(translate_cs_to_en(text)[0])
    print(score)
    return score['compound']

data['sentiment'] = data['message'].apply(analyze_sentiment)
print("\nData with Sentiment Scores:")
print(data[['message', 'sentiment']])

positive_threshold = 0.5
negative_threshold = -0.2
positive_messages = data[data['sentiment'] > positive_threshold]['message']
negative_messages = data[data['sentiment'] < negative_threshold]['message']

print("\nPozitivní zprávy:")
for msg in positive_messages:
    print("-", msg)

print("\nNegativní zprávy:")
for msg in negative_messages:
    print("-", msg)