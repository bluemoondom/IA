# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 19:47:20 2025

@author: dominika
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from transformers import pipeline, logging
import pandas as pd
logging.set_verbosity_error()

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
bert_pipeline = pipeline("sentiment-analysis", 
                             model=model_name,
                             tokenizer=model_name)

czech_texts = [
    "Jsem spoko s nákupem!",
    "Nejhorší nákup v mém životě",
    "Docela průměrné, nic moc",
    "Mám z toho smíšené pocity",
    "Skvělá kvalita, doporučujem",
    "Totální zklamání",
    "Celkem v pořádku",
    "Jako vždy"
]

results = []
for text in czech_texts:
    result = bert_pipeline(text)[0]
    results.append({
        'text': text,
        'sentiment': result['label'],
        'confidence': round(result['score'], 3)
    })

df = pd.DataFrame(results)
print("BERT sentiment analýza:")
print(df.to_string(index=False))

tw_pipeline = pipeline("sentiment-analysis", 
                                model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

results = []
for text in czech_texts:
    result = tw_pipeline(text)[0]
    results.append({
        'text': text,
        'sentiment': result['label'],
        'confidence': round(result['score'], 3)
    })
df = pd.DataFrame(results)
print("")
print("")
print("TW sentiment analýza:")
print(df.to_string(index=False)) 

