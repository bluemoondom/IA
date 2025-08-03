# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 20:47:09 2024

@author: dominika
"""

import feedparser
from transformers import MarianMTModel, MarianTokenizer, pipeline
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def translate_en_to_cs(mytext):
    tokenizer_en_cs = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-cs')
    model_en_cs = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-cs').to(device)
    translated_to_cs = model_en_cs.generate(**tokenizer_en_cs(mytext, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device))
    return[tokenizer_en_cs.decode(t, skip_special_tokens=True) for t in translated_to_cs]

def translate_en_to_fr(mytext):
    tokenizer_en_fr = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
    model_en_fr = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-fr').to(device)
    translated_to_fr = model_en_fr.generate(**tokenizer_en_fr(mytext, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device))
    return[tokenizer_en_fr.decode(t, skip_special_tokens=True) for t in translated_to_fr]

mytext = 'Hello, how are you?'
mytranslate = translate_en_to_cs(mytext)
print(mytranslate)

mytranslate = translate_en_to_fr(mytext)
print(mytranslate)
