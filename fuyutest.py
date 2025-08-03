# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 19:29:15 2025

@author: dominika
"""

from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b").to(device)
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Urbanway_18_GNV_%28RATP_Bus_95%29.jpg/960px-Urbanway_18_GNV_%28RATP_Bus_95%29.jpg"
image = Image.open(requests.get(url, stream=True).raw)
plt.figure()
plt.imshow(image) 
plt.show()
prompt = "Generate a coco-style caption.\n"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
outputs = model(**inputs)
generated_ids = model.generate(**inputs, max_new_tokens=7)
generation_text = processor.batch_decode(generated_ids[:, -7:], skip_special_tokens=True)

print(generation_text[0])
#popisovani obrazku