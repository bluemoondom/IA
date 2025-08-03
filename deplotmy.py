# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 15:28:43 2025

@author: dominika
"""

from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot").to(device)
processor = AutoProcessor.from_pretrained("google/deplot")
url = "https://img.ihned.cz/attachment.php/110/80017110/RTKwBpy2jfSnhNDrFEaVW5buGkU647vA/EK51-52_10_cesko_politika_graf.png"
image = Image.open(requests.get(url, stream=True).raw)
plt.figure()
plt.imshow(image) 
plt.show()
inputs = processor(images=image
                   , text="Generate underlying data table of the figure below:"
                   , return_tensors="pt"
                   , font_path="/kaggle/input/deplot-fonts/arial.ttf")
inputs = {key: value.to(device) for key, value in inputs.items()}
#inputs = processor(images=image
#                   , text="Generate underlying data table of the figure below:"
#                   , return_tensors="pt")
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))