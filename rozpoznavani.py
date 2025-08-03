# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 15:46:47 2024

@author: dominika
"""

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open("fa.jpg")

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
font = ImageFont.truetype(r'C:\Users\pille\AppData\Local\Microsoft\Windows\Fonts\Dominique.ttf', 20)
draw = ImageDraw.Draw(image)
print(results)
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    draw.rectangle((box), outline=(252, 194, 34), width=2)
    draw = ImageDraw.Draw(image)
    draw.text((box[0]+2, box[1]+2), model.config.id2label[label.item()], font = font, align ="left", fill=(252, 194, 34))  

image.save('./pillow_imagedraw.jpg', quality=95)
image.show()