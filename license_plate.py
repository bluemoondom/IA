# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:37:29 2024

@author: dominika
"""

from transformers import pipeline
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
import io

checkpoint = "google/owlv2-base-patch16-ensemble"
detector = pipeline(
    model=checkpoint,
    task="zero-shot-object-detection",
    device="cuda"
)
original_image = Image.open("car.jpg")
plt.figure()
plt.imshow(original_image) 
plt.show()
prediction = detector(
    original_image,
    candidate_labels=["license plate"],
)[0]
print(prediction)
temporary_image = original_image.copy()
draw = ImageDraw.Draw(temporary_image)
box = prediction["box"]
label = prediction["label"]
score = prediction["score"]
xmin, ymin, xmax, ymax = box.values()
draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
draw.text((xmin, ymin), f"{label}: {round(score, 2)}", fill="white")
cropped_image = original_image.crop([xmin, ymin, xmax, ymax])
plt.figure()
plt.imshow(cropped_image) 
plt.show()
cropped_image.save("carx.jpg")

tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained(
    'ucaslcl/GOT-OCR2_0'
    , trust_remote_code=True
    , low_cpu_mem_usage=True
    , device_map='cuda'
    , use_safetensors=True
)
model = model.eval().cuda()
image_file = "carx.jpg"
res = model.chat(
    tokenizer
    , image_file
    , ocr_type='ocr'
)
print("License plate is: " + res)