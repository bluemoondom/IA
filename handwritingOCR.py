# -*- coding: utf-8 -*-
"""
Created on Wed May 21 16:54:41 2025

@author: dominika
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

image_path = "hwtest3.jpg"  
image = cv2.imread(image_path)
plt.figure()
plt.imshow(image) 
plt.show()
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 5))
closed_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)
contour_image = np.zeros_like(closed_image)
contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contour_image, contours, -1, (255), thickness=2)

original_image = cv2.imread(image_path)
cropped_images = []
image_with_bboxes = original_image.copy()
mylist = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = original_image[y:y+h, x:x+w]
    pixel_values = processor(images=cropped_image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values,max_new_tokens=4000)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    #print(generated_text[0])
    mylist += [generated_text[0]]  
    cropped_images.append(cropped_image)
    cv2.rectangle(image_with_bboxes, (x, y), (x + w, y + h), (0, 255, 0), 2) 

mylistreverse = mylist[::-1]
mystringOCR = ' '.join(mylistreverse)
print("My OCR output is:" + mystringOCR)
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(image_with_bboxes, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


