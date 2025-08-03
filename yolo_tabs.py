# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 13:49:50 2025

@author: dominika
"""

from ultralyticsplus import YOLO, render_result
from PIL import Image as Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch
from tqdm.auto import tqdm
import numpy as np
import pytesseract
import csv

model = YOLO('foduucom/table-detection-and-extraction')
model.overrides['conf'] = 0.25
model.overrides['iou'] = 0.45
model.overrides['agnostic_nms'] = False
model.overrides['max_det'] = 1000

image = Image.open("fa.jpg")
results = model.predict(image)

print(results[0].boxes)
box = results[0].boxes.xyxy
i = 1
for myxyxy in zip (box):
    myxyxylist = myxyxy[0].tolist()
    #print(myxyxylist)
    new_image = image.crop(myxyxylist)
    plt.figure()
    plt.imshow(new_image) 
    plt.show()
    new_image.save("tab" + str(i) + ".jpg")
    i = i + 1

render = render_result(model=model, image=image, result=results[0])
#render.show()

model_name = "microsoft/table-transformer-detection"
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = TableTransformerForObjectDetection.from_pretrained(model_name,revision="no_timm")
structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
cropped_image = Image.open("fa.jpg")

inputs = image_processor(images = cropped_image, return_tensors="pt")
outputs = structure_model(**inputs)

target_sizes = torch.tensor([cropped_image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
features = []
for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
    box = [round(i, 2) for i in box.tolist()]
    score = score.item()
    label = structure_model.config.id2label[label.item()]
    cell_dict = {"label":label,
                  "score":score,
                  "bbox":box
                  }

    features.append(cell_dict)


def display_detected_features(cropped_image, features):
    cropped_table_visualized = cropped_image.copy()
    draw = ImageDraw.Draw(cropped_table_visualized)
    font = ImageFont.truetype("/content/roboto/Roboto-Bold.ttf", 15)
    for feature in features:
        draw.rectangle(feature["bbox"], outline="red")
        text_position = (feature["bbox"][0], feature["bbox"][1] - 3)
        draw.text(text_position, feature["label"], fill="blue", font = font)
        print(features)

    plt.figure()
    plt.imshow(cropped_table_visualized) 
    plt.show()

display_detected_features(cropped_image, features)

def get_cell_coordinates_by_row(table_data):
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox
    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'cell': cell_bbox})
        cell_coordinates.append({'cells': row_cells, 'cell_count': len(row_cells)})
    return cell_coordinates

cell_coordinates = get_cell_coordinates_by_row(features)

def apply_ocr(cell_coordinates, cropped_image):
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []
        for cell in row["cells"]:
            cell_image = np.array(cropped_image.crop(cell["cell"]))
            text = pytesseract.image_to_string(cell_image, lang='ces', config='--psm 6').strip()
            if text:
                row_text.append(text)
        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)
        data[idx] = row_text

    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data
        print(row_data)

    return data

data = apply_ocr(cell_coordinates, cropped_image)

with open('output.csv','w', encoding='utf-8') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    for row, row_text in data.items():
        wr.writerow(row_text)
