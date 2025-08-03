# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 21:25:13 2025

@author: dominika
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 13:49:50 2025

@author: dominika
"""

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from tqdm.auto import tqdm

model_name = "microsoft/table-transformer-detection"
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = TableTransformerForObjectDetection.from_pretrained(model_name,revision="no_timm")
structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
cropped_image = Image.open("tab.jpg")
img_temp = "temp.jpg"
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
    
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model2 = AutoModel.from_pretrained(
    'ucaslcl/GOT-OCR2_0'
    , trust_remote_code=True
    , low_cpu_mem_usage=True
    , device_map='cuda'
    , use_safetensors=True
)
model2 = model2.eval().cuda()

def display_table(cropped_image, features):
    ctv = cropped_image.copy()
    draw = ImageDraw.Draw(ctv)
    font = ImageFont.truetype("/content/roboto/Roboto-Bold.ttf", 15)
    for feature in features:
        draw.rectangle(feature["bbox"], outline="red")
        text_position = (feature["bbox"][0], feature["bbox"][1] - 3)
        draw.text(text_position, feature["label"], fill="blue", font = font)
        #print(text_position)

    plt.figure()
    plt.imshow(ctv) 
    plt.show()

display_table(cropped_image, features)

def get_cell_row(table_data):
    rows = [entry for entry in table_data if entry["label"] == "table row"]
    columns = [entry for entry in table_data if entry["label"] == "table column"]
    rows.sort(key=lambda x: x["bbox"][1])
    columns.sort(key=lambda x: x["bbox"][0])

    def find_cell(row, column):
        cell_bbox = [column["bbox"][0], row["bbox"][1], column["bbox"][2], row["bbox"][3]]
        return cell_bbox
    cell_positions = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell(row, column)
            row_cells.append({"cell": cell_bbox})
        cell_positions.append({"cells": row_cells, "cell_count": len(row_cells)})
    return cell_positions

cell_positions = get_cell_row(features)
print(cell_positions)

def apply_ocr(cell_positions, cropped_image, img_temp):
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(tqdm(cell_positions)):
        row_text = []
        for cell in row["cells"]:
            cell_image = cropped_image.crop(cell["cell"])
            cell_image.save(img_temp)
            text = model2.chat(
                tokenizer
                , img_temp
                , ocr_type='ocr'
            )
            
            if text:
                row_text.append(text)
        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)
        data[idx] = row_text

    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data
        #print(row_data)

    return data

data = apply_ocr(cell_positions, cropped_image, img_temp)
print(data)