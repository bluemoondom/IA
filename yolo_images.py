# -*- coding: utf-8 -*-
"""

@author: dominika
"""



from inference import get_model
import supervision as sv
from inference.core.utils.image_utils import load_image_bgr
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("hwtest2.png").convert("RGB")
#image = load_image_bgr("https://...cars.png")

model = get_model(model_id="yolo11x-640")
results = model.infer(image)[0]
results = sv.Detections.from_inference(results)
annotator = sv.BoxAnnotator(thickness=4)
annotated_image = annotator.annotate(image, results)
annotator = sv.LabelAnnotator(text_scale=2, text_thickness=2)
annotated_image = annotator.annotate(annotated_image, results)
print(results.xyxy)
for myxyxy, myclass_name, myconfidence in zip(results.xyxy, results.data["class_name"], results.confidence):
    print(myxyxy)
    print(myclass_name)
    print(myconfidence)
    new_image = annotated_image.crop(myxyxy)
    new_size = int((myxyxy[2] - myxyxy[0]) / 2), int((myxyxy[3] - myxyxy[1]) / 2)
    print(new_size)
    new_image = new_image.resize(new_size)
    sv.plot_image(new_image, (1, 1))
#sv.plot_image(annotated_image, (5, 5))
plt.figure()
plt.imshow(annotated_image) 
plt.show()