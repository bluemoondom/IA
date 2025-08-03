# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 14:02:20 2025

@author: dominika
"""

import timm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch
import matplotlib.pyplot as plt

model = timm.create_model('cspdarknet53', pretrained=True)
model.eval()
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

url = "https://resize-elle.ladmedia.fr/rcrop/638,,forcex/img/var/plain_site/storage/images/elle-a-table/recettes-de-cuisine/risotto-aux-asperges-2064378/21567538-2-fre-FR/Risotto-aux-asperges-vertes.jpg"
filename = "risotto.jpg"
urllib.request.urlretrieve(url, filename)
img = Image.open(filename).convert('RGB')
plt.figure()
plt.imshow(img) 
plt.show()
tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    out = model(tensor)

probabilities = torch.nn.functional.softmax(out[0], dim=0)
print(probabilities.shape)

url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename) 

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

top5_prob, top5_catid = torch.topk(probabilities, 5)

for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
    

