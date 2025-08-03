# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 09:01:04 2025

@author: dominika
"""

import pandas as pd
import numpy as np
import pymssql
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import mplcursors

def plot2d(x_values, y_values, text_labels):
    fig, ax = plt.subplots()
    mplcursors.cursor(hover=True)
    ax.scatter(x_values, y_values, label='Data Points')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('2D Plot of my embedding model')
    for i, label in enumerate(text_labels):
        ax.annotate(label, (x_values[i], y_values[i]))
    plt.show()

from sentence_transformers import SentenceTransformer
#model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
df = pd.read_csv('AG_news_samples.csv')

data = df.to_dict(orient='records')
descriptions = [row['description'] for row in data]
all_embeddings = model.encode(descriptions)
all_embeddings.shape
PCA_model = PCA(n_components = 2)
PCA_model.fit(all_embeddings)
embeddingspca = PCA_model.transform(all_embeddings)

target = "label_int"
conn = pymssql.connect('localhost', 'sa', 'Dominique745', 'my')
   
cur = conn.cursor()
query = """INSERT INTO dbo.mynews(ntitle
, ndescription
, nclassificationID
, nclassification
, nembedding
, nembeddingpca)
VALUES (%s, %s, %s, %s, %s, %s)"""
i = 0

for row, embedding, embeddingpca in zip(data, all_embeddings, embeddingspca):
    row["embedding"] = embedding
    row["embeddingpca"] = embeddingpca
    cur.execute(query, (row["title"]
                        , row["description"]
                        , row["label_int"]
                        , row["label"]
                        , str(row["embedding"])
                        , str(row["embeddingpca"]) ))
    conn.commit()
    if i == 0:
        a = np.array([[embeddingpca[0], embeddingpca[1], row["title"]]])
    if i > 0 and i < 20:
        b = np.array([[embeddingpca[0], embeddingpca[1], row["title"]]])
        a = np.concatenate((a, b), axis=0)
    i = i + 1
        
x = np.asarray(a[:,0], dtype=float)
y = np.asarray(a[:,1], dtype=float)
print(x)
print(y)
print(data[0])
plot2d(x, y, a[:,2])

#print(data[0])

cur.close()
conn.close()