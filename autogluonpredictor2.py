# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 08:47:57 2025

@author: dominika
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from autogluon.tabular import TabularDataset, TabularPredictor

predictor = TabularPredictor.load("AutogluonModels/ag-20250607_074512")
test_data = pd.read_csv('test.csv')
importance_df = predictor.feature_importance(test_data).reset_index()
plt.figure(figsize=(8,8))
plt.title('Feature Importance')
sns.set(style='darkgrid')
sns.barplot(
    data=importance_df,
    y='index',
    x='importance',
    orient='horizontal'
).set_ylabel('Feature')

plt.savefig('AutoML_with_AutoGluon_03.webp', bbox_inches='tight')

test_drive = {
  "age": 50,
  "workclass": " Private",
  "fnlwgt": 227890,
  "education": " HS-grad",
  "education-num": 13,
  "marital-status": " Married-civ-spouse",
  "occupation": " Sales",
  "relationship": " Husband",
  "race": " White",
  "sex": " Male",
  "capital-gain": 0,
  "capital-loss": 0,
  "hours-per-week": 50,
  "native-country": " United-States"
}

test_drive_df = TabularDataset([test_drive])

mypred = predictor.predict(test_drive_df)
print(mypred)