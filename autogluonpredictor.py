# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 07:32:18 2025

@author: dominika
"""

import pandas as pd
from autogluon.tabular import TabularPredictor

train_data = pd.read_csv('train.csv')
subsample_size = 5000  
if subsample_size is not None and subsample_size < len(train_data):
    train_data = train_data.sample(n=subsample_size, random_state=0)
test_data = pd.read_csv('test.csv')

tabpfnmix_default = {
    "model_path_classifier": "autogluon/tabpfn-mix-1.0-classifier",
    "model_path_regressor": "autogluon/tabpfn-mix-1.0-regressor",
    "n_ensembles": 1,
    "max_epochs": 10,
    }

hyperparameters = {
    "TABPFNMIX": [
        tabpfnmix_default,
    ],
    'GBM': [
        {'ag_args_fit': {'num_gpus': 1}}   #0 CPU 1 GPU
    ]}

label = "class"

predictor = TabularPredictor(label=label)
predictor = predictor.fit(
    train_data=train_data,
    num_gpus=1, #GPU
    hyperparameters=hyperparameters,
    verbosity=3,
    )

predictor.leaderboard(test_data, display=True)
predictor.feature_importance(data=test_data)

