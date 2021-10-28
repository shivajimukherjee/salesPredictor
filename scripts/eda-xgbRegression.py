#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 20:07:18 2021

@author: shivajimukherjee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

big_mart_data = pd.read_csv('~/DS Repository/bigMartSalesPredictor/Train.csv')
big_mart_data.head(0)
big_mart_data.info()
big_mart_data.select_dtypes(include='float64').head(0).columns