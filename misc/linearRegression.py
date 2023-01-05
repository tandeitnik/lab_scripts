# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 09:19:49 2023

@author: tandeitnik
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels import api

# generate an independent variable 
x = np.linspace(-10, 30, 100)
# generate a normally distributed residual
e = np.random.normal(10, 5, 100)
# generate ground truth
y = 10 + 4*x + e
df = pd.DataFrame({'x':x, 'y':y})

sns.regplot(x='x', y='y', data = df)
plt.show() 

features = api.add_constant(df.x)
model = api.OLS(y, features).fit()
model.summary()