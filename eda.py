import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#from ydata_profiling import ProfileReport

import polars as pl
print('polars_version:',pl.__version__)

from scipy.stats import f_oneway

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.dummy import DummyClassifier
#from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score,recall_score, precision_score, roc_curve, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('online_shoppers_intention.csv')
df_copy['Informational_Duration'].fillna(df_copy['Informational_Duration'].median(), inplace=True)
df_copy['ProductRelated_Duration'].fillna(df_copy['ProductRelated_Duration'].median(), inplace=True)
df_copy['ExitRates'].fillna(df_copy['ExitRates'].median(), inplace=True)