import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pydicom as dicom
import cv2
import ast

import glob
import re
import math
from tqdm.notebook import tqdm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from pydicom.pixel_data_handlers.util import apply_voi_lut
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import applications 

#Enlace de la base de datos
path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/'
os.listdir(path)


train_data = pd.read_csv(path+'train_labels.csv')
samp_subm  = pd.read_csv(path+'sample_submission.csv')

print(train_data.head(7)),print(samp_subm.head())

#Imprimir tamaño de carpetas por TRAIN Y TEST
print('Samples train:', len(train_data))
print('Samples test:', len(samp_subm))

train_data.head(7)

## analisis de datos faltantes
print(pd.isnull(train_data).sum()) 
print('___________')
print(pd.isnull(samp_subm ).sum())

train_data["MGMT_value"].value_counts()

to_exclude = [109, 123, 709]

train_data = train_data[~train_data['BraTS21ID'].isin(to_exclude)]
num_samples = train_data.shape[0]
num_positives = np.sum(train_data['MGMT_value'] == 1)
num_negatives = np.sum(train_data['MGMT_value'] == 0)


train_data["MGMT_value"].value_counts().head(2).plot(kind = 'pie',autopct='%1.1f%%', figsize=(8, 8)).legend()


train_data.hist(column="MGMT_value")

samp_subm.head()

#analizar una carpeta 100--->00150
folder = str(train_data.loc[100, 'BraTS21ID']).zfill(5)
## CONTENIDO DE LAS CARPETAS
os.listdir(path+'train/'+folder)