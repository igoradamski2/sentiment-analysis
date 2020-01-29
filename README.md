# Simple IMDB movie review sentiment analysis

This repository contains code to parse and build simple sentiment models. It is designed to only work on 2 classes of data: positive and negative. 

## Data

In order for the code to work properly, data folder must have the following structure:
```	
data -> {NEG,POS} -> \*.txt
```
meaning that the data folder contains 2 subfolders NEG and POS, that contain .txt files (with IMDB movie reviews for example). The NEG subfolder must contain the negative class files and POS the positive class files.

## Simple tutorial

To load the data and packages into python environment:

```{python}
import sys
sys.path.append('src/')

from utilities import *
from model import *
from metrics import *
import numpy as np
from scipy import stats

file_dir = 'full-path-to-your-data-folder'

data = DataHandler(file_dir)

# Read data into object
data.readOurData()

# Perform a 10-fold round-robin split for cross-validation
data.roundRobinSplit(10)

# Train a uni+bigram naive-bayes model (NBModel) and evaluate cross-validated accuracy
accuracy = Metrics.roundRobinCV(data,10,NBModel,threshold = [4,6], grams = ['uni', 'bi'], smoothing = 0.1)

```



