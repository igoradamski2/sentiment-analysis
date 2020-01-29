# Simple IMDB movie review sentiment analysis

This repository contains code to parse and build simple sentiment models. It is designed to only work on 2 classes of data: positive and negative. Results on a dataset from IMDB website containing 1000 positive and 1000 negative reviews can be viewed in the [imdb_benchmark_results_and_implementation.pdf](imdb_benchmark_results_and_implementation.pdf) file. It also contains all implementation details.

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
```
To load data and perform a 10-fold round-robin split:
```{python}
file_dir = 'full-path-to-your-data-folder'

data = DataHandler(file_dir)

# Read data into object
data.readOurData()

# Perform a 10-fold round-robin split for cross-validation
data.roundRobinSplit(10)
```
Finally to train a simple uni+bigram naive bayes model and evaluate its cross-validated performance on the round-robin splits:
```{python}
accuracy = Metrics.roundRobinCV(data,10,NBModel,threshold = [4,6], grams = ['uni', 'bi'], smoothing = 0.1)

```
The NBModel object takes in threshold (ignore certain uni or bigrams with frequency less than threshold), grams (uni or bi or both) and smoothing parameters (kernel smoothing). The implementation follows closely [this link](https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html) and can be found in the .pdf file in this repo.

## Models

Other models are also supported in the library such as MyDoc2Vec (wrapper for gensim Doc2Vec), MySVM (wrapper for sklearn implementation of SVM) and BoW2Vec (bag-of-words to vector model, uses sparse representation for efficiency). All models have train and predict methods. 


