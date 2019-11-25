from utilities import *
from model import *
from metrics import *
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from tqdm import tqdm
import time
import multiprocessing
from sklearn import svm


# First we need to load all data for the Doc2Vec

file_dir_stanford = "/Users/igoradamski/Documents/cambridge/MLMI/nlp/coursework/nlp/data-tagged/aclImdb"
file_dir_our      = "/Users/igoradamski/Documents/cambridge/MLMI/nlp/coursework/nlp/data"

# Load the data 
stanford_data = DataHandler.readStanfordData(file_dir_stanford)
our_data      = DataHandler.readOurData(file_dir_our)

# Process the data
stanford_data = [StanfordDocument(words, [1]) for words in stanford_data.x_data]

train, test   = our_data.blind_test()

cv_train      = DataHandler()
cv_train(train.x_train, train.y_train)

# Train models ===============================================================
cores     = multiprocessing.cpu_count()
svm_model = MySVM(gamma=0.01, kernel = 'rbf')

for dm in [0, 1]:

    for hs in [0, 1]:

        for min_count in [1, 5, 10]:

            for vector_size in [50, 100, 150, 200]:

                for window in [1, 2, 5]:

                    print("Calculating model_dm={},hs={},min_count={},vector_size={},window={} ....".format(dm,hs,min_count,vector_size,window))

                    doc2vec_model = MyDoc2Vec(dm=dm, \
                                              vector_size=vector_size, \
                                              min_count=min_count, \
                                              epochs=15, \
                                              workers=cores, \
                                              hs = hs, \
                                              window = window)
                    doc2vec_model.train(dataTag)
                    doc2vec_model.save("model_dm={},hs={},min_count={},vector_size={},window={}".format(dm,hs,min_count,vector_size,window))

                    # Now evaluate on our data
                    vector_data = DataHandler()
                    vector_data(DataHandler.applyDoc2Vec(cv_train.x_data, doc2vec_model.model), cv_train.y_data)
                    vector_data.roundRobinSplit(10)

                    svm_model.train(vector_data.x_data, vector_data.y_data)
                    predictions = svm_model.predict(DataHandler.applyDoc2Vec(test.x_train, doc2vec_model.model))

                    cv_accuracies   = np.mean(np.array(roundRobinCV(data, 10, MySVM, gamma=0.002, kernel = 'rbf')))
                    test_accuracies = Metrics.getAccuracy(predictions, test.y_train)
                    with open('optimisation_results.txt', 'a+') as f:
                        my_str = "model_dm={},hs={},min_count={},vector_size={},window={}".format(dm,hs,min_count,vector_size,window)
                        my_str += " accuracy = {}".format(Metrics.getAccuracy(predictions, test.y_train))
                        f.write(my_str)







