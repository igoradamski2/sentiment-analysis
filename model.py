from vocabulary import Vocabulary
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from sklearn import svm



class NBModel:

    def __init__(self, x_train, y_train, threshold, grams = ['uni', 'bi'], smoothing = False):
        self.x_train   = x_train
        self.y_train   = y_train
        self.threshold = threshold
        self.grams     = grams
        self.smoothing = smoothing
        #self.model     = NBModel.trainNB(x_train, y_train, threshold, grams = ['uni', 'bi'])

    def train(self, x_train, y_train):

        assert len(self.grams) > 0, "You must provide what n-grams to use"
        assert (self.grams == 'uni' or self.grams == 'bi' or \
                self.grams == ['uni', 'bi'] or self.grams == ['bi','uni']), \
                "Only uni or bi grams are implemented!"

        # First extract vocabulary
        if 'uni' in self.grams:
            unigrams = Vocabulary.getUniGrams(x_train)
        else:
            unigrams = set()

        if 'bi' in self.grams:
            bigrams = Vocabulary.getBiGrams(x_train)
        else:
            bigrams = set()


        allgrams = unigrams | bigrams

        assert len(unigrams) + len(bigrams) == len(allgrams)

        # Get number of total documents
        # Pc will be probability of class positive (class 1)
        N     = len(x_train)
        Pc    = sum(y_train)/N
        probs = {} 

        # Remember that y_train = 1 -> positive
        for cl in np.unique(y_train):
            # Get documents for that class
            cl_docs   = [x_train[i] for i in np.where(np.array(y_train) == cl)[0].tolist()]
            counts    = Vocabulary.getFullDict(cl_docs, allgrams, self.grams)

            # Implement the threshold - get rid of features 
            # which appear less than threshold number of times
            counts = {k: val for k,val in counts.items() if NBModel.thr_condition(k,\
                                                                                  val,\
                                                                                  unigrams,\
                                                                                  bigrams,\
                                                                                  self.threshold)}

            probs['{}_occr'.format(cl)] = counts
            probs['{}_tot'.format(cl)]  = sum(counts.values())
            probs['{}_Pc'.format(cl)]   = Pc if cl == 1 else 1 - Pc

        self.model = probs

    def predict(self, x_test):

        predictions = []
        prod_probs  = []
        smoothing   = self.smoothing if self.smoothing else 0

        for document in x_test:
            # Get the vocabulary of the document
            vocab = Vocabulary.getVocabularyByDocument(document, self.grams)
            probs = {}

            for cl in np.unique(self.y_train):
                # For each class calculate the class probability
                prob = np.log(self.model['{}_Pc'.format(cl)])

                for features in vocab.keys():
                    # Calculate probability of each feature
                    try:
                        freq = self.model['{}_occr'.format(cl)][features]
                    except:
                        freq = 0
                        #continue

                    total = self.model['{}_tot'.format(cl)]
                    sizeV = len(self.model['{}_occr'.format(cl)])
                    
                    calc  = np.log(freq + smoothing) if freq + smoothing > 0 else 0

                    prob += vocab[features]*calc

                probs[cl] = prob - sum(vocab.values()) * np.log(total + sizeV * smoothing)

            predictions.append(max(probs, key=probs.get))
            prod_probs.append(probs)

        return predictions, prod_probs


    @staticmethod
    def trainNB(x_train, y_train, threshold, grams = ['uni', 'bi']):

        assert len(grams) > 0, "You must provide what n-grams to use"
        assert (grams == 'uni' or grams == 'bi' or \
                grams == ['uni', 'bi'] or grams == ['bi','uni']), \
                "Only uni or bi grams are implemented!"

        # First extract vocabulary
        if 'uni' in grams:
            unigrams = Vocabulary.getUniGrams(x_train)
        else:
            unigrams = set()

        if 'bi' in grams:
            bigrams = Vocabulary.getBiGrams(x_train)
        else:
            bigrams = set()


        allgrams = unigrams | bigrams

        assert len(unigrams) + len(bigrams) == len(allgrams)

        # Get number of total documents
        # Pc will be probability of class positive (class 1)
        N     = len(x_train)
        Pc    = sum(y_train)/N
        probs = {} 

        # Remember that y_train = 1 -> positive
        for cl in np.unique(y_train):
            # Get documents for that class
            cl_docs   = [x_train[i] for i in np.where(np.array(y_train) == cl)[0].tolist()]
            counts    = Vocabulary.getFullDict(cl_docs, allgrams, grams)

            # Implement the threshold - get rid of features 
            # which appear less than threshold number of times
            counts = {k: val for k,val in counts.items() if NBModel.thr_condition(k,\
                                                                                  val,\
                                                                                  unigrams,\
                                                                                  bigrams,\
                                                                                  threshold)}

            probs['{}_occr'.format(cl)] = counts
            probs['{}_tot'.format(cl)]  = sum(counts.values())
            probs['{}_Pc'.format(cl)]   = Pc if cl == 1 else 1 - Pc

        return probs

    @staticmethod
    def thr_condition(k, val, unigrams, bigrams, threshold):
        if len(unigrams) != 0:
            if k in unigrams:
                if val >= threshold[0]:
                    return True
                else:
                    return False
            else:
                pass

        if len(bigrams) != 0:
            if k in bigrams:
                t = threshold[0] if len(threshold) == 1 else threshold[1]
                if val >= t:
                    return True
                else:
                    return False
            else:
                pass


class MyDoc2Vec:

    def __init__(self, **model_params):
        self.model = Doc2Vec(**model_params)

    def train(self, data):

        self.model.build_vocab(data)
        self.model.train(data, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def save(self, name):
        fname = get_tmpfile(name)
        self.model.save(fname)

    def load(name):
        name = get_tmpfile(name)
        return Doc2Vec.load(name)

class MySVM:

    def __init__(self, **model_params):
        self.model = svm.SVC(**model_params)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

                










        

