from vocabulary import Vocabulary
import numpy as np

class NBModel:

    @staticmethod
    def trainNB(x_train, y_train, threshold):
        # First extract vocabulary
        unigrams = Vocabulary.getUniGrams(x_train)
        bigrams  = Vocabulary.getBiGrams(x_train)

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
            counts    = Vocabulary.getFullDict(cl_docs, allgrams)

            # Implement the threshold - get rid of features 
            # which appear less than threshold number of times
            counts = {k: val for k,val in counts.items() if val >= threshold}

            probs['{}_occr'.format(cl)] = counts
            probs['{}_tot'.format(cl)]  = sum(counts.values())
            probs['{}_Pc'.format(cl)]   = Pc if cl == 1 else 1 - Pc

        return probs

    @staticmethod
    def predictNB(x_test, y_test, model, smoothing = False, eps = 0.01):

        predictions = []
        prod_probs  = []
        smoothing   = smoothing if smoothing else 0

        for document in x_test:
            # Get the vocabulary of the document
            vocab = Vocabulary.getVocabularyByDocument(document)
            probs = {}

            for cl in np.unique(y_test):
                # For each class calculate the class probability
                prob = np.log(model['{}_Pc'.format(cl)])

                for features in vocab.keys():
                    # Calculate probability of each feature
                    try:
                        freq = model['{}_occr'.format(cl)][features]
                    except:
                        freq = eps
                        #continue

                    total = model['{}_tot'.format(cl)]
                    sizeV = len(model['{}_occr'.format(cl)])
                    
                    calc  = np.log(freq + smoothing) - np.log(total + sizeV * smoothing)

                    prob += vocab[features]*calc

                probs[cl] = prob

            predictions.append(max(probs, key=probs.get))
            prod_probs.append(probs)

        return predictions, prod_probs



                










        

