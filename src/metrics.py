import numpy as np
import math as mt
from utilities import *
from model import *
from tqdm import tqdm
from scipy.sparse import csr_matrix as sparse


class Metrics:
    
    @staticmethod
    def getAccuracy(predictions, actual):

        accr = 0
        for idx, pred in enumerate(predictions):

            if pred == actual[idx]:
                accr += 1 

        return accr/len(predictions)


    @staticmethod
    def permutationTest(predictions1, predictions2, actual, R):

        scores1 = np.array([int(predictions1[i] == actual[i]) for i in range(len(actual))])
        scores2 = np.array([int(predictions2[i] == actual[i]) for i in range(len(actual))])

        # Get initial difference in means
        M0 = np.abs(np.mean(np.array(scores1)) - np.mean(np.array(scores2)))
        s  = 0

        for trial in range(R):

            permute = np.random.choice([0,1], len(scores1))
            curr_vec1 = [scores2[idx] if val == 1 else scores1[idx] for idx,val in enumerate(permute)]
            curr_vec2 = [scores1[idx] if val == 1 else scores2[idx] for idx,val in enumerate(permute)]

            M = np.abs(np.mean(np.array(curr_vec1)) - np.mean(np.array(curr_vec2)))

            if M >= M0:
                s += 1

        return (s+1)/(R+1)


    @staticmethod
    def signTest(predictions1, predictions2, actual):

        # ==========
        # This is a function to perform sign test to test
        # whether the performance of 1 is better than performance
        # of 2. So H0:performance1 = performance2
        # ==========

        # Get scores of each prediction
        scores1 = np.array([int(predictions1[i] == actual[i]) for i in range(len(actual))])
        scores2 = np.array([int(predictions2[i] == actual[i]) for i in range(len(actual))])

        uq, ct  = np.unique(scores1 - scores2, return_counts=True)
        p_diff  = {-1:0, 0:0, 1:0}
        for i,j in zip(uq, ct):
            p_diff[i] += j

        N = 2 * (-(-p_diff[0]//2)) + p_diff[-1] + p_diff[1]
        k = (-(-p_diff[0]//2)) + min(p_diff[-1], p_diff[1])
        q = 0.5

        p_val = 0

        for i in range(k+1):
            p_val = (mt.factorial(N)//(mt.factorial(N-i) * mt.factorial(i))) * (q ** i) * (1-q)**(N-i)

        p_val *= 2

        return p_val


    @staticmethod
    def roundRobinCV(data, N, model, **model_params):

        accuracies  = []

        for i in tqdm(range(N)):

            # We will leave out split i for testing
            x_train = []
            y_train = []
            x_test  = []
            y_test  = []

            for j in range(N):
                if j != i:
                    x_train += data.rrsplit[j].x_train
                    y_train += data.rrsplit[j].y_train
                else:
                    x_test += data.rrsplit[j].x_train
                    y_test += data.rrsplit[j].y_train

            trained_model  = model(**model_params)
            if data.sparse:
                x_train = sparse(x_train)
                trained_model.train(x_train, y_train)
            else:
                trained_model.train(x_train, y_train)
                
            predictions = trained_model.predict(x_test)
            
            accuracies.append(Metrics.getAccuracy(predictions, y_test))

        return accuracies


