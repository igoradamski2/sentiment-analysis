import os
import re
import numpy as np
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm

class DataLoader:
    @staticmethod
    def getFileList(file_dir):
        fileList = []
        for r, d, f in os.walk(file_dir):
            for file in f:
                fileList.append(os.path.join(file_dir, file))
            
        return fileList

    @staticmethod
    def getFilesInRange(file_dir, low, high):
        all_files = DataLoader.getFileList(file_dir)
        numbered  = [re.search("\\/cv\d\d\d\_",x).group(0) for x in all_files]
        numbered  = [re.findall(r'\d+', str) for str in numbered]
        numbered  = [int(x[0]) for x in numbered]

        flags     = [t >= low and t <= high for t in numbered]
        flags_idx = [i for i, x in enumerate(flags) if x]

        return [all_files[idx] for idx in flags_idx]

    @staticmethod
    def getFilesByIdx(file_dir, index):
        all_files = DataLoader.getFileList(file_dir)
        numbered  = [re.search("\\/cv\d\d\d\_",x).group(0) for x in all_files]
        numbered  = [re.findall(r'\d+', str) for str in numbered]
        numbered  = [int(x[0]) for x in numbered]

        flags     = [t in index for t in numbered]
        flags_idx = [i for i, x in enumerate(flags) if x]

        return [all_files[idx] for idx in flags_idx]

    @staticmethod
    def getFiles(file_dir, low, high = None):
        if high:
            all_files = DataLoader.getFilesInRange(file_dir, low, high)
        else:
            all_files = DataLoader.getFilesByIdx(file_dir, low)

        result=[]
        for fname in all_files:
            with open(fname) as f:
                result.append(f.read())
        return result

    @staticmethod
    def getAllFiles(file_dir):
        
        all_files = DataLoader.getFileList(file_dir)
        result=[]
        for fname in all_files:
            with open(fname) as f:
                result.append(f.read())
        return result

    @staticmethod
    def splitLines(all_files):
        # First get rid of empty lines
        result = [file.split("\n") for file in all_files]
        words  = []
        wtypes = []
        for review in result:
            curr_words, curr_wtypes = map(list, zip(*[x.split("\t") for x in review if x]))
            curr_words = [words.lower() for words in curr_words] 
            words.append(curr_words)
            wtypes.append(curr_wtypes)

        return words, wtypes

    @staticmethod
    def splitLinesNLTK(all_files):
        # First get rid of empty lines
        words = []
        tknzr = TweetTokenizer()
        for review in tqdm(all_files):
            curr_words = tknzr.tokenize(review)
            words.append([words.lower() for words in curr_words])

        return words


    @staticmethod
    def getStanfordFiles(file_dir):

        print("fetching /train/pos")
        data  = DataLoader.getAllFiles(file_dir + "/train/pos")
        print("fetching /train/neg")
        data += DataLoader.getAllFiles(file_dir + "/train/neg")
        print("fetching /train/unsup")
        data += DataLoader.getAllFiles(file_dir + "/train/unsup")
        print("fetching /test/pos")
        data += DataLoader.getAllFiles(file_dir + "/test/pos")
        print("fetching /test/neg")
        data += DataLoader.getAllFiles(file_dir + "/test/neg")

        return data

    @staticmethod
    def splitStanfordLines(all_files):
        # First get rid of empty lines
        result = [file.split("\n") for file in all_files]
        words  = []
        for review in result:
            rwords = review[0].split(" ")
            curr_words = [word.lower() for word in rwords]
            words.append(curr_words)

        return words

class DataHandler:

    # This class takes a directory path and returns whole data as object

    def __init__(self, file_dir = ''):
        self.file_dir = file_dir 

    def __call__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def readOurData(self):
        pos_train        = DataLoader.getAllFiles(self.file_dir + "/POS")
        neg_train        = DataLoader.getAllFiles(self.file_dir + "/NEG")

        x_train, y_train = DataHandler.mergeClasses(pos_train, neg_train)
        x_train          = DataLoader.splitLinesNLTK(x_train)

        self(x_train, y_train)

    def readStanfordData(self):
        data = DataLoader.getStanfordFiles(self.file_dir)
        self(DataLoader.splitLinesNLTK(data), [])


    def blind_test(self):

        self.roundRobinSplit(10)

        train_blind = Split([], [])
        test_blind  = Split([], [])
        for i in range(len(self.rrsplit)):
            if i == 0:
                test_blind.x_train += self.rrsplit[i].x_train
                test_blind.y_train += self.rrsplit[i].y_train

            else:
                train_blind.x_train += self.rrsplit[i].x_train
                train_blind.y_train += self.rrsplit[i].y_train

        return train_blind, test_blind


    def roundRobinSplit(self, N):

        uq, ct       = np.unique(self.y_data, return_counts=True)
        num_of_files = max(ct)

        splits       = DataHandler.getIndicesForRR(num_of_files, N)

        data_splits  = []
        for i in range(N):
            x_train = []
            y_train = []
            for ylabel in uq:

                # We are doing this so we have equal number of classes in splits
                this_class_idx  = [idx for idx, label in enumerate(self.y_data) if label == ylabel]
                this_class_data = [self.x_data[idx] for idx in this_class_idx]
                x_train += [this_class_data[idx] for idx in splits[i]]
                y_train += [ylabel for i in splits[i]]

            data_splits.append(Split(x_train, y_train))

        self.rrsplit = data_splits


    @staticmethod
    def applyDoc2Vec(data, model):

        vectors = [model.infer_vector(document) for document in data]
        return vectors

    @staticmethod
    def getIndicesForRR(num_of_files, N):
        splits = []
        for i in range(N):
            splits.append([i + N*k for k in range(((-(-num_of_files//N)))-1)])
        
        return splits


    @staticmethod
    def getTrainTestSet(file_dir, low_train, high_train=None, low_test=None, high_test=None):

        # The low and high arguments specify which files
        # will be the train files - the rest is test

        pos_train = DataLoader.getFiles(file_dir + "/POS", low_train, high_train)
        neg_train = DataLoader.getFiles(file_dir + "/NEG", low_train, high_train)

        x_train, y_train = DataHandler.mergeClasses(pos_train, neg_train)
        x_train, _ = DataLoader.splitLines(x_train)

        if low_test:
            pos_test  = DataLoader.getFiles(file_dir + "/POS", low_test, high_test)
            neg_test  = DataLoader.getFiles(file_dir + "/NEG", low_test, high_test)
            x_test, y_test   = DataHandler.mergeClasses(pos_test, neg_test)
            x_test , _ = DataLoader.splitLines(x_test)

            return x_train, y_train, x_test, y_test
        else:
            return x_train, y_train


    @staticmethod
    def mergeClasses(pos, neg):

        # This function merges 2 datasets into 1, 
        # giving the class vector: 1 - positive,
        # 0 - negative

        data  = pos+neg
        c_idx = [1] * len(pos)
        c_idx = c_idx + [0] * len(neg)

        return data, c_idx 


class Split:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train  

class StanfordDocument:

    def __init__(self, words, tags):
        self.words = words
        self.tags = tags
