from utilities import DataLoader
from model import NBModel
from metrics import Metrics

def getTrainTestSet(file_dir, low_train, high_train, low_test, high_test):

    # The low and high arguments specify which files
    # will be the train files - the rest is test

    pos_train = DataLoader.getFiles(file_dir + "/POS", low_train, high_train)
    neg_train = DataLoader.getFiles(file_dir + "/NEG", low_train, high_train)

    pos_test  = DataLoader.getFiles(file_dir + "/POS", low_test, high_test)
    neg_test  = DataLoader.getFiles(file_dir + "/NEG", low_test, high_test)

    return pos_train, neg_train, pos_test, neg_test

def mergeClasses(pos, neg):

    # This function merges 2 datasets into 1, 
    # giving the class vector: 1 - positive,
    # 0 - negative

    data  = pos+neg
    c_idx = [1] * len(pos)
    c_idx = c_idx + [0] * len(neg)

    return data, c_idx 



if __name__ == "__main__": 
    #Load data - assumes structure NEG and POS
    #file_dir = "/homes/ija23/nlp/data-tagged"
    file_dir = "/Users/igoradamski/Documents/cambridge/MLMI/nlp/coursework/nlp/data-tagged"

    pos_train, neg_train, pos_test, neg_test = getTrainTestSet(file_dir,0,899,900,999)

    x_train, y_train = mergeClasses(pos_train, neg_train)
    x_test, y_test   = mergeClasses(pos_test, neg_test)

    x_train, _ = DataLoader.splitLines(x_train)
    x_test , _ = DataLoader.splitLines(x_test)

    print("training")
    model      = NBModel.trainNB(x_train, y_train, 4)

    print("testing")
    predictions, prod_probs = NBModel.predictNB(x_test, y_test, model)

    accuracy = Metrics.getAccuracy(predictions, y_test)

    print(accuracy)


