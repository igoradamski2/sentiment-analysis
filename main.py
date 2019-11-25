from utilities import DataLoader, DataHandler
from model import NBModel
from metrics import Metrics



if __name__ == "__main__": 
    #Load data - assumes structure NEG and POS
    #file_dir = "/homes/ija23/nlp/data-tagged"
    file_dir = "/Users/igoradamski/Documents/cambridge/MLMI/nlp/coursework/nlp/data-tagged"

    pos_train, neg_train, pos_test, neg_test = DataHandler.getTrainTestSet(file_dir,0,899,900,999)

    x_train, y_train = DataHandler.mergeClasses(pos_train, neg_train)
    x_test, y_test   = DataHandler.mergeClasses(pos_test, neg_test)

    x_train, _ = DataLoader.splitLines(x_train)
    x_test , _ = DataLoader.splitLines(x_test)

    print("training")
    model      = NBModel.trainNB(x_train, y_train, 4)

    print("testing")
    predictions, prod_probs = NBModel.predictNB(x_test, y_test, model, smoothing = 0)
    predictions_sm, prod_probs_sm = NBModel.predictNB(x_test, y_test, model, smoothing = 0.5)

    accuracy = Metrics.getAccuracy(predictions, y_test)
    accuracy_sm = Metrics.getAccuracy(predictions_sm, y_test)


    print('Accuracy is {}. \nSmoothed accuracy is {}'.format(accuracy, accuracy_sm))


