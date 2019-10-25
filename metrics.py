
class Metrics:
    
    @staticmethod
    def getAccuracy(predictions, actual):

        accr = 0
        for idx, pred in enumerate(predictions):

            if pred == actual[idx]:
                accr += 1 

        return accr/len(predictions)