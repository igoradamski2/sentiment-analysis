from utilities import DataLoader

def loadReviews(file_dir):
    
    pos = DataLoader.getFiles(file_dir + "/POS")
    neg = DataLoader.getFiles(file_dir + "/NEG")

    return pos, neg




if __name__ == "__main__":
    # Load data - assumes structure NEG and POS
    




