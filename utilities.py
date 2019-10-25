import os
import re
import numpy as np

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
    def getFiles(file_dir, low, high):
        all_files = DataLoader.getFilesInRange(file_dir, low, high)
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
            words.append(curr_words)
            wtypes.append(curr_wtypes)

        return words, wtypes

        
