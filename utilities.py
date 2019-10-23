import glob
import os

class DataLoader:
    @staticmethod
    def getFileList(file_dir):
            result = sorted(glob.glob(file_name_string))
            return result

    @staticmethod
    def getFiles(file_dir):
        all_files = DataLoader.getFileList(file_name_string)
        result=[]
        for fname in all_files:
            with open(file_dir + '/' + fname) as f:
                result.append(f.read())
        return result