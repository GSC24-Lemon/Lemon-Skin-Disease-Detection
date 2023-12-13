import pandas as pd
import os
from sklearn.model_selection import train_test_split
class DataLoader :
    def __init__(self,data_path,split_size):
        self.data_path=data_path
        self.split_size=split_size
        self.classes=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    def split_data(self):
        files=[]
        labels=[]
        for cls in self.classes :
            path=os.path.join(self.data_path,cls)
            list_file=os.listdir(path)
            for file in list_file :
                file_path=os.path.join(path,file)
                files.append(file_path)
                labels.append(cls)
        filepath=pd.Series(files,name='path')
        labelpath=pd.Series(labels,name='label')
        dataset=pd.concat([filepath,labelpath],axis=1)
        strat = dataset['label']
        train_dataset, test_valid_dataset = train_test_split(dataset, test_size=self.split_size, shuffle=True, random_state=42,
                                                             stratify=strat)
        return train_dataset,test_valid_dataset


