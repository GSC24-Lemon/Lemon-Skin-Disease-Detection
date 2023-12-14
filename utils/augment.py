
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class AugmentData :
    def __init__(self,train_data,train_aug_path,val_aug_path,size):
        self.train_path=train_aug_path
        self.val_path=val_aug_path
        self.size=size
        self.dataset=train_data
        # self.make_folder=self.make_folder()
    def make_folder(self):
        if not os.path.exists(self.train_path):
            os.makedirs(self.train_path)

        if not os.path.exists(self.val_path):
            os.makedirs(self.val_path)

        # Create subfolders within self.train_path
        for folder in subfolders:
            train_subfolder_path = os.path.join(self.train_path, folder)
            if not os.path.exists(train_subfolder_path):
                os.makedirs(train_subfolder_path)

        # Create subfolders within self.val_path
        for folder in subfolders:
            val_subfolder_path = os.path.join(self.val_path, folder)
            if not os.path.exists(val_subfolder_path):
                os.makedirs(val_subfolder_path)
    def reduce_dataset(self):
        reduce=[]
        grouping=self.dataset.groupby("label")
        for i in self.dataset['label'].unique() :
            group_per_type=grouping.get_group(i)
            count=len(group_per_type)
            if count > self.size :
                sampling=group_per_type.sample(self.size,replace=False,weights=None, random_state=42,axis=0).reset_index(drop=True)
                reduce.append(sampling)
            else :
                reduce.append(group_per_type)
        reduced_dataset=pd.concat(reduce,axis=0).reset_index(drop=True)
        self.dataset=reduced_dataset
        # return reduced_dataset
    def create_generator(self):



