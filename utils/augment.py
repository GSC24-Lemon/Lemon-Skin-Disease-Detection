
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import re
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
        #
        # if not os.path.exists(self.val_path):
        #     os.makedirs(self.val_path)

        # Create subfolders within self.train_path
        for folder in self.dataset['label'].unique():
            train_subfolder_path = os.path.join(self.train_path, folder)
            if not os.path.exists(train_subfolder_path):
                os.makedirs(train_subfolder_path)

        # # Create subfolders within self.val_path
        # for folder in subfolders:
        #     val_subfolder_path = os.path.join(self.val_path, folder)
        #     if not os.path.exists(val_subfolder_path):
        #         os.makedirs(val_subfolder_path)
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
        return reduced_dataset
    def generate_data(self) :
        generator=ImageDataGenerator(horizontal_flip=True,rotation_range=90,width_shift_range=2,height_shift_range=2)
        grouping= self.dataset.groupby("label")
        for type in self.dataset['label'].unique() :
            group_per_type=grouping.get_group(type)
            count=len(group_per_type)
            if count < self.size:
                augment_data=0
                delta=self.size-count
                loc=os.path.join(self.train_path,type)
                aug=generator.flow_from_dataframe(group_per_type,x_col='path',y_col=None,target_size=(224,224), class_mode=None,
                                                                  batch_size=1, shuffle=False,save_to_dir=loc, save_prefix='augmented_',
                                                                  color_mode='rgb',save_format='jpg')
                while augment_data < delta :
                    images=next(aug)
                    augment_data+=len(images)

    def load_augment_data(self):
        files=[]
        labels=[]
        for i in self.dataset['label'].unique() :
            loc = os.path.join(self.train_path,i)
            list_files=os.listdir(loc)
            for images in list_files :
                images=os.path.join(self.train_path,images)
                files.append(images)
                labels.append(i)
        files=pd.Series(files,name='path')
        labels=pd.Series(labels,name='label')
        dataset_aug_final=pd.concat([files,labels],axis=1)
        dataset_final=pd.concat([self.dataset,dataset_aug_final],axis=0).reset_index(drop=True)
        def fix_path(x):
            return re.sub(r'\\', '/', x)
        dataset_final['path'] = dataset_final['path'].apply(fix_path)
        return dataset_final


    # def create_generator(self):



