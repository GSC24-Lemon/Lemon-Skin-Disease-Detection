import pandas as pd
import shutil
class ConvertData :
    def __init__(self,data_path):
        self.data_path=data_path
    def convert(self):
        annot=pd.read_csv(self.data_path)
        cls=annot.columns[1:]
        for i in cls :
            imgs_name=annot[annot[i]==1]['image'].to_list()
            for j in imgs_name :
                shutil.copy(f"dataset/images/{j}.jpg",
                           f"dataset/dataset/{i}" )
