# from utils.preprocessing import Preprocessing
# import matplotlib.pyplot as plt
# path='dataset/dataset/AKIEC/ISIC_0024329.jpg'
# label='dataset/GroundTruth.csv'
# preprocess=Preprocessing(path,label)
# prep=preprocess.process_all()


# data loader
from utils.data_loader import DataLoader
path="dataset/processed_dataset"
size=0.2
Loader=DataLoader(path,size)
a,b=Loader.split_data()
print(len(a),a,len(b))