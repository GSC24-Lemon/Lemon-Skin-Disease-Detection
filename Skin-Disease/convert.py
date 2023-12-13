from utils.data_converter import ConvertData
label_path='dataset/GroundTruth.csv'
convert_data=ConvertData(label_path)
convert_data.convert()