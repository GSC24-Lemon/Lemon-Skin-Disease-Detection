import cv2
import pandas as pd
class Preprocessing :
    def __init__(self,path,dataset_label):
        self.path=path
        self.label=dataset_label
    def process(self):
        gambar = cv2.imread(self.path,
                            cv2.IMREAD_COLOR)
        resize = cv2.resize(gambar, [224, 224])
        grayScale = cv2.cvtColor(resize, cv2.COLOR_RGB2GRAY)
        # Black hat filter
        kernel = cv2.getStructuringElement(1, (9, 9))
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
        # Gaussian blur
        bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
        # masking
        ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
        # Replace pixels of the mask
        dst = cv2.inpaint(resize, mask, 6, cv2.INPAINT_TELEA)
        return dst
    def process_all(self):
        classes=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        data=pd.read_csv(self.label)
        for cls in classes :
            images=data[data[cls]==1]['image'].tolist()
            for image in images :
                self.path=f"dataset/images/{image}.jpg"
                gambar=self.process()
                cv2.imwrite(f"dataset/processed_dataset/{cls}/{image}.jpg",gambar)
        print("All Images Have Been Processed")


