from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
import time
import csv
import glob
config = Cfg.load_config_from_name('vgg_transformer')
config['export'] = 'transformerocr_checkpoint.pth'
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False



detector = Predictor(config)


with open('data.csv', 'r') as csvinput:
    with open('output.csv', 'w',encoding='utf-8') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)
        all = []
        row = next(reader)
        row.append('8')
        all.append(row)
        i = 0
        path = glob.glob("cropWord/*.jpg")
        cv_img = []
        for i in range(len(path)):
            print('cropWord/999'+str(i)+'.jpg')
            n = Image.open('cropWord/999'+str(i)+'.jpg')
            print(str(detector.predict(n)))
            cv_img.append(str(detector.predict(n)))
        i = 0
        for row in reader:
            row.append(str(cv_img[i]))
            i = i + 1
            all.append(row)

        writer.writerows(all)


