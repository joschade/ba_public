import matplotlib.pyplot as plt
import boxed_image as bi
from matplotlib import patches
import evaluation as ev
import json
import numpy as np
from PIL import Image
import cv2
import evaluation as ev

# collect all pic names with boxes from val-batch
pics_ill_val = []
idx = 0
for i in range(1, 101):
    name = 'val100/0' + str(8000 + i) + '.jpg'
    img_box = json.load(open(name + '.json'))
    if img_box.get("pages") is not None:
        pics_ill_val.append([name, np.array(Image.open(name)).shape, idx])
        idx = idx + 1

# collect all pic names with boxes from train-batch
pics_ill_train = []
idx = 0
for i in range(1, 10):
    name = 'train100/0000' + str(i) + '.jpg'
    img_box = json.load(open(name + '.json'))
    if img_box.get("pages") is not None:
        pics_ill_train.append([name, np.array(Image.open(name)).shape, idx])
        idx = idx + 1
for i in range(10, 100):
    name = 'train100/000' + str(i) + '.jpg'
    img_box = json.load(open(name + '.json'))
    if img_box.get("pages") is not None:
        pics_ill_train.append([name, np.array(Image.open(name)).shape, idx])
        idx = idx + 1
name = 'train100/00100.jpg'
img_box = json.load(open(name + '.json'))
if img_box.get("pages") is not None:
    pics_ill_train.append([name, np.array(Image.open(name)).shape, idx])
    idx = idx + 1

intgrads_val = np.load('intgrads_val_gauss', allow_pickle=True)
intgrads_train = np.load('intgrads_train_gauss', allow_pickle=True)


batch = pics_ill_train

# recall for train-batch:

eval = ev.evaluate_pxl(pics_ill_train, pics_ill_val, intgrads_train, intgrads_val)
print(f'complete train: {eval}')

# Load pic-names with score over 0.8 (thus bigger score than baseline)
arr_under20_train= np.load('arr_under20_train', allow_pickle=True)
arr_under20_val = np.load('arr_under20_val', allow_pickle=True)

# Create array with pics from train with score under 0.8 AND boxes
pics_over80_train=[]
for pic in pics_ill_train:
    if pic[2] not in arr_under20_train:
        pics_over80_train.append(pic)

pics_over20_val=[]
for pic in pics_ill_val:
    if pic[2] not in arr_under20_val:
        pics_over20_val.append(pic)

# collecting all f1-scores for train<80

eval = ev.evaluate_pxl(pics_over80_train, pics_over20_val, intgrads_train, intgrads_val)
print(f'train>20: {eval}')

# Load pic-names with score over 0.5 (thus scored more likely healthy than ill)
arr_under50_train = np.load('arr_under50_train', allow_pickle=True)
arr_under50_val = np.load('arr_under50_val', allow_pickle=True)

# collecting all f1-scores for train>50
pics_under50_train=[]
for pic in pics_ill_train:
    if pic[2] not in arr_under50_train:
        pics_under50_train.append(pic)

pics_under50_val=[]
for pic in pics_ill_val:
    if pic[2] not in arr_under50_val:
        pics_under50_val.append(pic)

eval = ev.evaluate_pxl(pics_under50_train, pics_under50_val, intgrads_train, intgrads_val)
print(f'train<50: {eval}')




