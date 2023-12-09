import matplotlib.pyplot as plt
import boxed_image as bi
from matplotlib import patches
import evaluation as ev
import json
import numpy as np
from PIL import Image
import cv2

pics_ill_val = []
idx = 0
for i in range(1, 101):
    name = 'val100/0' + str(8000 + i) + '.jpg'
    img_box = json.load(open(name + '.json'))
    if img_box.get("pages") is not None:
        pics_ill_val.append([name, np.array(Image.open(name)).shape, idx])
        idx = idx + 1

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

# collecting all f1-scores for train
print(f' Maximizing f1-score for train:')
batch = pics_ill_train
f1scores=[]
for percentile in [70,75,80,85,90,95,99, 99.3, 99.5, 99.7, 99.9]:
    for thresh_area in [1, 5, 10, 15, 20, 30, 40]:
        eval =ev.evaluate_cv2box(
            pics_ill_train,
            pics_ill_val,
            intgrads_train,
            intgrads_val,
            percentile,
            thresh_area, only_train=True
        )
        f1 = eval['f1_train']

        f1scores.append([f1, percentile, thresh_area])

f1scores = np.array(f1scores)
max_f1 = f1scores[:,0].max()
index = np.where(f1scores[:,0] == max_f1)[0][0]
perc, area = f1scores[index, 1:3]

print(f'max f1-Socre is {max_f1}, percentile is {perc}, thresh_area is {area}')
eval = ev.evaluate_cv2box(pics_ill_train, pics_ill_val, intgrads_train, intgrads_val, perc, area)
print(eval)

# print(f'max f1-Socre is {max_f1}, percentile is {perc}, thresh_area is {area}')
pic = pics_ill_train[5]
intgrads = intgrads_train
percentile=95
thresh_area = 30
batch = [pic]

print(ev.tpfp_box(batch, intgrads, percentile, thresh_area))
print(ev.fn_box(batch, intgrads, percentile, thresh_area))


# intgrad = intgrads[pic[2]]
# intgrad=intgrads[pic[2]]
# coords = bi.get_box_coords(pic[0], pic[1])
# coords = bi.get_box_coords(pic[0], pic[1])
#
# perc = np.percentile(intgrad, percentile)
# perc = np.percentile(intgrads, percentile)
#
# intgrad = np.where(intgrad < perc, 0, 1)
# fig, ax = plt.subplots(1,1)
# ax.imshow(intgrad)
# labels, intgrad = cv2.connectedComponents(np.uint8(intgrad), 8, cv2.CV_32S)
# contours = cv2.findContours(np.uint8(intgrad), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
# for coord in coords:
#     box = ev.coord_to_cv2box(coord)
#     rect = patches.Rectangle((box[0], box[1]), box[2],
#                              box[3],
#                              linewidth=1, edgecolor='r',
#                              facecolor='none')
#     ax.add_patch(rect)
#
# n_contour=0
# for contour in contours:
#     box = cv2.boundingRect(contour)
#     if box[2]*box[3]>thresh_area:
#         n_contour=n_contour+1
#         box = patches.Rectangle(*ev.cv2box_to_rect(box),
#                                  linewidth=1, edgecolor='b',
#                                  facecolor='none')
#         ax.add_patch(box)
#
#
# for contour in contours:
#     box = cv2.boundingRect(contour)
#     if ev.intersect(box, ev.coord_to_cv2box(coords[4])) is True:
#         print('True')
#         break

