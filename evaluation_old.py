import json
import numpy as np
from PIL import Image
import make_integrated_gradients_model as mig
from tensorflow import keras
import gradvisualizer as gv
import boxed_image as bi
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import savetxt
import cv2


# model = keras.models.load_model('models/model_1/serve')

# Save all ill pics of val-batch:
pics_ill_val = []
idx=0
for i in range(1, 101):
    name = 'val100/0'+str(8000+i)+'.jpg'
    img_box = json.load(open(name+'.json'))
    if img_box.get("pages") is not None:
        pics_ill_val.append([name, np.array(Image.open(name)).shape, idx])
        idx = idx + 1

# save all pics of train-batch:
pics_ill_train = []
idx=0
for i in range(1, 10):
    name = 'train100/0000'+str(i)+'.jpg'
    img_box = json.load(open(name+'.json'))
    if img_box.get("pages") is not None:
        pics_ill_train.append([name, np.array(Image.open(name)).shape, idx])
        idx = idx + 1
for i in range(10, 100):
    name = 'train100/000'+str(i)+'.jpg'
    img_box = json.load(open(name+'.json'))
    if img_box.get("pages") is not None:
        pics_ill_train.append([name, np.array(Image.open(name)).shape, idx])
        idx = idx + 1
name = 'train100/00100.jpg'
img_box = json.load(open(name+'.json'))
if img_box.get("pages") is not None:
    pics_ill_train.append([name, np.array(Image.open(name)).shape, idx])
    idx = idx + 1

# Save filtered Integrated Gradients of val-batch:
# intgrads_val = []
# for pic in pics_ill_val:
#     img_processed = mig.process_image(pic[0])[0]
#     igrads = mig.get_igrads(model, img_processed)
#     vis = gv.GradVisualizer()
#     intgrads_val.append(vis.process_grads(img_processed['img'][0],
#                                           igrads[0],
#                                           morphological_cleanup=True,
#                                           clip_below_percentile=10,
#                                           overlay=False
#                                           ))
#
# intgrads_val = np.array(intgrads_val)
# intgrads_val.dump('intgrads_val')
intgrads_val_filtered=np.load('intgrads_val', allow_pickle=True)
intgrads_val_filtered.dump('intgrads_val_filtered')
test = np.load('intgrads_val_filtered', allow_pickle=True)
# save filtered intgrads for train-batch:
# intgrads_train = []
# for pic in pics_ill_train:
#     img_processed = mig.process_image(pic[0])[0]
#     igrads = mig.get_igrads(model, img_processed)
#     vis = gv.GradVisualizer()
#     intgrads_train.append(vis.process_grads(img_processed['img'][0],
#                                          igrads[0],
#                                          morphological_cleanup=True,
#                                          clip_below_percentile=10,
#                                          overlay=False
#                                          ))
#
# intgrads_train = np.array(intgrads_train)
# intgrads_train.dump('intgrads_train')
intgrads_train=np.load('intgrads_train_filtered', allow_pickle=True)

#
# for i in range(1, 101):
#     if i<10:
#         name='train100/0000'+str(i)+'.jpg'
#     elif i<100:
#         name = 'train100/000' + str(i) + '.jpg'
#     else:
#         name = 'train100/00' + str(i) + '.jpg'
#     img_box = json.load(open(name+'.json'))
#     if img_box.get("pages") is not None:
#         pics_ill.append([name, np.array(Image.open(name)).shape])


### counting colored pixels in box for val-batch:
thresh_area=0.1
thresh_pxl = 0.2


eval_pxl_val = []           # array containing every box>thresh (1st dimesnsion) for every picture (0th dimension)
total_rects_val=0       # count n_total of rects
for p in pics_ill_val:
    coords = bi.get_box_coords(p[0], p[1])
    eval_pic_pxl=[]
    for c in coords:
        n=0
        for i in range(c[0][0], c[3][0]+1):
            for j in range(c[0][1],c[1][1]+1):
                if np.all(np.array(intgrads_val[0][i,j][1]) >= thresh_pxl):
                    n=n+1
        total_rects_val = total_rects_val + 1
        if n/((c[3][0]-c[0][0]+1)*(c[1][1]-c[0][1]+1)) >= thresh_area:
            eval_pic_pxl.append(n / ((c[3][0] - c[0][0]+1) * (c[1][1] - c[0][1]+1)))
    eval_pxl_val.append(eval_pic_pxl)

recog_pics_pxl_val = 0
recog_rects_pxl_val= 0
for e in eval_pxl_val:
    if e != []:
        recog_pics_pxl_val = recog_pics_pxl_val + 1
        recog_rects_pxl_val = recog_rects_pxl_val + len(e)

print(f'val-batch. Total # pics:{len(pics_ill_val)}, Threshold (area): {thresh_area}, '
      f'# recognizes pics: {recog_pics_pxl_val}, total # boxes: {total_rects_val}, '
      f'# recognized boxes: {recog_rects_pxl_val}')
### counting colored pixels in box for train-batch:
eval_pxl_train = []
total_rects_train=0   # array containing every box>thresh (1st dimesnsion) for every picture (0th dimension)
for p in pics_ill_train:
    coords = bi.get_box_coords(p[0], p[1])
    eval_pic_pxl=[]
    for c in coords:
        n=0
        for i in range(c[0][0], c[3][0]+1):
            for j in range(c[0][1],c[1][1]+1):
                if np.all(np.array(intgrads_train[0][i,j][1]) >= thresh_pxl):
                    n=n+1
        total_rects_train = total_rects_train + 1
        if n/((c[3][0]-c[0][0]+1)*(c[1][1]-c[0][1]+1)) >= thresh_area:
            eval_pic_pxl.append(n / ((c[3][0] - c[0][0]+1) * (c[1][1] - c[0][1]+1)))
    eval_pxl_train.append(eval_pic_pxl)

recog_pics_pxl_train = 0
recog_rects_pxl_train= 0
for e in eval_pxl_train:
    if e != []:
        recog_pics_pxl_train = recog_pics_pxl_train + 1
        recog_rects_pxl_train = recog_rects_pxl_train + len(e)

print(f'train-batch. Total # pics:{len(pics_ill_train)}, Threshold (area): {thresh_area}, '
      f'# recognizes pics: {recog_pics_pxl_train}, total # boxes: {total_rects_train}, '
      f'# recognized boxes: {recog_rects_pxl_train}')

# evaluation with cv2-boxes

thresh_box = 10
connectivity=8


def center_of_cv2box(box):
    return (box[0]+0.5*box[2], box[1]+0.5*box[3])


def point_in_rect(rect,point):
    if rect[2][0] >= point[0] and rect[0][0]<= point[0] and rect[0][1]<=point[1] and rect[1][1]>= point[1]:
        return True
    else: return False


intgrads_boxes_val=[]
for intgrad in intgrads_val:
    ig_gray = cv2.cvtColor(np.float32(intgrad), cv2.COLOR_BGR2GRAY)
    ig_thresh = cv2.threshold(ig_gray, 0.2, 255, cv2.THRESH_BINARY)[1]
    ig_thresh = cv2.connectedComponents(np.uint8(ig_thresh), 8, cv2.CV_32S)
    contours = cv2.findContours(np.uint8(ig_thresh[1]), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    tmp = []
    for c in contours:
        box = cv2.boundingRect(c)
        if box[2]> thresh_box and box[3]>thresh_box:
            tmp.append(box)
    intgrads_boxes_val.append(tmp)

eval_box = []
for i in range(len(pics_ill_val)):
    coords = bi.get_box_coords(pics_ill_val[i][0], pics_ill_val[i][1])
    eval_pic_box = 0
    for c in coords:
        for box in intgrads_boxes_val[i]:
            if point_in_rect(c, center_of_cv2box(box)):
                eval_pic_box = eval_pic_box + 1
                break
    eval_box.append(eval_pic_box)

recog_rects_box_val = sum(eval_box)

while 0 in eval_box: eval_box.remove(0)

recog_pics_box_val= len(eval_box)

print(f'val-batch. Total # pics:{len(pics_ill_val)},  threshol (box lenght/width): {thresh_box}'
      f'# recognizes pics: {recog_pics_pxl_val}, total # boxes: {total_rects_val}, '
      f'# recognized boxes: {recog_rects_pxl_val}')

intgrads_boxes_train=[]
for intgrad in intgrads_train:
    ig_gray = cv2.cvtColor(np.float32(intgrad), cv2.COLOR_BGR2GRAY)
    ig_thresh = cv2.threshold(ig_gray, 0.2, 255, cv2.THRESH_BINARY)[1]
    ig_thresh = cv2.connectedComponents(np.uint8(ig_thresh), 8, cv2.CV_32S)
    contours = cv2.findContours(np.uint8(ig_thresh[1]), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    tmp = []
    for c in contours:
        box = cv2.boundingRect(c)
        if box[2]> thresh_box and box[3]>thresh_box:
            tmp.append(box)
    intgrads_boxes_train.append(tmp)

eval_box = []
for i in range(len(pics_ill_train)):
    coords = bi.get_box_coords(pics_ill_train[i][0], pics_ill_train[i][1])
    eval_pic_box = 0
    for c in coords:
        for box in intgrads_boxes_train[i]:
            if point_in_rect(c, center_of_cv2box(box)):
                eval_pic_box = eval_pic_box + 1
                break
    eval_box.append(eval_pic_box)

recog_rects_box_train = sum(eval_box)

while 0 in eval_box: eval_box.remove(0)

recog_pics_box_train= len(eval_box)

print(f'val-batch. Total # pics:{len(pics_ill_train)},  threshol (box lenght/width): {thresh_box}'
      f'# recognizes pics: {recog_pics_pxl_train}, total # boxes: {total_rects_train}, '
      f'# recognized boxes: {recog_rects_pxl_train}')