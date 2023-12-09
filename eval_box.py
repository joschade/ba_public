import evaluation as ev
import json
import numpy as np
from PIL import Image


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

# load pre-calculated gauss-filtered intgrads
intgrads_val = np.load('intgrads_val_gauss', allow_pickle=True)
intgrads_train = np.load('intgrads_train_gauss', allow_pickle=True)


# set batch to pics from train-batch with boxes
batch = pics_ill_train

# collecting all f1-scores for train
print(f'Maximizing f1-score for train:')
batch = pics_ill_train
f1scores=[]
for percentile in [70,75,80,85,90,95,99, 99.3, 99.5, 99.7, 99.9]:
    for thresh_area in [15, 20, 30, 40,50, 60, 70, 80 ,100]:
        tp, fp, n_box = ev.tpfp_box(batch, intgrads_train, percentile, thresh_area)
        fn, n_rects = ev.fn_box(batch, intgrads_train, percentile, thresh_area)
        prec = ev.precision(tp, fp)
        rec = ev.recall(tp, fn)
        f1 = ev.f1(prec, rec)

        f1scores.append([f1, percentile, thresh_area])
# find max f1:
f1scores = np.array(f1scores)
max_f1 = f1scores[:,0].max()
index = np.where(f1scores[:,0] == max_f1)[0][0]
perc, area = f1scores[index, 1:3]

print(f'max f1-Score is {max_f1}, percentile is {perc}, thresh_area is {area}')
tp, fp, n_box = ev.tpfp_box(batch, intgrads_train, perc, area)
fn, n_rects = ev.fn_box(batch, intgrads_train, perc, area)
prec = ev.precision(tp, fp)
rec = ev.recall(tp, fn)
print(f'precision for train is {prec}, recall is {rec}')
acc_val, tp_val, n_val = ev.accuracy_box(pics_ill_val, intgrads_val, perc, area)
acc_train, tp_train, n_train = ev.accuracy_box(pics_ill_train, intgrads_train, perc, area)
acc_comb = (tp_val+tp_train)/(n_val+n_train)
print(f'acc_val={acc_val}, acc_train={acc_train}, acc_comb={acc_comb}')

# Loading saved array with picnames that have higher score than baseline (appr. 0.8)
arr_over80s_train = np.load('arr_over80s_train', allow_pickle=True)
arr_over80s_val = np.load('arr_over80s_val', allow_pickle=True)
#
# pics_under80_train=[]
# for pic in pics_ill_train:
#     if pic[2] not in arr_over80s_train:
#         pics_under80_train.append(pic)
#
# pics_under80_val=[]
# for pic in pics_ill_val:
#     if pic[2] not in arr_over80s_val:
#         pics_under80_val.append(pic)
#
# # collecting all f1-scores for train<80
# print(f'Maximizing f1-score for train<80:')
# batch = pics_under80_train
# f1scores=[]
# for percentile in [70,75,80,85,90,95,99, 99.3, 99.5, 99.7, 99.9]:
#     for thresh_area in [1, 5, 10, 15, 20, 30, 40]:
#         tp, fp, n_box = ev.tpfp_box(batch, intgrads_train, percentile, thresh_area)
#         fn, n_rects = ev.fn_box(batch, intgrads_train, percentile, thresh_area)
#         prec = ev.precision(tp, fp)
#         rec = ev.recall(tp, fn)
#         f1 = ev.f1(prec, rec)
#
#         f1scores.append([f1, percentile, thresh_area])
# # find max f1:
# f1scores = np.array(f1scores)
# max_f1 = f1scores[:,0].max()
# index = np.where(f1scores[:,0] == max_f1)[0][0]
# perc, area = f1scores[index, 1:3]
#
# print(f'max f1-Score is {max_f1}, percentile is {perc}, thresh_area is {area}')
# tp, fp, n_box = ev.tpfp_box(batch, intgrads_train, perc, area)
# fn, n_rects = ev.fn_box(batch, intgrads_train, perc, area)
# prec = ev.precision(tp, fp)
# rec = ev.recall(tp, fn)
# print(f'precision is {prec}, recall is {rec}')
#
# acc_val, tp_val, n_val = ev.accuracy_box(pics_under80_val, intgrads_val, perc, area)
# acc_train, tp_train, n_train = ev.accuracy_box(pics_under80_train, intgrads_train, perc, area)
# acc_comb = (tp_val+tp_train)/(n_val+n_train)
#
# print(f'acc_val={acc_val}, acc_train={acc_train}, acc_comb={acc_comb}')
#
# Loading saved array with picnames that have higher score for ill than healthy (0.5)
arr_over50s_train = np.load('arr_over50s_train', allow_pickle=True)
arr_over50s_val = np.load('arr_over50s_val', allow_pickle=True)
#
# # collecting all f1-scores for train<50
# pics_under50_train=[]
# for pic in pics_ill_train:
#     if pic[2] not in arr_over50s_train:
#         pics_under50_train.append(pic)
#
# pics_under50_val=[]
# for pic in pics_ill_val:
#     if pic[2] not in arr_over50s_val:
#         pics_under50_val.append(pic)
#
# print(f'Maximizing f1-score for train<50:')
# batch = pics_under50_train
# f1scores=[]
# for percentile in [70,75,80,85,90,95,99, 99.3, 99.5, 99.7, 99.9]:
#     for thresh_area in [1, 5, 10, 15, 20, 30, 40]:
#         tp, fp, n_box = ev.tpfp_box(batch, intgrads_train, percentile, thresh_area)
#         fn, n_rects = ev.fn_box(batch, intgrads_train, percentile, thresh_area)
#         prec = ev.precision(tp, fp)
#         rec = ev.recall(tp, fn)
#         f1 = ev.f1(prec, rec)
#
#         f1scores.append([f1, percentile, thresh_area])
#
# # find max f1:
# f1scores = np.array(f1scores)
# max_f1 = f1scores[:,0].max()
# index = np.where(f1scores[:,0] == max_f1)[0][0]
# perc, area = f1scores[index, 1:3]
#
# print(f'max f1-Score is {max_f1}, percentile is {perc}, thresh_area is {area}')
# tp, fp, n_box = ev.tpfp_box(batch, intgrads_train, perc, area)
# fn, n_rects = ev.fn_box(batch, intgrads_train, perc, area)
# prec = ev.precision(tp, fp)
# rec = ev.recall(tp, fn)
# print(f'precision is {prec}, recall is {rec}')
# acc_val, tp_val, n_val = ev.accuracy_box(pics_under50_val, intgrads_val, perc, area)
# acc_train, tp_train, n_train = ev.accuracy_box(pics_under50_train, intgrads_train, perc, area)
# acc_comb = (tp_val+tp_train)/(n_val+n_train)
# print(f'acc_val={acc_val}, acc_train={acc_train}, acc_comb={acc_comb}')


# Calculate scores for pics with scores OVER 0.8
pics_over80_train=[]
for pic in pics_ill_train:
    if pic[2] in arr_over80s_train:
        pics_over80_train.append(pic)

pics_over80_val=[]
for pic in pics_ill_val:
    if pic[2] in arr_over80s_val:
        pics_over80_val.append(pic)

tp_train, fp_train, n_box_train = ev.tpfp_box(pics_over80_train, intgrads_train, 99.5, 70)
fn_train, n_rects_train = ev.fn_box(pics_over80_train, intgrads_train, 99.5, 70)
tp_val, fp_val, n_box_val = ev.tpfp_box(pics_over80_val, intgrads_val, 99.5, 70)
fn_val, n_rects_val = ev.fn_box(pics_over80_val, intgrads_val, 99.5, 70)
prec = ev.precision(tp_train+tp_val, fp_train+fp_val)
rec = ev.recall(tp_train+tp_val, fn_train+fn_val)
f1 = ev.f1(prec ,rec)
acc_train, tp_train, n_train = ev.accuracy_box(pics_over80_train, intgrads_train, 99.5, 70)
acc_val, tp_val, n_val = ev.accuracy_box(pics_over80_val, intgrads_val, 99.5, 70)
acc_comb = (tp_val+tp_train)/(n_val+n_train)
print(f'>0.8: f1={f1}, precision={prec}, recall={rec}, accuracy = {acc_comb}')

# Calculate scores for pics with scores OVER 0.5
pics_over50_train=[]
for pic in pics_ill_train:
    if pic[2] in arr_over50s_train:
        pics_over50_train.append(pic)

pics_over50_val=[]
for pic in pics_ill_val:
    if pic[2] in arr_over50s_val:
        pics_over50_val.append(pic)

tp_train, fp_train, n_box_train = ev.tpfp_box(pics_over50_train, intgrads_train, 99.5, 70)
fn_train, n_rects_train = ev.fn_box(pics_over50_train, intgrads_train, 99.5, 70)
tp_val, fp_val, n_box_val = ev.tpfp_box(pics_over50_val, intgrads_val, 99.5, 70)
fn_val, n_rects_val = ev.fn_box(pics_over50_val, intgrads_val, 99.5, 70)
prec = ev.precision(tp_train+tp_val, fp_train+fp_val)
rec = ev.recall(tp_train+tp_val, fn_train+fn_val)
f1 = ev.f1(prec ,rec)
acc_train, tp_train, n_train = ev.accuracy_box(pics_over50_train, intgrads_train, 99.5, 70)
acc_val, tp_val, n_val = ev.accuracy_box(pics_over50_val, intgrads_val, 99.5, 70)
acc_comb = (tp_val+tp_train)/(n_val+n_train)
print(f'>0.5: f1={f1}, precision={prec}, recall={rec}, accuracy = {acc_comb}')