import numpy as np
import integrated_gradients as ig
import copy
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

def process_image(path_to_img):
    img = Image.open(path_to_img)
    img = np.array(img)
    img_shape = img.shape
    img = cv2.resize(img, dsize=(512, 512))
    img = img / 255
    img = img.reshape(img.shape[0], img.shape[1], 1)
    return ({'img': np.asarray([img]), 'img_shape': np.asarray([img.shape])}, img_shape)

def get_igrads(model, img_processed, num_steps = 50, random=False, num_runs=2, top_pred_idx=1):
    'input: trained model, img of size ( n, n, 1), bool random'
    'returns image of size (1, n, n, 1)'

    img_size = img_processed['img_shape'][0]
    if not random:
        igrads = ig.get_integrated_gradients(
            model, copy.deepcopy(img_processed), top_pred_idx=top_pred_idx, num_steps=num_steps, size=img_size
         )

    else:
        igrads = ig.random_baseline_integrated_gradients(
            model, copy.deepcopy(img_processed), top_pred_idx=top_pred_idx, num_steps=num_steps, num_runs=num_runs, size=img_size
        )
    return igrads

# def include_box(img_overlay, path_to_img, img_shape, save = True):
#     fig, ax = plt.subplots(1, 1)
#     ax.imshow(img_overlay, cmap='gray')
#
#     img_box = json.load(open(path_to_img+'.json'))
#
#     if img_box.get("pages") is not None:
#         img_box = img_box['pages'][0]['regions']
#
#         box_coords = []
#         for i in range(len(img_box)):
#             img_box[i] = img_box[i]['coordinates'].split(';')
#
#             box_coords.append([])
#             for j in range(4):
#                 box_coords[i].append(img_box[i][j].split(','))
#                 box_coords[i][j] = (
#                     int(float(box_coords[i][j][0]) * 512.0 / img_shape[1]),
#                     int(float(box_coords[i][j][1]) * 512.0 / img_shape[0]))
#
#             rect = patches.Rectangle(box_coords[i][0], box_coords[i][1][1] - box_coords[i][0][1],
#                                      box_coords[i][2][0] - box_coords[i][1][0], linewidth=1, edgecolor='r',
#                                      facecolor='none')
#             ax.add_patch(rect)
#
# def get_box_coords(path_to_img, img_shape):
#     img_box = json.load(open(path_to_img + '.json'))
#     if img_box.get("pages") is not None:
#         img_box = img_box['pages'][0]['regions']
#
#         box_coords = []
#         for i in range(len(img_box)):
#             img_box[i] = img_box[i]['coordinates'].split(';')
#
#             box_coords.append([])
#             for j in range(4):
#                 box_coords[i].append(img_box[i][j].split(','))
#                 box_coords[i][j] = (
#                     int(float(box_coords[i][j][0]) * 512.0 / img_shape[1]),
#                     int(float(box_coords[i][j][1]) * 512.0 / img_shape[0]))
#     return box_coords





