import gradvisualizer as vis
import make_integrated_gradients_model as mig
import json
import matplotlib.pyplot as plt
from matplotlib import patches


# adds json-Boxes to overlay
def include_box(img_overlay, path_to_img, img_shape, save = True):
    fig, ax = plt.subplots(1, 1)
    ax.matshow(img_overlay, cmap='gray')
    img_box = json.load(open(path_to_img+'.json'))

    if img_box.get("pages") is not None:  # Only none if there are no coordinates
        img_box = img_box['pages'][0]['regions']

        box_coords = []
        for i in range(len(img_box)):
            img_box[i] = img_box[i]['coordinates'].split(';')

            box_coords.append([])
            for j in range(4):
                box_coords[i].append(img_box[i][j].split(','))
                box_coords[i][j] = (
                    int(float(box_coords[i][j][0]) * 512.0 / img_shape[1]),
                    int(float(box_coords[i][j][1]) * 512.0 / img_shape[0]))

            rect = patches.Rectangle(box_coords[i][0], box_coords[i][2][0] - box_coords[i][1][0],
                                     box_coords[i][1][1] - box_coords[i][0][1],
                                      linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)
            if save is True: plt.savefig(path_to_img[0:-4]+'boxed.png')

# Extracting box-coordinates from json-file
def get_box_coords(path_to_img, img_shape):
    img_box = json.load(open(path_to_img + '.json'))
    if img_box.get("pages") is not None:
        img_box = img_box['pages'][0]['regions']

        box_coords = []
        for i in range(len(img_box)):
            img_box[i] = img_box[i]['coordinates'].split(';')

            box_coords.append([])
            for j in range(4):
                box_coords[i].append(img_box[i][j].split(','))
                box_coords[i][j] = (
                    int(float(box_coords[i][j][0]) * 512.0 / img_shape[1]),
                    int(float(box_coords[i][j][1]) * 512.0 / img_shape[0]))
    return box_coords

# def get_boxed_img(
#         path_to_img,
#         model,
#         num_steps = 20,
#         clip_below_percentile = 10,
#         morphological_cleanup = True,
#         random = False,
#         save = False,
#         picname = "pic"):
#
#     (img_processed, img_shape) = mig.process_image(path_to_img)
#     igrads = mig.get_igrads(model, img_processed, num_steps=num_steps, random=random)
#     pic = vis.GradVisualizer()
#     overlay = pic.process_grads(img_processed['img'][0],
#                                 igrads[0].numpy(),
#                                 morphological_cleanup=morphological_cleanup,
#                                 clip_below_percentile=clip_below_percentile)
#     include_box(overlay, path_to_img, img_shape)
#
#     if save:
#         plt.savefig(picname + ".png")