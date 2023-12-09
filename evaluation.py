import numpy as np
import boxed_image as bi
import cv2

# returns true when two rectangles have a intersection>0
def intersect(rect1, rect2):
    x = max(rect1[0], rect2[0])
    y = max(rect1[1], rect2[1])
    w = min(rect1[0] + rect1[2], rect2[0] + rect2[2]) - x
    h = min(rect1[1] + rect1[3], rect2[1] + rect2[3]) - y
    if w<=0 or h<=0: return False
    else: return True

#Transforms canonical coordinates in the from required by cv2.BoundingRect
def coord_to_cv2box(coord):
    return coord[0][0], coord[0][1], coord[2][0] - coord[0][0], coord[1][1] - coord[0][1]

# Transforms cv2.Bounding-Rect coordinates to the form required by pypplot.patches.Rectangel
def cv2box_to_rect(box):
    return (box[0], box[1]), box[2], box[3]


def precision(tp, fp):
    return tp / (tp + fp)

def recall(tp, fn):
    return tp / (tp + fn)

def f1(prec, rec):
    return 2*prec*rec/(prec+rec)

# returns true positives, false positives if cluster-bounding rects are used
# true-positives: #cluster-bounding rects, that intersect with one rect from json-file
# false-positives: # cluster-bounding rects, that do not intersect with one rect from json-file
def tpfp_box(batch, intgrads, percentile, thresh_area):
    n_boxes=0
    tp = 0
    for pic in batch:
        intgrad = intgrads[pic[2]]
        coords = bi.get_box_coords(pic[0], pic[1])
        perc = np.percentile(intgrad, percentile)
        intgrad = np.where(intgrad < perc, 0, 1)
        labels, intgrad = cv2.connectedComponents(np.uint8(intgrad), 8, cv2.CV_32S)
        contours = cv2.findContours(np.uint8(intgrad), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            box = cv2.boundingRect(contour)
            if box[2] * box[3] > thresh_area:
                n_boxes = n_boxes + 1
            for coord in coords:
                if box[2] * box[3] > thresh_area:
                    if intersect(coord_to_cv2box(coord), box) is True: # if rectangle (from json-file) and box (contour-boundig Rect) intersect ->true
                        tp= tp + 1
                        break
    fp=n_boxes - tp
    return (tp, fp, n_boxes)

# returns false-negatives
# false-negatvies= # json-boxes, that do not intersect with at least one cluster-bounding rect
def fn_box(batch, intgrads, percentile, thresh_area):
    n_rects=0
    fn=0
    for pic in batch:
        coords = bi.get_box_coords(pic[0], pic[1])
        intgrad=intgrads[pic[2]]
        perc = np.percentile(intgrad, percentile)
        intgrad = np.where(intgrad < perc, 0, 1)
        labels, intgrad = cv2.connectedComponents(np.uint8(intgrad), 8, cv2.CV_32S)
        contours = cv2.findContours(np.uint8(intgrad), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        n_rects=n_rects+len(coords)
        for coord in coords:
            hits = False
            for contour in contours:
                box = cv2.boundingRect(contour)
                if intersect(coord_to_cv2box(coord), box) is True and box[2] * box[3] > thresh_area:
                        hits = True
                        break
            if hits is False:
                fn = fn+1
    return (fn, n_rects)

# calculates accuracy (pic-wide), if cluster-bounding rect is used
def accuracy_box(batch, intgrads, percentile, thresh_area):
    n_pics_hit = 0
    n_pics = len(batch)
    for pic in batch:
        intgrad = intgrads[pic[2]]
        coords = bi.get_box_coords(pic[0], pic[1])
        perc = np.percentile(intgrad, percentile)
        intgrad = np.where(intgrad < perc, 0, 1)
        labels, intgrad = cv2.connectedComponents(np.uint8(intgrad), 8, cv2.CV_32S)
        contours = cv2.findContours(np.uint8(intgrad), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        for coord in coords:
            n = 0
            for c in contours:
                box = cv2.boundingRect(c)
                if box[2] * box[3] > thresh_area:
                    if intersect(box, coord_to_cv2box(coord)):
                        n = n + 1
            if n > 0:
                n_pics_hit = n_pics_hit + 1
                break
    return (n_pics_hit/n_pics, n_pics_hit, n_pics)


# returns true-positives, false-positives if top pixel is used.
# true-pos: # top-pixel in json-Box
# false-pos: # top-pixel not in json-Box
def tpfp_pxl(batch, intgrads ,percentile):
    tp=0
    n_boxes = 0
    for pic in batch:
        intgrad = intgrads[pic[2]]
        coords = bi.get_box_coords(pic[0], pic[1])
        perc = np.percentile(intgrad, percentile)
        intgrad = np.where(intgrad < perc, 0, 1)
        location = np.where(intgrad==1)
        for i in range(len(location[0])):
            n_boxes=n_boxes + len(location[0])
            for coord in coords:
                x,y,w,h = coord_to_cv2box(coord)
                if location[0][i] >= y and location[0][i] <=y+h and location[1][i]>=x and location[1][i] <=x+w:
                    tp=tp+1
                    break
    fp = n_boxes-tp
    return (tp, fp, n_boxes)

# returns false-negatives.
# false-neg: #json-Boxes with no top-pixel inside
def fn_pxl(batch, intgrads ,percentile):
    fn = 0
    n_rect=0
    for pic in batch:
        intgrad = intgrads[pic[2]]
        coords = bi.get_box_coords(pic[0], pic[1])
        perc = np.percentile(intgrad, percentile)
        intgrad = np.where(intgrad < perc, 0, 1)
        for coord in coords:
            n_rect=n_rect+1
            x,y,w,h = coord_to_cv2box(coord)
            if intgrad[y:y+h+1,x:x+w+2].sum() < 1:
                fn = fn+1
    return (fn, n_rect)

#returns accuracy (pic-wide), # hit pics, # pics if top-pixel used
def accuracy_pxl(batch, intgrads, percentile):
    n_pics_hit=0
    n_pics=0
    for pic in batch:
        n_pics=n_pics+1
        coords = bi.get_box_coords(pic[0], pic[1])
        intgrad=intgrads[pic[2]]
        perc = np.percentile(intgrad, percentile)
        intgrad_threshed = np.where(intgrad < perc, 0, 1)
        tmp=0
        for c in coords:
            if intgrad_threshed[c[0][1]:c[1][1] + 1, c[0][0]:c[2][0] + 1].sum() > 0:
                tmp=tmp+1
                break
        if tmp>0: n_pics_hit=n_pics_hit+1
    return  (n_pics_hit/n_pics, n_pics_hit, n_pics)


def evaluate_pxl(train, val, igrads_train, igrads_val, percentile=100):
    tp, fp, n_boxes = tpfp_pxl(train, igrads_train, percentile)
    prec = precision(tp, fp)
    acc_train = accuracy_pxl(train, igrads_train, percentile)
    acc_val = accuracy_pxl(val, igrads_val, percentile)
    acc_comb = (acc_val[1] + acc_train[1]) / (acc_val[2] + acc_train[2])
    return {'precision':prec, 'acc_train':acc_train[0], 'acc_val':acc_val[0], 'acc_comb':acc_comb}

def evaluate_cv2box(train, val, igrads_train, igrads_val, percentile=99, thresh_area=5, only_train=False):
    tp_train, fp_train, n_box_train = tpfp_box(train, igrads_train, percentile, thresh_area)
    fn_train, n_rects_train = fn_box(train, igrads_train, percentile, thresh_area)
    prec_train = precision(tp_train, fp_train)
    rec_train = recall(tp_train, fn_train)
    f1_train = 2*prec_train*rec_train/(prec_train+rec_train)
    if only_train: return {'prec_train': prec_train, 'rec_train': rec_train, 'f1_train': f1_train}
    if not only_train:
        tp_val, fp_val, n_box_val = tpfp_box(train, igrads_train, percentile, thresh_area)
        fn_val, n_rects_val = fn_box(val, igrads_val, percentile, thresh_area)
        prec_val = precision(tp_val, fp_val)
        rec_val = recall(tp_val, fn_val)
        f1_val = 2 * prec_val * rec_val / (prec_val + rec_val)

        prec_comb = precision(tp_train+tp_val, fp_train+fp_val)
        rec_comb = recall(tp_train+tp_val, fn_train+fp_val)
        f1_comb = 2*prec_comb*rec_comb/(prec_comb+rec_comb)

        return {
            'prec_train': prec_train,
            'rec_train': rec_train,
            'f1_train': f1_train,
            'prec_val':prec_val,
            'rec_val': rec_val,
            'f1_val': f1_val,
            'prec_comb': prec_comb,
            'rec_comb': rec_comb,
            'f1_comb': f1_comb
        }