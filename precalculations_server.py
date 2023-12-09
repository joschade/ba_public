import json
import numpy as np
import json
from tensorflow import keras
import make_integrated_gradients_model as mig

model = keras.models.load_model('models/model_1/serve')
above_20_train = []
above_50_train = []
idx=0
for i in range(1, 8001):
        name = '/mnt/data_m2/intern02_data/object-cxr/data/train/' + str(i).zfill(5) + '.jpg'
        img_box = json.load(open(name + '.json'))
        if img_box.get("pages") is not None:
                idx = idx+1
                img_processed = mig.process_image(name)[0]
                if model(img_processed)['pred_obj'][0][1] > 0.2:
                        above_20_train.append(idx)
                if model(img_processed)['pred_obj'][0][1] > 0.5:
                        above_50_train.append(idx)


np.array(above_20_train).dump('above_20_train')
np.array(above_50_train).dump('above_50_train')


above_20_val = []
above_50_val = []
idx = 0
for i in range(8001, 9001):
        name = '/mnt/data_m2/intern02_data/object-cxr/data/val/' + str(i).zfill(5) + '.jpg'
        img_box = json.load(open(name + '.json'))
        if img_box.get("pages") is not None:
                idx = idx+1
                img_processed = mig.process_image(name)[0]
                if model(img_processed)['pred_obj'][0][1] > 0.2:
                        above_20_train.append(idx)
                if model(img_processed)['pred_obj'][0][1] > 0.5:
                        above_50_train.append(idx)

np.array(above_20_val).dump('above_20_val')
np.array(above_50_train).dump('above_50_val')
