import json
import numpy as np
import make_integrated_gradients_model as mig
from tensorflow import keras
from scipy.ndimage import gaussian_filter

model = keras.models.load_model('models/model_1/serve')

highes_index=4000

pics_ill_train=[]
igrads_gauss_train =[]

idx=0
for i in range(3001,highes_index+1):
        name = '/mnt/data_m2/intern02_data/object-cxr/data/train/'+str(i).zfill(5)+'.jpg'
        img_box = json.load(open(name + '.json'))
        if img_box.get("pages") is not None:
                idx = idx + 1
                img_processed = mig.process_image(name)[0]
                igrads_gauss_train.append(gaussian_filter(mig.get_igrads(model, img_processed)[0],4))


np.array(igrads_gauss_train).dump('igrads_gauss_train_'+str(highes_index))
