from tensorflow import keras
import boxed_image as bi
import integrated_gradients as ig

model = keras.models.load_model('models/model_1/serve')

bi.get_boxed_img('val100/08023.jpg',
                 model,
                 save=False,
                 num_steps=20,
                 morphological_cleanup=True,
                 clip_below_percentile=10)

