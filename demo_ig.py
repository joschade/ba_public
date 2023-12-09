from tensorflow import keras
import make_integrated_gradients_model as mig
import matplotlib.pyplot as plt
import gradvisualizer as gvis
import copy
import integrated_gradients as ig
import make_integrated_gradients_model as mig
import boxed_image as bi

path_to_img = 'val100/08023.jpg'
img_processed = mig.process_image(path_to_img)[0]

# img_box = bi.include_box(img_processed['img'][0], path_to_img, (512,512))

model = keras.models.load_model('models/model_1/serve')

bi.get_boxed_img(path_to_img, model)
#
#
#
# igrads = mig.get_igrads(model, img_processed, top_pred_idx=0, random=True)
#
# plt.imshow(igrads[0])
# plt.savefig('igrads_not_processed.png')
#
# vis = gvis.GradVisualizer()
# igrads_processed = vis.process_grads(img_processed['img'][0],
#                                      igrads[0],
#                                      morphological_cleanup=True,
#                                      clip_below_percentile=10,
#                                      overlay = True)
#
# plt.imshow(igrads_processed)
# plt.savefig('igrads_processed.png')
#
# overlay = vis.process_grads(img_processed['img'][0],
#                                      igrads[0],
#                                      morphological_cleanup=True,
#                                      clip_below_percentile=10)
#
# plt.imshow(overlay)
# plt.savefig('overlay.png')
#
# vis.visualize(img_processed['img'][0],
#                                      igrads[0],
#                                      morphological_cleanup=True,
#                                      clip_below_percentile=10)
#
#
#
#
# igrads_ill = get_igrads(model, img_processed)
# igrads_healthy = get_igrads(model, img_processed, top_pred_idx=0)
#
# vis.visualize(img_processed['img'][0], igrads_ill[0], morphological_cleanup=True, clip_below_percentile=10)
# vis.visualize(img_processed['img'][0], igrads_healthy[0], morphological_cleanup=True, clip_below_percentile=10)