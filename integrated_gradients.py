import numpy as np
import tensorflow as tf
from tensorflow import keras


def get_gradients(model, img_input, top_pred_idx):
    images = tf.cast(img_input, tf.float32)
    with tf.GradientTape() as tape:
        # 'dx':
        tape.watch(images)
        # 'df':
        top_class = model({'img': tf.cast([images[0]], tf.float32), 'img_shape': tf.cast([images[0].shape], tf.float32)})['pred_obj'][0, top_pred_idx]
    # 'df/dx':
    grads = tape.gradient(top_class, images)
    return grads


def get_integrated_gradients(model, img_input, top_pred_idx, baseline=None, num_steps=50, size=(299, 299)):
    if baseline is None:
        baseline = np.zeros(size).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)

    # Interpolation:
    img_input = img_input['img'].astype(np.float32)
    interpolated_image = [
        baseline + (step / num_steps) * (img_input - baseline)
        for step in range(num_steps + 1)
        ]
    interpolated_image = np.array(interpolated_image, dtype=np.float32)


    # Calculate gradient:
    grads = []
    for i, img in enumerate(interpolated_image):
        tf.expand_dims(img, axis=0)
        grad = get_gradients(model, img, top_pred_idx)
        grads.append(grad[0])

    grads = tf.convert_to_tensor(grads, dtype=tf.float32)

    # Integrate gradient with trapezoidal rule:
    grads = (grads[:-1] + grads[1:]) / 2
    avg_grads = tf.reduce_mean(grads, axis=0)

    # Caluclate integrated gradients:
    integrated_grads = (img_input - baseline) * avg_grads
    return integrated_grads


def random_baseline_integrated_gradients(
    model, img_input, top_pred_idx, size, num_steps=50, num_runs=2
):
    """Generates a number of random baseline images.

    Args:
        img_input (ndarray): 3D image
        top_pred_idx: Predicted label for the input image
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.
        num_runs: number of baseline images to generate

    Returns:
        Averaged integrated gradients for `num_runs` baseline images
        :param img_input:
        :param model:
    """
    # 1. List to keep track of Integrated Gradients (IG) for all the images
    integrated_grads = []

    # 2. Get the integrated gradients for all the baselines
    for run in range(num_runs):
        baseline = np.random.random(size) * 255
        igrads = get_integrated_gradients(
            model=model,
            img_input=img_input,
            top_pred_idx=top_pred_idx,
            baseline=baseline,
            num_steps=num_steps,
        )
        integrated_grads.append(igrads)

    # 3. Return the average integrated gradients for the image
    integrated_grads = tf.convert_to_tensor(integrated_grads)
    return tf.reduce_mean(integrated_grads, axis=0)