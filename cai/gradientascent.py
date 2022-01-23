# This code is based on https://www.tensorflow.org/tutorials/generative/deepdream

import tensorflow as tf
import numpy as np

def calc_sum_channel_mean_from_model(img, model):
  # Pass forward the image through the model to retrieve the activations.
  # Converts the image into a batch of size 1.
  img_batch = tf.expand_dims(img, axis=0)
  layer_activations = model(img_batch)
  if len(layer_activations) == 1:
    layer_activations = [layer_activations]
  
  losses = []
  for act in layer_activations:
    loss = tf.math.reduce_mean(act)
    losses.append(loss)

  return  tf.reduce_sum(losses)

class GradientAscent(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(
      input_signature=(
        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32),)
  )
  def __call__(self, img, steps, step_size):
      loss = tf.constant(0.0, dtype=tf.float32)
      for n in tf.range(steps):
        with tf.GradientTape() as tape:
          # This needs gradients relative to `img`
          # `GradientTape` only watches `tf.Variable`s by default
          tape.watch(img)
          loss = tf.cast(calc_sum_channel_mean_from_model(img, self.model), dtype=tf.float32)

        # Calculate the gradient of the loss with respect to the pixels of the input image.
        gradients = tape.gradient(loss, img)

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8 
        
        # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
        # You can update the image by directly adding the gradients (because they're the same shape!)
        img = img + gradients*step_size
        img = tf.clip_by_value(img, -1, 1)
      return loss, img

def run_gradient_ascent(img, partial_model, steps=100, step_size=0.01):
  ga = GradientAscent(partial_model)
  img = tf.convert_to_tensor(img)
  step_size = tf.convert_to_tensor(step_size)
  steps_remaining = steps
  step = 0
  while steps_remaining:
    if steps_remaining>100:
      run_steps = tf.constant(100)
    else:
      run_steps = tf.constant(steps_remaining)
    steps_remaining -= run_steps
    step += run_steps
    loss, img = ga(img, run_steps, tf.constant(step_size))
  return img

def run_gradient_ascent_octaves(img, partial_model, steps=50, step_size=0.01, octave_scale=1.3, low_range=-2, high_range=3):
  img = tf.constant(np.array(img))
  base_shape = tf.shape(img)[:-1]
  float_base_shape = tf.cast(base_shape, tf.float32)

  for n in range(low_range, high_range):
    new_shape = tf.cast(float_base_shape*(octave_scale**n), tf.int32)
    img = tf.image.resize(img, new_shape).numpy()
    img = run_gradient_ascent(img=img, partial_model=partial_model, steps=steps, step_size=step_size)
  return img
