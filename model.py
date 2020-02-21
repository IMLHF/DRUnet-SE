# -*- coding: utf-8 -*-
"""
20191017
"""
# Imports
import numpy as np
import tensorflow as tf

from helper import frame
from helper import over_lap_and_add
from bn import batch_normalization
from util import cosine_distance

def prelu(I, name='prelu'):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    alphas = tf.compat.v1.get_variable('alpha', I.get_shape()[-1],
                                       initializer=tf.constant_initializer(0.0),
                                       dtype=tf.float32)
    tf.summary.histogram(name='alpha', values=alphas)

    pos = tf.nn.relu(I)
    neg = alphas * (I - tf.abs(I)) * 0.5

    Output = pos + neg
  return Output

def subsample(I, i_c, o_c, ks, stri, training=True, name='subsample'):
  # Get input shape
  # shape = tf.shape(I)
  # kernel size
  k_h, k_w = ks

  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    # Kernel initialization selection initialized to diagonal array
    W = tf.compat.v1.get_variable(name='Weight',
                                  shape=[k_h, k_w, i_c, o_c],
                                  initializer=tf.orthogonal_initializer(),
                                  dtype=tf.float32,
                                  trainable=True)
    Output = tf.nn.conv2d(I,
                          W,
                          strides=[1, stri[0], stri[1], 1],
                          padding="SAME",
                          name='conv2d')
    # layer normalization
    Output = tf.contrib.layers.layer_norm(Output, reuse=tf.AUTO_REUSE, scope='layer_norm')
    Output = prelu(Output, name='prelu')
  return Output

def upsample(I, i_c, o_c, ks, stri, training=True, last_layer=False, name='upsample'):
  # input shape
  shape = tf.shape(I)

  # output shape
  out_height = shape[1] * stri[0]    # time
  out_width = shape[2] * stri[1]     # frequency

  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    W = tf.compat.v1.get_variable(name='Weight',
                                  shape=[ks[0], ks[1], o_c, i_c],
                                  initializer=tf.orthogonal_initializer(),
                                  dtype=tf.float32,
                                  trainable=True)
    Output = tf.nn.conv2d_transpose(I,
                                    W,
                                    output_shape=[
                                        shape[0], out_height, out_width, o_c],
                                    strides=[1, stri[0], stri[1], 1],
                                    padding="SAME",
                                    name='conv2d_transpose')
    if not last_layer:
      # layer normalization
      Output = tf.contrib.layers.layer_norm(Output, reuse=tf.AUTO_REUSE, scope='layer_norm')
      Output = prelu(Output, name='prelu')
  return Output

def Analysis(signals, frame_length, frame_step):
  """
  signals: shape=[batch, height]
  return
    spec_real: tf.float32, shape=[batch, n_frames, fft_length]"""
  with tf.name_scope("Analysis"):
    # frame
    framed_signals = frame(signals, frame_length, frame_step)
    # DFT
    spec = tf.signal.dct(framed_signals, type=2, norm='ortho')

    spec_real = tf.cast(spec, tf.float32)

  return spec_real

def Synthesis(spec, frame_length, frame_step):
  """
  spec: float32, shape=[batch, n_frames, fft_length]"""
  with tf.name_scope("Synthesis"):
    # iDFT
    signal_f = tf.signal.idct(spec, type=2, norm='ortho')
    # Recovery signal
    signals = over_lap_and_add(signal_f, frame_length, frame_step)
  return signals

def tf_trunc_as(x1, x2):
  # trunc x1 as x2.shape, x1.shape must be larger than x2.shape
  # x1, x2: [batch, time, feature, channel]
  x2_shape = tf.shape(x2)
  x1_new = x1[:x2_shape[0], :x2_shape[1], :x2_shape[2], :x2_shape[3]]
  return x1_new

def tf_pad_as(x1, x2):
  # pad x1 as x2.shape, x1.shape must be smaller than x2.shape
  x1_shape = tf.shape(x1)
  x2_shape = tf.shape(x2)
  diff = tf.expand_dims(x2_shape - x1_shape, -1)
  zeros = tf.zeros_like(diff)
  padding = tf.concat([zeros, diff], axis=-1)
  x1_new = tf.pad(x1, padding)
  return x1_new

class DRUnet:
  def __init__(self, noisy_wav_batch, clean_wav_batch, is_training, hparams, **kwargs):
    self.noisy_wav_batch = noisy_wav_batch
    self.clean_wav_batch = clean_wav_batch
    self.is_training = is_training
    self.hparams = hparams

    self.enhanced_wav_batch = self._forward()

    self.save_variables = []

    if clean_wav_batch is not None:
      # cosine distance
      height = self.hparams.selection
      half_in = tf.reshape(self.noisy_wav_batch, [-1, height // 16])
      half_clean = tf.reshape(self.clean_wav_batch, [-1, height // 16])
      half_out = tf.reshape(self.enhanced_wav_batch, [-1, height // 16])
      self.loss = cosine_distance(half_clean, half_out, half_in)

      if self.is_training:
        # TRAINING OPTIMIZER
        with tf.variable_scope('not_train_vars', reuse=tf.AUTO_REUSE):
          self.global_step = tf.compat.v1.get_variable('global_step', dtype=tf.int32,
                                                       initializer=tf.constant(1), trainable=False)
          self.lr = tf.compat.v1.get_variable('lr', dtype=tf.float32, trainable=False,
                                              initializer=tf.constant(self.hparams.lr))
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
          opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr,
                                                 beta1=0., beta2=0.999, epsilon=1e-8)
        self.save_variables.append(self.global_step)
        self.save_variables.append(self.lr)
        grads = opt.compute_gradients(self.loss, var_list=[var for var in tf.compat.v1.trainable_variables()])
        self.optimizer_op = opt.apply_gradients(grads, global_step=self.global_step)

    # Model saver
    self.save_variables.extend([var for var in tf.compat.v1.trainable_variables()])
    self.saver = tf.compat.v1.train.Saver(self.save_variables,
                                          max_to_keep=25,
                                          save_relative_paths=True)

  def _forward(self):
    frame_length = self.hparams.n_window
    frame_step = self.hparams.stride
    wav_n_samples = tf.shape(self.noisy_wav_batch)[-1]
    #
    # Stage 0
    #
    spec = Analysis(self.noisy_wav_batch, frame_length, frame_step)
    #
    # Stage 1
    #
    with tf.variable_scope('Model'):
      # Increase channel dimension
      Output = tf.expand_dims(spec, axis=-1)
      Output = self._getModel(Output)
      # Remove channel dimension
      Output = tf.squeeze(Output, axis=-1)
    #
    # Stage 2
    #
    with tf.name_scope('IRM'):
      irm = tf.math.tanh(0.5 * Output) * 2.
      tf.summary.histogram(name='irm', values=irm)
      output_spec = irm * spec
    #
    # Stage 3
    #
    Output = Synthesis(output_spec, frame_length, frame_step)
    Output = tf.slice(Output, [0,0], [-1, wav_n_samples])
    # Output_n_samples = tf.shape(Output)[-1]
    # Output = tf.pad(Output, [[0, 0], [0, wav_n_samples-Output_n_samples]])
    return Output

  def _getModel(self, Input):
    """Unet-10"""
    res = list()
    self.shape_list = list()
    self.shape_list.append(tf.shape(Input))
    # (128x1024x1)
    Output = subsample(Input, 1, 45, (5,7), (2,2), self.is_training, name='subsample_1')
    self.shape_list.append(tf.shape(Output))
    res.append(Output)
    # (64x512x32)
    Output = subsample(Output, 45, 90, (5,7), (2,2), self.is_training, name='subsample_2')
    self.shape_list.append(tf.shape(Output))
    res.append(Output)
    # (32x256x64)
    Output = subsample(Output, 90, 90, (3,5), (2,2), self.is_training, name='subsample_3')
    self.shape_list.append(tf.shape(Output))
    res.append(Output)
    # (16x128x64)
    Output = subsample(Output, 90, 90, (3,5), (2,2), self.is_training, name='subsample_4')
    res.append(Output)
    self.shape_list.append(tf.shape(Output))
    # (8x64x64)
    Output = subsample(Output, 90, 90, (3,5), (1,2), self.is_training, name='subsample_5')
    self.shape_list.append(tf.shape(Output))
    # (8x32x64)
    Output = upsample(Output, 90, 90, (3,5), (1,2), self.is_training, name='upsample_5')
    self.shape_list.append(tf.shape(Output))
    # (8x64x64)
    res_tmp = res.pop()
    # Output = tf_trunc_as(Output, res_tmp)
    res_tmp = tf_pad_as(res_tmp, Output)
    Output = tf.concat([Output, res_tmp], axis=-1, name='Concat_1')
    # (8x64x128)
    Output = upsample(Output, 180, 90, (3,5), (2,2), self.is_training, name='upsample_4')
    self.shape_list.append(tf.shape(Output))
    # (16x128x64)
    res_tmp = res.pop()
    # Output = tf_trunc_as(Output, res_tmp)
    res_tmp = tf_pad_as(res_tmp, Output)
    Output = tf.concat([Output, res_tmp], axis=-1, name='Concat_2')
    # (16x128x128)
    Output = upsample(Output, 180, 90, (3,5), (2,2), self.is_training, name='upsample_3')
    self.shape_list.append(tf.shape(Output))
    # (32x256x64)
    res_tmp = res.pop()
    # Output = tf_trunc_as(Output, res_tmp)
    res_tmp = tf_pad_as(res_tmp, Output)
    Output = tf.concat([Output, res_tmp], axis=-1, name='Concat_3')
    # (32x256x128)
    Output = upsample(Output, 180, 45, (5,7), (2,2), self.is_training, name='upsample_2')
    self.shape_list.append(tf.shape(Output))
    # (64x512x32)
    res_tmp = res.pop()
    # Output = tf_trunc_as(Output, res_tmp)
    res_tmp = tf_pad_as(res_tmp, Output)
    Output = tf.concat([Output, res_tmp], axis=-1, name='Concat_4')
    # (64x512x64)
    Output = upsample(Output, 90, 1, (5,7), (2,2), self.is_training, last_layer=True, name='upsample_1')
    self.shape_list.append(tf.shape(Output))
    Output = tf_trunc_as(Output, Input)
    # (128x1024x1)
    self.shape_list.append(tf.shape(Output))

    return Output
