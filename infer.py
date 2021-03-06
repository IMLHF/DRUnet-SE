# -*- coding: utf-8 -*-

# Imports
import os
import sys
import math
import numpy as np
from scipy.io import wavfile
import tensorflow as tf
from tqdm import tqdm

import config as cfg
from model import DRUnet
import data_prepare as dp

def infer(d, ckpt):
  batch_size = 1
  height = d.selection
  slice_len = d.selection

  strides = slice_len // 16
  #window = np.hamming(slice_len)

  # Defining model
  Input = tf.placeholder(tf.float32, shape=[batch_size, height], name='input')
  drunet_model = DRUnet(Input, None, False, d)
  Output = drunet_model.enhanced_wav_batch
  # INITIALIZE GPU CONFIG
  config=tf.ConfigProto()
  config.gpu_options.allow_growth=True
  sess=tf.compat.v1.Session(config=config)
  # Initialization parameters
  sess.run(tf.global_variables_initializer())
  # Model reading
  saver = tf.train.Saver()
  saver.restore(sess, os.path.join(d.workdir, ckpt))
  # Read test data
  noisy_testset_wav = os.path.join(d.workdir, "data/noisy_testset_wav")
  test_speech_names = [na.split(".")[0] for na in os.listdir(noisy_testset_wav) if na.lower().endswith(".wav")]
  test_speech_names.sort()
  denoised_dir = os.path.join(d.workdir, "denoised")
  dp.create_folder(denoised_dir)
  #################################################################################################
  for name in tqdm(test_speech_names):
    audio_noise, _ = dp.read_audio(os.path.join(noisy_testset_wav, "%s.wav" % name))

    n_samples = audio_noise.shape[0]
    slice_num = math.ceil((n_samples - slice_len) / strides) + 1
    out_wav = np.zeros(n_samples)
    win = np.zeros(n_samples)
    for j in range(slice_num):
      # When the last frame is less than a long time, some sample
      # points are obtained from the previous frame to fill
      if j == slice_num - 1:
        slice_noise = audio_noise[-slice_len:]
      else:
        slice_noise = audio_noise[j * strides: j * strides + slice_len]
      #slice_noise *= window
      inputData = slice_noise[np.newaxis, ...]

      # from model import Analysis
      # spec = Analysis(Input, d.n_window, d.stride)
      # tmp = sess.run(spec, feed_dict={Input: inputData})
      # print(np.shape(tmp))

      output_slice = sess.run(Output, feed_dict={Input: inputData})
      output_slice = np.array(output_slice).squeeze()
      #output_slice /= window

      if j == slice_num - 1:
        output_slice = output_slice[j * strides - n_samples:]
        out_wav[j * strides:] += output_slice
        win[j * strides:] += np.ones(output_slice.shape[0])
      else:
        out_wav[j * strides: j * strides + slice_len] += output_slice
        win[j * strides: j * strides + slice_len] += np.ones(slice_len)
    out_wav /= win
    assert out_wav.shape[0] == n_samples
    out_wav *= (2 ** 15)
    out_wav = out_wav.astype(np.int16)
    wavfile.write(os.path.join(denoised_dir, "%s.wav" % name), d.sample_rate, out_wav)
if __name__ == "__main__":
  ckpt = sys.argv[1]
  infer(cfg, ckpt)
