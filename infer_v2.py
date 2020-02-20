# -*- coding: utf-8 -*-

# Imports
import os
import sys
import numpy as np
import soundfile as sf
import tensorflow as tf
from tqdm import tqdm

import config as cfg
from model import end_to_end
import data_prepare as dp

def infer(d, ckpt):
  batch_size = 1
  # height = d.selection

  # Defining model
  Input = tf.placeholder(tf.float32, shape=[batch_size, None], name='input')
  Output = end_to_end(Input, False, d)
  # INITIALIZE GPU CONFIG
  config=tf.ConfigProto()
  config.gpu_options.allow_growth=True
  sess=tf.Session(config=config)
  # Initialization parameters
  sess.run(tf.global_variables_initializer())
  # Model reading
  saver = tf.train.Saver()
  saver.restore(sess, os.path.join(d.workdir, ckpt))
  # Read test data
  noisy_testset_wav = os.path.join(d.workdir, "data/noisy_testset_wav")
  test_speech_names = [na.split(".")[0] for na in os.listdir(noisy_testset_wav) if na.lower().endswith(".wav")]
  test_speech_names.sort()
  denoised_dir = os.path.join(d.workdir, "data/denoised")
  dp.create_folder(denoised_dir)
  #################################################################################################
  for name in tqdm(test_speech_names):
    audio_noise, _ = dp.read_audio(os.path.join(noisy_testset_wav, "%s.wav" % name))

    # n_samples = audio_noise.shape[0]
    inputData = audio_noise[np.newaxis, ...]

    # from model import Analysis
    # spec = Analysis(Input, d.n_window, d.stride)
    # tmp = sess.run(spec, feed_dict={Input: inputData})
    # print(np.shape(tmp))

    outputData = sess.run(Output, feed_dict={Input: inputData})
    outputData = np.array(outputData).squeeze()

    sf.write(os.path.join(denoised_dir, "%s.wav" % name), outputData, d.sample_rate)
if __name__ == "__main__":
  ckpt = sys.argv[1]
  infer(cfg, ckpt)
