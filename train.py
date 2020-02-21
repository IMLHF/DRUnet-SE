# -*- coding: utf-8 -*-

# Imports
import os
import time
import numpy as np
import tensorflow as tf
import pickle
import math

import config as cfg
from model import DRUnet
from util import cosine_distance
import data_prepare as dp
# Get training data and validation data sets
train_set = os.path.join(cfg.workdir, "packs/train_set")
train_speech_names = [os.path.join(train_set, na) for na in os.listdir(train_set) if na.lower().endswith(".p")]
val_set = os.path.join(cfg.workdir, "packs/val_set")             #
val_speech_names = [os.path.join(val_set, na) for na in os.listdir(val_set) if na.lower().endswith(".p")]

def read_pickle(file_path):
  clean, noisy = pickle.load(open(file_path.decode(), 'rb'))
  return clean, noisy

def func(file_path):
  clean, noisy = tf.py_func(read_pickle, [file_path], [tf.float32, tf.float32])
  return clean, noisy

def train(d):
  """train"""
  batch_size = d.batch_size

  #
  log_file = open(os.path.join(d.workdir, "logfile.txt"), 'w+')
  # log_device_file = open(os.path.join(d.workdir, "devicefile.log"), 'w+')
  # Model save path
  model_path = os.path.join(d.workdir, "ckpt")
  dp.create_folder(model_path)
  # initialize dataset
  with tf.name_scope('dataset'):
    trainSpeechNames = tf.placeholder(tf.string, shape=[None], name="train_speech_names")
    dataset = tf.data.Dataset.from_tensor_slices(trainSpeechNames) \
        .map(func).batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    Ref, Input = iterator.get_next()

    valSpeechNames = tf.placeholder(tf.string, shape=[None], name="val_speech_names")
    val_dataset = tf.data.Dataset.from_tensor_slices(valSpeechNames) \
        .map(func).batch(batch_size)
    val_iterator = val_dataset.make_initializable_iterator()
    val_Ref, val_Input = val_iterator.get_next()
  drunet_model_tr = DRUnet(Input, Ref, True, d)
  drunet_model_val = DRUnet(val_Ref, val_Input, False, d)

  # INITIALIZE GPU CONFIG
  config=tf.ConfigProto()
  config.gpu_options.allow_growth=True
  #config.log_device_placement = True
  sess=tf.compat.v1.Session(config=config)
  # train_writer = tf.summary.FileWriter(os.path.join(d.workdir, "log/train"), sess.graph)
  sess.run(tf.global_variables_initializer())
  sess.run(iterator.initializer, feed_dict={trainSpeechNames: train_speech_names})

  # drunet_model_tr.saver.restore(sess, os.path.join(d.workdir, "models/se_model16_15000.ckpt"))
  # sess.run(tf.assign(drunet_model_tr.lr, d.lr))

  # train_batchs = len(train_speech_names) // batch_size   # Training set batch number
  val_batchs = math.ceil(len(val_speech_names) / batch_size)       # Verification set batch number
  loss_val = np.zeros(val_batchs)

  tmp_tr_loss = 0.0
  while True:
    # TRAINING ITERATION
    try:
      _, loss_vec, gs, shapes = sess.run([drunet_model_tr.optimizer_op,
                                          drunet_model_tr.loss,
                                          drunet_model_tr.global_step,
                                          drunet_model_tr.shape_list])
      # print(shapes)
      # exit(0)
    except tf.errors.OutOfRangeError:
      np.random.seed()
      np.random.shuffle(train_speech_names)
      sess.run(iterator.initializer, feed_dict={trainSpeechNames: train_speech_names})
      continue
    tmp_tr_loss += loss_vec

    # if gs % 50 == 0:
    #   train_writer.add_summary(summary, gs)
    n_prt = 5000
    if gs % n_prt == 0:
      sess.run(val_iterator.initializer, feed_dict={valSpeechNames: val_speech_names})

      while True:
        i = 0
        try:
          val_loss = sess.run(drunet_model_val.loss)
          loss_val[i] = val_loss
          i += 1
        except tf.errors.OutOfRangeError:
          break
      val_loss_mean = np.mean(loss_val)
      mean_loss_train = tmp_tr_loss / n_prt
      tmp_tr_loss = 0.0
      print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+
            "\tbatch: %d\ttrain loss: %.4f\tvalidation loss: %.4f\n" %
            (gs, mean_loss_train, val_loss_mean))
      log_file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+
                     "\tbatch: %d\ttrain loss: %.4f\tvalidation loss: %.4f\n" %
                     (gs, mean_loss_train, val_loss_mean))
      log_file.flush()
      drunet_model_tr.saver.save(sess, os.path.join(model_path, "step_%d.ckpt" % gs))
    if gs >= d.max_train_step:
     break
  log_file.close()

if __name__ == "__main__":
  train(cfg)


# [array([16,  143, 1024,  1], dtype=int32),
#  array([16,  72,  512,  45], dtype=int32),
#  array([16,  36,  256,  90], dtype=int32),
#  array([16,  18,  128,  90], dtype=int32),
#  array([16,  9,   64,   90], dtype=int32),
#  array([16,  9,   32,   90], dtype=int32),
#  array([16,  9,   64,   90], dtype=int32),
#  array([16,  18,  128,  90], dtype=int32),
#  array([16,  36,  256,  90], dtype=int32),
#  array([16,  72,  512,  45], dtype=int32),
#  array([16,  144, 1024,  1], dtype=int32),
#  array([16,  143, 1024,  1], dtype=int32)]
