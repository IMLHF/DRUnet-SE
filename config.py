# -*- coding: utf-8 -*-

workdir = '.'

#
# Network Optimization
#
lr = 1e-3      # Initial learning rate

#
# Define short-term spectral parameters
#
sample_rate = 16000         # Sampling Rate
n_window = 1024             # Window length
stride = 64              # Step size
selection = 9152            # sequence length

#
batch_size = 16

#
max_train_step = 200000

#
feature_type = "DCT" # DCT | AbsDFT | ConcatDFT
mask_type = "IRM" # IRM | cIRM | polar_cIRM
frame_pad_end = True
