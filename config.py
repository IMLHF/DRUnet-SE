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
